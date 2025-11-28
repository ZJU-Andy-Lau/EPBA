import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from scipy import datasets
import h5py
import datetime
import time
from tqdm import tqdm,trange
from skimage.transform import AffineTransform
from skimage.measure import ransac
import random
from functools import partial
import cv2
from copy import deepcopy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset,DataLoader,DistributedSampler

from torchvision import transforms

from load_data import TrainDataset, ImageSampler
from model.encoder import Encoder
from model.gru import GRUBlock
from model.ctx_decoder import ContextDecoder
from criterion.pretrain_loss import Loss
from scheduler import MultiStageOneCycleLR
from utils.utils import str2bool,feats_pca,vis_conf,get_current_time,check_grad
from solve.solve_windows import Windows

def print_on_main(msg, rank):
    if rank == 0:
        print(msg)

def deb_print(msg):
    rank = dist.get_rank()
    print(f"[Rank {rank}]:{msg}")

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
    return model

def load_model_state_dict(model:nn.Module,state_dict_path:str):
    state_dict = torch.load(state_dict_path,map_location='cpu')
    state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def load_data(args):
    #加载数据
    pprint("Loading Dataset")
    rank = dist.get_rank()
    if not args.dataset_select is None:
        args.dataset_num = len(args.dataset_select.split(','))
    dataset_indices = torch.empty(args.dataset_num,dtype=torch.long,device=args.device)
    if rank == 0:
        with h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r') as f:
            total_num = len(f.keys())
        if args.dataset_select is None:
            dataset_indices = torch.randperm(total_num)[:args.dataset_num].to(args.device)
        else:
            dataset_indices = torch.tensor([int(i) for i in args.dataset_select.split(',')],dtype=int,device=args.device)
    dist.barrier()
    dist.broadcast(dataset_indices,src=0)
    dataset_indices = dataset_indices.cpu().numpy()

    dataset = TrainDataset(root = args.dataset_path,
                           dataset_idxs=dataset_indices,
                           batch_size = args.batch_size,
                           downsample = 16,
                           input_size = 512,
                           norm_coefs={
                                'mean':(0.485, 0.456, 0.406),
                                'std':(0.229, 0.224, 0.225)
                           },
                           mode='train')
    # sampler = ImageSampler(dataset,shuffle=True)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=1,num_workers=4,drop_last=False,pin_memory=False,shuffle=False)

    return dataset,dataloader,sampler

def load_models(args):
    pprint("Loading Models")
    
    encoder = Encoder(dino_weight_path = args.dino_weight_path,embed_dim=256,ctx_dim=128)
    ctx_decoder = ContextDecoder(ctx_dim=128)
    
    adapter_optimizer = optim.AdamW(params = list(encoder.adapter.parameters()) + list(ctx_decoder.parameters()),lr = args.lr_encoder_max) # 同时优化adapter和ctx_decoder
    
    if not args.adapter_path is None:
        encoder.load_adapter(os.path.join(args.adapter_path))
        pprint("Adapter Loaded")

    if not args.decoder_path is None:
        load_model_state_dict(ctx_decoder,args.decoder_path)
        pprint("Decoder Loaded")
    
    encoder = encoder.to(args.device)
    ctx_decoder = ctx_decoder.to(args.device)
    
    for state in adapter_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)
    
    encoder = distibute_model(encoder,args.local_rank)
    ctx_decoder = distibute_model(ctx_decoder,args.local_rank)
    
    return encoder,ctx_decoder,adapter_optimizer

def get_loss(args,encoder:Encoder,ctx_decoder:ContextDecoder,data,loss_funcs:Loss,epoch,get_debuf_info = False):
    imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b = data
    imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b = [i.squeeze(0).to(device = args.device,dtype = torch.float32) for i in [imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b]]

    feats_1,feats_2 = encoder(imgs1_train,imgs2_train)
    match_feats_1,ctx_feats_1,confs_1 = feats_1
    match_feats_2,ctx_feats_2,confs_2 = feats_2

    imgs_pred_1 = ctx_decoder(ctx_feats_1)
    imgs_pred_2 = ctx_decoder(ctx_feats_2)

    loss_input = {
        'imgs_1':imgs1_label,
        'imgs_2':imgs2_label,
        'feats_1':feats_1,
        'feats_2':feats_2,
        'residual_1':residual1,
        'residual_2':residual2,
        'imgs_pred_1':imgs_pred_1,
        'imgs_pred_2':imgs_pred_2,
        'Hs_a':Hs_a,
        'Hs_b':Hs_b,
        'M_a_b':M_a_b
    }

    loss,loss_details = loss_funcs(loss_input)

    if get_debuf_info:
        #====================准备debug info===========================
        #train_imgs
        train_img_1 = imgs1_train[0].permute(1,2,0).detach().cpu().numpy()
        train_img_2 = imgs2_train[0].permute(1,2,0).detach().cpu().numpy()
        train_img_1 = 255. * (train_img_1 - train_img_1.min()) / (train_img_1.max() - train_img_1.min())
        train_img_2 = 255. * (train_img_2 - train_img_2.min()) / (train_img_2.max() - train_img_2.min())
        train_img_1 = train_img_1.astype(np.uint8)
        train_img_2 = train_img_2.astype(np.uint8)

        #pred_imgs
        img_pred_1 = imgs_pred_1[0].permute(1,2,0).detach().cpu().numpy()
        img_pred_2 = imgs_pred_2[0].permute(1,2,0).detach().cpu().numpy()
        img_pred_1 = 255. * (img_pred_1 - img_pred_1.min()) / (img_pred_1.max() - img_pred_1.min())
        img_pred_2 = 255. * (img_pred_2 - img_pred_2.min()) / (img_pred_2.max() - img_pred_2.min())
        img_pred_1 = img_pred_1.astype(np.uint8)
        img_pred_2 = img_pred_2.astype(np.uint8)

        #match_feats
        match_feat_1 = match_feats_1[0].permute(1,2,0).detach().cpu().numpy()
        match_feat_2 = match_feats_2[0].permute(1,2,0).detach().cpu().numpy()
        match_feat_pca = feats_pca(np.stack([match_feat_1,match_feat_2],axis=0))
        match_feat_img_1 = match_feat_pca[0]
        match_feat_img_2 = match_feat_pca[1]

        #ctx_feats
        ctx_feat_1 = ctx_feats_1[0].permute(1,2,0).detach().cpu().numpy()
        ctx_feat_2 = ctx_feats_2[0].permute(1,2,0).detach().cpu().numpy()
        ctx_feat_pca = feats_pca(np.stack([ctx_feat_1,ctx_feat_2],axis=0))
        ctx_feat_img_1 = ctx_feat_pca[0]
        ctx_feat_img_2 = ctx_feat_pca[1]

        #conf
        conf_1 = confs_1[0][0].detach().cpu().numpy()
        conf_2 = confs_2[0][0].detach().cpu().numpy()
        _,conf_img_1 = vis_conf(conf_1,train_img_1,16)
        _,conf_img_2 = vis_conf(conf_2,train_img_2,16)


        debug_info ={
            'imgs':{
                'train_img_1':train_img_1,
                'train_img_2':train_img_2,
                'img_pred_1':img_pred_1,
                'img_pred_2':img_pred_2,
                'match_feat_img_1':match_feat_img_1,
                'match_feat_img_2':match_feat_img_2,
                'ctx_feat_img_1':ctx_feat_img_1,
                'ctx_feat_img_2':ctx_feat_img_2,
                'conf_img_1':conf_img_1,
                'conf_img_2':conf_img_2
            },
            'values':{
            }
        }

        #=============================================================
    else:
        debug_info = {
            "imgs":{},
            "values":{}
        }

    return loss,loss_details,debug_info     


def main(args):
    os.makedirs('./log',exist_ok=True)
    os.makedirs(args.model_save_path,exist_ok=True)
    os.makedirs(args.checkpoints_path,exist_ok=True)
    pprint = partial(print_on_main, rank=dist.get_rank())
    num_gpus = dist.get_world_size()
    rank = dist.get_rank()
    pprint(f"Using {num_gpus} GPUS")

    min_loss = args.min_loss
    epoch = 0
    log_name = args.log_prefix

    #构建logger
    if rank == 0:
        logger = SummaryWriter(log_dir=os.path.join('./log',f'{log_name}_tensorboard'))
    else:
        logger = None

    dataset,dataloader,sampler = load_data(args)
    
    args.dataset_num = dataset.dataset_num
    # train_images = dataset.get_train_images()
    # if rank == 0:
    #     for i,img in enumerate(train_images):
    #         tag = f'train_imgs/{i}'
    #         logger.add_image(tag,img,0,dataformats='HWC')

    encoder,ctx_decoder,adapter_optimizer = load_models(args)

    adapter_scheduler = MultiStageOneCycleLR(optimizer=adapter_optimizer,
                                             total_steps=args.max_epoch * len(dataloader),
                                             warmup_ratio=min(5. / args.max_epoch,.1),
                                             cooldown_ratio=.9)

    loss_funcs = Loss(downsample_factor = dataset.DOWNSAMPLE,
                      temperature = 0.07)

    for epoch in range(args.max_epoch):
        pprint(f'\nEpoch:{epoch}')
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        records = {
            "loss":0,
            "loss_sim":0,
            "loss_conf":0,
            "loss_ctx":0,
            "count":0
        }
        encoder.train()

        for batch_idx,data in enumerate(dataloader):
            adapter_optimizer.zero_grad()

            loss,loss_details,debug_info = get_loss(args,encoder,ctx_decoder,data,loss_funcs,epoch,get_debuf_info = (epoch % 5 == 0 and batch_idx == len(dataloader) - 1))

            loss_is_nan = not torch.isfinite(loss).all()
            loss_status_tensor = torch.tensor([loss_is_nan], dtype=torch.float32, device=rank)
            dist.all_reduce(loss_status_tensor, op=dist.ReduceOp.SUM)
            if loss_status_tensor.item() > 0:
                pprint(f"--- 检测到 NaN！Epoch {epoch}, batch {batch_idx}. 所有进程将一起跳过此次更新。---")
                del loss,loss_details
                adapter_scheduler.step()
                # backbone_scheduler.step()
                continue 
            
            loss.backward()

            adapter_optimizer.step()

            adapter_scheduler.step()

            for key in loss_details.keys():
                dist.all_reduce(loss_details[key],dist.ReduceOp.AVG)
            dist.barrier()
            for key in loss_details.keys():
                records[key] += loss_details[key]
            records['count'] += 1

            if rank == 0:
                info = (
                    f"epoch:{epoch} \t"
                    f"batch:{batch_idx+1} / {len(dataloader)} \t"
                    f"loss:{loss_details['loss'].item():.2f} \t"
                    f"l_sim:{loss_details['loss_sim'].item():.2f} \t"
                    f"l_cons:{loss_details['loss_consist'].item():.2f} \t"
                    f"l_ctx:{loss_details['loss_ctx'].item():.2f} \t"
                    f"lr_encoder:{adapter_scheduler.get_last_lr()[0]:.2e} \t"
                )
                print(info)

        if rank == 0:
            for key in loss_details.keys():
                records[key] /= records['count']
            info = (
                    f"epoch:{epoch} \t"
                    f"loss:{records['loss'].item():.2f} \t"
                    f"l_sim:{records['loss_sim'].item():.2f} \t"
                    f"l_conf:{records['loss_conf'].item():.2f} \t"
                    f"l_ctx:{records['loss_ctx'].item():.2f} \t"
                    f"min_loss:{min_loss:.2f} \t"
                )
            print(info)

            for key in records:
                if key == 'count':
                    continue
                logger.add_scalar(f"loss/{key}",records[key].item(),epoch)

            for key in debug_info['imgs']:
                logger.add_image(f"imgs/{key}",debug_info['imgs'][key],epoch,dataformats='HWC')
            
            for key in debug_info['values']:
                print(f"{key} : {debug_info['values'][key]}")
            
            # 保存模型
            if loss < min_loss:
                min_loss = loss
                encoder.module.save_adapter(os.path.join(args.model_save_path,'adapter.pth'))
                torch.save(ctx_decoder.state_dict(),os.path.join(args.model_save_path,'ctx_decoder.pth'))
                print("Best Updated")


        
            

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--adapter_path',type=str,default=None)
    parser.add_argument('--decoder_path',type=str,default=None)
    parser.add_argument('--model_save_path',type=str,default=f'./weights/{get_current_time()}')
    parser.add_argument('--checkpoints_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--resume_training',type=str2bool,default=False)
    parser.add_argument('--max_epoch',type=int,default=1000)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=1e-3)
    parser.add_argument('--min_loss',type=float,default=1e8)
    parser.add_argument('--log_prefix',type=str,default='')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()
    # gpus = os.environ['CUDA_VISIBLE_DEVICES']
    # args.multi_gpu = len(gpus.split(',')) > 1

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    pprint = partial(print_on_main, rank=dist.get_rank())

    pprint("==============================configs==============================")
    for k,v in vars(args).items():
        pprint(f"{k}:{v}")
    pprint("===================================================================")
    main(args)



