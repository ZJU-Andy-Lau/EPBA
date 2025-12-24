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
from model.gru_mf import GRUBlock
from model.ctx_decoder import ContextDecoder
from criterion.train_loss import Loss
from scheduler import MultiStageOneCycleLR
from shared.utils import str2bool,feats_pca,vis_conf,get_current_time,check_grad,load_model_state_dict
import shared.visualize as visualizer # 引入新的可视化模块
from solve.solve_windows import WindowSolver

def print_on_main(msg, rank):
    if rank == 0:
        print(msg)

def deb_print(msg):
    rank = dist.get_rank()
    print(f"[Rank {rank}]:{msg}")

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
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
    gru = GRUBlock(corr_levels=2,corr_radius=4,context_dim=128,hidden_dim=128)
    ctx_decoder = ContextDecoder(ctx_dim=128)
    
    adapter_optimizer = optim.AdamW(params = list(encoder.adapter.parameters()) + list(ctx_decoder.parameters()),lr = args.lr_encoder_max) # 同时优化adapter和ctx_decoder
    gru_optimizer = optim.AdamW(params = gru.parameters(),lr = args.lr_gru_max)
    
    if not args.adapter_path is None:
        encoder.load_adapter(os.path.join(args.adapter_path))
        pprint("Encoder Loaded")
    
    if not args.gru_path is None:
        load_model_state_dict(gru,args.gru_path)
        pprint("GRU Loaded")
    
    if not args.decoder_path is None:
        load_model_state_dict(ctx_decoder,args.decoder_path)
        pprint("Decoder Loaded")
    
    encoder = encoder.to(args.device)
    gru = gru.to(args.device)
    ctx_decoder = ctx_decoder.to(args.device)
    
    for state in adapter_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)
    for state in gru_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)
    
    encoder = distibute_model(encoder,args.local_rank)
    gru = distibute_model(gru,args.local_rank)
    ctx_decoder = distibute_model(ctx_decoder,args.local_rank)
    
    return encoder,gru,ctx_decoder,adapter_optimizer,gru_optimizer

def get_loss(args,encoder:Encoder,gru:GRUBlock,ctx_decoder:ContextDecoder,data,loss_funcs:Loss,epoch,get_debuf_info = False):
    imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b = data
    imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b = [i.squeeze(0).to(device = args.device,dtype = torch.float32) for i in [imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,Hs_a,Hs_b,M_a_b]]

    B,H,W = imgs1_train.shape[0],imgs1_train.shape[-2],imgs1_train.shape[-1]

    feats_1,feats_2 = encoder(imgs1_train,imgs2_train)
    match_feats_1,ctx_feats_1,confs_1 = feats_1
    match_feats_2,ctx_feats_2,confs_2 = feats_2

    imgs_pred_1 = ctx_decoder(ctx_feats_1)
    imgs_pred_2 = ctx_decoder(ctx_feats_2)
    
    windowsolver = WindowSolver(B,H,W,gru,feats_1,feats_2,Hs_a,Hs_b,gru_max_iter=args.gru_max_iter)
    
    preds_ab = windowsolver.solve(flag = 'ab')
    preds_ba = windowsolver.solve(flag = 'ba')

    loss_input = {
        'epoch':epoch,
        'max_epoch':args.max_epoch,
        'imgs_1':imgs1_label,
        'imgs_2':imgs2_label,
        'feats_1':feats_1,
        'feats_2':feats_2,
        'preds_1':preds_ab,
        'preds_2':preds_ba,
        'residual_1':residual1,
        'residual_2':residual2,
        'imgs_pred_1':imgs_pred_1,
        'imgs_pred_2':imgs_pred_2,
        'Hs_a':Hs_a,
        'Hs_b':Hs_b,
        'M_a_b':M_a_b,
        'norm_factor_a':windowsolver.norm_factors_a,
        'norm_factor_b':windowsolver.norm_factors_b,
    }

    # [修改] 传递 get_debuf_info 给 Loss，获取 Affine Loss 的详细信息 (trajectory)
    loss,loss_details,extra_info = loss_funcs(loss_input, return_details=get_debuf_info)

    debug_info = {
        'imgs': {},
        'values': {}
    }

    if get_debuf_info:
        # =================== Part 1: 恢复原有的 Basic Visuals ===================
        # train_imgs
        train_img_1 = imgs1_train[0].permute(1,2,0).detach().cpu().numpy()
        train_img_2 = imgs2_train[0].permute(1,2,0).detach().cpu().numpy()
        train_img_1 = 255. * (train_img_1 - train_img_1.min()) / (train_img_1.max() - train_img_1.min())
        train_img_2 = 255. * (train_img_2 - train_img_2.min()) / (train_img_2.max() - train_img_2.min())
        train_img_1 = train_img_1.astype(np.uint8)
        train_img_2 = train_img_2.astype(np.uint8)

        # pred_imgs
        img_pred_1 = imgs_pred_1[0].permute(1,2,0).detach().cpu().numpy()
        img_pred_2 = imgs_pred_2[0].permute(1,2,0).detach().cpu().numpy()
        img_pred_1 = 255. * (img_pred_1 - img_pred_1.min()) / (img_pred_1.max() - img_pred_1.min())
        img_pred_2 = 255. * (img_pred_2 - img_pred_2.min()) / (img_pred_2.max() - img_pred_2.min())
        img_pred_1 = img_pred_1.astype(np.uint8)
        img_pred_2 = img_pred_2.astype(np.uint8)

        # match_feats
        match_feat_1 = match_feats_1[0].permute(1,2,0).detach().cpu().numpy()
        match_feat_2 = match_feats_2[0].permute(1,2,0).detach().cpu().numpy()
        match_feat_pca = feats_pca(np.stack([match_feat_1,match_feat_2],axis=0))
        match_feat_img_1 = match_feat_pca[0]
        match_feat_img_2 = match_feat_pca[1]

        # ctx_feats
        ctx_feat_1 = ctx_feats_1[0].permute(1,2,0).detach().cpu().numpy()
        ctx_feat_2 = ctx_feats_2[0].permute(1,2,0).detach().cpu().numpy()
        ctx_feat_pca = feats_pca(np.stack([ctx_feat_1,ctx_feat_2],axis=0))
        ctx_feat_img_1 = ctx_feat_pca[0]
        ctx_feat_img_2 = ctx_feat_pca[1]

        # conf
        conf_1 = confs_1[0][0].detach().cpu().numpy()
        conf_2 = confs_2[0][0].detach().cpu().numpy()
        _,conf_img_1 = vis_conf(conf_1,train_img_1,16)
        _,conf_img_2 = vis_conf(conf_2,train_img_2,16)

        debug_info['imgs'].update({
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
        })
        
        # =================== Part 2: 收集全场景透明化所需数据 ===================
        # 取 Batch 0
        vis_data = {
            'img_a': imgs1_label[0],     # Tensor (3, H, W) - 原始归一化图
            'img_b': imgs2_label[0],     # Tensor (3, H, W)
            'feat_a': match_feats_1[0],   # Tensor (C, H, W)
            'feat_b': match_feats_2[0],   # Tensor (C, H, W)
            'conf_a': confs_1[0][0],      # Tensor (H, W)
            'gt_affine': M_a_b,        # Tensor (2, 3) RC
            # [新增] 记录 H_a 和 H_b 用于全局坐标恢复
            'H_a': Hs_a[0],               # Tensor (3, 3) RC
            'H_b': Hs_b[0],               # Tensor (3, 3) RC
        }
        if 'affine_details' in extra_info:
            # (Steps+1, 2, 3)
            vis_data['pred_affines_list'] = extra_info['affine_details']['pred_affines_list'][0]
            
        debug_info['vis_data'] = vis_data

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
    batch_num = len(dataloader)

    encoder,gru,ctx_decoder,adapter_optimizer,gru_optimizer = load_models(args)

    adapter_scheduler = MultiStageOneCycleLR(optimizer=adapter_optimizer,
                                             total_steps=args.max_epoch * batch_num,
                                             warmup_ratio=min(5. / args.max_epoch,.1),
                                             cooldown_ratio=max((1. - .5 / args.max_epoch, .9)) - (5. / args.max_epoch))
    
    gru_scheduler = MultiStageOneCycleLR(optimizer=gru_optimizer,
                                         total_steps=args.max_epoch * batch_num,
                                         warmup_ratio=min(5. / args.max_epoch,.1),
                                         cooldown_ratio=max((1. - .5 / args.max_epoch, .9)) - (5. / args.max_epoch))

    loss_funcs = Loss(img_size = (dataset.input_size,dataset.input_size),
                      downsample_factor = dataset.DOWNSAMPLE,
                      temperature = 0.07,
                      decay_rate = 0.8,
                      reg_weight = 1e-3,
                      parallax_border = (args.parallax_border_left,args.parallax_border_right),
                      device = args.device)

    start_time = time.perf_counter()
    step_count = 0
    
    # [新增] 定义可视化频率
    VIS_FREQ = 5

    for epoch in range(args.max_epoch):
        pprint(f'\nEpoch:{epoch}')
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        records = {
            "loss":0,
            "loss_sim":0,
            "loss_conf":0,
            "loss_affine":0,
            "loss_affine_last":0,
            "loss_consist":0,
            "loss_ctx":0,
        }
        encoder.train()
        gru.train()

        for batch_idx,data in enumerate(dataloader):
            adapter_optimizer.zero_grad()
            gru_optimizer.zero_grad()

            # [修改] 触发条件：Rank 0, 指定 Epoch, 第一个 Batch
            is_vis_step = (rank == 0) and (epoch % VIS_FREQ == 0) and (batch_idx == 0)

            loss,loss_details,debug_info = get_loss(args,encoder,gru,ctx_decoder,data,loss_funcs,epoch, get_debuf_info = is_vis_step)

            loss_is_nan = not torch.isfinite(loss).all()
            loss_status_tensor = torch.tensor([loss_is_nan], dtype=torch.float32, device=rank)
            dist.all_reduce(loss_status_tensor, op=dist.ReduceOp.SUM)
            if loss_status_tensor.item() > 0:
                pprint(f"--- 检测到 NaN！Epoch {epoch}, batch {batch_idx}. 所有进程将一起跳过此次更新。---")
                del loss,loss_details
                adapter_scheduler.step()
                gru_scheduler.step()
                continue 
            
            loss.backward()

            adpater_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.module.adpater.parameters(), max_norm=1.0)
            gru_grad_norm = torch.nn.utils.clip_grad_norm_(gru.parameters(), max_norm=1.0)

            print(f"adapter grad norm:{adpater_grad_norm:.6f} \t gru grad norm:{gru_grad_norm:.6f}")

            adapter_optimizer.step()
            gru_optimizer.step()

            adapter_scheduler.step()
            gru_scheduler.step()

            for key in loss_details.keys():
                dist.all_reduce(loss_details[key],dist.ReduceOp.AVG)
            dist.barrier()
            for key in loss_details.keys():
                records[key] += loss_details[key]
            step_count += 1

            # [新增] 在主进程处理可视化逻辑 (不影响训练流)
            if is_vis_step and rank == 0:
                vis_data = debug_info.get('vis_data', None)
                if vis_data:
                    # 数据准备
                    img_a_np = visualizer.denormalize_image(vis_data['img_a'])
                    img_b_np = visualizer.denormalize_image(vis_data['img_b'])
                    
                    # [修改] 提取所需的矩阵 (转为 numpy)
                    gt_aff_rc = vis_data['gt_affine'].detach().cpu().numpy()
                    pred_affs_rc = vis_data['pred_affines_list'].detach().cpu().numpy()
                    final_pred_rc = pred_affs_rc[-1]
                    H_a_rc = vis_data['H_a'].detach().cpu().numpy()
                    H_b_rc = vis_data['H_b'].detach().cpu().numpy()
                    
                    feat_a_np = vis_data['feat_a'].detach().cpu().numpy()
                    feat_b_np = vis_data['feat_b'].detach().cpu().numpy()
                    conf_np = vis_data['conf_a'].detach().cpu().numpy()

                    # --- Panel A: 配准全景 (使用 Global Warp) ---
                    target_wh = (img_b_np.shape[1], img_b_np.shape[0])
                    
                    # 1. Origin (Aligned by GT)
                    # 使用 GT 矩阵将 Image A 变换到 Image B
                    img_a_aligned_gt = visualizer.warp_image_by_global_affine(
                        img_a_np, H_a_rc, H_b_rc, gt_aff_rc, target_wh
                    )
                    strip_origin = visualizer.vis_registration_strip(img_a_aligned_gt, img_b_np, "Origin (Aligned by GT)")
                    logger.add_image('Visual_Adv/A1_Origin_Alignment', strip_origin, epoch, dataformats='HWC')
                    
                    # 2. Training Input (Misaligned)
                    # 这里的 "Training Input" 展示原始的小图
                    img_a_misaligned = visualizer.warp_image_by_global_affine(
                        img_a_np, H_a_rc, H_b_rc, np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]]), target_wh
                    )
                    strip_input = visualizer.vis_registration_strip(img_a_misaligned, img_b_np, "Training Input (Raw Crops)")
                    logger.add_image('Visual_Adv/A2_Training_Input', strip_input, epoch, dataformats='HWC')
                    
                    # 3. Prediction (Aligned by Pred)
                    img_a_aligned_pred = visualizer.warp_image_by_global_affine(
                        img_a_np, H_a_rc, H_b_rc, final_pred_rc, target_wh
                    )
                    strip_pred = visualizer.vis_registration_strip(img_a_aligned_pred, img_b_np, "Prediction (Corrected)")
                    logger.add_image('Visual_Adv/A3_Prediction_Result', strip_pred, epoch, dataformats='HWC')

                    # --- Panel B: 特征匹配 ---
                    img_match = visualizer.vis_sparse_match(img_a_np, img_b_np, feat_a_np, feat_b_np, conf_np)
                    logger.add_image('Visual_Adv/B_Feature_Matching', img_match, epoch, dataformats='HWC')

                    # --- Panel C: 置信度 & 响应 ---
                    img_conf_over = visualizer.vis_confidence_overlay(img_a_np, conf_np)
                    logger.add_image('Visual_Adv/C1_Confidence_Overlay', img_conf_over, epoch, dataformats='HWC')
                    
                    img_resp = visualizer.vis_pyramid_response(feat_a_np, feat_b_np, level_num=gru.module.corr_levels)
                    logger.add_image('Visual_Adv/C2_Pyramid_Response', img_resp, epoch, dataformats='HWC')

                    # --- Panel D: 轨迹 (使用 Fixed Size + Zoom Inset) ---
                    H, W = img_a_np.shape[:2]
                    # [修改] 传入 H_a_rc 进行坐标还原
                    img_traj = visualizer.vis_trajectory_fixed_size(pred_affs_rc, gt_aff_rc, H_a_rc, H, W)
                    logger.add_image('Visual_Adv/D_Iterative_Trajectory', img_traj, epoch, dataformats='HWC')

                    # [恢复] 将原有的 Debug 图像写入 Visual_Basic
                    if 'imgs' in debug_info:
                        for key in debug_info['imgs']:
                            logger.add_image(f"Visual_Basic/{key}", debug_info['imgs'][key], epoch, dataformats='HWC')

            if rank == 0:
                curtime = time.perf_counter()
                curstep = step_count
                remain_step = (args.max_epoch - epoch)  * batch_num - batch_idx - 1
                cost_time = curtime - start_time
                remain_time = remain_step * cost_time / curstep
                info = (
                    f"epoch:{epoch} "
                    f"batch:{batch_idx+1} / {batch_num} \t"
                    f"l_sim:{loss_details['loss_sim'].item():.2f} \t"
                    f"l_conf:{loss_details['loss_conf'].item():.2f} \t"
                    f"l_af:{loss_details['loss_affine'].item():.2f} / {loss_details['loss_affine_last'].item():.2f} \t"
                    f"l_cons:{loss_details['loss_consist'].item():.2f} \t"
                    f"l_ctx:{loss_details['loss_ctx'].item():.2f} \t"
                    f"lr_enc:{adapter_scheduler.get_last_lr()[0]:.2e} "
                    f"lr_gru:{gru_scheduler.get_last_lr()[0]:.2e} \t"
                    f"time:{str(datetime.timedelta(seconds=round(cost_time)))} "
                    f"ETA:{str(datetime.timedelta(seconds=round(remain_time)))} \t"
                )
                print(info)

        if rank == 0:
            for key in loss_details.keys():
                records[key] /= batch_num
            info = (
                    f"epoch:{epoch} \t"
                    f"loss:{records['loss'].item():.2f} \t"
                    f"l_sim:{records['loss_sim'].item():.2f} \t"
                    f"l_conf:{records['loss_conf'].item():.2f} \t"
                    f"l_af:{records['loss_affine'].item():.2f} / {records['loss_affine_last'].item():.2f} \t"
                    f"l_cons:{records['loss_consist'].item():.2f} \t"
                    f"l_ctx:{records['loss_ctx'].item():.2f} \t"
                    f"min_loss:{min_loss:.2f} \t"
                )
            print(info)

            for key in records:
                logger.add_scalar(f"loss/{key}",records[key].item(),epoch)
            logger.add_scalar(f"lr/adp_lr",adapter_scheduler.get_lr()[0],epoch)
            logger.add_scalar(f"lr/gru_lr",gru_scheduler.get_lr()[0],epoch)

            
            if 'values' in debug_info:
                for key in debug_info['values']:
                    print(f"{key} : {debug_info['values'][key]}")
        
            if records['loss_affine_last'] < min_loss:
                min_loss = records['loss_affine_last']
                encoder.module.save_adapter(os.path.join(args.model_save_path,'adapter.pth'))
                torch.save(gru.state_dict(),os.path.join(args.model_save_path,'gru.pth'))
                torch.save(ctx_decoder.state_dict(),os.path.join(args.model_save_path,'ctx_decoder.pth'))
                print("Best Updated")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--adapter_path',type=str,default=None)
    parser.add_argument('--gru_path',type=str,default=None)
    parser.add_argument('--decoder_path',type=str,default=None)
    parser.add_argument('--model_save_path',type=str,default=f'./weights/{get_current_time()}')
    parser.add_argument('--checkpoints_path',type=str,default=None)
    parser.add_argument('--vis_img_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--gru_max_iter',type=int,default=10)
    parser.add_argument('--resume_training',type=str2bool,default=False)
    parser.add_argument('--max_epoch',type=int,default=1000)
    parser.add_argument('--parallax_border_left',type=float,default=2.0)
    parser.add_argument('--parallax_border_right',type=float,default=10.0)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=1e-3)
    parser.add_argument('--lr_gru_min',type=float,default=1e-7)
    parser.add_argument('--lr_gru_max',type=float,default=1e-3) 
    parser.add_argument('--min_loss',type=float,default=1e8)
    parser.add_argument('--log_prefix',type=str,default='')
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()
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