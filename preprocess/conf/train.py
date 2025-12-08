import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import cv2
import h5py
import numpy as np
import random
from functools import partial

from preprocess.conf.model import ConfHead
from scheduler import MultiStageOneCycleLR
from shared.utils import get_current_time

def print_on_main(msg, rank):
    if rank == 0:
        print(msg)

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
    return model

def load_model_state_dict(model:nn.Module,state_dict_path:str):
    state_dict = torch.load(state_dict_path,map_location='cpu')
    state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def get_conf_label(residual):
    conf_label = torch.full(residual.shape,.5,device=residual.device,dtype=residual.dtype)
    valid_residual = residual[residual >= 0]
    res_mid = torch.median(valid_residual)
    conf_label[residual > res_mid] = .1
    conf_label[(residual <= res_mid) & (residual >= 0)] = .9
    conf_label[residual < 0] = .1
    return conf_label

class TrainDataset(Dataset):
    def __init__(self,root,downsample = 16,input_size = 512,batch_size = 1):
        super().__init__()
        self.root = root
        self.database = h5py.File(os.path.join(self.root,'train_data.h5'),'r')
        self.database_keys = list(self.database.keys())
        self.DONWSAMPLE = downsample
        self.img_size = self.database[self.database_keys[0]]['images']['image_0'][:].shape[0]
        self.input_size = input_size
        self.batch_size = batch_size
        self.rank = dist.get_rank()
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.epoch = 0

    def get_random_crop_box(self, H, W, min_size, max_size):
        if min_size > H or min_size > W:
            raise ValueError(f"min_size ({min_size}) 不能大于图像尺寸 (H={H}, W={W})")
        real_max_size = min(H, W, max_size)
        if real_max_size < min_size:
            raise ValueError(f"调整后的 max_size ({real_max_size}) 小于 min_size ({min_size})")
        size = random.randint(min_size, real_max_size)
        min_row = random.randint(0, H - size)
        min_col = random.randint(0, W - size)
        max_row = min_row + size
        max_col = min_col + size

        return min_row, min_col, max_row, max_col
    
    def residual_average(self, arr:np.ndarray, a:int) -> np.ndarray:
        is_2d = arr.ndim == 2
        if is_2d:
            arr = arr[..., np.newaxis]
        H, W, C = arr.shape
        new_H = ((H + a - 1) // a) * a
        new_W = ((W + a - 1) // a) * a
        padded = np.pad(arr, ((0, new_H - H), (0, new_W - W), (0, 0)),constant_values=np.nan)
        reshaped = padded.reshape(new_H//a, a, new_W//a, a, C)
        output = np.nanmedian(reshaped,axis=(1,3))
        if is_2d:
            output = output.squeeze(axis=-1)
        output[np.isnan(output)] = -1.
        return output
        
    def set_epoch(self,epoch):
        self.epoch = epoch
        
    def __getitem__(self, index):
        seed = int(index + self.epoch * 10000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        key = self.database_keys[index]
        img_num = len(self.database[key]['images'])
        img_idx = np.random.choice(img_num,1)[0]
        img_full = self.database[key]['images'][f"image_{img_idx}"][:]
        res_full = self.database[key]['residuals'][f"residual_{img_idx}"][:]
        imgs = []
        residuals = []
        for k in range(self.batch_size):
            tlr,tlc,brr,brc = self.get_random_crop_box(self.img_size,self.img_size,128,2048)
            img_crop = cv2.resize(img_full[tlr:brr,tlc:brc],(self.input_size,self.input_size),interpolation=cv2.INTER_LINEAR)
            img_crop = np.stack([img_crop]*3,axis=-1)
            res_crop = cv2.resize(res_full[tlr:brr,tlc:brc],(self.input_size,self.input_size),
                                interpolation=cv2.INTER_NEAREST)
            imgs.append(img_crop)
            residuals.append(res_crop)
        
        imgs = torch.stack([self.transform(img) for img in imgs],dim=0)
        residuals = np.stack([self.residual_average(residual,self.DONWSAMPLE) for residual in residuals],axis=0)
        residuals[np.isnan(residuals)] = -1
        residuals = torch.from_numpy(residuals).unsqueeze(1)

        return imgs,residuals
    
    def __len__(self):
        return len(self.database_keys)
    
def main(args):
    os.makedirs(args.model_save_path,exist_ok=True)
    pprint = partial(print_on_main, rank=dist.get_rank())
    num_gpus = dist.get_world_size()
    rank = dist.get_rank()
    pprint(f"Using {num_gpus} GPUS")
    min_loss = 1e9
    epoch = 0

    dataset = TrainDataset(root = args.dataset_path,
                           downsample = 16,
                           input_size = 512,
                           batch_size = args.batch_size)
    sampler = DistributedSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            batch_size=1,
                            num_workers=4,
                            drop_last=False,
                            pin_memory=False,
                            shuffle=False)
    batch_num = len(dataloader)
    
    conf_head = ConfHead(args.dino_weight_path)
    optimizer = optim.AdamW(params=conf_head.head.parameters(),lr = args.lr_max)
    if not args.conf_head_path is None:
        load_model_state_dict(conf_head.head,args.conf_head_path)

    conf_head = conf_head.to(args.device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)
    conf_head = distibute_model(conf_head,args.local_rank)
    
    scheduler = MultiStageOneCycleLR(optimizer = optimizer,
                                     total_steps = args.max_epoch * batch_num,
                                     warmup_ratio=min(5. / args.max_epoch, .1),
                                     cooldown_ratio=.9)
    
    bce = nn.BCELoss()
    
    for epoch in range(args.max_epoch):
        pprint(f'\nEpoch:{epoch}')
        sampler.set_epoch(epoch)
        dataset.set_epoch(epoch)
        total_loss = 0

        conf_head.train()

        for batch_idx,data in enumerate(dataloader):
            optimizer.zero_grad()
            imgs,residuals = data
            imgs,residuals = imgs.squeeze(0).to(device = args.device,dtype = torch.float32),residuals.squeeze(0).to(device = args.device,dtype = torch.float32)
            
            B,_,H,W = imgs.shape

            conf_pred = conf_head(imgs)
            conf_label = get_conf_label(residuals)
            loss = bce(conf_pred,conf_label) * 100.

            loss.backward()
            
            pprint(f"epoch:{epoch} \t batch:{batch_idx+1}/{batch_num} \t loss:{loss.item():.2f}")
            total_loss += loss.item()

            optimizer.step()
            scheduler.step()

            loss_rec = loss.clone().detach()
            dist.all_reduce(loss_rec,dist.ReduceOp.AVG)
            dist.barrier()

            total_loss = loss_rec.item()
        
        if rank == 0:
            print(f"epoch:{epoch} \t total_loss:{total_loss:.2f} \t min_loss:{min_loss}")
            if total_loss < min_loss:
                min_loss = total_loss
                conf_head.module.save_head(os.path.join(args.model_save_path,'conf_head.pth'))
                print("Best Updated")
   






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--conf_head_path',type=str,default=None)
    parser.add_argument('--model_save_path',type=str,default=f'./weights/conf_head_{get_current_time()}')
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--max_epoch',type=int,default=1000)
    parser.add_argument('--lr_max',type=float,default=1e-3)
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