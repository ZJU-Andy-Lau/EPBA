import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import h5py
import cv2

import torch
from torchvision import transforms

from load_data import generate_affine_matrices,xy2rc_mat
from model.encoder import Encoder
from model.gru import GRUBlock
from shared.utils import get_current_time,load_model_state_dict,load_config
import shared.visualize as visualizer
from solve.solve_windows import WindowSolver

def load_data(args):
    """
    return:
        imgs_a, imgs_b : (N,3,H,W) torch.Tensor
        Hs_a, Hs_b: (N,3,3) torch.Tensor
        Ms:(N,2,3) torch.Tensor
    """
    file = h5py.File(args.dataset_path,'r')
    all_keys = list(file.keys())
    select_keys = []
    if not args.dataset_select is None:
        idxs = [int(i) for i in args.dataset_select.split(',')]
        select_keys = [all_keys[i] for i in idxs]
    else:
        idxs = np.random.choice(len(all_keys),args.dataset_num)
        select_keys = [all_keys[i] for i in idxs]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    imgs_a = []
    imgs_b = []
    Hs_a = []
    Hs_b = []
    Ms = []

    dsize = (512,512)

    for key in select_keys:
        img_num = len(file[key]['images'])
        idx1,idx2 = np.random.choice(img_num,2)
        if idx1 == idx2:
            idx2 = (idx1 + 1) % img_num
        img_1_full = file[key]['images'][f'{idx1}'][:]
        img_2_full = file[key]['images'][f'{idx2}'][:]
        img_1_full = np.stack([img_1_full]*3,axis=-1)
        img_2_full = np.stack([img_2_full]*3,axis=-1)
        H,W = img_1_full.shape[:2]

        Hs_a_xy,Hs_b_xy,M_xy = generate_affine_matrices((H,W),(256,2048),dsize,1)
        img_2_full = cv2.warpAffine(img_2_full,M_xy,(W,H),flags=cv2.INTER_LINEAR)
        
        img_1_warp = cv2.warpPerspective(img_1_full,Hs_a_xy[0],dsize,flags=cv2.INTER_LINEAR)
        img_2_warp = cv2.warpPerspective(img_2_full,Hs_b_xy[0],dsize,flags=cv2.INTER_LINEAR)

        H_a_rc = xy2rc_mat(Hs_a_xy)[0]
        H_b_rc = xy2rc_mat(Hs_b_xy)[0]
        M_rc = xy2rc_mat(M_xy)

        imgs_a.append(img_1_warp)
        imgs_b.append(img_2_warp)
        Hs_a.append(H_a_rc)
        Hs_b.append(H_b_rc)
        Ms.append(M_rc)
    
    imgs_a = torch.stack([transform(img) for img in imgs_a],dim=0).to(device=args.device,dtype=torch.float32)
    imgs_b = torch.stack([transform(img) for img in imgs_b],dim=0).to(device=args.device,dtype=torch.float32)
    Hs_a = torch.from_numpy(np.stack(Hs_a,axis=0)).to(device=args.device,dtype=torch.float32)
    Hs_b = torch.from_numpy(np.stack(Hs_b,axis=0)).to(device=args.device,dtype=torch.float32)
    Ms = torch.from_numpy(np.stack(Ms,axis=0)).to(device=args.device,dtype=torch.float32)

    return imgs_a,imgs_b,Hs_a,Hs_b,Ms

def load_models(args):
    model_configs = load_config(args.model_config_path)

    encoder = Encoder(dino_weight_path=args.dino_path,
                      embed_dim=model_configs['encoder']['embed_dim'],
                      ctx_dim=model_configs['encoder']['ctx_dim'])
    
    encoder.load_adapter(args.adapter_path)
    
    gru = GRUBlock(corr_levels=model_configs['gru']['corr_levels'],
                   corr_radius=model_configs['gru']['corr_radius'],
                   context_dim=model_configs['gru']['ctx_dim'],
                   hidden_dim=model_configs['gru']['hidden_dim'])
    
    load_model_state_dict(gru,args.gru_path)
    
    args.gru_iter_num = model_configs['gru']['iter_num']
    
    encoder = encoder.to(args.device).eval()
    gru = gru.to(args.device).eval()

    print("Models Loaded")

    return encoder,gru

def solve(args,encoder:Encoder,gru:GRUBlock,data):
    imgs_a,imgs_b,Hs_a,Hs_b,Ms = data
    B,_,H,W = imgs_a.shape

    feats_1,feats_2 = encoder(imgs_a,imgs_b)

    solver = WindowSolver(B,H,W,gru,feats_1,feats_2,Hs_a,Hs_b,gru_max_iter=args.gru_iter_num)

    M_a_b_pred = solver.solve(flag='ab',final_only=True) # N,2,3

    report(args,M_a_b_pred,Ms,imgs_a,imgs_b)

def report(args,Ms_pred:torch.Tensor,Ms_gt:torch.Tensor,imgs_1:torch.Tensor,imgs_2:torch.Tensor):
    pass
    

def main(args):
    data = load_data(args)
    encoder,gru = load_models(args)
    
    solve(args,encoder,gru,data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=10)
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--adapter_path',type=str,default=None)
    parser.add_argument('--gru_path',type=str,default=None)
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--output_path',type=str,default='./results')
    parser.add_argument('--device',type=str,default='cuda')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path,get_current_time())
    os.makedirs(args.output_path,exist_ok=True)

    print("==============================configs==============================")
    for k,v in vars(args).items():
        print(f"{k}:{v}")
    print("===================================================================")
    main(args)