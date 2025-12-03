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
from typing import List
import itertools
import yaml


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

from model.encoder import Encoder
from model.gru import GRUBlock
from model.ctx_decoder import ContextDecoder
from utils.utils import str2bool,get_current_time
from .utils import is_overlap,convert_pair_dicts_to_solver_inputs,load_config,get_error_report
from pair import Pair
from solve.global_affine_solver import GlobalAffineSolver
from rs_image import RSImage

def init_random_seed(args):
    seed = args.random_seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images(args) -> List[RSImage] :
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs] #按照字典序的idx
    images = []
    for idx,folder in enumerate(img_folders):
        img_path = os.path.join(base_path,folder)
        images.append(RSImage(args,img_path,idx))
        print(f"Loaded Image {idx} from {folder}")
    print(f"Totally {len(images)} Images Loaded")
    args.image_num = len(images)
    return images

def build_pairs(images:List[RSImage]) -> List[Pair]:
    images_num = len(images)
    pairs = []
    for i,j in itertools.combinations(range(images_num),2):
        if is_overlap(images[i],images[j],args.min_window_size ** 2):
            pair = Pair(images[i],images[j],images[i].id,images[j].id,device=args.device)
            pairs.append(pair)
    print(f"Totally {len(pairs)} Pairs")
    return pairs

def load_models(args):
    model_configs = load_config(args.model_config_path)

    encoder = Encoder(dino_weight_path=args.dino_path,
                      embed_dim=model_configs['encoder']['embed_dim'],
                      ctx_dim=model_configs['encoder']['ctx_dim'])
    
    gru = GRUBlock(corr_levels=model_configs['gru']['corr_levels'],
                   corr_radius=model_configs['gru']['corr_radius'],
                   context_dim=model_configs['gru']['ctx_dim'],
                   hidden_dim=model_configs['gru']['hidden_dim'])
    
    args.gru_iter_num = model_configs['gru']['iter_num']
    
    encoder = encoder.to(args.device).eval()
    gru = gru.to(args.device).eval()

    return encoder,gru

def solve(args,pairs:List[Pair],encoder:Encoder,gru:GRUBlock) -> torch.Tensor:
    results = []
    for pair in pairs:
        affine_ab,affine_ba = pair.solve_affines(encoder,gru)
        result = {
            str(pair.id_a):affine_ab,
            str(pair.id_b):affine_ba
        }
        results.append(result)
    solver_configs = load_config(args.solver_config_path)
    solver = GlobalAffineSolver(num_nodes=args.image_num,
                                lambda_anchor=solver_configs['lambda_anchor'],
                                lambda_A=solver_configs['lambda_A'],
                                lambda_t=solver_configs['lambda_t'],
                                sigma_A=solver_configs['sigma_A'],
                                sigma_t=solver_configs['sigma_t'],
                                device=args.device,
                                dtype=torch.float32)
    edge_src, edge_dst, A_ij, t_ij, w_ij = convert_pair_dicts_to_solver_inputs(results,device=args.device,dtype=torch.float32)
    Ms = solver.solve(edge_src=edge_src,
                      edge_dst=edge_dst,
                      A_ij=A_ij,
                      t_ij=t_ij,
                      w_ij=w_ij,
                      anchor_indices=[0])
    Ms_23 = Ms[:,:2,]
    return Ms_23


def main(args):
    init_random_seed(args)

    images = load_images(args)
    pairs = build_pairs(images)
    encoder,gru = load_models(args)

    Ms = solve(args,pairs,encoder,gru)
    for image in images:
        M = Ms[image.id]
        image.rpc.Update_Adjust(M)
        image.rpc.Merge_Adjust()
    
    report = get_error_report(pairs)
    print("\n" + "--- Global Error Report (Summary) ---")
    print(f"Total tie points checked: {report['count']}")
    print(f"Mean Error:   {report['mean']:.4f} m")
    print(f"Median Error: {report['median']:.4f} m")
    print(f"Max Error:    {report['max']:.4f} m")
    print(f"RMSE:         {report['rmse']:.4f} m")
    print(f"< 1.0 m: {report['<1m_percent']:.2f} %")
    print(f"< 3.0 m: {report['<3m_percent']:.2f} %")
    print(f"< 5.0 m: {report['<5m_percent']:.2f} %")


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #==============================数据相关设置=====================================

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    
    parser.add_argument('--select_imgs',type=str,default='0,1')

    #==============================================================================


    #==============================模型相关设置=====================================

    parser.add_argument('--dino_path', type=str, default='weights')

    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')

    parser.add_argument('--gru_path', type=str, default='weights/gru.pth')

    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')

    #==============================================================================


    #==============================求解相关设置=====================================

    parser.add_argument('--max_window_size', type=int, default=8000)

    parser.add_argument('--min_window_size', type=int, default=500)

    parser.add_argument('--max_window_num', type=int, default=256)

    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)

    parser.add_argument('--solver_config_path', type=str, default='configs/global_solver_config.yaml')

    #================================================================================


    #==============================输出相关设置=====================================
    
    parser.add_argument('--output_path', type=str, default='results')
    
    parser.add_argument('--vis_resolution', type=float, default=1.0, 
                        help='Resolution (in meters) for output orthophotos and checkerboards.')
    
    #===============================================================================


    #==============================实验相关设置======================================
    
    parser.add_argument('--experiment_id', type=str, default=None)

    parser.add_argument('--random_seed',type=int,default=42)

    #==============================================================================


    args = parser.parse_args()

    args.device = 'cuda'

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    
    args.output_path = os.path.join(args.output_path,args.experiment_id)
    os.makedirs(args.output_path,exist_ok=True)

    main(args)
    


