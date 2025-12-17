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
from shared.utils import str2bool,get_current_time,load_model_state_dict,load_config
from utils import is_overlap,convert_pair_dicts_to_solver_inputs,get_error_report,partition_pairs
from pair import Pair
from solve.global_affine_solver import GlobalAffineSolver,TopologicalAffineSolver
from rs_image import RSImage,RSImageMeta,vis_registration

def init_random_seed(args):
    seed = args.random_seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images_meta(args) -> List[RSImageMeta]:
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs] #按照字典序的idx
    metas = []
    for idx,folder in enumerate(img_folders):
        img_path = os.path.join(base_path,folder)
        metas.append(RSImageMeta(args,img_path,idx,args.device))
        print(f"[rank{dist.get_rank()}]Loaded Image Meta {idx} from {folder}")
    print(f"[rank{dist.get_rank()}]Totally {len(metas)} Images' Meta Loaded")
    return metas

def load_images(args,metas:List[RSImageMeta]) -> List[RSImage] :
    images = [RSImage(meta,device=args.device) for meta in metas]
    print(f"[rank{dist.get_rank()}]Totally {len(images)} Images Loaded")
    args.image_num = len(images)
    return images

def get_pairs(args,metas:List[RSImageMeta]):
    pair_idxs = []
    for i,j in itertools.combinations(range(len(metas)),2):
        if is_overlap(metas[i],metas[j],args.min_window_size ** 2):
            pair_idxs.append((i,j))
    return pair_idxs

def build_pairs(args,images:List[RSImage],pair_ids = None) -> List[Pair]:
    images_num = len(images)
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
    }
    pairs = []
    if pair_ids is None:
        for i,j in itertools.combinations(range(images_num),2):
            if is_overlap(images[i],images[j],args.min_window_size ** 2):
                configs['output_path'] = os.path.join(args.output_path,f"pair_{images[i].id}_{images[j].id}")
                pair = Pair(images[i],images[j],images[i].id,images[j].id,configs,device=args.device)
                pairs.append(pair)
    else:
        for i,j in pair_ids:
            configs['output_path'] = os.path.join(args.output_path,f"pair_{images[i].id}_{images[j].id}")
            pair = Pair(images[i],images[j],images[i].id,images[j].id,configs,device=args.device)
            pairs.append(pair)

    print(f"[rank{dist.get_rank()}]Totally {len(pairs)} Pairs")
    return pairs

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
    
    encoder = encoder.to(args.device).eval().half()
    gru = gru.to(args.device).eval()

    # encoder = torch.compile(encoder,mode='max-autotune')
    # gru = torch.compile(gru,mode="max-autotune")

    print(f"[rank{dist.get_rank()}]Models Loaded")

    return encoder,gru

def main(args):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    args.device = f"cuda:{local_rank}"

    init_random_seed(args)

    metas = []
    if rank == 0:
        metas = load_images_meta(args)
        pairs_ids_all = get_pairs(args,metas)
        pairs_ids_chunks = partition_pairs(pairs_ids_all,world_size) # TODO:考虑pairs num < world size的情况
    
    #ddp同步metas
    broadcast_container = [metas]
    dist.broadcast_object_list(broadcast_container,src=0)
    metas = broadcast_container[0]

    # ddp分发pair ids
    scatter_recive = [None]
    dist.scatter_object_list(scatter_recive,pairs_ids_chunks if rank == 0 else None, src=0)
    pairs_ids = scatter_recive[0]

    image_ids = sorted(set(x for t in pairs_ids for x in t))

    print(f"[rank{rank}]: pair_ids:{pairs_ids} \t image_ids:{image_ids} \n")

    images = load_images(args,[metas[i] for i in image_ids])
    pairs = build_pairs(args,images,pairs_ids)
    encoder,gru = load_models(args)

    local_results = []
    for pair in pairs:
        print(f"[rank{rank}]Solving Pair {pair.id_a} - {pair.id_b}")
        affine_ab,affine_ba = pair.solve_affines(encoder,gru)
        result = {
            pair.id_a:affine_ab,
            pair.id_b:affine_ba
        }
        local_results.append(result)

    # ddp收集results
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results,local_results)
    
    if rank == 0:
        all_results = [item for sublist in all_results for item in sublist]
        image_ids = sorted(set(x for t in pairs_ids_all for x in t))
        images = load_images(args,[metas[i] for i in image_ids])
        pairs = build_pairs(args,images)
        if args.solver == 'global':
            solver = GlobalAffineSolver(images=images,
                                    device=args.device,
                                    anchor_indices=[0],
                                    max_iter=100,
                                    converge_tol=1e-6)
        else:
            solver = TopologicalAffineSolver(images=images,
                                            device=args.device,
                                            anchor_indices=[0])
        Ms = solver.solve(all_results)
        Ms = Ms[:,:2,]

        for i,image in enumerate(images):
            M = Ms[i]
            print(f"Affine Matrix of Image {image.id}\n{M}\n")
            image.rpc.Update_Adjust(M)
            image.rpc.Merge_Adjust()
        
        for i,j in itertools.combinations(range(len(images)),2):
            vis_registration(image_a=images[i],image_b=images[j],output_path=args.output_path,device=args.device)
        
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

    parser.add_argument('--solver',type=str,default='global')

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

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())
    
    args.output_path = os.path.join(args.output_path,args.experiment_id)
    os.makedirs(args.output_path,exist_ok=True)

    main(args)
    


