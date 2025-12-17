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
from typing import List, Dict
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
from utils import is_overlap,convert_pair_dicts_to_solver_inputs,get_error_report
from pair import Pair
from solve.global_affine_solver import GlobalAffineSolver,TopologicalAffineSolver
from rs_image import RSImage,vis_registration

def init_random_seed(args):
    seed = args.random_seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images(args, lazy=False) -> List[RSImage] :
    """
    lazy: If True, load in lightweight mode (metadata only).
    """
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs] #按照字典序的idx
    images = []
    for idx,folder in enumerate(img_folders):
        img_path = os.path.join(base_path,folder)
        # Pass lazy flag
        images.append(RSImage(args,img_path,idx,args.device, lazy=lazy))
        if not lazy: 
            print(f"Loaded Image {idx} from {folder}")
    
    if lazy and dist.is_initialized() and dist.get_rank() == 0:
        print(f"[Rank 0] Meta-loaded {len(images)} Images (Lazy Mode)")
    elif not lazy:
        print(f"Totally {len(images)} Images Loaded")
        
    args.image_num = len(images)
    return images

def build_pair_indices(args, images: List[RSImage]) -> List[Dict]:
    """
    Construct list of metadata dictionaries for pairs instead of Pair objects.
    This runs on Rank 0 efficiently.
    """
    images_num = len(images)
    configs = {
        'max_window_num': args.max_window_num,
        'min_window_size': args.min_window_size,
        'max_window_size': args.max_window_size,
        'min_area_ratio': args.min_cover_area_ratio,
    }
    pair_indices = []
    for i, j in itertools.combinations(range(images_num), 2):
        # is_overlap relies on corner_xys, which is computed in RSImage.__init__ even in lazy mode
        if is_overlap(images[i], images[j], args.min_window_size ** 2):
            output_path = os.path.join(args.output_path, f"pair_{images[i].id}_{images[j].id}")
            # Clone config for each pair to set path
            pair_config = configs.copy()
            pair_config['output_path'] = output_path
            
            pair_indices.append({
                'id_a': images[i].id,
                'id_b': images[j].id,
                'configs': pair_config
            })
            
    if dist.is_initialized() and dist.get_rank() == 0:
        print(f"[Rank 0] Identified {len(pair_indices)} pairs")
    
    return pair_indices

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

    return encoder,gru

def main(args):
    # ========================== 1. DDP Initialization ==========================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
        is_ddp = True
    else:
        rank = 0
        world_size = 1
        is_ddp = False
        print("Running in Single-GPU mode.")

    init_random_seed(args)

    # ========================== 2. Data Preparation (Rank 0) ==========================
    all_images = [] 
    scatter_list = None
    
    if rank == 0:
        # Rank 0 loads all images in lazy mode (metadata only)
        all_images = load_images(args, lazy=True)
        
        # Build pair indices (metadata only)
        all_pair_indices = build_pair_indices(args, all_images)
        
        # Sort to ensure deterministic order
        all_pair_indices.sort(key=lambda p: (p['id_a'], p['id_b']))
        
        # Split into chunks
        pairs_chunks = np.array_split(all_pair_indices, world_size)
        scatter_list = [chunk.tolist() for chunk in pairs_chunks]
        
        print(f"[Rank 0] Distributing {len(all_pair_indices)} tasks to {world_size} ranks.")
    else:
        # Workers need empty container for all_images broadcast
        # But since pickle works, we can just broadcast the object list
        pass

    # ========================== 3. Broadcast Image List & Scatter Tasks ==========================
    if is_ddp:
        # Broadcast the Lazy Image List so all workers have metadata/references
        # Note: pickling lazy RSImage is fast as it contains no heavy arrays
        broadcast_container = [all_images]
        dist.broadcast_object_list(broadcast_container, src=0)
        all_images = broadcast_container[0]

        # Scatter the pair indices
        output_list = [None]
        dist.scatter_object_list(output_list, scatter_list if rank == 0 else None, src=0)
        my_pair_indices = output_list[0]
    else:
        # Single GPU mode: run everything
        if rank == 0: # Should be true
            # In single GPU we need to convert build_pair_indices result to Pair objects later?
            # Or just follow the same flow.
            my_pair_indices = all_pair_indices
        else:
            my_pair_indices = []

    # ========================== 4. Worker: Load Heavy Data & Construct Pairs ==========================
    # Identify which images this rank needs
    my_unique_image_ids = set()
    for meta in my_pair_indices:
        my_unique_image_ids.add(meta['id_a'])
        my_unique_image_ids.add(meta['id_b'])
            
    # Load actual pixels for needed images
    # Update device for local images
    # tqdm.write(f"[Rank {rank}] Loading heavy data for {len(my_unique_image_ids)} images...")
    for img_id in my_unique_image_ids:
        img = all_images[img_id]
        img.device = args.device # Set to local rank device
        img.load_heavy_data()    # Actual IO

    # Construct Pair objects locally
    my_pairs = []
    for meta in my_pair_indices:
        img_a = all_images[meta['id_a']]
        img_b = all_images[meta['id_b']]
        pair = Pair(img_a, img_b, meta['id_a'], meta['id_b'], meta['configs'], device=args.device)
        my_pairs.append(pair)

    # ========================== 5. Load Models ==========================
    encoder, gru = load_models(args)

    # ========================== 6. Parallel Inference ==========================
    local_results = []
    
    # Setup progress bar
    pbar = tqdm(my_pairs, position=rank, leave=True, ncols=140,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
    
    def status_update(direction, w_size, step):
        if isinstance(w_size, (int, float)):
            w_str = f"{w_size}m"
        else:
            w_str = str(w_size)
        pbar.set_description(f"[R{rank}] {direction} | {w_str} | {step}")

    for pair in pbar:
        pbar.set_description(f"[R{rank}] Pair {pair.id_a}-{pair.id_b} Init")
        affine_ab, affine_ba = pair.solve_affines(encoder, gru, status_callback=status_update)
        local_results.append({
            pair.id_a: affine_ab,
            pair.id_b: affine_ba
        })
    
    pbar.close()

    # ========================== 7. Gather Results ==========================
    if is_ddp:
        all_results_lists = [None for _ in range(world_size)]
        dist.all_gather_object(all_results_lists, local_results)
    else:
        all_results_lists = [local_results]

    # ========================== 8. Global Solve (Rank 0) ==========================
    Ms_global = None
    
    if rank == 0:
        print("\n[Rank 0] Aggregating results and solving global affine...")
        
        flat_results = [item for sublist in all_results_lists for item in sublist]
        
        # Load DEMs for ALL images for global solver projection
        print("[Rank 0] Loading DEMs for Global Solver...")
        for img in tqdm(all_images, desc="Loading DEMs"):
            img.device = args.device
            img.load_dem_only() # Lightweight load
            
        solver_configs = load_config(args.solver_config_path)
        solver = GlobalAffineSolver(images=all_images,
                                    device=args.device,
                                    anchor_indices=[0],
                                    max_iter=100,
                                    converge_tol=1e-6)
        
        Ms_global = solver.solve(flat_results)
        Ms_global = Ms_global[:, :2, ] 
    
    # ========================== 9. Broadcast Result ==========================
    if is_ddp:
        broadcast_list = [Ms_global]
        dist.broadcast_object_list(broadcast_list, src=0)
        Ms_global = broadcast_list[0]

    # ========================== 10. Update, Output & Report ==========================
    # 10.1 Update Local RPCs & Vis
    for img_id in my_unique_image_ids:
        img = all_images[img_id]
        M = Ms_global[img.id]
        img.rpc.Update_Adjust(M)
        img.rpc.Merge_Adjust()

    for pair in my_pairs:
        vis_registration(image_a=pair.rs_image_a, 
                         image_b=pair.rs_image_b, 
                         output_path=args.output_path, 
                         device=args.device)
    
    # 10.2 Rank 0 Error Report
    if rank == 0:
        print("\n[Rank 0] Calculating Final Error Report...")
        # Update RPCs for ALL images in Rank 0 (they have DEM loaded now)
        for i, img in enumerate(all_images):
            M = Ms_global[i]
            img.rpc.Update_Adjust(M)
            img.rpc.Merge_Adjust()
            
        # Reconstruct light pairs for error checking (Pixel data not needed)
        # We assume build_pair_indices holds all pairs in order
        # Need to reconstruct the full list of indices since we scattered them
        # all_pair_indices is still available on Rank 0
        
        eval_pairs = []
        for meta in all_pair_indices:
            img_a = all_images[meta['id_a']]
            img_b = all_images[meta['id_b']]
            # Device doesn't matter much for check_error as it uses CPU/Numpy mostly or lightweight tensors
            # But ensure consistency
            p = Pair(img_a, img_b, meta['id_a'], meta['id_b'], meta['configs'], device=args.device)
            eval_pairs.append(p)
            
        report = get_error_report(eval_pairs)
        print("\n" + "="*50)
        print("Final Registration Error Report")
        print("="*50)
        # report is a dict, print nicely
        for k, v in report.items():
            print(f"{k}: {v}")
        print("="*50)

    if is_ddp:
        dist.destroy_process_group()

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

    parser.add_argument('--device',type=str,default='cuda')

    #==============================================================================


    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())
    
    args.output_path = os.path.join(args.output_path,args.experiment_id)
    os.makedirs(args.output_path,exist_ok=True)

    main(args)