import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
import random
from typing import List,Tuple
import itertools
import traceback # 新增导入
import cv2


import torch
import torch.distributed as dist


from model.encoder import Encoder
from model.gru import GRUBlock
from shared.utils import str2bool,get_current_time,load_model_state_dict,load_config
from utils import is_overlap,convert_pair_dicts_to_solver_inputs,get_error_report,get_report_dict,partition_pairs
from pair import Pair
from rs_image import RSImage,RSImageMeta,vis_registration
from infer.monitor import StatusMonitor, StatusReporter # 新增导入

def init_random_seed(args):
    seed = args.random_seed 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images_meta(args, reporter) -> Tuple[List[RSImageMeta],List[RSImageMeta]]:
    reporter.update(current_step="Loading Meta")

    adjust_base_path = os.path.join(args.root, 'adjust_images')
    ref_base_path = os.path.join(args.root, 'ref_images')

    adjust_img_folders = sorted([d for d in os.listdir(adjust_base_path) if os.path.isdir(os.path.join(adjust_base_path, d))])
    if args.select_adjust_imgs != '-1':
        adjust_select_img_idxs = [int(i) for i in args.select_adjust_imgs.split(',')]
    else:
        adjust_select_img_idxs = range(len(adjust_img_folders))
    adjust_img_folders = [adjust_img_folders[i] for i in adjust_select_img_idxs]

    ref_img_folders = sorted([d for d in os.listdir(ref_base_path) if os.path.isdir(os.path.join(ref_base_path, d))])
    if args.select_ref_imgs != '-1':
        ref_select_img_idxs = [int(i) for i in args.select_ref_imgs.split(',')]
    else:
        ref_select_img_idxs = range(len(ref_img_folders))
    ref_img_folders = [ref_img_folders[i] for i in ref_select_img_idxs]

    adjust_metas = []
    ref_metas = []
    for idx,folder in enumerate(adjust_img_folders):
        img_path = os.path.join(adjust_base_path,folder)
        adjust_metas.append(RSImageMeta(args,img_path,idx,args.device))
    for idx,folder in enumerate(ref_img_folders):
        img_path = os.path.join(ref_base_path,folder)
        ref_metas.append(RSImageMeta(args,img_path,idx,args.device))
    
    return adjust_metas,ref_metas

def load_images(args,metas:List[RSImageMeta], reporter) -> List[RSImage] :
    reporter.update(current_step="Loading Images")
    images = [RSImage(meta,device=args.device) for meta in metas]
    return images

def get_ref_lists(args,adjust_metas:List[RSImageMeta],ref_metas:List[RSImageMeta], reporter) -> List[List]:
    reporter.update(current_step="Filtering Ref")
    ref_lists = []
    for i in range(len(adjust_metas)):
        ref_list = []
        for j in range(len(ref_metas)):
            if is_overlap(adjust_metas[i],ref_metas[j],args.min_window_size ** 2):
                ref_list.append(j)
        ref_lists.append(ref_list)
    return ref_lists

def build_adj_ref_pairs(args,adjust_image:RSImage,ref_images:List[RSImage], reporter) -> List[Pair]:
    reporter.update(current_step="Building Pairs")
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
    }
    pairs = []
    for ref_image in ref_images:
        configs['output_path'] = os.path.join(args.output_path,f"pair_{adjust_image.id}_{ref_image.id}")
        pair = Pair(adjust_image,ref_image,adjust_image.id,ref_image.id,configs,device=args.device,dual=False,reporter=reporter)
        pairs.append(pair)
    return pairs

def build_pairs(args,images:List[RSImage], reporter) -> List[Pair]:
    reporter.update(current_step="Building Pairs")
    images_num = len(images)
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
    }
    pairs = []
    for i,j in itertools.combinations(range(images_num),2):
        if is_overlap(images[i],images[j],args.min_window_size ** 2):
            configs['output_path'] = os.path.join(args.output_path,f"pair_{images[i].id}_{images[j].id}")
            pair = Pair(images[i],images[j],images[i].id,images[j].id,configs,device=args.device,check_error_only=True,reporter=reporter)
            pairs.append(pair)

    return pairs

def load_models(args, reporter):
    reporter.update(current_step="Loading Models")
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

@torch.no_grad()
def main(args):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    args.device = f"cuda:{local_rank}"

    # --- Monitor & Reporter Initialization ---
    experiment_id_clean = str(args.experiment_id).replace(":", "_").replace(" ", "_")
    monitor = None
    if rank == 0:
        monitor = StatusMonitor(world_size, experiment_id_clean)
        monitor.start()
    
    reporter = StatusReporter(rank, world_size, experiment_id_clean, monitor)
    # -----------------------------------------

    try:
        init_random_seed(args)

        metas = []
        if rank == 0:
            adjust_metas_all,ref_metas_all = load_images_meta(args, reporter)
            ref_lists = get_ref_lists(args,adjust_metas_all,ref_metas_all,reporter)
            reporter.log(f"ref lists:{ref_lists}")
            ref_metas_lists = [[ref_metas_all[i] for i in sub_list] for sub_list in ref_lists]
            adjust_metas_chunk = np.array_split(np.array(adjust_metas_all,dtype=object),world_size) # (world_size,K)
            ref_metas_chunk = np.array_split(np.array(ref_metas_lists,dtype=object),world_size) # (world_size,K,N)
            
        
        #ddp分发metas
        reporter.update(current_step="Syncing Meta")
        scatter_adjust_metas = [None]
        dist.scatter_object_list(scatter_adjust_metas,adjust_metas_chunk if rank == 0 else None, src=0)
        adjust_metas = scatter_adjust_metas[0] # K
        scatter_ref_metas = [None]
        dist.scatter_object_list(scatter_ref_metas,ref_metas_chunk if rank == 0 else None, src=0)
        ref_metas = scatter_ref_metas[0] # K,N
        
        local_results = {}
        local_dis = []
        reporter.update(current_task="Ready", progress=f"-", level="-", current_step="Ready")
        
        if len(adjust_metas) > 0 and len(ref_metas) > 0:
            reporter.update(current_task="Loading Images")
            pairs:List[Pair] = []
            adjust_images:List[RSImage] = []
            for i in range(len(adjust_metas)):
                adjust_image = RSImage(adjust_metas[i],device=args.device)
                ref_images = [RSImage(ref_meta,device=args.device) for ref_meta in ref_metas[i]]
                pairs_tmp = build_adj_ref_pairs(args,adjust_image,ref_images,reporter)
                pairs.extend(pairs_tmp)
                adjust_images.append(adjust_image)

            encoder,gru = load_models(args, reporter)

            for idx, pair in enumerate(pairs):
                reporter.update(progress=f"{idx+1}/{len(pairs)}")
                affine = pair.solve_affines(encoder,gru).detach().cpu()
                pair.rs_image_a.affine_list.append(affine)
            
            reporter.update(current_task="Baking RPC", level="-", current_step="-")

            for idx,adjust_image in enumerate(adjust_images):
                reporter.update(progress=f"{idx+1}/{len(adjust_images)}")
                affine = adjust_image.merge_affines()
                reporter.log(f"Affine Matrix of Image {adjust_image.id}\n{affine}\n")
                adjust_image.rpc.Update_Adjust(affine)
                adjust_image.rpc.Merge_Adjust()
                local_results[adjust_image.id] = affine
            
            reporter.update(current_task="Check Error", level="-", current_step="-")
            for idx, pair in enumerate(pairs):
                reporter.update(progress=f"{idx+1}/{len(pairs)}")
                ref_points = pair.rs_image_b.get_ref_points()
                dis = pair.rs_image_a.check_error(ref_points)
                checkpoint_vis = pair.rs_image_a.vis_checkpoints(ref_points)
                reporter.log(f"checkpoint_vis shape:{checkpoint_vis.shape}")
                local_dis.append(dis)
                cv2.imwrite(os.path.join(args.output_path,f"ckpts_{pair.id_a}_{pair.id_b}.png"),checkpoint_vis)
                report = get_report_dict(dis)
                reporter.log("\n" + f"--- Adj {pair.id_a} => Ref {pair.id_b}  Error Report ---")
                reporter.log(f"Total tie points checked: {report['count']}")
                reporter.log(f"Mean Error:   {report['mean']:.4f} m")
                reporter.log(f"Median Error: {report['median']:.4f} m")
                reporter.log(f"Max Error:    {report['max']:.4f} m")
                reporter.log(f"RMSE:         {report['rmse']:.4f} m")
                reporter.log(f"< 1.0 pix: {report['<1m_percent']:.2f} %")
                reporter.log(f"< 3.0 pix: {report['<3m_percent']:.2f} %")
                reporter.log(f"< 5.0 pix: {report['<5m_percent']:.2f} %")
            
            local_dis = np.concatenate(local_dis)

                
                # reporter.log(f"{adjust_image.id}:\n{local_results[adjust_image.id]}")
            
            reporter.update(current_task="Finished", progress=f"{len(pairs)}/{len(pairs)}", level="-", current_step="Cleanup")
 
            del encoder
            del gru
            for image in adjust_images:
                del image
            for pair in pairs:
                del pair.rs_image_b
                del pair
            pairs = None
            adjust_images = None
            ref_images = None
            encoder = None
            gru = None
            
        # ddp收集results
        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results = [None for _ in range(world_size)]
            all_distances = [None for _ in range(world_size)]
        else:
            all_results = None
            all_distances = None
        dist.gather_object(local_results, all_results if rank == 0 else None, dst=0)
        dist.gather_object(local_dis, all_distances if rank == 0 else None, dst=0)
        
        if rank == 0:
            all_distances = np.concatenate(all_distances)
            report = get_report_dict(all_distances)
            reporter.log("\n" + "--- Global Error Report (Summary) ---")
            reporter.log(f"Total tie points checked: {report['count']}")
            reporter.log(f"Mean Error:   {report['mean']:.4f} m")
            reporter.log(f"Median Error: {report['median']:.4f} m")
            reporter.log(f"Max Error:    {report['max']:.4f} m")
            reporter.log(f"RMSE:         {report['rmse']:.4f} m")
            reporter.log(f"< 1.0 pix: {report['<1m_percent']:.2f} %")
            reporter.log(f"< 3.0 pix: {report['<3m_percent']:.2f} %")
            reporter.log(f"< 5.0 pix: {report['<5m_percent']:.2f} %")

            # all_results = {k:v for d in all_results if d for k,v in d.items()}
            # images = load_images(args,adjust_metas_all, reporter)
            # for image in images:
            #     M = all_results[image.id]
            #     reporter.log(f"Affine Matrix of Image {image.id}\n{M}\n")
            #     image.rpc.Update_Adjust(M)
            #     image.rpc.Merge_Adjust()
            

            # pairs = build_pairs(args,images, reporter)
            
            # reporter.update(current_step="Visualizing")
            # for i,j in itertools.combinations(range(len(images)),2):
            #     vis_registration(image_a=images[i],image_b=images[j],output_path=args.output_path,device=args.device)
            
            # report = get_error_report(pairs)
            # reporter.log("\n" + "--- Global Error Report (Summary) ---")
            # reporter.log(f"Total tie points checked: {report['count']}")
            # reporter.log(f"Mean Error:   {report['mean']:.4f} m")
            # reporter.log(f"Median Error: {report['median']:.4f} m")
            # reporter.log(f"Max Error:    {report['max']:.4f} m")
            # reporter.log(f"RMSE:         {report['rmse']:.4f} m")
            # reporter.log(f"< 1.0 m: {report['<1m_percent']:.2f} %")
            # reporter.log(f"< 3.0 m: {report['<3m_percent']:.2f} %")
            # reporter.log(f"< 5.0 m: {report['<5m_percent']:.2f} %")
            
    except Exception as e:
        error_msg = traceback.format_exc()
        if reporter:
            reporter.update(current_task="ERROR", error=error_msg)
        raise e  
    
    finally:
        if monitor:
            monitor.stop()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #==============================数据相关设置=====================================

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    
    parser.add_argument('--select_adjust_imgs',type=str,default='-1')

    parser.add_argument('--select_ref_imgs',type=str,default='-1')

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

    #================================================================================


    #==============================输出相关设置=====================================
    
    parser.add_argument('--output_path', type=str, default='results')
    
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