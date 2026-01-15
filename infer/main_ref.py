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
import time
import datetime


import torch
import torch.distributed as dist


from model.encoder import Encoder
from model.predictor import Predictor
from shared.utils import str2bool,get_current_time,load_model_state_dict,load_config
from utils import is_overlap,convert_pair_dicts_to_solver_inputs,get_error_report,get_report_dict,partition_pairs
from pair import Pair
from rs_image import RSImage,RSImageMeta,RSImage_Error_Check,vis_registration
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

def get_ref_list(args,adjust_meta:RSImageMeta,ref_metas:List[RSImageMeta], reporter) -> List[List]:
    reporter.update(current_step="Filtering Ref")
    ref_list = []
    for i in range(len(ref_metas)):
        if is_overlap(adjust_meta,ref_metas[i],args.min_window_size ** 2):
            ref_list.append(i)

    return ref_list

def build_adj_ref_pair(args,adjust_image:RSImage,ref_image:RSImage, reporter) -> Pair:
    reporter.update(current_step="Building Pairs")
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
        'quad_split_times':args.quad_split_times,
        'iter_num':args.predictor_iter_num,
    }
    configs['output_path'] = os.path.join(args.output_path,f"pair_{adjust_image.id}_{ref_image.id}")
    pair = Pair(adjust_image,ref_image,adjust_image.id,ref_image.id,configs,device=args.device,dual=False,reporter=reporter)
    return pair

def build_pairs(args,images:List[RSImage], reporter) -> List[Pair]:
    reporter.update(current_step="Building Pairs")
    images_num = len(images)
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
        'quad_split_times':args.quad_split_times,
        'iter_num':args.predictor_iter_num,
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
                      ctx_dim=model_configs['encoder']['ctx_dim'],
                      use_adapter=args.use_adapter,
                      use_conf=args.use_conf)
    
    encoder.load_adapter(args.adapter_path)
    
    predictor = Predictor(corr_levels=model_configs['predictor']['corr_levels'],
                   corr_radius=model_configs['predictor']['corr_radius'],
                   context_dim=model_configs['predictor']['ctx_dim'],
                   hidden_dim=model_configs['predictor']['hidden_dim'],
                   use_mtf=args.use_mtf)
    
    load_model_state_dict(predictor,args.predictor_path)
    
    if args.predictor_iter_num is None:
        args.predictor_iter_num = model_configs['predictor']['iter_num']
    
    encoder = encoder.to(args.device).eval().half()
    predictor = predictor.to(args.device).eval()

    return encoder,predictor

@torch.no_grad()
def main(args):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",timeout=datetime.timedelta(minutes=120))
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

        ref_metas_all = []
        if rank == 0:
            adjust_metas_all,ref_metas_all = load_images_meta(args, reporter)
            adjust_metas_chunk = np.array_split(np.array(adjust_metas_all,dtype=object),world_size) # (world_size,K)
            
        #ddp分发adjust metas
        reporter.update(current_step="Syncing Meta")
        scatter_adjust_metas = [None]
        dist.scatter_object_list(scatter_adjust_metas,adjust_metas_chunk if rank == 0 else None, src=0)
        adjust_metas:List[RSImageMeta] = scatter_adjust_metas[0] # K

        #ddp同步ref metas
        broadcast_container = [ref_metas_all]
        dist.broadcast_object_list(broadcast_container,src=0)
        ref_metas:List[RSImageMeta] = broadcast_container[0]

        dist.barrier()
        
        local_results = {}
        reporter.update(current_task="Ready", progress=f"-", level="-", current_step="Ready")

        model_time = 0
        
        if len(adjust_metas) > 0 and len(ref_metas) > 0:
            encoder,predictor = load_models(args, reporter)
            for adjust_idx,adjust_meta in enumerate(adjust_metas):
                reporter.update(progress=f"{adjust_idx}/{len(adjust_metas)}")
                ref_list = get_ref_list(args,adjust_meta,ref_metas,reporter)
                reporter.log(f"ref list for img_{adjust_meta.id} : {ref_list}")
                reporter.update(current_step="Loading Adjust Image")
                adjust_image = RSImage(adjust_meta,device=args.device)
                pairs:List[Pair] = []
                for ref_idx in ref_list:
                    reporter.update(current_step="Loading Ref Image")
                    ref_image = RSImage(ref_metas[ref_idx],device=args.device)

                    pair = build_adj_ref_pair(args,adjust_image,ref_image,reporter)
                    pairs.append(pair)

                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    affine = pair.solve_affines(encoder,predictor).detach().cpu()
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    model_time += end_time - start_time
                    adjust_image.affine_list.append(affine)

                reporter.update(current_task="Baking RPC", level="-")
                total_affine = adjust_image.merge_affines()
                reporter.log(f"Affine Matrix of Image {adjust_image.id}\n{affine}\n")
                adjust_image.rpc.Update_Adjust(total_affine)
                local_results[adjust_image.id] = total_affine.detach().cpu()

                reporter.update(current_task="Check Error", level="-", current_step="-")
                if not adjust_image.tie_points is None:
                    for pair in pairs:
                        ref_points = pair.rs_image_b.get_ref_points()
                        dis = pair.rs_image_a.check_error(ref_points)
                        checkpoint_vis = pair.rs_image_a.vis_checkpoints(ref_points)
                        cv2.imwrite(os.path.join(args.output_path,f"ckpts_{pair.id_a}_{pair.id_b}.png"),checkpoint_vis)
                        report = get_report_dict(dis)
                        reporter.log("\n" + f"--- Adj {pair.id_a} => Ref {pair.id_b}  Error Report ---")
                        reporter.log(f"Total tie points checked: {report['count']}")
                        reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
                        reporter.log(f"Median Error: {report['median']:.4f} pix")
                        reporter.log(f"Max Error:    {report['max']:.4f} pix")
                        reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
                        reporter.log(f"< 1.0 pix: {report['<1pix_percent']:.2f} %")
                        reporter.log(f"< 3.0 pix: {report['<3pix_percent']:.2f} %")
                        reporter.log(f"< 5.0 pix: {report['<5pix_percent']:.2f} %")
                else:
                    for pair in pairs:
                        try:
                            vis_registration(pair.rs_image_a,pair.rs_image_b,os.path.join(args.output_path),device=args.device)
                        except:
                            reporter.log(f"{pair.rs_image_a.id} --- {pair.rs_image_b.id} vis registration error, pass")
                            pass
                

            reporter.update(current_task="Finished", progress=f"{len(adjust_metas)}/{len(adjust_metas)}", level="-", current_step="Cleanup")
            reporter.log(f"model time: {model_time} s")
            del encoder
            del predictor
            for pair in pairs:
                del pair.rs_image_a
                del pair.rs_image_b
                del pair
            encoder = None
            predictor = None
            pair = None
            
        # ddp收集results
        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results = [None for _ in range(world_size)]
        else:
            all_results = None

        dist.gather_object(local_results, all_results if rank == 0 else None, dst=0)
        dist.barrier()
        
        if rank == 0:
            all_results = {k:v for d in all_results if d for k,v in d.items()}
            reporter.update(current_step="Loading Images")
            images = [RSImage(meta,device=args.device) for meta in adjust_metas_all]
            for image in images:
                M = all_results[image.id]
                image.rpc.Update_Adjust(M)
                if args.output_rpc:
                    image.rpc.Merge_Adjust()
                    image.rpc.save_rpc_to_file(os.path.join(args.output_path,f"{image.root.replace('/','_')}_rpc.txt"))
                # image.rpc.Merge_Adjust()

            if not images[0].tie_points is None:
                ref_image = RSImage_Error_Check(ref_metas_all[0],device=args.device)
                heights = ref_image.heights
                all_distances = []
                for i,j in itertools.combinations(range(len(images)),2):
                    ref_points = images[i].get_ref_points(heights)
                    distances = images[j].check_error(ref_points)
                    all_distances.append(distances)
                        
                all_distances = np.concatenate(all_distances)

                report = get_report_dict(all_distances)
                reporter.log("\n" + "--- Global Error Report (Summary) ---")
                reporter.log(f"Total tie points checked: {report['count']}")
                reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
                reporter.log(f"Median Error: {report['median']:.4f} pix")
                reporter.log(f"Max Error:    {report['max']:.4f} pix")
                reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
                reporter.log(f"< 1.0 pix: {report['<1pix_percent']:.2f} %")
                reporter.log(f"< 3.0 pix: {report['<3pix_percent']:.2f} %")
                reporter.log(f"< 5.0 pix: {report['<5pix_percent']:.2f} %")
                
            
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

    parser.add_argument('--predictor_path', type=str, default='weights/predictor.pth')

    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')

    parser.add_argument('--predictor_iter_num', type=int, default=None)

    parser.add_argument('--use_adapter',type=str2bool,default=True)

    parser.add_argument('--use_conf',type=str2bool,default=True)
    
    parser.add_argument('--use_mtf',type=str2bool,default=True)

    #==============================================================================


    #==============================求解相关设置=====================================

    parser.add_argument('--max_window_size', type=int, default=8000)

    parser.add_argument('--min_window_size', type=int, default=500)

    parser.add_argument('--max_window_num', type=int, default=256)

    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)

    parser.add_argument('--quad_split_times', type=int, default=1)

    #================================================================================


    #==============================输出相关设置=====================================
    
    parser.add_argument('--output_path', type=str, default='results')
    
    #===============================================================================


    #==============================实验相关设置======================================
    
    parser.add_argument('--experiment_id', type=str, default=None)

    parser.add_argument('--random_seed',type=int,default=42)

    parser.add_argument('--output_rpc',type=str2bool,default=False)

    parser.add_argument('--usgs_dem',type=str2bool,default=False)

    #==============================================================================


    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())
    
    args.output_path = os.path.join(args.output_path,args.experiment_id)
    os.makedirs(args.output_path,exist_ok=True)

    main(args)