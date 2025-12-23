import sys
import os

# 将项目根目录添加到 pythonpath
sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
from typing import List, Tuple
import itertools
import traceback
import cv2
import torch
import torch.distributed as dist
import torch.nn.functional as F

# 尝试导入 kornia
try:
    from kornia.feature import LoFTR
except ImportError:
    print("Error: kornia not installed. Please run 'pip install kornia' to use LoFTR.")
    sys.exit(1)

# 复用现有的工具函数
from shared.utils import str2bool, get_current_time
from shared.rpc import RPCModelParameterTorch
from infer.utils import is_overlap, get_report_dict, find_intersection, find_squares, apply_H
from infer.rs_image import RSImage, RSImageMeta, RSImage_Error_Check
from infer.monitor import StatusMonitor, StatusReporter

def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images_meta(args, reporter) -> Tuple[List[RSImageMeta], List[RSImageMeta]]:
    """加载影像元数据"""
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
    for idx, folder in enumerate(adjust_img_folders):
        img_path = os.path.join(adjust_base_path, folder)
        adjust_metas.append(RSImageMeta(args, img_path, idx, args.device))
    for idx, folder in enumerate(ref_img_folders):
        img_path = os.path.join(ref_base_path, folder)
        ref_metas.append(RSImageMeta(args, img_path, idx, args.device))
    return adjust_metas, ref_metas

def get_ref_lists(args, adjust_metas, ref_metas, reporter):
    ref_lists = []
    for i in range(len(adjust_metas)):
        ref_list = []
        for j in range(len(ref_metas)):
            if is_overlap(adjust_metas[i], ref_metas[j], args.min_window_size ** 2):
                ref_list.append(j)
        ref_lists.append(ref_list)
    return ref_lists

@torch.no_grad()
def run_loftr_matching(loftr_model, img_a_np, img_b_np, device):
    """
    运行 LoFTR 模型
    Args:
        img_a_np: (H, W, 3) uint8 numpy
        img_b_np: (H, W, 3) uint8 numpy
    Returns:
        pts_a: (N, 2) numpy
        pts_b: (N, 2) numpy
    """
    # 预处理：转灰度 -> Tensor -> Normalize -> Batch
    img0 = cv2.cvtColor(img_a_np, cv2.COLOR_RGB2GRAY)
    img1 = cv2.cvtColor(img_b_np, cv2.COLOR_RGB2GRAY)
    
    img0 = torch.from_numpy(img0).float().to(device) / 255.0
    img1 = torch.from_numpy(img1).float().to(device) / 255.0
    
    # 增加 batch 和 channel 维: (1, 1, H, W)
    batch = {'image0': img0.unsqueeze(0).unsqueeze(0), 'image1': img1.unsqueeze(0).unsqueeze(0)}
    
    # 推理
    correspondences = loftr_model(batch)
    
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    # mconf = correspondences['confidence'].cpu().numpy() # 如需基于置信度再次筛选

    if len(mkpts0) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
        
    # 几何验证 (RANSAC)
    M, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, 3.0, 0.99)
    if M is None:
        return np.empty((0, 2)), np.empty((0, 2))
        
    mask = mask.ravel().astype(bool)
    return mkpts0[mask], mkpts1[mask]

def solve_pair_affine_loftr(args, loftr_model, adjust_image: RSImage, ref_image: RSImage, reporter):
    # 1. 窗口生成
    corners_a = adjust_image.corner_xys
    corners_b = ref_image.corner_xys
    polygon_corners = find_intersection(np.stack([corners_a, corners_b], axis=0))
    window_diags = find_squares(polygon_corners, args.max_window_size, args.min_window_size, args.min_cover_area_ratio)
    
    if len(window_diags) == 0: return None

    # 限制窗口数量
    if args.max_window_num > 0 and len(window_diags) > args.max_window_num:
        idxs = np.random.choice(range(len(window_diags)), args.max_window_num, replace=False)
        window_diags = window_diags[idxs]

    # 2. 裁切 (LoFTR 需要能够被 8 整除的尺寸，这里 crop_size 设为 640 或 800)
    corners_linesamps_a = adjust_image.convert_diags_to_corners(window_diags)
    corners_linesamps_b = ref_image.convert_diags_to_corners(window_diags, ref_image.rpc)

    imgs_a, _, Hs_a = adjust_image.crop_windows(corners_linesamps_a, output_size=(args.crop_size, args.crop_size))
    imgs_b, _, Hs_b = ref_image.crop_windows(corners_linesamps_b, output_size=(args.crop_size, args.crop_size))

    all_pts_a_global = []
    all_pts_b_global = []

    # 3. 匹配
    for i in range(len(imgs_a)):
        if np.isnan(imgs_a[i]).any() or np.isnan(imgs_b[i]).any(): continue
            
        pts_a_crop, pts_b_crop = run_loftr_matching(loftr_model, imgs_a[i], imgs_b[i], args.device)
        
        if len(pts_a_crop) == 0: continue

        # 4. 坐标还原
        pts_a_rc = pts_a_crop[:, [1,0]] # (x,y) -> (row,col)
        pts_b_rc = pts_b_crop[:, [1,0]]
        
        pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(args.device)
        pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(args.device)
        
        H_a_inv = torch.from_numpy(np.linalg.inv(Hs_a[i])).unsqueeze(0).to(args.device, dtype=torch.float32)
        H_b_inv = torch.from_numpy(np.linalg.inv(Hs_b[i])).unsqueeze(0).to(args.device, dtype=torch.float32)
        
        pts_a_global = apply_H(pts_a_rc_t, H_a_inv, args.device).squeeze(0).cpu().numpy()
        pts_b_global = apply_H(pts_b_rc_t, H_b_inv, args.device).squeeze(0).cpu().numpy()
        
        all_pts_a_global.append(pts_a_global)
        all_pts_b_global.append(pts_b_global)

    if not all_pts_a_global: return None

    pts_a_obs = np.concatenate(all_pts_a_global, axis=0) # (row, col)
    pts_b_obs = np.concatenate(all_pts_b_global, axis=0)

    # 5. RPC 投影
    heights = ref_image.dem_interp(pts_b_obs[:, ::-1]) 
    lons, lats = ref_image.rpc.RPC_PHOTO2OBJ(pts_b_obs[:, 1], pts_b_obs[:, 0], heights, 'numpy')
    samps_proj, lines_proj = adjust_image.rpc.RPC_OBJ2PHOTO(lons, lats, heights, 'numpy')
    pts_a_proj = np.stack([lines_proj, samps_proj], axis=-1)

    # 6. 计算仿射变换
    src_pts = pts_a_obs[:, ::-1].astype(np.float32)
    dst_pts = pts_a_proj[:, ::-1].astype(np.float32)
    
    affine_xy, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if affine_xy is None: return None
        
    affine_rc = np.array([
        [affine_xy[1, 1], affine_xy[1, 0], affine_xy[1, 2]],
        [affine_xy[0, 1], affine_xy[0, 0], affine_xy[0, 2]]
    ], dtype=np.float32)

    return torch.from_numpy(affine_rc)

def main(args):
    # DDP 初始化
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        dist.init_process_group(backend="nccl")
    else:
        args.device = "cpu"
        dist.init_process_group(backend="gloo")

    experiment_id_clean = str(args.experiment_id).replace(":", "_").replace(" ", "_")
    monitor = None
    if rank == 0:
        monitor = StatusMonitor(world_size, experiment_id_clean)
        monitor.start()
    reporter = StatusReporter(rank, world_size, experiment_id_clean, monitor)

    try:
        init_random_seed(args.random_seed)

        # 加载 LoFTR 模型
        reporter.update(current_step="Loading Model")
        
        # 针对无外网环境：如果提供了权重路径，从本地加载
        if args.loftr_weight_path is not None:
            if not os.path.exists(args.loftr_weight_path):
                raise FileNotFoundError(f"LoFTR weights not found at: {args.loftr_weight_path}")
            
            reporter.log(f"Loading LoFTR weights from local path: {args.loftr_weight_path}")
            # pretrained=None 避免下载，默认配置通常兼容 outdoor 权重结构
            loftr_model = LoFTR(pretrained=None)
            
            # 加载权重文件
            checkpoint = torch.load(args.loftr_weight_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            loftr_model.load_state_dict(state_dict)
            loftr_model = loftr_model.to(args.device).eval()
        else:
            # 默认尝试下载 (如果服务器有网)
            reporter.log("No local weights provided, attempting to download 'outdoor' weights...")
            loftr_model = LoFTR(pretrained='outdoor').to(args.device).eval()

        adjust_metas_all = []
        if rank == 0:
            adjust_metas_all, ref_metas_all = load_images_meta(args, reporter)
            ref_lists = get_ref_lists(args, adjust_metas_all, ref_metas_all, reporter)
            
            ref_metas_lists = [[ref_metas_all[i] for i in sub_list] for sub_list in ref_lists]
            adjust_metas_chunk = np.array_split(np.array(adjust_metas_all, dtype=object), world_size)
            ref_metas_chunk = np.array_split(np.array(ref_metas_lists, dtype=object), world_size)
        
        reporter.update(current_step="Syncing Meta")
        scatter_adjust_metas = [None]
        dist.scatter_object_list(scatter_adjust_metas, adjust_metas_chunk if rank == 0 else None, src=0)
        adjust_metas = scatter_adjust_metas[0]
        
        scatter_ref_metas = [None]
        dist.scatter_object_list(scatter_ref_metas, ref_metas_chunk if rank == 0 else None, src=0)
        ref_metas = scatter_ref_metas[0]

        local_results = {}
        
        if len(adjust_metas) > 0:
            reporter.update(current_task="LoFTR Matching")
            
            for i in range(len(adjust_metas)):
                adj_meta = adjust_metas[i]
                current_refs = ref_metas[i]
                
                if len(current_refs) == 0: continue
                    
                adjust_image = RSImage(adj_meta, device=args.device)
                reporter.update(progress=f"{i+1}/{len(adjust_metas)}")
                
                for ref_meta in current_refs:
                    ref_image = RSImage(ref_meta, device=args.device)
                    reporter.update(current_step=f"{adjust_image.id}=>{ref_image.id}")
                    affine = solve_pair_affine_loftr(args, loftr_model, adjust_image, ref_image, reporter)
                    
                    if affine is not None:
                        adjust_image.affine_list.append(affine)
                    
                    del ref_image
                
                if len(adjust_image.affine_list) > 0:
                    merged_affine = adjust_image.merge_affines()
                    local_results[adjust_image.id] = merged_affine.cpu()
                
                del adjust_image

        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results_list = [None for _ in range(world_size)]
        else:
            all_results_list = None
        dist.gather_object(local_results, all_results_list if rank == 0 else None, dst=0)

        if rank == 0:
            reporter.update(current_task="Evaluation", current_step="Checking Errors")
            full_results = {}
            for res in all_results_list:
                if res: full_results.update(res)
            
            images = [RSImage_Error_Check(meta, device=args.device) for meta in adjust_metas_all]
            all_distances = []
            
            for image in images:
                if image.id in full_results:
                    M = full_results[image.id].to(args.device)
                    image.rpc.Update_Adjust(M)
            
            for i, j in itertools.combinations(range(len(images)), 2):
                if images[i].tie_points is not None and images[j].tie_points is not None:
                     ref_points = images[i].get_ref_points()
                     distances = images[j].check_error(ref_points)
                     all_distances.append(distances)
            
            if len(all_distances) > 0:
                all_distances = np.concatenate(all_distances)
                report = get_report_dict(all_distances)
                reporter.log("\n" + "--- LoFTR Baseline Global Error Report ---")
                reporter.log(f"Total tie points checked: {report['count']}")
                reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
                reporter.log(f"Median Error: {report['median']:.4f} pix")
                reporter.log(f"Max Error:    {report['max']:.4f} pix")
                reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
                reporter.log(f"< 1.0 pix: {report['<1m_percent']:.2f} %")
                reporter.log(f"< 3.0 pix: {report['<3m_percent']:.2f} %")
                reporter.log(f"< 5.0 pix: {report['<5m_percent']:.2f} %")
            else:
                reporter.log("No valid validation pairs found.")

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
    
    parser.add_argument('--root', type=str, help='path to dataset')
    parser.add_argument('--select_adjust_imgs', type=str, default='-1')
    parser.add_argument('--select_ref_imgs', type=str, default='-1')
    parser.add_argument('--output_path', type=str, default='results_loftr')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    
    parser.add_argument('--max_window_size', type=int, default=8000)
    parser.add_argument('--min_window_size', type=int, default=500)
    parser.add_argument('--max_window_num', type=int, default=64)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    
    # LoFTR 的 crop_size 最好能被 8 整除，推荐 640 或 840
    parser.add_argument('--crop_size', type=int, default=640, help="LoFTR input patch size")
    
    # 新增模型权重路径参数
    parser.add_argument('--loftr_weight_path', type=str, default=None, 
                        help="Path to local LoFTR weights file (e.g., loftr_outdoor.ckpt). If None, tries to download.")

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time() + "_LoFTR"
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())
    
    args.output_path = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)