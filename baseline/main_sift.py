import sys
import os

# 将项目根目录添加到 pythonpath，确保能引用 infer 和 shared 中的模块
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
from tqdm import tqdm

import torch
import torch.distributed as dist

# 复用现有的工具函数
from shared.utils import str2bool, get_current_time, load_config
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
    """加载影像元数据 (与 infer/main_ref.py 逻辑一致)"""
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

def get_ref_lists(args, adjust_metas: List[RSImageMeta], ref_metas: List[RSImageMeta], reporter) -> List[List]:
    """筛选重叠影像对"""
    reporter.update(current_step="Filtering Ref")
    ref_lists = []
    for i in range(len(adjust_metas)):
        ref_list = []
        for j in range(len(ref_metas)):
            if is_overlap(adjust_metas[i], ref_metas[j], args.min_window_size ** 2):
                ref_list.append(j)
        ref_lists.append(ref_list)
    return ref_lists

def run_sift_matching(img_a, img_b):
    """
    在两个图像块之间运行 SIFT 匹配
    Args:
        img_a: (H, W, 3) uint8
        img_b: (H, W, 3) uint8
    Returns:
        pts_a: (N, 2)
        pts_b: (N, 2)
    """
    # 转灰度
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    # 简单的直方图均衡化增强对比度
    gray_a = cv2.equalizeHist(gray_a)
    gray_b = cv2.equalizeHist(gray_b)

    # SIFT 提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_a, None)
    kp2, des2 = sift.detectAndCompute(gray_b, None)

    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    # 匹配
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return np.empty((0, 2)), np.empty((0, 2))

    pts_a = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_b = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # 局部 RANSAC 几何验证 (去除局部误匹配)
    M, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 3.0, 0.99)
    if M is None:
        return np.empty((0, 2)), np.empty((0, 2))
    
    mask = mask.ravel().astype(bool)
    return pts_a[mask], pts_b[mask]

def solve_pair_affine_sift(args, adjust_image: RSImage, ref_image: RSImage, reporter):
    """
    针对一对 (Adj, Ref) 使用分块 SIFT + RPC 投影计算仿射变换
    """
    # 1. 计算重叠区域并生成窗口
    corners_a = adjust_image.corner_xys
    corners_b = ref_image.corner_xys
    polygon_corners = find_intersection(np.stack([corners_a, corners_b], axis=0))
    
    # 在重叠区内生成窗口 (使用 infer/utils.py 中的逻辑)
    window_diags = find_squares(polygon_corners, args.max_window_size, args.min_window_size, args.min_cover_area_ratio)
    
    if len(window_diags) == 0:
        return None

    # 限制窗口数量
    if args.max_window_num > 0 and len(window_diags) > args.max_window_num:
        idxs = np.random.choice(range(len(window_diags)), args.max_window_num, replace=False)
        window_diags = window_diags[idxs]

    # 2. 裁切数据
    # RSImage 的 convert_diags_to_corners 会处理坐标转换
    corners_linesamps_a = adjust_image.convert_diags_to_corners(window_diags)
    corners_linesamps_b = ref_image.convert_diags_to_corners(window_diags, ref_image.rpc)

    # crop_windows 返回: 图像, DEMs, 以及用于 Crop 的单应矩阵 Hs
    # Hs: Global(row,col) -> Crop(row,col)
    imgs_a, _, Hs_a = adjust_image.crop_windows(corners_linesamps_a, output_size=(args.crop_size, args.crop_size))
    imgs_b, _, Hs_b = ref_image.crop_windows(corners_linesamps_b, output_size=(args.crop_size, args.crop_size))

    all_pts_a_global = []
    all_pts_b_global = []

    # 3. 遍历窗口进行 SIFT 匹配
    for i in range(len(imgs_a)):
        if np.isnan(imgs_a[i]).any() or np.isnan(imgs_b[i]).any():
            continue
            
        # 匹配 (返回的是 Crop 坐标系下的 x, y)
        pts_a_crop, pts_b_crop = run_sift_matching(imgs_a[i], imgs_b[i])
        
        if len(pts_a_crop) == 0:
            continue

        # 4. 坐标还原: Crop (x,y) -> Global (row, col)
        # 注意: crop返回的Hs是 (row, col) 体系的
        # pts_a_crop 是 (x, y) 即 (col, row)
        
        # 转换为 (row, col)
        pts_a_rc = pts_a_crop[:, [1,0]]
        pts_b_rc = pts_b_crop[:, [1,0]]
        
        # 转换为 Tensor 以利用 infer/utils 中的 apply_H
        pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(args.device) # (1, N, 2)
        pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(args.device)
        
        H_a_inv = torch.from_numpy(np.linalg.inv(Hs_a[i])).unsqueeze(0).to(args.device, dtype=torch.float32)
        H_b_inv = torch.from_numpy(np.linalg.inv(Hs_b[i])).unsqueeze(0).to(args.device, dtype=torch.float32)
        
        # 应用逆变换还原到全图坐标 (row, col)
        pts_a_global = apply_H(pts_a_rc_t, H_a_inv, args.device).squeeze(0).cpu().numpy()
        pts_b_global = apply_H(pts_b_rc_t, H_b_inv, args.device).squeeze(0).cpu().numpy()
        
        all_pts_a_global.append(pts_a_global)
        all_pts_b_global.append(pts_b_global)

    if not all_pts_a_global:
        return None

    # 合并所有窗口的点
    pts_a_obs = np.concatenate(all_pts_a_global, axis=0) # Adj Image Observations (row, col)
    pts_b_obs = np.concatenate(all_pts_b_global, axis=0) # Ref Image Observations (row, col)

    # 5. 构建几何约束 (Ref Point -> Ground -> Adj Image Project)
    # 获取 Ref 点的高程
    # dem_interp 输入需为 (samp, line) 即 (x, y)，pts_b_obs 是 (line, samp)
    heights = ref_image.dem_interp(pts_b_obs[:, ::-1]) 
    
    # 反投影 Ref: Image -> Ground (Lon, Lat, H)
    lons, lats = ref_image.rpc.RPC_PHOTO2OBJ(pts_b_obs[:, 1], pts_b_obs[:, 0], heights, 'numpy')
    
    # 正投影 Adj: Ground -> Image (Projected ideal position)
    # 我们希望 Adj 上的点 pts_a_obs 移动到 pts_a_proj
    samps_proj, lines_proj = adjust_image.rpc.RPC_OBJ2PHOTO(lons, lats, heights, 'numpy')
    pts_a_proj = np.stack([lines_proj, samps_proj], axis=-1)

    # 6. 计算仿射变换
    # 模型: pts_a_proj = M * pts_a_obs
    # 输入 estimateAffine2D 需要是 (x, y) 格式
    src_pts = pts_a_obs[:, ::-1].astype(np.float32) # (col, row) -> (x, y)
    dst_pts = pts_a_proj[:, ::-1].astype(np.float32)
    
    # 使用全局 RANSAC 剔除误匹配和高程误差导致的外点
    affine_xy, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if affine_xy is None:
        return None
        
    # 将 (x,y) 仿射变换转为 (row,col) 仿射变换
    # M_xy = [[sx, shy, tx], [shx, sy, ty]]
    # M_rc = [[sy, shx, ty], [shy, sx, tx]]
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

    # Monitor
    experiment_id_clean = str(args.experiment_id).replace(":", "_").replace(" ", "_")
    monitor = None
    if rank == 0:
        monitor = StatusMonitor(world_size, experiment_id_clean)
        monitor.start()
    reporter = StatusReporter(rank, world_size, experiment_id_clean, monitor)

    try:
        init_random_seed(args.random_seed)

        # 1. 加载元数据
        adjust_metas_all = []
        if rank == 0:
            adjust_metas_all, ref_metas_all = load_images_meta(args, reporter)
            ref_lists = get_ref_lists(args, adjust_metas_all, ref_metas_all, reporter)
            
            # 构建分发列表
            ref_metas_lists = [[ref_metas_all[i] for i in sub_list] for sub_list in ref_lists]
            adjust_metas_chunk = np.array_split(np.array(adjust_metas_all, dtype=object), world_size)
            ref_metas_chunk = np.array_split(np.array(ref_metas_lists, dtype=object), world_size)
        
        # 2. 同步数据
        reporter.update(current_step="Syncing Meta")
        scatter_adjust_metas = [None]
        dist.scatter_object_list(scatter_adjust_metas, adjust_metas_chunk if rank == 0 else None, src=0)
        adjust_metas = scatter_adjust_metas[0]
        
        scatter_ref_metas = [None]
        dist.scatter_object_list(scatter_ref_metas, ref_metas_chunk if rank == 0 else None, src=0)
        ref_metas = scatter_ref_metas[0]

        local_results = {}
        
        # 3. 开始处理
        if len(adjust_metas) > 0:
            reporter.update(current_task="SIFT Matching")
            
            for i in range(len(adjust_metas)):
                adj_meta = adjust_metas[i]
                current_refs = ref_metas[i]
                
                if len(current_refs) == 0:
                    continue
                    
                adjust_image = RSImage(adj_meta, device=args.device)
                reporter.update(progress=f"{i+1}/{len(adjust_metas)}")
                
                # 遍历该 adjust image 对应的所有 ref image
                for ref_meta in current_refs:
                    ref_image = RSImage(ref_meta, device=args.device)
                    reporter.update(current_step=f"{adjust_image.id}=>{ref_image.id}")
                    
                    # 核心求解
                    affine = solve_pair_affine_sift(args, adjust_image, ref_image, reporter)
                    
                    if affine is not None:
                        adjust_image.affine_list.append(affine)
                    
                    del ref_image
                
                # 融合结果并更新
                if len(adjust_image.affine_list) > 0:
                    merged_affine = adjust_image.merge_affines()
                    local_results[adjust_image.id] = merged_affine.cpu()
                    # 可以在这里打印矩阵
                    # reporter.log(f"Image {adjust_image.id} SIFT Affine:\n{merged_affine}")
                
                del adjust_image

        # 4. 收集结果
        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results_list = [None for _ in range(world_size)]
        else:
            all_results_list = None
        dist.gather_object(local_results, all_results_list if rank == 0 else None, dst=0)

        # 5. 统一评估
        if rank == 0:
            reporter.update(current_task="Evaluation", current_step="Checking Errors")
            # 合并字典
            full_results = {}
            for res in all_results_list:
                if res:
                    full_results.update(res)
            
            # 重新加载影像进行检查 (只用 Error_Check 轻量类)
            images = [RSImage_Error_Check(meta, device=args.device) for meta in adjust_metas_all]
            all_distances = []
            
            for image in images:
                if image.id in full_results:
                    # 更新 RPC
                    M = full_results[image.id].to(args.device)
                    image.rpc.Update_Adjust(M)
            
            # 计算全连接误差
            # 这里的逻辑是：如果两张 adjust image 都有 tie_points 且地理重叠，则互相检查
            # 简化逻辑：直接利用 image.check_error(ref_points) 
            # 这里复用 main_ref.py 的逻辑: 所有的 adjust images 互相作为 ref 进行检查
            # 但前提是 adjust_images 实际上包含了 ref_images 或者我们加载了专门的 ref
            # 为了严谨复现 main_ref.py 的评估逻辑：
            for i, j in itertools.combinations(range(len(images)), 2):
                # 只有重叠且有 tie_points 才计算
                # 这里假设所有 adjust images 都在同一区域
                if images[i].tie_points is not None and images[j].tie_points is not None:
                     ref_points = images[i].get_ref_points()
                     distances = images[j].check_error(ref_points)
                     all_distances.append(distances)
            
            if len(all_distances) > 0:
                all_distances = np.concatenate(all_distances)
                report = get_report_dict(all_distances)
                reporter.log("\n" + "--- SIFT Baseline Global Error Report ---")
                reporter.log(f"Total tie points checked: {report['count']}")
                reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
                reporter.log(f"Median Error: {report['median']:.4f} pix")
                reporter.log(f"Max Error:    {report['max']:.4f} pix")
                reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
                reporter.log(f"< 1.0 pix: {report['<1pix_percent']:.2f} %")
                reporter.log(f"< 3.0 pix: {report['<3pix_percent']:.2f} %")
                reporter.log(f"< 5.0 pix: {report['<5pix_percent']:.2f} %")
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
    
    # 数据与路径
    parser.add_argument('--root', type=str, help='path to dataset')
    parser.add_argument('--select_adjust_imgs', type=str, default='-1')
    parser.add_argument('--select_ref_imgs', type=str, default='-1')
    parser.add_argument('--output_path', type=str, default='results_sift')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    
    # 窗口设置 (影响 SIFT 的输入)
    parser.add_argument('--max_window_size', type=int, default=8000)
    parser.add_argument('--min_window_size', type=int, default=500)
    parser.add_argument('--max_window_num', type=int, default=64)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    
    # Baseline 特有设置
    parser.add_argument('--crop_size', type=int, default=512, help="SIFT input patch size")

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time() + "_SIFT"
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())

    args.output_path = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)