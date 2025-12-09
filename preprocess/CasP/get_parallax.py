import argparse
import itertools
import math
import os
import sys
import gc
import shutil
import time
from collections import defaultdict
import multiprocessing as mp
from queue import Empty

import h5py
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

# 确保可以导入 src 模块
sys.path.append(os.getcwd())
try:
    from src.models.nets import CasP
except ImportError:
    print("Error: Could not import CasP. Make sure you are in the project root.")
    sys.exit(1)

# ====================================================================
# 1. CasP 模型与图像处理工具函数
# ====================================================================

def load_matcher(config_path, ckpt_path, device):
    """加载CasP模型"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        # 实际部署时建议下载或报错
        print(f"Warning: Weights not found at {ckpt_path}")
    
    config = OmegaConf.load(config_path).config
    config.threshold = 0.2 
    
    # 抑制加载过程中的一些冗余输出
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        matcher = CasP(config)
        matcher.load_state_dict(torch.load(ckpt_path, map_location=device))
        matcher = matcher.eval().to(device)
    finally:
        sys.stdout = original_stdout
        
    return matcher

def preprocess_image(image, device):
    """(H, W) or (H, W, 3) -> (1, C, H, W) Tensor"""
    if image.ndim == 2:
        image = image[:, :, None]
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)[None]

def get_windows(image_shape, window_size=1152):
    """自适应计算最少覆盖窗口坐标"""
    h, w = image_shape[:2]
    
    if h <= window_size:
        h_starts = [0]
    else:
        num_h = math.ceil(h / window_size)
        h_starts = np.linspace(0, h - window_size, num_h).astype(int)
        h_starts = np.unique(h_starts)

    if w <= window_size:
        w_starts = [0]
    else:
        num_w = math.ceil(w / window_size)
        w_starts = np.linspace(0, w - window_size, num_w).astype(int)
        w_starts = np.unique(w_starts)

    # 生成 (top, left) 列表
    window_coords = []
    for top in h_starts:
        for left in w_starts:
            window_coords.append((top, left))
    return window_coords

def clear_matcher_cache(matcher):
    """清理 CasP 内部缓存"""
    if hasattr(matcher, "fine_reg_matching"):
        fine = matcher.fine_reg_matching
        if hasattr(fine, "coords0") and isinstance(fine.coords0, dict): fine.coords0.clear()
        if hasattr(fine, "coords1") and isinstance(fine.coords1, dict): fine.coords1.clear()
        if hasattr(fine, "points") and isinstance(fine.points, dict): fine.points.clear()
        if hasattr(fine, "four_point_disp") and isinstance(fine.four_point_disp, dict): fine.four_point_disp.clear()

def match_pair_windows(matcher, img1, img2, window_coords, window_size, device):
    """匹配一对图像的所有窗口，返回全局坐标匹配点"""
    all_pts1 = []
    all_pts2 = []
    
    for top, left in window_coords:
        bottom = top + window_size
        right = left + window_size
        
        crop1 = img1[top:bottom, left:right]
        crop2 = img2[top:bottom, left:right]
        
        data = {
            "image0": preprocess_image(crop1, device),
            "image1": preprocess_image(crop2, device)
        }
        
        with torch.no_grad():
            results = matcher(data)
            
        p1 = results["points0"].cpu().numpy()
        p2 = results["points1"].cpu().numpy()
        
        if len(p1) > 0:
            # 还原为全局坐标
            p1[:, 0] += left
            p1[:, 1] += top
            p2[:, 0] += left
            p2[:, 1] += top
            
            all_pts1.append(p1)
            all_pts2.append(p2)
            
    if not all_pts1:
        return None, None
        
    return np.concatenate(all_pts1), np.concatenate(all_pts2)

# ====================================================================
# 2. Worker 进程逻辑
# ====================================================================

def worker_process(gpu_id, task_queue, args, temp_dir):
    """
    Worker 进程：负责在指定 GPU 上处理分配到的 dataset_key
    """
    # 1. 初始化环境
    try:
        # 设置只使用分配的 GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        
        # 加载模型
        matcher = load_matcher(args.config_path, args.weights_path, device)
        
        # 窗口参数
        win_size = args.window_size
        
        processed_count = 0
        
        while True:
            try:
                # 设置 timeout 防止死锁，如果 5 秒没拿到任务且队列为空则退出
                dataset_key = task_queue.get(timeout=5)
            except Empty:
                break
            
            # print(f"[GPU {gpu_id}] Processing: {dataset_key}")
            
            # 2. 读取数据 (只读模式，支持并发)
            with h5py.File(args.dataset_path, 'r') as f:
                if dataset_key not in f or 'images' not in f[dataset_key]:
                    continue
                
                img_group = f[dataset_key]['images']
                img_ids = list(img_group.keys())
                
                if len(img_ids) < 2:
                    continue
                
                # 加载图像到内存
                images = {}
                for iid in img_ids:
                    images[iid] = img_group[iid][:]
            
            # 3. 计算逻辑
            h, w = images[img_ids[0]].shape[:2]
            win_coords = get_windows((h, w), win_size)
            
            # 初始化累加器
            accumulators = defaultdict(lambda: defaultdict(list))
            
            # 全排列两两匹配
            pairs = list(itertools.combinations(img_ids, 2))
            
            for id1, id2 in pairs:
                pts1, pts2 = match_pair_windows(
                    matcher, images[id1], images[id2], win_coords, win_size, device
                )
                
                if pts1 is None:
                    continue
                
                # 计算距离
                dists = np.linalg.norm(pts1 - pts2, axis=1)
                
                # 取整坐标
                pts1_round = np.round(pts1).astype(int)
                pts2_round = np.round(pts2).astype(int)
                
                # 边界过滤
                valid1 = (pts1_round[:, 0] >= 0) & (pts1_round[:, 0] < w) & \
                         (pts1_round[:, 1] >= 0) & (pts1_round[:, 1] < h)
                valid2 = (pts2_round[:, 0] >= 0) & (pts2_round[:, 0] < w) & \
                         (pts2_round[:, 1] >= 0) & (pts2_round[:, 1] < h)
                
                # 记录到累加器
                for k in np.where(valid1)[0]:
                    accumulators[id1][(pts1_round[k, 1], pts1_round[k, 0])].append(dists[k])
                    
                for k in np.where(valid2)[0]:
                    accumulators[id2][(pts2_round[k, 1], pts2_round[k, 0])].append(dists[k])
                
                # 清理模型缓存
                clear_matcher_cache(matcher)
            
            # 4. 生成视差图并保存为 .npy
            # 文件名安全处理: 将 '/' 替换为 '___'
            safe_key = dataset_key.replace('/', '___')
            
            for iid in img_ids:
                parallax_map = np.full((h, w), np.nan, dtype=np.float32)
                acc = accumulators[iid]
                
                for (y, x), val_list in acc.items():
                    parallax_map[y, x] = np.median(val_list)
                
                # 保存临时文件
                # 格式: temp_dir / {safe_key}___{img_id}.npy
                npy_filename = f"{safe_key}___{iid}.npy"
                npy_path = os.path.join(temp_dir, npy_filename)
                np.save(npy_path, parallax_map)
            
            processed_count += 1
            # 显式垃圾回收
            del images, accumulators
            gc.collect()
            torch.cuda.empty_cache()
            
        # print(f"[GPU {gpu_id}] Finished. Processed {processed_count} datasets.")
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()

# ====================================================================
# 3. 主控逻辑
# ====================================================================

def run_pipeline(args):
    # 1. 检查和准备
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset not found at {args.dataset_path}")
        return

    temp_dir = os.path.join(os.path.dirname(args.dataset_path), "_temp_parallax_npy")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"Temporary directory created at: {temp_dir}")

    # 2. 扫描任务
    with h5py.File(args.dataset_path, 'r') as f:
        all_keys = list(f.keys())
    
    print(f"Found {len(all_keys)} datasets to process.")
    
    # 3. 设置多进程队列
    task_queue = mp.Queue()
    for key in all_keys:
        task_queue.put(key)
        
    # 4. 启动 Workers
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU detected, switching to CPU (1 process).")
        num_gpus = 1
        # 注意：CPU模式下 worker_process 中的 set_device 会被忽略或报错，这里假设有 GPU
        # 如果是纯 CPU 环境，需修改 worker_process 移除 cuda 相关调用
    else:
        print(f"Launching {num_gpus} worker processes...")

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, args, temp_dir))
        p.start()
        processes.append(p)
    
    # 5. 等待所有 Workers 完成
    for p in processes:
        p.join()
        
    print("All workers finished. Starting aggregation...")

    # 6. 聚合结果写入 HDF5
    # 遍历临时目录下的所有 npy 文件
    npy_files = os.listdir(temp_dir)
    
    if not npy_files:
        print("No results generated.")
        shutil.rmtree(temp_dir)
        return

    with h5py.File(args.dataset_path, 'r+') as f:
        for npy_file in tqdm(npy_files, desc="Aggregating to H5"):
            if not npy_file.endswith('.npy'):
                continue
            
            # 解析文件名: {safe_key}___{img_id}.npy
            # 注意: safe_key 本身可能包含 '___' 如果原始 key 有特殊构造，但这里假设 ___ 是分隔符
            # 使用 rsplit 确保只分割最后一部分作为 img_id
            basename = os.path.splitext(npy_file)[0]
            parts = basename.rsplit('___', 1)
            
            if len(parts) != 2:
                print(f"Skipping malformed filename: {npy_file}")
                continue
            
            safe_key, img_id = parts
            # 还原 key
            original_key = safe_key.replace('___', '/')
            
            # 读取数据
            npy_path = os.path.join(temp_dir, npy_file)
            parallax_map = np.load(npy_path)
            
            # 写入 HDF5
            if original_key in f:
                group = f[original_key]
                if 'parallax' not in group:
                    group.create_group('parallax')
                
                parallax_group = group['parallax']
                
                if img_id in parallax_group:
                    del parallax_group[img_id]
                
                parallax_group.create_dataset(
                    img_id, 
                    data=parallax_map, 
                    compression="gzip", 
                    compression_opts=4
                )
            
            # 删除已处理的临时文件 (节省空间)
            os.remove(npy_path)

    # 7. 清理临时目录
    shutil.rmtree(temp_dir)
    print("Processing complete. Temporary files cleaned up.")

def main():
    # 必须设置 spawn 启动方式以支持 CUDA 多进程
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Multi-GPU Parallax Generation for HDF5")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--config_path", type=str, default="preprocess/CasP/configs/model/net/casp.yaml")
    parser.add_argument("--weights_path", type=str, default="preprocess/CasP/weights/casp_outdoor.pth")
    parser.add_argument("--window_size", type=int, default=1152)
    
    args = parser.parse_args()
    
    run_pipeline(args)

if __name__ == "__main__":
    main()