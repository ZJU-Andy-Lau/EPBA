import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List
import itertools

from global_affine_solver import GlobalAffineSolver,TopologyAffineSolver

# 为了测试方便，我们在这里定义辅助函数，实际运行时请确保您的环境中可以访问 RSImage 等类
from infer.rs_image import RSImage
from infer.utils import haversine_distance

def apply_affine_torch(M, pts):
    # pts: (N, 2)
    # M: (2, 3)
    pts_homo = torch.cat([pts, torch.ones(pts.shape[0], 1, device=pts.device)], dim=1).to(torch.float32)
    M = M.to(torch.float32)
    return (M @ pts_homo.T).T

def get_heights_torch(img, pts):
    # 模拟从 img 获取高程，调用 img.dem_interp
    pts_np = pts.detach().cpu().numpy()
    return torch.from_numpy(img.dem_interp(pts_np)).float().to(pts.device)

def check_error(rs_image_a:RSImage,rs_image_b:RSImage):
    lines_i = rs_image_a.tie_points[:,0]
    samps_i = rs_image_a.tie_points[:,1]
    heights_i = rs_image_a.dem[lines_i,samps_i]
    lats_i, lons_i = rs_image_a.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
    coords_i = np.stack([lats_i, lons_i], axis=-1)
    
    lines_j = rs_image_b.tie_points[:,0]
    samps_j = rs_image_b.tie_points[:,1]
    heights_j = rs_image_b.dem[lines_j,samps_j]
    lats_j, lons_j = rs_image_b.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
    coords_j = np.stack([lats_j, lons_j], axis=-1)

    distances = haversine_distance(coords_i, coords_j)
    return distances

def evaluate_and_visualize(images: List[RSImage], Ms: torch.Tensor):
    for i,image in enumerate(images):
        image.rpc.Update_Adjust(Ms[i])
        image.rpc.Merge_Adjust()
    all_distances = []
    for i,j in itertools.combinations(range(len(images)),2):
        image_a = images[i]
        image_b = images[j]
        distance = check_error(image_a,image_b)
        all_distances.append(distance)
    all_distances = np.concatenate(all_distances)
    total_points = len(all_distances)
    report = {
        'mean': float(np.mean(all_distances)),
        'median': float(np.median(all_distances)),
        'max': float(np.max(all_distances)),
        'rmse': float(np.sqrt(np.mean(all_distances**2))),
        'count': int(total_points),
        '<1m_percent': float(((all_distances < 1.0).sum() / total_points) * 100),
        '<3m_percent': float(((all_distances < 3.0).sum() / total_points) * 100),
        '<5m_percent': float(((all_distances < 5.0).sum() / total_points) * 100),
    }
    for key in report.keys():
        print(f"{key}:{report[key]}")
    

def run_test_with_provided_data(images: List[RSImage], pair_results: List[dict]):
    """
    用户调用此函数传入真实数据进行测试
    Args:
        images: 真实的 RSImage 对象列表
        pair_results: 真实的配对预测结果列表
    """
    if not images or not pair_results:
        print("Error: Empty data provided.")
        return

    print("Running Global Affine Solver Test with provided data...")
    device = images[0].device
    
    # 实例化求解器
    # solver = GlobalAffineSolver(images, device=device,anchor_indices=[2],converge_tol=1e-8,max_iter=50)
    solver = TopologyAffineSolver(images, device=device,anchor_indices=[2])
    
    # 求解
    Ms = solver.solve(pair_results)
    
    print("\nResult Affine Matrices (First 3):")
    print(Ms[:3])
    
    # 评估
    evaluate_and_visualize(images, Ms)

if __name__ == "__main__":
    root = './datasets/wv_test_error_5/adjust_images'
    img_paths = os.listdir(root)
    images = [RSImage({},os.path.join(root,img_paths[i]),i,device='cuda') for i in range(len(img_paths))]
    pair_results = [
        {
            0:torch.tensor([
                [1.0,-1.1280e-5,7.6001],
                [2.0645e-5,9.9988e-1,2.9490]
            ],device='cuda'),
            1:torch.tensor([
                [9.9997e-1,1.7625e-5,-7.6724],
                [-1.6173e-5,1.0001,-3.0122]
            ],device='cuda'),
        },
        {
            0:torch.tensor([
                [0.99999,6.6777e-5,1.9881],
                [1.5201e-5,0.99999,8.1751]
            ],device='cuda'),
            2:torch.tensor([
                [1.0,-6.6743e-5,-1.9009],
                [-2.3943e-5,1.0,-8.1431]
            ],device='cuda'),
        },{
            1:torch.tensor([
                [9.9994e-1,1.6867e-5,-5.0975],
                [-1.4066e-5,1.0,5.9996]
            ],device='cuda'),
            2:torch.tensor([
                [1.0001,-5.0414e-6,5.0313],
                [6.0914e-6,0.99999,-5.8980]
            ],device='cuda'),
        }
    ]
    # print("Test script loaded. Use 'run_test_with_provided_data(images, pair_results)' to execute.")
    run_test_with_provided_data(images,pair_results)