import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List


from global_affine_solver import GlobalAffineSolver

# 为了测试方便，我们在这里定义辅助函数，实际运行时请确保您的环境中可以访问 RSImage 等类
from infer.rs_image import RSImage

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

def evaluate_and_visualize(images: List[RSImage], pair_results: List[dict], Ms: torch.Tensor, output_dir: str):
    """
    对求解结果进行定量评估和可视化
    """
    os.makedirs(output_dir, exist_ok=True)
    
    errors_before = []
    errors_after = []
    
    print("Evaluating results...")
    
    # 随机选择一些锚点进行验证
    grid_size = 5
    
    for pair_idx, pair in enumerate(pair_results):
        ids = list(pair.keys())
        id_i, id_j = int(ids[0]), int(ids[1])
        
        M_ij_net = pair[id_i] # 网络预测的相对变换 (i->j)
        
        M_i_opt = Ms[id_i] # 优化后的全局变换 i
        M_j_opt = Ms[id_j] # 优化后的全局变换 j
        
        img_i = images[id_i]
        img_j = images[id_j]
        
        # 生成测试点 (img_i 上的像素)
        h, w = img_i.H, img_i.W
        x = np.linspace(0, w-1, grid_size)
        y = np.linspace(0, h-1, grid_size)
        xx, yy = np.meshgrid(x, y)
        pts_i = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        pts_i_tensor = torch.from_numpy(pts_i).float().to(img_i.device)
        
        # --- 计算误差 ---
        
        # 1. 目标真值 (Target): 基于网络预测的相对关系
        # Target = Project(M_ij_net * P_i) -> projected to j
        pts_i_prime_net = apply_affine_torch(M_ij_net, pts_i_tensor)
        # 获取高程用于投影
        heights = get_heights_torch(img_i, pts_i_tensor)
        # 投影到 J
        lats, lons = img_i.rpc.RPC_PHOTO2OBJ(pts_i_prime_net[:,0], pts_i_prime_net[:,1], heights)
        samps_j_target, lines_j_target = img_j.rpc.RPC_OBJ2PHOTO(lats, lons, heights)
        pts_j_target = torch.stack([samps_j_target, lines_j_target], dim=-1)
        
        # 2. 优化前 (Before): 直接投影 P_i 到 j (假设初始M=Identity)
        lats_raw, lons_raw = img_i.rpc.RPC_PHOTO2OBJ(pts_i_tensor[:,0], pts_i_tensor[:,1], heights)
        samps_j_raw, lines_j_raw = img_j.rpc.RPC_OBJ2PHOTO(lats_raw, lons_raw, heights)
        pts_j_raw = torch.stack([samps_j_raw, lines_j_raw], dim=-1)
        
        # 3. 优化后 (After): 
        # 我们验证方程的一致性: M_j_opt * Ideal_Target_in_J 是否接近 Project(M_i_opt * P_i)
        # 这里的 Ideal_Target_in_J 实际上就是 P_j_target (未经过 M_j 修正的观测值)
        # 我们的方程是 M_j * P_j_hat = Project(M_i * P_i)
        
        # Term 1: M_j_opt * pts_j_target
        term1 = apply_affine_torch(M_j_opt, pts_j_target)
        
        # Term 2: Project(M_i_opt * P_i)
        pts_i_opt = apply_affine_torch(M_i_opt, pts_i_tensor)
        heights_opt = get_heights_torch(img_i, pts_i_opt) # 高程应该查新的位置
        lats_opt, lons_opt = img_i.rpc.RPC_PHOTO2OBJ(pts_i_opt[:,0], pts_i_opt[:,1], heights_opt)
        samps_j_opt, lines_j_opt = img_j.rpc.RPC_OBJ2PHOTO(lats_opt, lons_opt, heights_opt)
        term2 = torch.stack([samps_j_opt, lines_j_opt], dim=-1)
        
        # 残差 (Pixels)
        diff_after = torch.norm(term1 - term2, dim=1).detach().cpu().numpy()
        # 初始残差 (假设 M=I)
        diff_before = torch.norm(pts_j_target - pts_j_raw, dim=1).detach().cpu().numpy()
        
        errors_before.extend(diff_before)
        errors_after.extend(diff_after)
        
        # --- 可视化 ---
        if pair_idx < 5: # 只画前5对
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # 图1：初始误差
            # 箭头起点：pts_j_target (理想目标点)
            # 箭头指向：pts_j_raw (未修正的原始投影点)
            # 红色箭头表示：由于未修正导致的偏差
            ax[0].set_title(f"Pair {id_i}->{id_j}: Initial Discrepancy\nMean: {np.mean(diff_before):.2f} px")
            ax[0].quiver(pts_j_target[:,0].cpu(), pts_j_target[:,1].cpu(), 
                         (pts_j_raw - pts_j_target)[:,0].cpu(), (pts_j_raw - pts_j_target)[:,1].cpu(),
                         color='r', angles='xy', scale_units='xy', scale=0.1)
            ax[0].invert_yaxis()
            ax[0].set_aspect('equal')
            
            # 图2：优化后残差
            # 箭头起点：term1 (M_j 修正后的目标点)
            # 箭头指向：term2 (M_i 修正后投影过来的点)
            # 绿色箭头表示：优化后的一致性误差 (应该很小)
            ax[1].set_title(f"Optimized Residual\nMean: {np.mean(diff_after):.2f} px")
            ax[1].quiver(term1[:,0].cpu(), term1[:,1].cpu(), 
                         (term2 - term1)[:,0].cpu(), (term2 - term1)[:,1].cpu(),
                         color='g', angles='xy', scale_units='xy', scale=1) # scale=1 means exact pixel error
            ax[1].invert_yaxis()
            ax[1].set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pair_{id_i}_{id_j}_error.png"))
            plt.close()

    print("\n=== Global Solver Evaluation ===")
    print(f"Mean Residual Before: {np.mean(errors_before):.4f} pixels")
    print(f"Mean Residual After:  {np.mean(errors_after):.4f} pixels")
    if np.mean(errors_before) > 0:
        print(f"Error Reduction: {(1 - np.mean(errors_after)/np.mean(errors_before))*100:.2f}%")
    else:
        print("Initial error is zero.")

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
    solver = GlobalAffineSolver(images, device=device, lambda_anchor=1e7)
    
    # 求解
    Ms = solver.solve(pair_results)
    
    print("\nResult Affine Matrices (First 3):")
    print(Ms[:3])
    
    # 评估
    evaluate_and_visualize(images, pair_results, Ms, "./results/test_solver_output")

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