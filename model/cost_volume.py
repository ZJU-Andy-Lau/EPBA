import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.utils import debug_print
import time

class CostVolume:
    def __init__(self, fmap_query, fmap_ref, num_levels=2, radius=4):
        """
        构建全对相关性金字塔。
        
        策略：保持 Query 分辨率不变，对 Ref 维度的相关性进行池化。
        
        Args:
            fmap_query (Image A): [B, C, H, W]
            fmap_ref (Image B): [B, C, H, W]
        """
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # 1. 准备数据
        B, C, H, W = fmap_query.shape
        self.H_query, self.W_query = H, W
        
        # [B, C, H, W] -> [B, C, H*W]
        query_flat = fmap_query.reshape(B, C, -1).contiguous()
        ref_flat = fmap_ref.reshape(B, C, -1).contiguous()
        
        # 2. 计算全对相关性 (All-Pairs Correlation)
        # Output: [B, H*W (Query), H*W (Ref)]
        corr = torch.matmul(query_flat.transpose(1, 2), ref_flat)
        # 4. 重塑为 [B, H, W, 1, H, W] 以便后续处理
        # 前两个 H,W 对应 Query 像素位置（保持不变）
        # 后两个 H,W 对应 Ref 像素位置（将被池化）
        corr = corr.reshape(B, H, W, 1, H, W).contiguous()
        
        self.corr_pyramid.append(corr)

        # 5. 构建金字塔 (对 Ref 维度进行池化)
        for _ in range(num_levels - 1):
            # 获取当前 Ref 维度
            curr_h_ref = corr.shape[-2]
            curr_w_ref = corr.shape[-1]
            
            # Reshape 为 [N, C, H_ref, W_ref] 以使用 avg_pool2d
            # N = B * H_query * W_query
            corr_reshaped = corr.reshape(-1, 1, curr_h_ref, curr_w_ref).contiguous()
            
            # 执行池化
            pooled = F.avg_pool2d(corr_reshaped, kernel_size=2, stride=2)
            
            # 恢复形状: [B, H, W, 1, H_new, W_new]
            _, _, new_h_ref, new_w_ref = pooled.shape
            corr = pooled.reshape(B, H, W, 1, new_h_ref, new_w_ref).contiguous()
            
            self.corr_pyramid.append(corr)

    def lookup(self, coords):
        """
        在金字塔中查找邻域相关性。
        
        Args:
            coords: [B, H, W, 2] - Query像素在Ref坐标系下的归一化坐标 [-1, 1] (row,col)
            
        Returns:
            out: [B, Channels, H, W] - 查表特征
            out_coords: [B, Channels, 2, H, W] - 采样点在Level 0 Ref图上的像素坐标
        """
        r = self.radius
        B, H, W, _ = coords.shape
        
        # 生成局部邻域网格偏移 delta: [-r, ..., r]
        dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
        dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
        # meshgrid('ij') -> dy (row), dx (col)
        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1).reshape(-1, 2).contiguous()
        
        out_pyramid = []
        out_coords_list = []
        
        # 获取 Level 0 Ref 维度 (未经下采样的 Ref 尺寸)
        _, _, _, _, H_ref_0, W_ref_0 = self.corr_pyramid[0].shape
        
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i] # [B, H, W, 1, H_ref, W_ref]
            _, _, _, _, H_ref_lvl, W_ref_lvl = corr.shape
            
            # 1. 坐标映射: 归一化 [-1, 1] -> 像素坐标 [0, W-1]
            coords_lvl = coords.clone()
            coords_lvl[..., 0] = (coords_lvl[..., 0] + 1.0) * (H_ref_lvl - 1.0) / 2.0
            coords_lvl[..., 1] = (coords_lvl[..., 1] + 1.0) * (W_ref_lvl - 1.0) / 2.0
            
            # 2. 加上局部偏移: [B, H, W, 1, 2] + [1, 1, 1, N, 2] -> [B, H, W, N, 2]
            coords_new = coords_lvl.unsqueeze(-2) + delta.reshape(1, 1, 1, -1, 2) # [B, H, W, N, 2]

            # debug_print(f"===========lvl {i} img32===========")
            # debug_print(f"{coords_new[0,2,2,[0,10,20,30,40,50,60,70,80]]}\n\n{coords_new[0,16,16,[0,10,20,30,40,50,60,70,80]]}\n\n")
            
            # 3. 映射回归一化坐标 [-1, 1] 用于 grid_sample
            coords_norm = coords_new.clone() #[B, H, W, N, 2]
            coords_norm[..., 0] = 2.0 * coords_norm[..., 0] / (H_ref_lvl - 1.0) - 1.0
            coords_norm[..., 1] = 2.0 * coords_norm[..., 1] / (W_ref_lvl - 1.0) - 1.0

            # debug_print(f"===========lvl {i} norm===========")
            # debug_print(f"{coords_norm[0,2,2,[0,10,20,30,40,50,60,70,80]]}\n\n{coords_norm[0,16,16,[0,10,20,30,40,50,60,70,80]]}\n\n")
            
            # --- 计算 Level 0 像素坐标 ---
            # 将归一化坐标映射回 Level 0 像素空间
            coords_lvl0 = torch.zeros_like(coords_norm) # line,samp
            coords_lvl0[..., 0] = (coords_norm[..., 0] + 1.0) * (H_ref_0 * 16 - 1.0) / 2.0 # 16为提取特征图时的下采样倍率，乘以16转化为原图尺寸
            coords_lvl0[..., 1] = (coords_norm[..., 1] + 1.0) * (W_ref_0 * 16 - 1.0) / 2.0
            
            # debug_print(f"===========lvl {i} img512===========")
            # debug_print(f"{coords_lvl0[0,2,2,[0,10,20,30,40,50,60,70,80]]}\n\n{coords_lvl0[0,16,16,[0,10,20,30,40,50,60,70,80]]}\n\n")

            # N 对应 Channels 维度的一部分
            out_coords_list.append(coords_lvl0) #[B, H, W, N, 2]
            # ----------------------------
            
            # 4. Grid Sample
            # Input: [B*H*W, 1, H_ref, W_ref]
            # Grid:  [B*H*W, N, 1, 2]
            
            # 准备 Input
            corr_reshaped = corr.reshape(-1, 1, H_ref_lvl, W_ref_lvl).contiguous() # [B*H*W, 1, H_ref, W_ref]

            # 准备 Grid: N 个点排成一行 (N, 1)
            num_points = delta.shape[0]
            grid_reshaped = coords_norm.reshape(-1, num_points, 1, 2) # [B*H*W, N, 1, 2]
            grid_xy = grid_reshaped[..., [1, 0]].contiguous() # (row,col) -> (x,y)
            
            # 采样
            sampled = F.grid_sample(corr_reshaped, grid_xy, align_corners=True, mode='bilinear', padding_mode='zeros')
            
            # Output: [B*H*W, 1, N, 1] -> [B, H, W, N]
            sampled = sampled.view(B, H, W, -1)
            out_pyramid.append(sampled)
            
        # 拼接所有层级特征: [B, H, W, Total_Channels]
        out = torch.cat(out_pyramid, dim=-1)
        
        # 调整维度顺序: [B, Channels, H, W] 适应 CNN 输入
        out = out.permute(0, 3, 1, 2).contiguous()
        
        # 拼接所有层级坐标
        out_coords = torch.cat(out_coords_list, dim=3).contiguous() #[B, H, W, Total_Channels, 2]
        
        return out, out_coords