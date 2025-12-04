import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os

from model.encoder import Encoder
from model.gru import GRUBlock
from model.cost_volume import CostVolume
from shared.rpc import RPCModelParameterTorch,project_linesamp
from shared.utils import debug_print,check_invalid_tensors
from shared.visualize import vis_pyramid_correlation
from criterion.utils import merge_affine

class WindowSolver():
    def __init__(self,B,H,W,
                 gru:GRUBlock,
                 feats_a,feats_b,
                 H_as:torch.Tensor,H_bs:torch.Tensor,
                 rpc_a:RPCModelParameterTorch = None,rpc_b:RPCModelParameterTorch = None,
                 height:torch.Tensor = None,
                 gru_max_iter:int = 10):
        self.gru = gru
        self.gru_access = gru.module if hasattr(gru,'module') else gru
        self.match_feats_a, self.ctx_feats_a, self.confs_a = feats_a
        self.match_feats_b, self.ctx_feats_b, self.confs_b = feats_b
        self.cost_volume_ab = None
        self.cost_volume_ba = None
        self.H_as = H_as # B,3,3
        self.H_bs = H_bs
        self.rpc_a = rpc_a
        self.rpc_b = rpc_b
        self.height = height # B,h,w

        self.gru_max_iter = gru_max_iter
        self.B,self.H,self.W = B,H,W
        self.h,self.w = self.ctx_feats_a.shape[-2:]
        self.device = self.ctx_feats_a.device

        self.cost_volume_ab = CostVolume(self.match_feats_a,self.match_feats_b,num_levels=self.gru_access.corr_levels)
        self.cost_volume_ba = CostVolume(self.match_feats_b,self.match_feats_a,num_levels=self.gru_access.corr_levels)

        self.Ms_a_b = torch.tensor([
            [
                [1.0,0.0,0.0],
                [0.0,1.0,0.0]
            ]
        ] * self.B).to(device=self.device,dtype=torch.float32)

        self.Ms_b_a = torch.tensor([
            [
                [1.0,0.0,0.0],
                [0.0,1.0,0.0]
            ]
        ] * self.B).to(device=self.device,dtype=torch.float32)

        self.norm_factors_a = self.calculate_original_extent(self.B,self.H,self.W,self.H_as) # B,
        self.norm_factors_b = self.calculate_original_extent(self.B,self.H,self.W,self.H_bs)         
        

    def calculate_original_extent(self,B,H,W,Hs) -> torch.Tensor:
        device = self.device
        dtype = Hs.dtype

        Hs_inv = torch.linalg.inv(Hs)

        corners_row = [0.0, 0.0, float(H), float(H)]
        corners_col = [0.0, float(W), float(W), 0.0]
        ones = torch.ones(4, dtype=dtype, device=device)
        rows = torch.tensor(corners_row, dtype=dtype, device=device)
        cols = torch.tensor(corners_col, dtype=dtype, device=device)
        corners_homo = torch.stack([rows, cols, ones], dim=0)
        corners_batch = corners_homo.unsqueeze(0).expand(B, -1, -1)
        p_big_homo = torch.bmm(Hs_inv, corners_batch)
        w = p_big_homo[:, 2:3, :] 
        eps = 1e-7
        p_big = p_big_homo / (w + eps)

        rows_big = p_big[:, 0, :]  # (B, 4)
        cols_big = p_big[:, 1, :]  # (B, 4)

        row_span = rows_big.max(dim=1).values - rows_big.min(dim=1).values
        col_span = cols_big.max(dim=1).values - cols_big.min(dim=1).values
        max_extent = torch.maximum(row_span, col_span)

        return max_extent
        
    def sample_from_coords(self, Coords, Values, H, W, align_corners=True):
        """
        根据像素坐标从特征图中采样数值。
        
        参数:
            Coords: (B, N, 2) tensor, 记录了 (row, col) 格式的坐标。
                    坐标是相对于原始尺寸 (H, W) 的。
            Values: (B, h, w) tensor, 被采样的特征图/数值图。
            H: int, 原始图像高度。
            W: int, 原始图像宽度。
            align_corners: bool, grid_sample 的参数。
                        默认为 True，意味着坐标会被归一化使得 -1 指向像素中心还是边缘。
                        通常对于像素精确坐标，使用 True (映射 0 -> -1, size-1 -> 1)。
        
        返回:
            Samples: (B, N) tensor, 采样后的结果。
        """
        features = Values.unsqueeze(1) #(B, h, w) -> (B, 1, h, w)

        rows = Coords[..., 0]
        cols = Coords[..., 1]
        norm_rows = 2 * (rows / (H - 1)) - 1
        norm_cols = 2 * (cols / (W - 1)) - 1
        grid = torch.stack((norm_cols, norm_rows), dim=-1)
        grid = grid.unsqueeze(1) # Grid: (B, N, 2) -> (B, 1, N, 2)

        sampled_features = F.grid_sample(
            features, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=align_corners
        ) # (B, C, 1, N) -> (B, 1, 1, N)

        samples = sampled_features.squeeze(1).squeeze(1) # (B, 1, 1, N) -> (B, N)
        
        return samples

    def _get_coord_mat(self,h,w,b = 0,ds = 1,device = 'cpu'):
        grid_row, grid_col = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
        coords_row = grid_row * ds + ds / 2.0
        coords_col = grid_col * ds + ds / 2.0
        coords = torch.stack([coords_row,coords_col],dim=-1) # h,w,2
        if b > 0:
            coords = torch.stack([coords] * b,dim=0) # b,h,w,2
        return coords

    def coord_norm(self,coords:torch.Tensor,norm_factor:torch.Tensor):
        target_shape = [coords.shape[0]] + [1] * (coords.ndim - 1)
        norm_factor = norm_factor.view(target_shape)
        return coords / norm_factor # 0~1
    
    def coord_norm_inv(self,coords:torch.Tensor,norm_factor:torch.Tensor):
        target_shape = [coords.shape[0]] + [1] * (coords.ndim - 1)
        norm_factor = norm_factor.view(target_shape)
        return coords * norm_factor

    def apply_H(self,coords:torch.Tensor,Hs:torch.Tensor,device:str = 'cpu'):
        """
        coords: B,N,2
        Hs: B,3,3

        return: B,N,2
        """
        B,N = coords.shape[:2]
        coords = coords.permute(0,2,1) # B,2,N
        ones = torch.ones(B, 1, N, device=device)
        coords_homo = torch.cat([coords,ones],dim=1) # B,3,N
        coords_trans_homo = torch.bmm(Hs,coords_homo) # (B,3,3) @ (B,3,N) -> (B,3,N)
        eps = 1e-7
        z = coords_trans_homo[:, 2:3, :]
        coords_trans = coords_trans_homo[:, :2, :] / (z + eps) # B,2,N
        return coords_trans.permute(0,2,1) # B,N,2
    
    def apply_M(self,coords:torch.Tensor,Ms:torch.Tensor,device:str = 'cpu'):
        """
        coords: B,N,2
        Ms: B,2,3

        return: B,N,2
        """
        B,N = coords.shape[:2]
        coords = coords.permute(0,2,1) # B,2,N
        ones = torch.ones(B, 1, N, device=device)
        coords_homo = torch.cat([coords,ones],dim=1) # B,3,N
        coords_trans = torch.bmm(Ms,coords_homo) # (B,2,3) @ (B,3,N) -> (B,2,N)
        return coords_trans.permute(0,2,1) # B,N,2

    def merge_M(self,Ma: torch.Tensor, Mb: torch.Tensor) -> torch.Tensor:

        # 1. 创建用于填充的 [0, 0, 1] 行
        # 确保它在与输入张量相同的设备和数据类型上
        pad_row = torch.zeros((self.B, 1, 3), device=Ma.device, dtype=Ma.dtype)
        pad_row[..., 2] = 1.0  # (B, 1, 3)

        # 2. 将 A 和 B 转换为 (B, 3, 3) 的齐次矩阵
        A_hom = torch.cat([Ma, pad_row], dim=1)  # (B, 3, 3)
        B_hom = torch.cat([Mb, pad_row], dim=1)  # (B, 3, 3)

        # 3. 计算复合矩阵 C_hom = B_hom @ A_hom
        # 注意：顺序是 B @ A，因为 B(A(p)) 对应 H_B * (H_A * p) = (H_B * H_A) * p
        # torch.matmul (或 @) 会自动处理批量矩阵乘法
        C_hom = B_hom @ A_hom

        # 4. 从齐次矩阵 C_hom 中提取 (B, 2, 3) 的仿射部分
        C = C_hom[:, :2, :]

        return C

    def prepare_data(self,cost_volume:CostVolume,Hs_1:torch.Tensor,Hs_2:torch.Tensor,Ms:torch.Tensor,norm_factor:torch.Tensor,rpc_1:RPCModelParameterTorch = None,rpc_2:RPCModelParameterTorch = None,height:torch.Tensor = None):
        """
        height: B,h,w
        """
        anchor_coords_in_1 = self._get_coord_mat(self.h,self.w,self.B,ds=16,device=self.device) # (B,h,w,2)
        anchor_coords_in_1_flat = anchor_coords_in_1.flatten(1,2) # B,h*w,2

        anchor_coords_in_big_1_flat = self.apply_H(anchor_coords_in_1_flat,torch.linalg.inv(Hs_1),device=self.device)
        anchor_coords_in_big_1_flat_af = self.apply_M(anchor_coords_in_big_1_flat,Ms,device=self.device)
        anchor_coords_in_big_1_af = anchor_coords_in_big_1_flat_af.reshape(self.B,self.h,self.w,2) # B,h,w,2

        if not rpc_1 is None and not rpc_2 is None:
            anchor_lines_in_big_1_af = anchor_coords_in_big_1_flat_af[...,0].ravel()
            anchor_samps_in_big_1_af = anchor_coords_in_big_1_flat_af[...,1].ravel()
            anchor_lines_in_big_2, anchor_samps_in_big_2 = project_linesamp(rpc_1,rpc_2,anchor_lines_in_big_1_af,anchor_samps_in_big_1_af,height.ravel())
            anchor_coords_in_big_2_flat = torch.stack([anchor_lines_in_big_2,anchor_samps_in_big_2],dim=-1).reshape(self.B,-1,2).to(torch.float32) # B,h*w,2
        else:
            anchor_coords_in_big_2_flat = anchor_coords_in_big_1_flat_af # B,h*w,2
        
        anchor_coords_in_2_flat = self.apply_H(anchor_coords_in_big_2_flat,Hs_2,device=self.device) # B,h*w,2
        anchor_coords_in_2 = anchor_coords_in_2_flat.reshape(self.B,self.h,self.w,2) # B,h,w,2
        anchor_coords_in_2[...,0] = ((anchor_coords_in_2[...,0] / (self.H - 1)) * 2.) - 1.
        anchor_coords_in_2[...,1] = ((anchor_coords_in_2[...,1] / (self.W - 1)) * 2.) - 1.

        corr_simi, corr_coords = cost_volume.lookup(anchor_coords_in_2) #corr_simi(B,N,h,w), corr_coords(B,h,w,N,2)

        corr_coords_in_2_flat = corr_coords.flatten(1,3) # (B,h*w*N,2) b小图坐标系下采样点坐标
        corr_coords_in_big_2_flat = self.apply_H(corr_coords_in_2_flat,torch.linalg.inv(Hs_2),device=self.device) # B,h*w*N,2

        if not rpc_1 is None and not rpc_2 is None:
            corr_heights = self.sample_from_coords(corr_coords_in_2_flat,height,self.H,self.W) # B,h*w*N
            corr_lines_in_big_2 = corr_coords_in_big_2_flat[...,0].ravel() # B*h*w*N
            corr_samps_in_big_2 = corr_coords_in_big_2_flat[...,1].ravel()
            corr_lines_in_big_1, corr_samps_in_big_1 = project_linesamp(rpc_2,rpc_1,corr_lines_in_big_2,corr_samps_in_big_2,corr_heights.ravel())
            corr_coords_in_big_1_flat = torch.stack([corr_lines_in_big_1,corr_samps_in_big_1],dim=-1).reshape(self.B,-1,2).to(torch.float32) # B,h*w*N,2
        else:
            corr_coords_in_big_1_flat = corr_coords_in_big_2_flat # B,h*w*N,2

        corr_coords_in_big_1 = corr_coords_in_big_1_flat.reshape(self.B,self.h,self.w,-1,2) # B,h,w,N,2
        
        corr_offset = corr_coords_in_big_1 - anchor_coords_in_big_1_af.unsqueeze(3) # B,h,w,N,2
        print(f"corr_offset:\n{corr_offset[0,15,15]}")
        corr_offset = corr_offset.flatten(3,4).permute(0,3,1,2) # (B,N*2,h,w)
        corr_offset = self.coord_norm(corr_offset,norm_factor) # 将offset进行归一化

        # check_invalid_tensors([anchor_coords_in_1,anchor_coords_in_big_1_flat,anchor_coords_in_big_1_flat_af,anchor_coords_in_big_1_af,anchor_lines_in_big_2,anchor_coords_in_big_2_flat,anchor_coords_in_2_flat,
        #                        corr_simi,corr_coords,corr_coords_in_big_2_flat,corr_heights,corr_coords_in_big_1_flat,corr_offset],"[prepare data]: ")

        return corr_simi,corr_offset

    def solve(self,flag = 'ab',final_only = False, return_vis=False):
        """
        返回 preds = [delta_Ms_0, delta_Ms_1, ... , delta_Ms_N]  (B,steps,2,3)
        """
        hidden_state = torch.zeros((self.B,self.gru_access.hidden_dim),dtype=self.ctx_feats_a.dtype,device=self.device)
        preds = []
        vis_dict = {}

        for iter in range(self.gru_max_iter):
            # 计算a->b的仿射
            if flag == 'ab':
                corr_simi_ab,corr_offset_ab = self.prepare_data(self.cost_volume_ab,self.H_as,self.H_bs,self.Ms_a_b,self.norm_factors_a,self.rpc_a,self.rpc_b,self.height)
                if return_vis and iter == 0:
                    vis_dict = vis_pyramid_correlation(
                        corr_simi_ab, 
                        corr_offset_ab, 
                        self.norm_factors_a,
                        num_levels=self.gru_access.corr_levels,
                        radius=self.gru_access.corr_radius
                    )

                delta_affines_ab, hidden_state = self.gru(corr_simi_ab,
                                                          corr_offset_ab,
                                                          self.ctx_feats_a,
                                                          self.confs_a.detach(),
                                                          hidden_state)
                
                delta_affines_ab[...,2] = self.coord_norm_inv(delta_affines_ab[...,2] , self.norm_factors_a)
                preds.append(delta_affines_ab)
                self.Ms_a_b = self.merge_M(self.Ms_a_b,delta_affines_ab)
                # check_invalid_tensors([corr_simi_ab,corr_offset_ab,self.norm_factors_a,delta_affines_ab,hidden_state,self.Ms_a_b],f"[solve gru iter {iter}]: ")


            # 计算b->a的仿射
            if flag == 'ba':
                corr_simi_ba,corr_offset_ba = self.prepare_data(self.cost_volume_ba,self.H_bs,self.H_as,self.Ms_b_a,self.norm_factors_b,self.rpc_b,self.rpc_a,self.height)
                delta_affines_ba, hidden_state = self.gru(corr_simi_ba,
                                                          corr_offset_ba,
                                                          self.ctx_feats_b,
                                                          self.confs_b.detach(),
                                                          hidden_state)
                
                delta_affines_ba[...,2] = self.coord_norm_inv(delta_affines_ba[...,2] , self.norm_factors_b)
                preds.append(delta_affines_ba)
                self.Ms_b_a = self.merge_M(self.Ms_b_a,delta_affines_ba)

        preds = torch.stack(preds,dim=1)

        if final_only:
            final = torch.eye(2, 3, device=self.device, dtype=preds.dtype).unsqueeze(0).repeat(self.B, 1, 1)
            for t in range(self.gru_max_iter):
                pred = preds[:,t]
                final = merge_affine(final,pred)
            return final # B,2,3
        
        if return_vis:
            return preds, vis_dict # [修改] 返回元组
        else:
            return preds        
