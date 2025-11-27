import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from model.encoder import Encoder
from model.gru import GRUBlock
from model.cost_volume import CostVolume
from utils.rpc import RPCModelParameterTorch
from utils.utils import debug_print

class Windows():
    def __init__(self,B,H,W,
                 gru:GRUBlock,
                 feats_a,feats_b,
                 H_as:torch.Tensor,H_bs:torch.Tensor,
                 rpc_a:RPCModelParameterTorch = None,rpc_b:RPCModelParameterTorch = None,
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

        self.gru_max_iter = gru_max_iter
        self.B,self.H,self.W = B,H,W
        self.h,self.w = self.ctx_feats_a.shape[-2:]
        self.device = self.ctx_feats_a.device

        self.cost_volume_ab = CostVolume(self.match_feats_a,self.match_feats_b)
        self.cost_volume_ba = CostVolume(self.match_feats_b,self.match_feats_a)

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


    def transform_coords_mat(
        self,
        B: int,
        h: int,
        w: int,
        Hs_1: torch.Tensor,
        Hs_2: torch.Tensor,
        Ms: torch.Tensor,
        rpc_1:RPCModelParameterTorch = None,
        rpc_2:RPCModelParameterTorch = None,
        stride: int = 16,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        将a图中patch的中心点坐标投影到b图坐标系
        
        参数:
            B: Batch size
            h: 特征图高度
            w: 特征图宽度
            Hs_1: (B, 3, 3) 单应变换矩阵 (大图 -> Imgs_1), 定义在 (row, col) 空间
            Hs_2: (B, 3, 3) 单应变换矩阵 (大图 -> Imgs_2), 定义在 (row, col) 空间
            Ms:   (B, 2, 3) 仿射变换矩阵 (在大图坐标系下应用), 定义在 (row, col) 空间
            stride: 下采样倍率，默认16
            device: 运行设备
            
        返回:
            imgs_a_coords_b: (B, h, w, 2) 变换后的坐标，格式为 (row, col)
        """
        
        y_range = torch.arange(h, device=device, dtype=torch.float32)
        x_range = torch.arange(w, device=device, dtype=torch.float32)
        grid_row, grid_col = torch.meshgrid(y_range, x_range, indexing='ij')
        coords_row = grid_row * stride + stride / 2.0
        coords_col = grid_col * stride + stride / 2.0

        N = h * w
        ones = torch.ones(B, 1, N, device=device)
        coords_row_flat = coords_row.reshape(1, -1).expand(B, -1) # (h, w) -> (N,) -> (B, N)
        coords_col_flat = coords_col.reshape(1, -1).expand(B, -1)
        imgs_a_coords_homo = torch.stack([coords_row_flat, coords_col_flat, ones.squeeze(1)], dim=1) # imgs_a_coords_homo: (B, 3, N) -> Stack as [row, col, 1]

        Hs_1_inv = torch.inverse(Hs_1).to(torch.float32) # (B, 3, 3)
        coords_ori_homo = torch.bmm(Hs_1_inv, imgs_a_coords_homo) #(B, 3, 3) @ (B, 3, N) -> (B, 3, N)
        eps = 1e-7
        z_ori = coords_ori_homo[:, 2:3, :]
        coords_ori_rc = coords_ori_homo[:, :2, :] / (z_ori + eps) # (B, 2, N), channel 0 is row, 1 is col
        
        coords_ori_rehomo = torch.cat([coords_ori_rc, ones], dim=1) # (B, 3, N)
        coords_ori_af = torch.bmm(Ms, coords_ori_rehomo) # (B,2,3) @ (B,3,N) -> (B,2,N)
        
        #==========================================
        # RPC 投影与反投影 （TODO）
        #==========================================
        if not rpc_1 is None and not rpc_2 is None:
            pass
            raise ValueError("Not Impleted")
        else:
            coords_b_ori = coords_ori_af # (B,2,N)

        coords_b_ori_homo = torch.cat([coords_b_ori, ones], dim=1) # (B, 3, N)
        coords_b_homo = torch.bmm(Hs_2, coords_b_ori_homo) # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
        z_b = coords_b_homo[:, 2:3, :]
        coords_b_rc = coords_b_homo[:, :2, :] / (z_b + eps) # (B, 2, N) -> [row, col]
        
        imgs_a_coords_b = coords_b_rc.view(B, 2, h, w).permute(0, 2, 3, 1) # (B, 2, N) -> (B, 2, h, w) -> (B, h, w, 2)
        
        return imgs_a_coords_b

    def transform_points_coords(
        self,
        points: torch.Tensor,
        Hs_1: torch.Tensor,
        Hs_2: torch.Tensor,
        rpc_1:RPCModelParameterTorch = None,
        rpc_2:RPCModelParameterTorch = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        将 imgs_b 中的采样点坐标变换回 imgs_a 的坐标系。
        不包含仿射变换。
        所有坐标和矩阵基于 (row, col) 格式。

        参数:
            B: Batch size
            coords_b: (B, N, 2) 采样点坐标，格式为 (row, col)
            H_as: (B, 3, 3) 单应变换矩阵 (大图 -> Imgs_A)
            H_bs: (B, 3, 3) 单应变换矩阵 (大图 -> Imgs_B)
            device: 运行设备
            
        返回:
            coords_a: (B, N, 2) 变换后的坐标，格式为 (row, col)
        """
        
        # 获取采样点数量
        B,N = points.shape[:2]
        ones = torch.ones(B, 1, N, device=device)
        
        # 1. 准备数据: (B, N, 2) -> (B, 2, N) -> (B, 3, N) [row, col, 1]
        # 注意：输入 coords_1 最后一维是 (row, col)
        coords_1_permuted = points.permute(0, 2, 1) 
        coords_1_homo = torch.cat([coords_1_permuted, ones], dim=1)
        
        
        Hs_1_inv = torch.inverse(Hs_1).to(torch.float32)
        coords_ori_homo = torch.bmm(Hs_1_inv, coords_1_homo)
        
        # 透视除法
        eps = 1e-7
        z_ori = coords_ori_homo[:, 2:3, :]
        coords_ori_rc = coords_ori_homo[:, :2, :] / (z_ori + eps) # (B, 2, N)

        #==========================================
        # RPC 投影与反投影 （TODO）
        #==========================================
        if not rpc_1 is None and not rpc_2 is None:
            pass
            raise ValueError("Not Impleted")
        else:
            coords_2_ori = coords_ori_rc

        
        # 重新构建齐次坐标 (B, 3, N)
        coords_2_ori_rehomo = torch.cat([coords_2_ori, ones], dim=1)
        
        coords_2_homo = torch.bmm(Hs_2, coords_2_ori_rehomo)
        
        # 透视除法
        z_2 = coords_2_homo[:, 2:3, :]
        coords_2_rc = coords_2_homo[:, :2, :] / (z_2 + eps) # (B, 2, N)
        
        # 4. 转换回 (B, N, 2)
        coords_2 = coords_2_rc.permute(0, 2, 1)
        
        return coords_2

    def prepare_data(self,cost_volume:CostVolume,Hs_1,Hs_2,Ms,norm_factor,rpc_1:RPCModelParameterTorch = None,rpc_2:RPCModelParameterTorch = None):
        imgs_1_coords_2 = self.transform_coords_mat(self.B,self.h,self.w,Hs_1,Hs_2,Ms,rpc_1,rpc_2,device=self.device) # 得到a的坐标网格投影到b后的坐标
        imgs_1_coords_2[...,0] = ((imgs_1_coords_2[...,0] / self.H) * 2.) - 1.
        imgs_1_coords_2[...,1] = ((imgs_1_coords_2[...,1] / self.W) * 2.) - 1.

        corr_simi, corr_coords = cost_volume.lookup(imgs_1_coords_2) #通过a投影到b中的归一化坐标在代价体中查询相似性，并且记录采样点在b中坐标，corr_simi(B,N,h,w),corr_coords(B,N,2,h,w)

        # 将b中的采样点通过 Hb-1 -> RPC_b -> RPC_a -> Ha 投影回到a中的坐标
        corr_coords_in_2 = corr_coords.permute(0,3,4,1,2).flatten(1,3) # (B,H*W*N,2)
        corr_coords_in_1 = self.transform_points_coords(corr_coords_in_2,Hs_2,Hs_1,rpc_1,rpc_2,device=self.device) # (B,h*w*N,2)
        corr_coords_in_1 = corr_coords_in_1.reshape(self.B,self.h,self.w,-1,2) # (B,h,w,N,2)

        #得到每组采样点的基准点（也就是a中的网格点），然后相减，得到每个采样点相对于其基准点的offset
        coords_1 = self._get_coord_mat(self.h,self.w,self.B,ds=16,device=self.device) # (B,h,w,2)
        corr_offset = corr_coords_in_1 - coords_1.unsqueeze(3) # (B,h,w,N,2)
        corr_offset = corr_offset.permute(0,3,4,1,2).flatten(1,2) # (B,N*2,h,w)
        corr_offset = self.coord_norm(corr_offset,norm_factor) # 将offset进行归一化

        return corr_simi,corr_offset
    
    def get_flow(self,Hs,Ms,norm_factor,device):
        Hs_inv = torch.inverse(Hs).to(torch.float32)
        ones = torch.ones(self.B, self.h * self.w, 1, device=device)
        coords_local = self._get_coord_mat(self.h,self.w,self.B,ds=16,device=device) # B,h,w,2
        coords_local_homo = torch.cat([coords_local.flatten(1,2),ones],dim=-1).permute(0,2,1) # B,3,h*w
        coords_ori_homo = torch.bmm(Hs_inv,coords_local_homo) # B,3,3 @ B,3,h*w -> B,3,h*w
        eps = 1e-7
        z_ori = coords_ori_homo[:, 2:3, :]
        coords_ori = coords_ori_homo[:, :2, :] / (z_ori + eps) # B,2,h*w
        
        ones = torch.ones(self.B, 1, self.h * self.w, device=device)
        coords_ori_homo = torch.cat([coords_ori,ones],dim=1)    
        coords_ori_af = torch.bmm(Ms,coords_ori_homo) # B,2,3 @ B,3,h*w -> B,2,h*w
        coords_ori_af = coords_ori_af.reshape(self.B,2,self.h,self.w) # B,2,h,w
        coords_ori = coords_ori.reshape(self.B,2,self.h,self.w) # B,2,h,w
        flow = coords_ori_af - coords_ori # B,2,h,w
        flow = self.coord_norm(flow,norm_factor) # B,2,h,w

        return flow

        

    def solve(self,flag = 'ab'):
        """
        返回 preds = [delta_Ms_0, delta_Ms_1, ... , delta_Ms_N]  (B,steps,2,3)
        """
        hidden_state = torch.zeros((self.B,self.gru_access.hidden_dim),dtype=self.ctx_feats_a.dtype,device=self.device)
        flow = torch.zeros((self.B,2,self.h,self.w),dtype=self.ctx_feats_a.dtype,device=self.device)
        preds = []
        for iter in range(self.gru_max_iter):
            # 计算a->b的仿射
            if flag == 'ab':
                corr_simi_ab,corr_offset_ab = self.prepare_data(self.cost_volume_ab,self.H_as,self.H_bs,self.Ms_a_b,self.norm_factors_a,self.rpc_a,self.rpc_b)
                delta_affines_ab, new_hidden_states = self.gru(corr_simi_ab,
                                                               corr_offset_ab,
                                                               flow,
                                                               self.ctx_feats_a,
                                                               self.confs_a,
                                                               hidden_state)
                
                delta_affines_ab[...,2] = self.coord_norm_inv(delta_affines_ab[...,2] , self.norm_factors_a)
                preds.append(delta_affines_ab)
                self.Ms_a_b = self.Ms_a_b + delta_affines_ab
                hidden_state = new_hidden_states
                flow = self.get_flow(self.H_as,self.Ms_a_b,self.norm_factors_a,device=self.device)


            # 计算b->a的仿射
            if flag == 'ba':
                corr_simi_ba,corr_offset_ba = self.prepare_data(self.cost_volume_ba,self.H_bs,self.H_as,self.Ms_b_a,self.norm_factors_b,self.rpc_b,self.rpc_a)
                delta_affines_ba, new_hidden_states = self.gru(corr_simi_ba,
                                                               corr_offset_ba,
                                                               flow,
                                                               self.ctx_feats_b,
                                                               self.confs_b,
                                                               hidden_state)
                
                delta_affines_ba[...,2] = self.coord_norm_inv(delta_affines_ba[...,2] , self.norm_factors_b)
                preds.append(delta_affines_ba)
                self.Ms_b_a = self.Ms_b_a + delta_affines_ba
                hidden_state = new_hidden_states
                flow = self.get_flow(self.H_bs,self.Ms_b_a,self.norm_factors_b,device=self.device)
        preds = torch.stack(preds,dim=1)
        
        return preds            
