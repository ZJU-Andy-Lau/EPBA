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
from utils.visualize import vis_pyramid_correlation
from criterion.utils import invert_affine_matrix

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

    def apply_H(self,coords:torch.Tensor,Hs:torch.Tensor,device:str = 'cpu'):
        """
        coords: B,N,2
        Hs: B,3,3

        return: B,N,2
        """
        B,N = coords.shape[:2]
        coords = coords.permute(0,2,1) # B,2,N
        ones = torch.ones(B, 1, N, device=device)
        coords_homo = torch.cat([coords_homo,ones],dim=1) # B,3,N
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
        coords_homo = torch.cat([coords_homo,ones],dim=1) # B,3,N
        coords_trans = torch.bmm(Ms,coords_homo) # (B,2,3) @ (B,3,N) -> (B,2,N)
        return coords_trans.permute(0,2,1) # B,N,2


    def prepare_data(self,cost_volume:CostVolume,Hs_1,Hs_2,Ms,norm_factor,rpc_1:RPCModelParameterTorch = None,rpc_2:RPCModelParameterTorch = None):
        anchor_coords_in_1 = self._get_coord_mat(self.h,self.w,self.B,ds=16,device=self.device) # (B,h,w,2)
        anchor_coords_in_1_flat = anchor_coords_in_1.flatten(1,2) # B,h*w,2

        anchor_coords_in_big_1_flat = self.apply_H(anchor_coords_in_1_flat,torch.linalg.inv(Hs_1),device=self.device)
        anchor_coords_in_big_1_flat_af = self.apply_M(anchor_coords_in_big_1_flat,Ms,device=self.device)
        anchor_coords_in_big_1_af = anchor_coords_in_big_1_flat_af.reshape(self.B,self.h,self.w,2) # B,h,w,2

        if not rpc_1 is None and not rpc_2 is None:
            pass
        else:
            anchor_coords_in_big_2 = anchor_coords_in_big_1_flat_af
        
        anchor_coords_in_2_flat = self.apply_H(anchor_coords_in_big_2,Hs_2,device=self.device) # B,h*w,2
        anchor_coords_in_2 = anchor_coords_in_2_flat.reshape(self.B,self.h,self.w,2) # B,h,w,2
        anchor_coords_in_2[...,0] = ((anchor_coords_in_2[...,0] / (self.H - 1)) * 2.) - 1.
        anchor_coords_in_2[...,1] = ((anchor_coords_in_2[...,1] / (self.W - 1)) * 2.) - 1.

        corr_simi, corr_coords = cost_volume.lookup(anchor_coords_in_2) #corr_simi(B,N,h,w), corr_coords(B,h,w,N,2)

        corr_coords_in_2_flat = corr_coords.flatten(1,3) # (B,h*w*N,2) b小图坐标系下采样点坐标
        corr_coords_in_big_2_flat = self.apply_H(corr_coords_in_2_flat,torch.linalg.inv(Hs_2),device=self.device) # B,h*w*N,2

        if not rpc_1 is None and not rpc_2 is None:
            pass
        else:
            corr_coords_in_big_1_flat = corr_coords_in_big_2_flat

        corr_coords_in_big_1 = corr_coords_in_big_1_flat.reshape(self.B,self.h,self.w,-1,2) # B,h,w,N,2
        
        corr_offset = corr_coords_in_big_1 - anchor_coords_in_big_1_af.unsqueeze(3) # B,h,w,N,2
        corr_offset = corr_offset.flatten(3,4).permute(0,3,1,2) # (B,N*2,h,w)
        corr_offset = self.coord_norm(corr_offset,norm_factor) # 将offset进行归一化


        return corr_simi,corr_offset
    
    # def get_flow(self,Hs,Ms,norm_factor,device):
    #     Hs_inv = torch.inverse(Hs).to(torch.float32)
    #     ones = torch.ones(self.B, self.h * self.w, 1, device=device)
    #     coords_local = self._get_coord_mat(self.h,self.w,self.B,ds=16,device=device) # B,h,w,2
    #     coords_local_homo = torch.cat([coords_local.flatten(1,2),ones],dim=-1).permute(0,2,1) # B,3,h*w
    #     coords_ori_homo = torch.bmm(Hs_inv,coords_local_homo) # B,3,3 @ B,3,h*w -> B,3,h*w
    #     eps = 1e-7
    #     z_ori = coords_ori_homo[:, 2:3, :]
    #     coords_ori = coords_ori_homo[:, :2, :] / (z_ori + eps) # B,2,h*w
        
    #     ones = torch.ones(self.B, 1, self.h * self.w, device=device)
    #     coords_ori_homo = torch.cat([coords_ori,ones],dim=1)    
    #     coords_ori_af = torch.bmm(Ms,coords_ori_homo) # B,2,3 @ B,3,h*w -> B,2,h*w
    #     coords_ori_af = coords_ori_af.reshape(self.B,2,self.h,self.w) # B,2,h,w
    #     coords_ori = coords_ori.reshape(self.B,2,self.h,self.w) # B,2,h,w
    #     flow = coords_ori_af - coords_ori # B,2,h,w
    #     flow = self.coord_norm(flow,norm_factor) # B,2,h,w

    #     return flow

        

    def solve(self,flag = 'ab', return_vis=False):
        """
        返回 preds = [delta_Ms_0, delta_Ms_1, ... , delta_Ms_N]  (B,steps,2,3)
        """
        hidden_state = torch.zeros((self.B,self.gru_access.hidden_dim),dtype=self.ctx_feats_a.dtype,device=self.device)
        # flow = torch.zeros((self.B,2,self.h,self.w),dtype=self.ctx_feats_a.dtype,device=self.device)
        preds = []
        vis_dict = {}

        for iter in range(self.gru_max_iter):
            # 计算a->b的仿射
            if flag == 'ab':
                corr_simi_ab,corr_offset_ab = self.prepare_data(self.cost_volume_ab,self.H_as,self.H_bs,self.Ms_a_b,self.norm_factors_a,self.rpc_a,self.rpc_b)
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
                                                            #    flow,
                                                               self.ctx_feats_a,
                                                               self.confs_a,
                                                               hidden_state)
                
                delta_affines_ab[...,2] = self.coord_norm_inv(delta_affines_ab[...,2] , self.norm_factors_a)
                preds.append(delta_affines_ab)
                self.Ms_a_b = self.Ms_a_b + delta_affines_ab
                # flow = self.get_flow(self.H_as,self.Ms_a_b,self.norm_factors_a,device=self.device)


            # 计算b->a的仿射
            if flag == 'ba':
                corr_simi_ba,corr_offset_ba = self.prepare_data(self.cost_volume_ba,self.H_bs,self.H_as,self.Ms_b_a,self.norm_factors_b,self.rpc_b,self.rpc_a)
                delta_affines_ba, hidden_state = self.gru(corr_simi_ba,
                                                               corr_offset_ba,
                                                            #    flow,
                                                               self.ctx_feats_b,
                                                               self.confs_b,
                                                               hidden_state)
                
                delta_affines_ba[...,2] = self.coord_norm_inv(delta_affines_ba[...,2] , self.norm_factors_b)
                preds.append(delta_affines_ba)
                self.Ms_b_a = self.Ms_b_a + delta_affines_ba
                # flow = self.get_flow(self.H_bs,self.Ms_b_a,self.norm_factors_b,device=self.device)
        # print(f"norm_factor_a:{self.norm_factors_a}")
        # print(f"norm_factor_b:{self.norm_factors_b}")
        preds = torch.stack(preds,dim=1)
        
        if return_vis:
            return preds, vis_dict # [修改] 返回元组
        else:
            return preds        
