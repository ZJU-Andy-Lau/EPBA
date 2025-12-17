import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time

from model.encoder import Encoder
from model.gru import GRUBlock
from model.cost_volume import CostVolume
from shared.rpc import RPCModelParameterTorch,project_linesamp
from shared.utils import debug_print,check_invalid_tensors,avg_downsample
from shared.visualize import vis_pyramid_correlation
from criterion.utils import merge_affine
from infer.utils import Reporter

class WindowSolver():
    def __init__(self,B,H,W,
                 gru:GRUBlock,
                 feats_a,feats_b,
                 H_as:torch.Tensor,H_bs:torch.Tensor,
                 rpc_a:RPCModelParameterTorch = None,rpc_b:RPCModelParameterTorch = None,
                 height:torch.Tensor = None,
                 test_imgs_a = None, test_imgs_b = None,
                 gru_max_iter:int = 10,
                 reporter:Reporter = None): # [新增] reporter 参数
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
        self.height_ds = avg_downsample(height,16) if not height is None else None
        self.gru_max_iter = gru_max_iter
        self.reporter = reporter # [新增]
        self.B,self.H,self.W = B,H,W
        self.h,self.w = self.ctx_feats_a.shape[-2:]
        self.device = self.ctx_feats_a.device

        self.test_imgs_a = test_imgs_a
        self.test_imgs_b = test_imgs_b

        self.cost_volume_ab = CostVolume(self.match_feats_a,self.match_feats_b,num_levels=self.gru_access.corr_levels)
        self.cost_volume_ba = CostVolume(self.match_feats_b,self.match_feats_a,num_levels=self.gru_access.corr_levels)

        self.Ms_a_b = torch.eye(2, 3, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B,2,3)

        self.Ms_b_a = torch.eye(2, 3, dtype=torch.float32, device=self.device).unsqueeze(0).expand(B,2,3)

        self.norm_factors_a = self.calculate_original_extent(self.B,self.H,self.W,self.H_as) # B,
        self.norm_factors_b = self.calculate_original_extent(self.B,self.H,self.W,self.H_bs)         

    def calculate_original_extent(self,B,H,W,Hs) -> torch.Tensor:
        device = self.device
        dtype = Hs.dtype
        torch.cuda.synchronize()
        Hs_inv = torch.inverse(Hs)
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
    
    def calculate_current_centroid(self, B, H, W, Hs, Ms) -> torch.Tensor:
        """
        [新增] 计算当前仿射状态下，小图对应的大图四边形的质心 (O_loc)。
        
        Args:
            Hs: (B, 3, 3) 初始单应性矩阵 (Img -> Big) 的逆矩阵?
                correct logic: P_big = Hs^{-1} * P_img
                所以 Hs 应该是 Img -> Big? 
                如果是 calculate_original_extent 中的逻辑，Hs_inv 用于 Img -> Big
        Returns:
            centroid: (B, 2) [row, col]
        """
        device = self.device
        dtype = Ms.dtype
        
        Hs_inv = torch.linalg.inv(Hs)
        
        # 1. 原始小图的四个角点
        corners_row = [0.0, 0.0, float(H), float(H)]
        corners_col = [0.0, float(W), float(W), 0.0]
        ones = torch.ones(4, dtype=dtype, device=device)
        rows = torch.tensor(corners_row, dtype=dtype, device=device)
        cols = torch.tensor(corners_col, dtype=dtype, device=device)
        corners_homo = torch.stack([rows, cols, ones], dim=0) # (3, 4)
        corners_batch = corners_homo.unsqueeze(0).expand(B, -1, -1) # (B, 3, 4)
        
        # 2. 变换到大图坐标系 (Reference Domain)
        # P_big = Hs^{-1} * P_img
        p_big_homo = torch.bmm(Hs_inv, corners_batch) # (B, 3, 4)
        
        # [修改] 显式处理齐次坐标，转换为 (B, N, 2) 格式供 apply_M 使用
        eps = 1e-7
        w = p_big_homo[:, 2:3, :]
        p_big = p_big_homo[:, :2, :] / (w + eps) # (B, 2, 4)
        p_big = p_big.permute(0, 2, 1) # (B, 4, 2)
        
        # 3. 应用当前的仿射变换 M
        p_curr = self.apply_M(p_big, Ms, device=device) # (B, 4, 2)
        
        # 4. 计算质心
        centroid = p_curr.mean(dim=1) # (B, 2)
        
        return centroid

    def get_position_features(self, coords, centroid, norm_factor):
        """
        [新增] 生成相对位置编码特征。
        
        Args:
            coords: (B, h, w, 2) 当前特征图网格点在大图中的坐标
            centroid: (B, 2) 当前局部原点 O_loc
            norm_factor: (B,) 归一化因子
        Returns:
            pos_features: (B, 2, h, w)
        """
        B, h, w, _ = coords.shape
        
        # 扩展 centroid: (B, 2) -> (B, 1, 1, 2)
        centroid_expanded = centroid.view(B, 1, 1, 2)
        
        # 计算相对位移
        diff = coords - centroid_expanded
        
        # 归一化
        norm_factor_expanded = norm_factor.view(B, 1, 1, 1)
        pos_norm = diff / (norm_factor_expanded + 1e-7)
        
        # 调整维度 (B, h, w, 2) -> (B, 2, h, w)
        pos_features = pos_norm.permute(0, 3, 1, 2)
        
        return pos_features

    def convert_local_to_global(self, delta_local, centroid, norm_factor):
        """
        [新增] 将 GRU 预测的局部仿射增量转换为全局仿射增量。
        
        Args:
            delta_local: (B, 2, 3) GRU 输出 [A_loc | t_loc_norm]
            centroid: (B, 2) 局部原点 O_loc
            norm_factor: (B,)
        Returns:
            delta_global: (B, 2, 3) [A_glob | t_glob]
        """
        B = delta_local.shape[0]
        device = delta_local.device
        
        # 1. 分解局部参数
        A_loc = delta_local[:, :, :2] # (B, 2, 2)
        
        # 注意：GRU 输出的 t 是经过 coord_norm_inv 处理后的物理尺度吗？
        # 在 solve 循环里，我们在调用此函数前，会先执行 coord_norm_inv。
        # 所以这里的 t_loc 应该是物理尺度的。
        t_loc = delta_local[:, :, 2] # (B, 2)
        
        # 2. 计算全局平移补偿
        # 公式: t_glob = t_loc + (I - A_loc) * O_loc
        I = torch.eye(2, device=device).unsqueeze(0).expand(B, -1, -1)
        
        # (B, 2, 2) @ (B, 2, 1) -> (B, 2, 1)
        diff_term = torch.bmm(I - A_loc, centroid.unsqueeze(-1)).squeeze(-1)
        
        t_glob = t_loc + diff_term # (B, 2)
        
        # 3. 组装全局增量
        # A_glob = A_loc
        delta_global = torch.cat([A_loc, t_glob.unsqueeze(-1)], dim=-1) # (B, 2, 3)
        
        return delta_global
        
    def sample_from_coords(self, Coords, Values, H, W, align_corners=True):
        """
        根据像素坐标从特征图中采样数值。
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
        """
        Return:
            coords: torch.Tensor (B,H,W,2) (row,col)
        """
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
    
    def test(self,Ms:torch.Tensor):
        """
        Ms: torch.Tensor (B,2,3)
        
        Return:
            imgs_a: np.ndarray (B,H,W,3)
            sampled_imgs_b: np.ndarray (B,H,W,3)
        """
        if self.test_imgs_a is None or self.test_imgs_b is None:
            return None
        imgs_a,imgs_b = self.test_imgs_a,self.test_imgs_b
        B,H,W = imgs_a.shape[:3]
        coords_in_a = self._get_coord_mat(H,W,B,ds=1,device=self.device) # B,H,W,2
        coords_in_a_flat = coords_in_a.flatten(1,2) # B,H*W,2
        coords_in_big_a_flat = self.apply_H(coords_in_a_flat,torch.linalg.inv(self.H_as),device=self.device)
        coords_in_big_a_flat_af = self.apply_M(coords_in_big_a_flat,Ms,device=self.device) # B,H*W,2
        if not self.rpc_a is None and not self.rpc_b is None and not self.height is None:
            lines_in_big_a_af = coords_in_big_a_flat_af[...,0].ravel() # B*H*W
            samps_in_big_a_af = coords_in_big_a_flat_af[...,1].ravel()
            lines_in_big_b,samps_in_big_b = project_linesamp(self.rpc_a,self.rpc_b,lines_in_big_a_af,samps_in_big_a_af,self.height.ravel())
            coords_in_big_b_flat = torch.stack([lines_in_big_b,samps_in_big_b],dim=-1).reshape(B,-1,2).to(torch.float32) # B,H*W,2
        else:
            coords_in_big_b_flat = coords_in_big_a_flat_af
        
        coords_in_b_flat = self.apply_H(coords_in_big_b_flat,self.H_bs,device=self.device)
        coords_in_b = coords_in_b_flat.reshape(B,H,W,2)
        sample_coords = coords_in_b[...,[1,0]] # (row,col) -> (x,y)
        sample_coords[...,0] = 2.0 * sample_coords[...,0] / (W - 1) - 1.0 # 缩放到 -1 ~ 1
        sample_coords[...,1] = 2.0 * sample_coords[...,1] / (H - 1) - 1.0
        input_imgs = torch.from_numpy(imgs_b).to(device=self.device,dtype=torch.float32).permute(0,3,1,2) # B,3,H,W
        sampled_img = F.grid_sample(input_imgs,sample_coords,mode='bilinear',
                                    padding_mode='zeros',align_corners=True)
        sampled_img = sampled_img.permute(0,2,3,1).cpu().numpy() # B,H,W,3

        return imgs_a,sampled_img     

            
    def prepare_data(self,cost_volume:CostVolume,Hs_1:torch.Tensor,Hs_2:torch.Tensor,Ms:torch.Tensor,norm_factor:torch.Tensor,rpc_1:RPCModelParameterTorch = None,rpc_2:RPCModelParameterTorch = None,height:torch.Tensor = None):
        """
        [修改] 返回值增加 anchor_coords_in_big_1_af (用于位置编码)
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
        corr_offset = corr_offset.flatten(3,4).permute(0,3,1,2) # (B,N*2,h,w)
        corr_offset = self.coord_norm(corr_offset,norm_factor) # 将offset进行归一化

        # [修改] 返回 anchor_coords_in_big_1_af
        return corr_simi, corr_offset, anchor_coords_in_big_1_af

    def solve(self,flag = 'ab',final_only = False, return_vis=False):
        """
        返回 preds = [delta_Ms_0, delta_Ms_1, ... , delta_Ms_N]  (B,steps,2,3)
        """
        hidden_state = torch.zeros((self.B,self.gru_access.hidden_dim),dtype=self.ctx_feats_a.dtype,device=self.device)
        preds = []
        vis_dict = {}

        for iter in range(self.gru_max_iter):
            # [新增] 汇报进度
            if self.reporter is not None:
                self.reporter.update(step=f"GRU {iter+1}/{self.gru_max_iter}")

            # 计算a->b的仿射
            if flag == 'ab':
                # 1. 准备数据 (接收新增的 anchor_coords)
                corr_simi_ab, corr_offset_ab, anchor_coords_ab = self.prepare_data(
                    self.cost_volume_ab, self.H_as, self.H_bs, self.Ms_a_b, 
                    self.norm_factors_a, self.rpc_a, self.rpc_b, self.height_ds
                )
                
                # 2. 计算当前局部原点 (O_loc)
                centroid_ab = self.calculate_current_centroid(self.B, self.H, self.W, self.H_as, self.Ms_a_b)
                
                # 3. 生成位置特征
                pos_features_ab = self.get_position_features(anchor_coords_ab, centroid_ab, self.norm_factors_a)

                if return_vis and iter == 0:
                    vis_dict = vis_pyramid_correlation(
                        corr_simi_ab, 
                        corr_offset_ab, 
                        self.norm_factors_a,
                        num_levels=self.gru_access.corr_levels,
                        radius=self.gru_access.corr_radius
                    )

                # 4. GRU 预测局部仿射 (输入新增 pos_features)
                delta_affines_local, hidden_state = self.gru(
                    corr_simi_ab,
                    corr_offset_ab,
                    self.ctx_feats_a,
                    pos_features_ab, # [新增]
                    self.confs_a.detach(),
                    hidden_state
                )
                
                # 5. 反归一化局部平移 (恢复物理尺度)
                delta_affines_local[...,2] = self.coord_norm_inv(delta_affines_local[...,2] , self.norm_factors_a)
                
                # 6. 将局部仿射增量转换为全局增量
                delta_affines_global = self.convert_local_to_global(delta_affines_local, centroid_ab, self.norm_factors_a)
                
                preds.append(delta_affines_global)
                
                # 7. 更新全局状态
                self.Ms_a_b = self.merge_M(self.Ms_a_b, delta_affines_global)


            # 计算b->a的仿射
            if flag == 'ba':
                # 1. 准备数据
                corr_simi_ba, corr_offset_ba, anchor_coords_ba = self.prepare_data(
                    self.cost_volume_ba, self.H_bs, self.H_as, self.Ms_b_a, 
                    self.norm_factors_b, self.rpc_b, self.rpc_a, self.height_ds
                )
                
                # 2. 计算当前局部原点
                centroid_ba = self.calculate_current_centroid(self.B, self.H, self.W, self.H_bs, self.Ms_b_a)
                
                # 3. 生成位置特征
                pos_features_ba = self.get_position_features(anchor_coords_ba, centroid_ba, self.norm_factors_b)

                # 4. GRU 预测
                delta_affines_local, hidden_state = self.gru(
                    corr_simi_ba,
                    corr_offset_ba,
                    self.ctx_feats_b,
                    pos_features_ba, # [新增]
                    self.confs_b.detach(),
                    hidden_state
                )
                
                # 5. 反归一化
                delta_affines_local[...,2] = self.coord_norm_inv(delta_affines_local[...,2] , self.norm_factors_b)
                
                # 6. 转换全局
                delta_affines_global = self.convert_local_to_global(delta_affines_local, centroid_ba, self.norm_factors_b)
                
                preds.append(delta_affines_global)
                
                # 7. 更新全局状态
                self.Ms_b_a = self.merge_M(self.Ms_b_a, delta_affines_global)

        preds = torch.stack(preds,dim=1)

        if final_only:
            if self.reporter is not None:
                self.reporter.update(step="Merging") # [新增] 汇报Merge状态
            
            final = torch.eye(2, 3, dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.B,2,3)
            for t in range(self.gru_max_iter):
                pred = preds[:,t]
                final = self.merge_M(final,pred)
            preds = final

            if return_vis and flag == 'ab':
                imgs_a,imgs_b = self.test(preds)
                vis_dict['test'] = {
                    'imgs_a':imgs_a,
                    'imgs_b':imgs_b
                }
        
        if return_vis:
            return preds, vis_dict
        else:
            return preds