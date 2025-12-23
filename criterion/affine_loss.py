import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import merge_affine

class AffineLoss(nn.Module):
    def __init__(self, img_size, grid_stride=64, decay_rate=0.8, reg_weight=1e-3, device='cuda'):
        """
        Args:
            img_size (tuple): 裁切影像的尺寸 (H, W)
            grid_stride (int): 网格采样步长。
            decay_rate (float): 时间步权重的衰减系数，默认为 0.8
            reg_weight (float): 正则化项的权重，默认为 1e-3
            device (str): 计算设备
        """
        super(AffineLoss, self).__init__()
        self.H, self.W = img_size
        self.grid_stride = grid_stride
        self.decay_rate = decay_rate
        self.reg_weight = reg_weight
        self.device = device
        self.epsilon = 1e-7

        # 1. 预计算本地像素坐标网格 (1, 3, N)
        # 覆盖 [0, H] 和 [0, W] 范围，格式为齐次坐标 [row, col, 1]^T
        self.register_buffer('local_grid', self._create_local_grid())

    def _create_local_grid(self):
        """生成本地像素坐标系的稀疏网格 (RC坐标系)"""
        y_range = torch.arange(0, self.H, self.grid_stride, dtype=torch.float32, device=self.device)
        x_range = torch.arange(0, self.W, self.grid_stride, dtype=torch.float32, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        N = grid_y.numel()
        # [修改] 堆叠为齐次坐标 [row, col, 1]^T
        grid = torch.stack([
            grid_y.reshape(-1),
            grid_x.reshape(-1),
            torch.ones(N, dtype=torch.float32, device=self.device)
        ], dim=0)

        return grid.unsqueeze(0) # (1, 3, N)

    def get_reference_grid_in_large_coords(self, Hs_a):
        """
        将本地像素网格投影回大图坐标系，构建物理参考网格。
        
        Args:
            Hs_a: (B, 3, 3) Large A -> Img A 的单应矩阵 (RC坐标系)
        Returns:
            ref_grid: (B, 3, N) 大图坐标系下的齐次坐标 [row_large, col_large, 1]^T
        """
        B = Hs_a.shape[0]
        
        # 1. 计算逆矩阵: Img A -> Large A
        try:
            Hs_a_inv = torch.linalg.inv(Hs_a)
        except RuntimeError:
            Hs_a_inv = torch.inverse(Hs_a)
            
        # 2. 扩展本地网格以匹配 Batch
        local_grid_batch = self.local_grid.expand(B, -1, -1) # (B, 3, N)
        
        # 3. 投影: P_large_homo = H_inv @ P_local (均为 RC)
        large_points_homo = torch.bmm(Hs_a_inv, local_grid_batch) # (B, 3, N)
        
        # 4. 透视除法 (Perspective Division)
        z = large_points_homo[:, 2:3, :] + self.epsilon
        row_large = large_points_homo[:, 0:1, :] / z
        col_large = large_points_homo[:, 1:2, :] / z
        
        # 5. 重新构建为仿射计算所需的齐次坐标 [row, col, 1]
        ones = torch.ones_like(row_large)
        ref_grid = torch.cat([row_large, col_large, ones], dim=1) # (B, 3, N)
        
        return ref_grid

    def forward(self, delta_affines, Hs_a, Hs_b, M_a_b, norm_factor, conf_weights = None, return_details=False):
        """
        Args:
            delta_affines: (B, steps, 2, 3) 预测的仿射变换增量 (RC)
            Hs_a: (B, 3, 3) Img A -> Large A 的单应矩阵 (RC)
            Hs_b: (B, 3, 3) Img B -> Large B 的单应矩阵 (RC)
            M_a_b: (B, 2, 3) Large A -> Large B 的真值仿射变换 (RC)
            norm_factor: (B,)
            return_details: (bool) 是否返回可视化所需的详细信息
        """
        B, steps, _, _ = delta_affines.shape
        scale = 512. / norm_factor
        if conf_weights is None:
            conf_weights = torch.ones((delta_affines.shape[0],),device=delta_affines.device)

        # --- 1. 构建参考网格 (Reference Grid) ---
        ref_grid = self.get_reference_grid_in_large_coords(Hs_a)
        
        # --- 2. 计算 Ground Truth 目标点 ---
        if M_a_b.dim() == 2:
            M_gt = M_a_b.unsqueeze(0).expand(B, -1, -1)
        else:
            M_gt = M_a_b
            
        # 使用真值矩阵变换参考网格
        with torch.no_grad():
            target_points = torch.bmm(M_gt, ref_grid)

        # --- 3. 迭代计算预测损失 ---
        current_affine = torch.eye(2, 3, device=self.device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)
        identity_linear = torch.eye(2, device=self.device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)
        
        step_losses = []
        last_loss = None
        
        # [修改] 始终记录轨迹，用于可视化返回
        affine_trajectory = []
        if return_details:
            affine_trajectory.append(current_affine.clone())
        
        for t in range(steps):
            delta = delta_affines[:, t]
            current_affine = merge_affine(current_affine, delta)
            
            if return_details:
                affine_trajectory.append(current_affine.clone())
            
            # A. 几何距离损失
            pred_points = torch.bmm(current_affine, ref_grid)
            loss_dist = torch.norm(pred_points - target_points, dim=1) * conf_weights # B,N
            last_loss = torch.mean(loss_dist * scale.unsqueeze(-1) * conf_weights)
            
            # B. 正则化损失
            pred_linear = current_affine[:, :, :2]
            loss_reg = torch.norm(pred_linear - identity_linear, p='fro', dim=(1, 2)).mean()
            
            # C. 单步总 Loss
            total_step_loss = torch.mean(loss_dist) + self.reg_weight * loss_reg
            step_losses.append(total_step_loss)
            
        # --- 4. 时间加权聚合 ---
        step_losses_tensor = torch.stack(step_losses)
        exponents = torch.arange(steps - 1, -1, -1, device=self.device, dtype=torch.float32)
        weights = torch.pow(self.decay_rate, exponents)
        
        weighted_loss = (step_losses_tensor * weights).sum() / weights.sum()
        
        # --- 5. 返回结果与可视化信息 ---
        if return_details:
            details = {
                # 返回整个轨迹 (B, Steps+1, 2, 3)
                'pred_affines_list': torch.stack(affine_trajectory, dim=1),
                'gt_affine': M_gt.detach(),
                'coords_a': ref_grid.detach(), # 保留以备不时之需
                'Hs_b': Hs_b.detach()
            }
            return weighted_loss, last_loss, details
            
        return weighted_loss, last_loss