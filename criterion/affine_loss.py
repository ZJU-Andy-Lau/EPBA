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
                               建议设置较大(如 32 或 64)以生成稀疏网格(如 8x8)，
                               既能大幅减少计算量，又能充分捕捉全局几何变换（平移/旋转/缩放）。
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
        # 覆盖 [0, H] 和 [0, W] 范围，格式为齐次坐标 [x, y, 1]^T
        self.register_buffer('local_grid', self._create_local_grid())

    def _create_local_grid(self):
        """生成本地像素坐标系的稀疏网格"""
        y_range = torch.arange(0, self.H, self.grid_stride, dtype=torch.float32, device=self.device)
        x_range = torch.arange(0, self.W, self.grid_stride, dtype=torch.float32, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        N = grid_y.numel()
        # 堆叠为齐次坐标 [x, y, 1]^T
        grid = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.ones(N, dtype=torch.float32, device=self.device)
        ], dim=0)

        return grid.unsqueeze(0) # (1, 3, N)

    def get_reference_grid_in_large_coords(self, Hs_a):
        """
        将本地像素网格投影回大图坐标系，构建物理参考网格。
        
        Args:
            Hs_a: (B, 3, 3) Large A -> Img A 的单应矩阵
        Returns:
            ref_grid: (B, 3, N) 大图坐标系下的齐次坐标 [x_large, y_large, 1]^T
        """
        B = Hs_a.shape[0]
        
        # 1. 计算逆矩阵: Img A -> Large A
        # Hs_a 将大图坐标映射到小图像素坐标，其逆矩阵将像素坐标还原为大图坐标
        try:
            Hs_a_inv = torch.linalg.inv(Hs_a)
        except RuntimeError:
            # 处理奇异矩阵或旧版PyTorch兼容性
            Hs_a_inv = torch.inverse(Hs_a)
            
        # 2. 扩展本地网格以匹配 Batch
        local_grid_batch = self.local_grid.expand(B, -1, -1) # (B, 3, N)
        
        # 3. 投影: P_large_homo = H_inv @ P_local
        large_points_homo = torch.bmm(Hs_a_inv, local_grid_batch) # (B, 3, N)
        
        # 4. 透视除法 (Perspective Division)
        # 这一步至关重要，因为单应变换是非线性的，必须除以 z 才能得到真实的物理坐标
        z = large_points_homo[:, 2:3, :] + self.epsilon
        x_large = large_points_homo[:, 0:1, :] / z
        y_large = large_points_homo[:, 1:2, :] / z
        
        # 5. 重新构建为仿射计算所需的齐次坐标 [x, y, 1]
        # 用于后续与仿射矩阵 M 相乘
        ones = torch.ones_like(x_large)
        ref_grid = torch.cat([x_large, y_large, ones], dim=1) # (B, 3, N)
        
        return ref_grid

    def forward(self, delta_affines, Hs_a, Hs_b, M_a_b, return_details=False):
        """
        Args:
            delta_affines: (B, steps, 2, 3) 预测的仿射变换增量
            Hs_a: (B, 3, 3) Img A -> Large A 的单应矩阵
            Hs_b: (B, 3, 3) Img B -> Large B 的单应矩阵 (本方案中仅保留接口兼容性，不实际参与计算)
            M_a_b: (B, 2, 3) 或 (2, 3) Large A -> Large B 的真值仿射变换
            return_details: (bool) 是否返回可视化所需的详细信息
        """
        B, steps, _, _ = delta_affines.shape
        
        # --- 1. 构建参考网格 (Reference Grid) ---
        # 将 Img A 的网格还原到大图物理空间
        # ref_grid: (B, 3, N)
        ref_grid = self.get_reference_grid_in_large_coords(Hs_a)
        
        # --- 2. 计算 Ground Truth 目标点 ---
        # 确保 M_a_b 的 Batch 维度匹配
        if M_a_b.dim() == 2:
            M_gt = M_a_b.unsqueeze(0).expand(B, -1, -1)
        else:
            M_gt = M_a_b
            
        # 使用真值矩阵变换参考网格，得到目标点
        # target_points: (B, 2, N)
        with torch.no_grad():
            target_points = torch.bmm(M_gt, ref_grid)

        # --- 3. 迭代计算预测损失 ---
        # 初始化当前累积变换为单位阵
        current_affine = torch.eye(2, 3, device=self.device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)
        identity_linear = torch.eye(2, device=self.device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)
        
        step_losses = []
        
        for t in range(steps):
            delta = delta_affines[:, t]
            # 更新累积仿射矩阵 (M_new = M_old + delta_new 或 M_new = delta_new @ M_old，取决于 merge_affine 实现)
            # 这里沿用原工程的 merge_affine 工具函数
            current_affine = merge_affine(current_affine, delta)
            
            # A. 几何距离损失 (Physical Grid Distance)
            # 使用预测矩阵变换参考网格
            # pred_points: (B, 2, N)
            pred_points = torch.bmm(current_affine, ref_grid)
            
            # 计算 L1 距离
            # 在大图坐标系下，坐标数值可能很大 (e.g. 10000+)，L1 Loss 相比 MSE (L2^2) 梯度更稳定，不易爆炸
            loss_dist = torch.mean(torch.abs(pred_points - target_points))
            
            # B. 正则化损失 (Regularization)
            # 约束仿射变换的线性部分 (旋转/缩放/剪切) 接近单位阵，防止过拟合或病态扭曲
            pred_linear = current_affine[:, :, :2]
            # 使用 Frobenious 范数
            loss_reg = torch.norm(pred_linear - identity_linear, p='fro', dim=(1, 2)).mean()
            
            # C. 单步总 Loss
            total_step_loss = loss_dist + self.reg_weight * loss_reg
            step_losses.append(total_step_loss)
            
        # --- 4. 时间加权聚合 ---
        # 赋予后期迭代更高的权重
        step_losses_tensor = torch.stack(step_losses) # (steps,)
        exponents = torch.arange(steps - 1, -1, -1, device=self.device, dtype=torch.float32)
        weights = torch.pow(self.decay_rate, exponents)
        
        weighted_loss = (step_losses_tensor * weights).sum() / weights.sum()
        
        # --- 5. 返回结果与可视化信息 ---
        if return_details:
            details = {
                'pred_affine': current_affine.detach(),
                'gt_affine': M_gt.detach(),
                'coords_a': ref_grid.detach(), # 返回大图坐标系下的参考点
                'Hs_b': Hs_b.detach()          # 保留 Hs_b 以便在可视化函数中投影回 Img B
            }
            return weighted_loss, details
            
        return weighted_loss