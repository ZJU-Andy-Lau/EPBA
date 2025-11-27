import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineLoss(nn.Module):
    def __init__(self, img_size, grid_stride=16, decay_rate=0.8, reg_weight=1e-3,device = 'cuda'):
        """
        初始化 AffineLoss (基于同名点匹配)

        Args:
            img_size (tuple): 裁切影像的尺寸 (H, W)
            grid_stride (int): 网格采样步长，越小点越密计算量越大，默认为 16
            decay_rate (float): 时间步权重的衰减系数，默认为 0.8
            reg_weight (float): 正则化项的权重，默认为 1e-3
        """
        super(AffineLoss, self).__init__()
        self.H, self.W = img_size
        self.grid_stride = grid_stride
        self.decay_rate = decay_rate
        self.reg_weight = reg_weight
        self.device = device
        self.epsilon = 1e-6

        # 1. 预计算标准化的像素网格 (1, 3, N)
        # 格式为齐次坐标 [x, y, 1]^T
        self.register_buffer('base_grid', self._create_grid())

    def _create_grid(self):
        """生成稀疏的像素采样网格"""
        # 使用 meshgrid 生成坐标
        y_range = torch.arange(0, self.H, self.grid_stride, dtype=torch.float32, device = self.device)
        x_range = torch.arange(0, self.W, self.grid_stride, dtype=torch.float32, device = self.device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        # 展平
        N = grid_y.numel()
        # 堆叠为 (3, N) -> x, y, 1
        grid = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.ones(N, dtype=torch.float32, device=self.device)
        ], dim=0)

        return grid.unsqueeze(0) # (1, 3, N)

    def apply_homography(self, H, points):
        """
        应用单应变换 H (3x3) 到点集 points (3xN)
        并执行透视除法
        """
        # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
        proj_points = torch.bmm(H, points)
        
        # 透视除法: x' = x/z, y' = y/z
        z = proj_points[:, 2:3, :] + self.epsilon # 避免除零
        return proj_points / z

    def apply_affine(self, M, points):
        """
        应用仿射变换 M (2x3) 到点集 points (3xN)
        无需透视除法，输出为 (B, 2, N)
        """
        # (B, 2, 3) @ (B, 3, N) -> (B, 2, N)
        return torch.bmm(M, points)

    def get_ground_truth_and_mask(self, Hs_a, Hs_b, M_a_b):
        """
        计算 Ground Truth 目标点和有效区域 Mask
        """
        B = Hs_a.shape[0]
        N = self.base_grid.shape[2]
        device = Hs_a.device

        # --- 1. 准备数据 ---
        # 扩展网格到 Batch
        grid_a = self.base_grid.expand(B, -1, -1) # (B, 3, N)

        # 确保 M_a_b 是 (B, 2, 3)
        if M_a_b.dim() == 2:
            M_a_b = M_a_b.unsqueeze(0).repeat(B, 1, 1)

        # --- 2. 计算 Img A -> Large A 的坐标 (起点) ---
        # coords_a = Hs_a_inv @ grid_a
        try:
            Hs_a_inv = torch.linalg.inv(Hs_a).to(torch.float32)
        except RuntimeError:
            Hs_a_inv = torch.inverse(Hs_a).to(torch.float32)
        
        # 这里是大图坐标，数值可能很大
        coords_a = self.apply_homography(Hs_a_inv, grid_a) # (B, 3, N)

        # --- 3. 计算 Large A -> Large B 的坐标 (GT 目标点) ---
        # coords_b = M_a_b @ coords_a
        # 输出是 (B, 2, N)，这是我们在 Loss 中要回归的目标物理位置
        target_coords_b_2d = self.apply_affine(M_a_b, coords_a) # (B, 2, N)

        # 为了后续投影回 Img B 计算 Mask，我们需要将其变回齐次坐标 (B, 3, N)
        target_coords_b_3d = torch.cat([
            target_coords_b_2d, 
            torch.ones(B, 1, N, device=device, dtype=target_coords_b_2d.dtype)
        ], dim=1)

        # --- 4. 计算 Mask (投影回 Img B 检查边界) ---
        # proj_b = Hs_b @ coords_b
        points_in_img_b = self.apply_homography(Hs_b, target_coords_b_3d) # (B, 3, N)
        
        # 检查是否在 [0, W] 和 [0, H] 范围内
        x_b = points_in_img_b[:, 0, :]
        y_b = points_in_img_b[:, 1, :]
        
        # 定义稍宽松的边界或严格边界
        mask = (x_b >= 0) & (x_b <= self.W - 1) & \
               (y_b >= 0) & (y_b <= self.H - 1)
        
        mask = mask.float().unsqueeze(1) # (B, 1, N)

        return coords_a, target_coords_b_2d, mask

    def forward(self, delta_affines, Hs_a, Hs_b, M_a_b):
        """
        Args:
            delta_affines: (B, steps, 2, 3) 预测的仿射变换增量
            Hs_a: (B, 3, 3) Img A -> Large A 的单应矩阵
            Hs_b: (B, 3, 3) Img B -> Large B 的单应矩阵
            M_a_b: (B, 2, 3) 或 (2, 3) Large A -> Large B 的真值仿射变换
        """
        B, steps, _, _ = delta_affines.shape
        device = delta_affines.device

        # 1. 预计算 Ground Truth 和 Mask
        # coords_a: (B, 3, N) - 起点 (Large coords)
        # target_coords: (B, 2, N) - 终点 (Large coords)
        # mask: (B, 1, N) - 有效点掩膜
        coords_a, target_coords, mask = self.get_ground_truth_and_mask(Hs_a, Hs_b, M_a_b)
        
        # 统计有效点数量，用于 Loss 归一化 (避免除以0)
        mask_sum = mask.sum(dim=2) # (B, 1)
        valid_batch_mask = (mask_sum > 0).float() # 标记哪些 batch 有有效点

        # 2. 初始化累积仿射变换 (单位阵)
        current_affine = torch.eye(2, 3, device=device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)
        identity_linear = torch.eye(2, device=device, dtype=delta_affines.dtype).unsqueeze(0).repeat(B, 1, 1)

        step_losses = []

        # 3. 序列预测循环
        for t in range(steps):
            delta = delta_affines[:, t, :, :]
            current_affine = current_affine + delta

            # --- A. 距离 Loss ---
            # 预测点位置: pred = current_affine @ coords_a
            pred_coords = self.apply_affine(current_affine, coords_a) # (B, 2, N)

            # 欧氏距离: ||pred - target||_2
            dist = torch.norm(pred_coords - target_coords, p=2, dim=1, keepdim=True) # (B, 1, N)

            # 应用 Mask
            masked_dist = dist * mask # (B, 1, N)
            
            # 计算平均距离 (仅在有效区域内)
            # sum over N, divide by mask_sum
            loss_dist_batch = masked_dist.sum(dim=2) / (mask_sum + self.epsilon) # (B, 1)
            loss_dist = (loss_dist_batch * valid_batch_mask).mean() # Batch 维度平均

            # --- B. 正则化 Loss ---
            # 约束 M 的线性部分接近单位阵
            pred_linear = current_affine[:, :, :2]
            diff = pred_linear - identity_linear
            loss_reg = torch.norm(diff, p='fro', dim=(1, 2)).mean()

            # --- C. 单步总 Loss ---
            total_step_loss = loss_dist + self.reg_weight * loss_reg
            step_losses.append(total_step_loss)

        # 4. 时间加权聚合
        step_losses_tensor = torch.stack(step_losses) # (steps,)
        
        # 生成逆向衰减权重 [decay^(T-1), ..., 1]
        exponents = torch.arange(steps - 1, -1, -1, device=device, dtype=torch.float32)
        weights = torch.pow(self.decay_rate, exponents)
        
        weighted_loss = (step_losses_tensor * weights).sum() / weights.sum()

        return weighted_loss