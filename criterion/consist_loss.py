import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistLoss(nn.Module):
    def __init__(self, img_size, grid_stride=32, decay_rate=0.8):
        """
        初始化 ConsistLoss (循环一致性损失)

        Args:
            img_size (tuple): 影像尺寸 (H, W)
            grid_stride (int): 网格采样步长。由于仿射变换自由度低，稀疏网格(如32或64)即可捕捉平移和旋转。
            decay_rate (float): 时间步权重的衰减系数，默认为 0.8
        """
        super(ConsistLoss, self).__init__()
        self.H, self.W = img_size
        self.grid_stride = grid_stride
        self.decay_rate = decay_rate
        
        # 注册不需要梯度的缓冲区：基础采样网格
        self.register_buffer('base_grid', self._create_grid())

    def _create_grid(self):
        """生成稀疏的像素采样网格 (3, N)"""
        # 使用较大的 stride 生成稀疏点，减少计算量
        # 即使是 3x3 的网格也能很好地约束仿射变换
        y_range = torch.arange(0, self.H, self.grid_stride, dtype=torch.float32)
        x_range = torch.arange(0, self.W, self.grid_stride, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        N = grid_y.numel()
        # 堆叠为齐次坐标 [x, y, 1]^T
        grid = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1),
            torch.ones(N, dtype=torch.float32)
        ], dim=0)

        return grid.unsqueeze(0) # (1, 3, N)

    def _to_homogeneous_matrix(self, affine_2x3):
        """将 (B, 2, 3) 仿射矩阵填充为 (B, 3, 3)"""
        B = affine_2x3.shape[0]
        device = affine_2x3.device
        
        # 构造最后一行 [0, 0, 1]
        last_row = torch.tensor([0, 0, 1], dtype=affine_2x3.dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
        
        # 拼接
        matrix_3x3 = torch.cat([affine_2x3, last_row], dim=1)
        return matrix_3x3

    def forward(self, delta_affine_a, delta_affine_b):
        """
        Args:
            delta_affine_a: (B, steps, 2, 3) 预测的 A -> B 增量
            delta_affine_b: (B, steps, 2, 3) 预测的 B -> A 增量
        Returns:
            weighted_loss: 标量损失
        """
        B, steps, _, _ = delta_affine_a.shape
        device = delta_affine_a.device
        N = self.base_grid.shape[2]

        # 1. 扩展网格到 Batch (B, 3, N)
        points_identity = self.base_grid.expand(B, -1, -1)

        # 2. 初始化累积仿射变换 (B, 2, 3) 单位阵
        current_M_ab = torch.eye(2, 3, device=device, dtype=delta_affine_a.dtype).unsqueeze(0).repeat(B, 1, 1)
        current_M_ba = torch.eye(2, 3, device=device, dtype=delta_affine_b.dtype).unsqueeze(0).repeat(B, 1, 1)

        step_losses = []

        # 3. 序列迭代
        for t in range(steps):
            # --- 更新变换矩阵 ---
            # M_t = M_{t-1} + Delta_t
            current_M_ab = current_M_ab + delta_affine_a[:, t]
            current_M_ba = current_M_ba + delta_affine_b[:, t]

            # --- 扩展为 3x3 以进行矩阵乘法 ---
            M_ab_3x3 = self._to_homogeneous_matrix(current_M_ab)
            M_ba_3x3 = self._to_homogeneous_matrix(current_M_ba)

            # --- 计算循环变换矩阵 (Cycle Matrix) ---
            # 路径 1: A -> B -> A (先乘 AB，再乘 BA)
            # transform = M_ba @ M_ab
            cycle_M_aba = torch.bmm(M_ba_3x3, M_ab_3x3)
            
            # 路径 2: B -> A -> B (先乘 BA，再乘 AB)
            # transform = M_ab @ M_ba
            cycle_M_bab = torch.bmm(M_ab_3x3, M_ba_3x3)

            # --- 应用变换 ---
            # 坐标点: (B, 3, N)
            # 结果 P': (B, 3, N)
            points_recon_a = torch.bmm(cycle_M_aba, points_identity)
            points_recon_b = torch.bmm(cycle_M_bab, points_identity)

            # --- 计算几何距离损失 ---
            # 只取前两维 (x, y) 计算欧氏距离
            # Target 是原始点 points_identity
            
            # Dir 1: || P_recon_a - P_origin ||
            diff_a = points_recon_a[:, :2, :] - points_identity[:, :2, :]
            loss_cycle_a = torch.norm(diff_a, p=2, dim=1).mean() # Mean over points and batch

            # Dir 2: || P_recon_b - P_origin ||
            diff_b = points_recon_b[:, :2, :] - points_identity[:, :2, :]
            loss_cycle_b = torch.norm(diff_b, p=2, dim=1).mean()

            # 双向损失取平均
            total_step_loss = (loss_cycle_a + loss_cycle_b) / 2.0
            step_losses.append(total_step_loss)

        # 4. 时间加权聚合
        step_losses_tensor = torch.stack(step_losses) # (steps,)
        
        # 权重: [decay^(T-1), ..., 1]
        exponents = torch.arange(steps - 1, -1, -1, device=device, dtype=torch.float32)
        weights = torch.pow(self.decay_rate, exponents)
        
        weighted_loss = (step_losses_tensor * weights).sum() / weights.sum()

        return weighted_loss