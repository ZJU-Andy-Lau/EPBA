import torch
import torch.nn as nn
import torch.nn.functional as F

class SimLoss(nn.Module):
    def __init__(self, downsample_factor=16, temperature=0.07):
        """
        初始化 SimLoss
        Args:
            downsample_factor (int): 特征图相对于原图的下采样倍数 (默认 16)
            temperature (float): InfoNCE 的温度参数
        """
        super(SimLoss, self).__init__()
        self.scale = downsample_factor
        self.temp = temperature
        self.epsilon = 1e-8

    def get_transform_matrix(self, Hs_a, Hs_b, M_a_b):
        """
        构建从 Feat_A 到 Feat_B 的复合变换矩阵 T
        变换链: Feat_A -> Img_A -> Large_A -> Large_B -> Img_B -> Feat_B
        
        Args:
            Hs_a: (B, 3, 3) 大图A -> Img_A 的单应矩阵
            Hs_b: (B, 3, 3) 大图B -> Img_B 的单应矩阵
            M_a_b: (2, 3) 大图A -> 大图B 的仿射矩阵 (全局共享)
        Returns:
            T: (B, 3, 3) 复合变换矩阵
        """
        B = Hs_a.shape[0]
        device = Hs_a.device

        # 1. 处理 M_a_b 并扩展为 (B, 3, 3)
        # M_a_b: (2, 3) -> (3, 3) -> (B, 3, 3)
        if M_a_b.dim() == 2:
            # 添加最后一行 [0, 0, 1] 使其变为 (3, 3)
            last_row = torch.tensor([0, 0, 1], dtype=Hs_a.dtype, device=device).unsqueeze(0) # (1, 3)
            M_3x3 = torch.cat([M_a_b, last_row], dim=0) # (3, 3)
            # 广播到 Batch 维度
            M_pad = M_3x3.unsqueeze(0).repeat(B, 1, 1) # (B, 3, 3)
        else:
            # 兼容 (B, 2, 3) 的输入情况，虽然题目指明是 (2, 3)
            last_row = torch.tensor([0, 0, 1], dtype=Hs_a.dtype, device=device).view(1, 1, 3).repeat(B, 1, 1)
            M_pad = torch.cat([M_a_b, last_row], dim=1) # (B, 3, 3)

        # 2. 构建缩放矩阵 S (Feat -> Img) 和 S_inv (Img -> Feat)
        # S 将特征图坐标 (u,v) 放大 scale 倍到图像坐标
        S = torch.eye(3, dtype=Hs_a.dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        S[:, 0, 0] = self.scale
        S[:, 1, 1] = self.scale
        
        S_inv = torch.eye(3, dtype=Hs_a.dtype, device=device).unsqueeze(0).repeat(B, 1, 1)
        S_inv[:, 0, 0] = 1.0 / self.scale
        S_inv[:, 1, 1] = 1.0 / self.scale

        # 3. 计算 Hs_a 的逆矩阵 (Img_A -> Large_A)
        # 注意: torch.inverse 可能不稳定，如果 H 奇异需处理，这里假设 H 都是良好的
        Ha_inv = torch.inverse(Hs_a)

        # 4. 组合变换矩阵 (注意矩阵乘法顺序: 右乘列向量，故 T = Last @ ... @ First)
        # P_feat_b = S_inv @ H_b @ M_pad @ Ha_inv @ S @ P_feat_a
        T = S_inv @ Hs_b @ M_pad @ Ha_inv @ S
        
        return T

    def warp_features(self, feats_b, T, target_h, target_w):
        """
        利用变换矩阵 T，将 feats_b 采样到 feats_a 的网格上
        """
        B, D, H_b, W_b = feats_b.shape
        device = feats_b.device

        # 1. 生成 feats_a 的像素坐标网格 (y, x)
        # 形状: (H, W)
        y_range = torch.arange(target_h, dtype=torch.float32, device=device)
        x_range = torch.arange(target_w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')

        # 堆叠为齐次坐标 (B, 3, H*W) -> [x, y, 1]^T
        # 注意: grid_sample 坐标系是 (x, y)，所以我们构建 [x, y, 1]
        ones = torch.ones_like(grid_x)
        coords = torch.stack([grid_x, grid_y, ones], dim=0) # (3, H, W)
        coords = coords.view(3, -1).unsqueeze(0).repeat(B, 1, 1) # (B, 3, N)

        # 2. 应用变换 T
        # T: (B, 3, 3), coords: (B, 3, N) -> new_coords: (B, 3, N)
        new_coords = torch.bmm(T, coords)

        # 3. 归一化齐次坐标 (处理透视除法)
        z = new_coords[:, 2:3, :] + self.epsilon # 避免除零
        x_trans = new_coords[:, 0:1, :] / z
        y_trans = new_coords[:, 1:2, :] / z

        # 4. 转换为 grid_sample 所需的归一化坐标 [-1, 1]
        # 公式: norm = 2 * pixel / (size - 1) - 1
        x_norm = 2 * x_trans / (W_b - 1) - 1
        y_norm = 2 * y_trans / (H_b - 1) - 1

        # 拼接为 (B, H, W, 2)
        grid = torch.cat([x_norm, y_norm], dim=1).view(B, 2, target_h, target_w).permute(0, 2, 3, 1)

        # 5. 生成 Mask (标记变换后落在 feats_b 范围内的点)
        # grid 在 [-1, 1] 之间为有效
        valid_mask = (grid[..., 0] >= -1) & (grid[..., 0] <= 1) & \
                     (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
        valid_mask = valid_mask.float().unsqueeze(1) # (B, 1, H, W)

        # 6. 重采样 (Padding 模式设为 0)
        warped_feats_b = F.grid_sample(feats_b, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return warped_feats_b, valid_mask

    def forward(self, feats_a, feats_b, Hs_a, Hs_b, M_a_b):
        """
        Args:
            feats_a (Tensor): (B, D, h, w)
            feats_b (Tensor): (B, D, h, w)
            Hs_a (Tensor): (B, 3, 3) Img_A 对应的单应矩阵
            Hs_b (Tensor): (B, 3, 3) Img_B 对应的单应矩阵
            M_a_b (Tensor): (2, 3) A到B的仿射变换 (全局共享)
        """
        B, D, h, w = feats_a.shape
        N = h * w
        
        # 1. 计算总变换矩阵
        T = self.get_transform_matrix(Hs_a, Hs_b, M_a_b)

        # 2. 几何对齐: 将 feats_b warp 到 feats_a 的视角
        warped_b, mask = self.warp_features(feats_b, T, h, w)

        # 3. 准备 InfoNCE 输入
        # Flatten: (B, D, h, w) -> (B, D, N) -> permute to (B, N, D)
        feats_a_flat = feats_a.view(B, D, N).permute(0, 2, 1)      # Query
        warped_b_flat = warped_b.view(B, D, N).permute(0, 2, 1)    # Key
        mask_flat = mask.view(B * N)                               # Mask (Flattened for loss)

        # 4. L2 归一化 (Cosine Similarity 前置)
        # 注意：padding 区域是全0向量，normalize 后可能会产生 0 或 NaN (取决于 epsilon)
        # 我们依赖 F.normalize 的 epsilon 避免除零错误
        feats_a_norm = F.normalize(feats_a_flat, p=2, dim=2)
        warped_b_norm = F.normalize(warped_b_flat, p=2, dim=2)

        # 5. 计算相似度矩阵 logits
        # (B, N, D) @ (B, D, N) -> (B, N, N)
        logits = torch.bmm(feats_a_norm, warped_b_norm.transpose(1, 2)) / self.temp

        # 6. 计算 InfoNCE Loss (使用 F.cross_entropy)
        # 我们将 Batch 和 N 维度合并，视为 B*N 个分类任务
        logits_reshaped = logits.view(-1, N) # (B*N, N)
        
        # 目标标签: 对于 Query i，正样本就是 Key i
        # labels: [0, 1, ..., N-1, 0, 1, ..., N-1, ...]
        labels = torch.arange(N, device=feats_a.device).repeat(B) # (B*N)

        # 使用 reduction='none' 获取每个样本的 loss
        loss_per_sample = F.cross_entropy(logits_reshaped, labels, reduction='none') # (B*N)

        # 7. 应用 Mask 并求平均
        # 只有当 feats_a 中的点确实变换到了 feats_b 内部时，才计算 Loss
        valid_loss = loss_per_sample * mask_flat
        
        num_valid = mask_flat.sum()
        if num_valid > 0:
            final_loss = valid_loss.sum() / num_valid
        else:
            final_loss = torch.tensor(0.0, device=feats_a.device, requires_grad=True)

        return final_loss