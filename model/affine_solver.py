import torch
import torch.nn as nn

class AffineSolver(nn.Module):
    def __init__(self, damping=1e-6):
        """
        可微仿射求解器 (Differentiable Affine Solver)
        
        功能:
            将多个局部窗口的流场预测 (Local Flow) 聚合为
            全局仿射参数 (Global Affine Parameters)。
            使用加权最小二乘法 (WLS) 的解析解，支持反向传播。
            
        Args:
            damping (float): 阻尼系数 (Levenberg-Marquardt style)，防止矩阵奇异。
        """
        super().__init__()
        self.damping = damping

    def forward(self, patch_coords, predicted_flow, weights):
        """
        Args:
            patch_coords: [B, N, 2] Patch 中心在全图的归一化坐标 (x, y)
                          N 是 Patch 的数量。
            predicted_flow: [B, N, 2] predictor 预测的局部残差位移 (du, dv)
            weights: [B, N, 1] predictor 预测的置信度权重
            
        Returns:
            delta_affine: [B, 6] 全局仿射参数增量 [da, db, dc, dd, dtx, dty]
        """
        B, N, _ = patch_coords.shape
        device = patch_coords.device
        
        # 1. 构建设计矩阵 A [B, N, 2, 6]
        # Eq: u = x*da + y*db + 1*dtx
        #     v = x*dc + y*dd + 1*dty
        
        x = patch_coords[..., 0] # [B, N]
        y = patch_coords[..., 1] # [B, N]
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        
        # Row 1: [x, y, 0, 0, 1, 0] (注意参数顺序: a, b, c, d, tx, ty)
        # 这里为了矩阵构建方便，我们调整参数顺序为: [a, b, tx, c, d, ty]
        # 对应矩阵结构:
        # [x, y, 1, 0, 0, 0]
        # [0, 0, 0, x, y, 1]
        
        row1 = torch.stack([x, y, ones, zeros, zeros, zeros], dim=-1) # [B, N, 6]
        row2 = torch.stack([zeros, zeros, zeros, x, y, ones], dim=-1) # [B, N, 6]
        
        A = torch.stack([row1, row2], dim=-2) # [B, N, 2, 6]
        
        # Reshape A to [B, 2N, 6]
        A_flat = A.view(B, 2*N, 6)
        
        # 2. 构建观测向量 b [B, 2N, 1]
        b_flat = predicted_flow.view(B, 2*N, 1)
        
        # 3. 构建权重矩阵 W [B, 2N, 1]
        # 每个 Patch 的 u 和 v 共享同一个权重
        w_expanded = weights.repeat(1, 1, 2).view(B, 2*N, 1)
        
        # 4. 加权最小二乘求解: (A^T * W * A + lambda*I) * h = A^T * W * b
        
        # 计算 A_T_W = A^T * W (利用广播)
        # [B, 6, 2N] * [B, 2N, 1] (broadcasting logic differs, use mul)
        # A_flat: [B, 2N, 6]
        # w_expanded: [B, 2N, 1]
        A_weighted = A_flat * w_expanded # [B, 2N, 6]
        
        # LHS = A^T * W * A -> A_weighted^T * A
        # [B, 6, 2N] @ [B, 2N, 6] -> [B, 6, 6]
        ATA = torch.matmul(A_weighted.transpose(1, 2), A_flat)
        
        # RHS = A^T * W * b
        # [B, 6, 2N] @ [B, 2N, 1] -> [B, 6, 1]
        ATb = torch.matmul(A_weighted.transpose(1, 2), b_flat)
        
        # 5. 添加阻尼项 (正则化)
        # 这一步保证了即使 Patch 很少 (N<3) 也能解出结果 (偏向于 0)
        I = torch.eye(6, device=device).unsqueeze(0)
        ATA_damped = ATA + self.damping * I
        
        # 6. 求解线性方程组
        # h = ATA_inv * ATb
        # 使用 torch.linalg.solve 比 inv 更快更稳
        solution = torch.linalg.solve(ATA_damped, ATb).squeeze(-1) # [B, 6]
        
        # solution 顺序是 [a, b, tx, c, d, ty]
        # 我们的系统期望输出顺序 [da, db, dc, dd, dtx, dty]
        # 需要重排一下
        # 当前: 0->a, 1->b, 2->tx, 3->c, 4->d, 5->ty
        # 目标: 0->a, 1->b, 2->c, 3->d, 4->tx, 5->ty
        
        delta_affine = torch.stack([
            solution[:, 0], # a
            solution[:, 1], # b
            solution[:, 3], # c
            solution[:, 4], # d
            solution[:, 2], # tx
            solution[:, 5]  # ty
        ], dim=1)
        
        return delta_affine