import torch
import numpy as np

def invert_affine_matrix(M):
    """
    计算仿射变换矩阵的逆矩阵 (从 B -> A)。
    
    参数:
        M (torch.Tensor): 形状为 (2, 3) 或 (B, 2, 3) 的仿射变换矩阵。
                          描述从 A -> B 的变换。
    
    返回:
        M_inv (torch.Tensor): 形状与输入相同的逆变换矩阵。
                              描述从 B -> A 的变换。
    """
    if M.dim() == 2:
        M = M.unsqueeze(0)  # 变为 (1, 2, 3) 处理
        is_batch = False
    else:
        is_batch = True
        
    batch_size = M.shape[0]
    device = M.device
    dtype = M.dtype

    # 1. 构建齐次坐标矩阵 (B, 3, 3)
    # 底部添加一行 [0, 0, 1]
    bottom_row = torch.tensor([0, 0, 1], device=device, dtype=dtype).view(1, 1, 3)
    bottom_row = bottom_row.expand(batch_size, -1, -1) # (B, 1, 3)
    
    M_homogeneous = torch.cat([M, bottom_row], dim=1)  # (B, 3, 3)
    
    # 2. 计算逆矩阵
    try:
        M_inv_homogeneous = torch.linalg.inv(M_homogeneous)
    except RuntimeError as e:
        print("错误: 矩阵不可逆 (可能存在奇异矩阵)。")
        raise e
        
    # 3. 取前两行作为结果 (B, 2, 3)
    M_inv = M_inv_homogeneous[:, :2, :]
    
    if not is_batch:
        return M_inv.squeeze(0)
    
    return M_inv

def merge_affine(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    计算两个仿射变换的复合变换 C，使得 C(p) = B(A(p))。

    参数:
        A (torch.Tensor): 第一个仿射变换矩阵，形状为 (B, 2, 3)。
        B (torch.Tensor): 第二个仿射变换矩阵，形状为 (B, 2, 3)。

    返回:
        torch.Tensor: 复合仿射变换矩阵 C，形状为 (B, 2, 3)。
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"Batch size a ({A.shape[0]}) 与 B ({B.shape[0]}) 不匹配。")
    if A.shape[1:] != (2, 3) or B.shape[1:] != (2, 3):
        raise ValueError(f"Tensor 形状必须为 (B, 2, 3)。")

    batch_size = A.shape[0]

    # 1. 创建用于填充的 [0, 0, 1] 行
    # 确保它在与输入张量相同的设备和数据类型上
    pad_row = torch.zeros((batch_size, 1, 3), device=A.device, dtype=A.dtype)
    pad_row[..., 2] = 1.0  # (B, 1, 3)

    # 2. 将 A 和 B 转换为 (B, 3, 3) 的齐次矩阵
    A_hom = torch.cat([A, pad_row], dim=1)  # (B, 3, 3)
    B_hom = torch.cat([B, pad_row], dim=1)  # (B, 3, 3)

    # 3. 计算复合矩阵 C_hom = B_hom @ A_hom
    # 注意：顺序是 B @ A，因为 B(A(p)) 对应 H_B * (H_A * p) = (H_B * H_A) * p
    # torch.matmul (或 @) 会自动处理批量矩阵乘法
    C_hom = B_hom @ A_hom

    # 4. 从齐次矩阵 C_hom 中提取 (B, 2, 3) 的仿射部分
    C = C_hom[:, :2, :]

    return C

def residual_to_conf(residual:torch.Tensor,left:float,right:float) -> torch.Tensor:
    """
    residual: (B,H,W) torch.Tensor

    returns: conf (B,H,W) torch.Tensor
    """
    mid = (left + right) * 0.5
    a = np.log(9) / ((right - left) * 0.5)
    residual[residual < 0] = residual.max()
    residual = torch.clamp(residual,min=0.,max=right + mid)
    conf = 1. / (1. + torch.exp(a * (residual - mid)))
    return conf