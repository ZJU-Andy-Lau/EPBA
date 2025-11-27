import torch

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