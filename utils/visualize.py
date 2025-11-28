import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io

def fig_to_numpy(fig):
    """将 matplotlib figure 转换为 numpy array (H, W, 3) RGB"""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=100)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def vis_pyramid_correlation(
    corr_simi: torch.Tensor,    # [B, C_total, H, W]
    corr_offset: torch.Tensor,  # [B, C_total*2, H, W]
    norm_factor: torch.Tensor,  # [B]
    num_levels: int = 4,
    radius: int = 4,
    anchor_points: list = [(0.3, 0.3), (0.7, 0.7)] # 相对坐标 (row_ratio, col_ratio)
):
    """
    可视化多层级相关性金字塔的采样点分布与相似度。
    Args:
        corr_simi: 展平后的相似度张量
        corr_offset: 展平后的偏移量张量 (归一化坐标)
        norm_factor: 归一化因子，用于还原像素坐标
        num_levels: 金字塔层数
        radius: 采样半径
    Returns:
        imgs_dict: { 'level_0': np.ndarray, ... }
    """
    B, _, H, W = corr_simi.shape
    device = corr_simi.device
    
    # 1. 计算参数
    diameter = 2 * radius + 1
    points_per_level = diameter ** 2 # e.g. 9*9=81
    
    # 2. 拆分金字塔层级
    # simi: [B, L*P, H, W] -> List of [B, P, H, W]
    simi_levels = torch.split(corr_simi, points_per_level, dim=1)
    
    # offset: [B, L*P*2, H, W] -> 重塑为 [B, L, P, 2, H, W]
    # 注意：原始 offset 排列可能是 interleaved (x1, y1, x2, y2...)
    # 在 solve_windows.py/prepare_data 中: 
    # corr_offset (B, h, w, N, 2) -> permute(0,3,4,1,2) -> flatten(1,2) -> (B, N*2, h, w)
    # 因此这里的 channel 维度是 [p1_row, p1_col, p2_row, p2_col, ...]
    offset_reshaped = corr_offset.view(B, num_levels, points_per_level, 2, H, W)
    
    imgs_dict = {}
    
    # 取 Batch 0 进行可视化
    b_idx = 0
    norm_scale = norm_factor[b_idx].item() if norm_factor.numel() > 1 else norm_factor.item()

    for lvl in range(num_levels):
        fig, axes = plt.subplots(1, len(anchor_points), figsize=(5 * len(anchor_points), 5))
        if len(anchor_points) == 1: axes = [axes]
        
        current_simi = simi_levels[lvl][b_idx]     # [P, H, W]
        current_offset = offset_reshaped[b_idx, lvl] # [P, 2, H, W]
        
        # 获取本层下采样后的 stride (近似)，仅用于设置坐标轴范围参考，不做计算依赖
        # Level 0: stride=1(相对于feature map), Level 1: stride=2 ...
        lvl_stride_approx = 2 ** lvl 
        
        for ax_idx, (r_ratio, c_ratio) in enumerate(anchor_points):
            # 确定基准像素坐标
            py = int(r_ratio * H)
            px = int(c_ratio * W)
            
            # 提取该点的采样数据
            # simi_vals: [P]
            simi_vals = current_simi[:, py, px].detach().cpu().numpy()
            
            # offsets: [P, 2] -> (row_offset, col_offset)
            # 乘以 norm_scale 还原为像素单位 (相对于 Feature Map 坐标系或原图坐标系，取决于 norm_factor 定义)
            # 通常 norm_factor 是原图尺寸，所以还原的是原图尺度的位移
            offsets_norm = current_offset[:, :, py, px].detach().cpu().numpy()
            offsets_pixel = offsets_norm * norm_scale
            
            # 准备绘图数据 (matplotlib scatter 使用 x, y)
            # offsets_pixel 是 (row, col) -> (dy, dx)
            dy = offsets_pixel[:, 0]
            dx = offsets_pixel[:, 1]
            
            ax = axes[ax_idx]
            
            # 绘制中心点
            ax.scatter(0, 0, c='black', marker='+', s=100, label='Center')
            
            # 绘制采样点 (颜色映射相似度)
            # 使用 vmin/vmax 固定颜色范围便于比较，假设 simi 是 logits 或概率
            sc = ax.scatter(dx, dy, c=simi_vals, cmap='jet', s=20, alpha=0.8)
            
            ax.set_title(f"Lvl {lvl} @ ({px},{py})\nrange: {offsets_pixel.min():.1f}~{offsets_pixel.max():.1f}")
            ax.set_xlabel("Offset X (px)")
            ax.set_ylabel("Offset Y (px)")
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.invert_yaxis() # 图像坐标系 Y 向下
            
            # 强制坐标轴比例相等，真实反映几何分布
            ax.set_aspect('equal', adjustable='datalim')
            
            # 只有最后一个图加 colorbar 防止挤压
            if ax_idx == len(anchor_points) - 1:
                plt.colorbar(sc, ax=ax, label='Similarity')

        plt.tight_layout()
        img = fig_to_numpy(fig)
        plt.close(fig)
        
        imgs_dict[f'level_{lvl}'] = img
        
    return imgs_dict