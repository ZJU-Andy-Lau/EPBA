import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import cv2
import io

def fig_to_numpy(fig):
    """将 matplotlib figure 转换为 numpy array (H, W, 3) RGB"""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
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
        
        for ax_idx, (r_ratio, c_ratio) in enumerate(anchor_points):
            # 确定基准像素坐标
            py = int(r_ratio * H)
            px = int(c_ratio * W)
            
            # 提取该点的采样数据
            # simi_vals: [P]
            simi_vals = current_simi[:, py, px].detach().cpu().numpy()
            
            # offsets: [P, 2] -> (row_offset, col_offset)
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

def vis_affine_prediction(
    pred_matrix: torch.Tensor,  # [2, 3] (RC)
    gt_matrix: torch.Tensor,    # [2, 3] (RC)
    source_points: torch.Tensor,# [3, N] (Large Coords, Homo, RC)
    Hs_b: torch.Tensor,         # [3, 3] (Proj Matrix to Img B, RC)
    canvas_size: tuple = (512, 512),
    grid_side_len: int = 8      # 假设是 8x8 的网格
):
    """
    可视化仿射变换预测结果 (Multi-scale Structured Visualization)。
    注意：输入均为 RC 坐标系，绘图时需转为 XY 坐标系。
    """
    H, W = canvas_size
    
    # --- 数据准备 ---
    pred_matrix = pred_matrix.detach().cpu()
    gt_matrix = gt_matrix.detach().cpu()
    source_points = source_points.detach().cpu() # [3, N]
    Hs_b = Hs_b.detach().cpu()

    # 1. 投影到 Img B 像素坐标系 (全程在 RC 体系下计算)
    def project_to_img(pts_large_3d, H_mat, affine_mat):
        # Step 1: Apply Affine (Large A -> Large B)
        pts_large_b_2d = affine_mat @ pts_large_3d # [2, N]
        
        # 变回齐次坐标 [row, col, 1]
        ones = torch.ones(1, pts_large_b_2d.shape[1])
        pts_large_b_3d = torch.cat([pts_large_b_2d, ones], dim=0) # [3, N]
        
        # Step 2: Apply Homography (Large B -> Img B)
        pts_proj = H_mat @ pts_large_b_3d # [3, N]
        
        # 透视除法
        z = pts_proj[2:3, :] + 1e-7
        return pts_proj[:2, :] / z # [2, N] (row, col)

    # 计算所有点的 GT 和 Pred 像素坐标 (Row, Col)
    gt_pts_pix = project_to_img(source_points, Hs_b, gt_matrix).numpy()     # [2, N]
    pred_pts_pix = project_to_img(source_points, Hs_b, pred_matrix).numpy() # [2, N]

    # 2. 重塑为二维网格结构 [2, 8, 8]
    try:
        gt_grid = gt_pts_pix.reshape(2, grid_side_len, grid_side_len)     # [2, H_grid, W_grid]
        pred_grid = pred_pts_pix.reshape(2, grid_side_len, grid_side_len)
    except ValueError:
        print(f"Error: Grid size mismatch. Expected {grid_side_len**2} points, got {gt_pts_pix.shape[1]}")
        return None, None

    # --- 绘图初始化 ---
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 5, height_ratios=[2.2, 1], hspace=0.35, wspace=0.3)

    # ==========================
    # 1. Global View (Top)
    # ==========================
    ax_global = fig.add_subplot(gs[0, :])
    ax_global.set_title(f"Global Deformation Field ({grid_side_len}x{grid_side_len} Grid) - Green: GT, Red: Pred", fontsize=14, fontweight='bold')
    
    # [修改] 绘制全图网格点 (RC -> XY)
    # X = Col (index 1), Y = Row (index 0)
    ax_global.scatter(gt_pts_pix[1], gt_pts_pix[0], c='green', s=15, alpha=0.6, label='GT Grid')
    ax_global.scatter(pred_pts_pix[1], pred_pts_pix[0], c='red', s=15, alpha=0.6, label='Pred Grid')
    
    # [修改] 绘制全图误差场 (Quiver)
    # U (dx) = pred_col - gt_col
    # V (dy) = pred_row - gt_row
    U = pred_pts_pix[1] - gt_pts_pix[1]
    V = pred_pts_pix[0] - gt_pts_pix[0]
    
    ax_global.quiver(gt_pts_pix[1], gt_pts_pix[0], 
                     U, V,
                     angles='xy', scale_units='xy', scale=1, 
                     color='red', alpha=0.4, width=0.002, headwidth=3, label='Error Vector')

    ax_global.set_xlim(0, W)
    ax_global.set_ylim(H, 0) # Y轴向下
    ax_global.grid(True, linestyle=':', alpha=0.3)
    ax_global.legend(loc='upper right')

    # ==========================
    # 2. Local Views (Bottom)
    # ==========================
    
    roi_configs = [
        ("Top-Left", slice(0, 2), slice(0, 2)),
        ("Top-Right", slice(0, 2), slice(grid_side_len-2, grid_side_len)),
        ("Center", slice(grid_side_len//2-1, grid_side_len//2+1), slice(grid_side_len//2-1, grid_side_len//2+1)),
        ("Bottom-Left", slice(grid_side_len-2, grid_side_len), slice(0, 2)),
        ("Bottom-Right", slice(grid_side_len-2, grid_side_len), slice(grid_side_len-2, grid_side_len))
    ]

    poly_order = [0, 2, 3, 1, 0] 

    for i, (name, r_slice, c_slice) in enumerate(roi_configs):
        ax_local = fig.add_subplot(gs[1, i])
        
        # 提取局部 2x2 数据 [2, 2, 2] -> [2, 4]
        # grid是 (row, col)，但 scatter 需要 (x, y)
        # local_gt_row -> Y
        # local_gt_col -> X
        local_gt_row = gt_grid[0, r_slice, c_slice].flatten()
        local_gt_col = gt_grid[1, r_slice, c_slice].flatten()
        local_pred_row = pred_grid[0, r_slice, c_slice].flatten()
        local_pred_col = pred_grid[1, r_slice, c_slice].flatten()
        
        # 计算平均误差
        diff = np.sqrt((local_gt_col - local_pred_col)**2 + (local_gt_row - local_pred_row)**2)
        mean_err = np.mean(diff)
        
        # 绘制点 (X=col, Y=row)
        ax_local.scatter(local_gt_col, local_gt_row, c='green', s=60, edgecolors='white', zorder=3)
        ax_local.scatter(local_pred_col, local_pred_row, c='red', s=60, edgecolors='white', zorder=3)
        
        # 绘制结构多边形
        ax_local.plot(local_gt_col[poly_order], local_gt_row[poly_order], 'g-', linewidth=2, alpha=0.5)
        ax_local.plot(local_pred_col[poly_order], local_pred_row[poly_order], 'r--', linewidth=2, alpha=0.5)
        
        # 绘制对应点连线
        for k in range(4):
            ax_local.annotate("", 
                              xy=(local_pred_col[k], local_pred_row[k]), 
                              xytext=(local_gt_col[k], local_gt_row[k]),
                              arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.8))

        # 自动调整视口范围
        # 使用 col 作为 X, row 作为 Y
        min_x, max_x = local_gt_col.min(), local_gt_col.max()
        min_y, max_y = local_gt_row.min(), local_gt_row.max()
        
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        span_x, span_y = max_x - min_x, max_y - min_y
        
        margin = max(40, span_x * 1.2, span_y * 1.2, mean_err * 2.5) 
        
        ax_local.set_xlim(center_x - margin, center_x + margin)
        ax_local.set_ylim(center_y + margin, center_y - margin) # Y轴向下
        
        # 样式设置
        title_color = 'green' if mean_err < 1.0 else ('red' if mean_err > 20.0 else 'black')
        ax_local.set_title(f"{name}\nErr: {mean_err:.1f}px", fontsize=10, color=title_color, fontweight='bold')
        ax_local.grid(True, linestyle='--', alpha=0.5)
        ax_local.set_xticks([])
        ax_local.set_yticks([])
        
        # 在 Global View 上绘制对应的 ROI 框
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                                 linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
        ax_global.add_patch(rect)
        ax_global.text(min_x, min_y - 5, name, fontsize=8, color='blue', ha='left')

    plt.tight_layout()
    
    img_out = fig_to_numpy(fig)
    plt.close(fig)
    
    return img_out, img_out