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
    pred_matrix: torch.Tensor,  # [2, 3]
    gt_matrix: torch.Tensor,    # [2, 3]
    source_points: torch.Tensor,# [3, N] (Large Coords, Homo)
    Hs_b: torch.Tensor,         # [3, 3] (Proj Matrix to Img B)
    canvas_size: tuple = (512, 512),
    grid_side_len: int = 8      # 假设是 8x8 的网格
):
    """
    可视化仿射变换预测结果 (Multi-scale Structured Visualization)。
    
    视图设计：
    1. Global View: 展示 8x8 完整矢量场，锁定在 512x512 图像范围，标记 5 个 ROI 区域。
    2. Local View: 展示 5 个关键区域（四角+中心），每个区域包含 2x2 的子网格。
       绘制连接 4 点的四边形，直观展示局部的旋转、缩放和位移误差。
    """
    H, W = canvas_size
    
    # --- 数据准备 ---
    pred_matrix = pred_matrix.detach().cpu()
    gt_matrix = gt_matrix.detach().cpu()
    source_points = source_points.detach().cpu() # [3, N]
    Hs_b = Hs_b.detach().cpu()

    # 1. 投影到 Img B 像素坐标系
    def project_to_img(pts_large_3d, H_mat, affine_mat):
        # Step 1: Apply Affine (Large A -> Large B)
        # pts_large_3d: [3, N], affine_mat: [2, 3]
        pts_large_b_2d = affine_mat @ pts_large_3d # [2, N]
        
        # 变回齐次坐标 [x, y, 1]
        ones = torch.ones(1, pts_large_b_2d.shape[1])
        pts_large_b_3d = torch.cat([pts_large_b_2d, ones], dim=0) # [3, N]
        
        # Step 2: Apply Homography (Large B -> Img B)
        pts_proj = H_mat @ pts_large_b_3d # [3, N]
        
        # 透视除法
        z = pts_proj[2:3, :] + 1e-7
        return pts_proj[:2, :] / z # [2, N] (x, y)

    # 计算所有点的 GT 和 Pred 像素坐标
    gt_pts_pix = project_to_img(source_points, Hs_b, gt_matrix).numpy()     # [2, N]
    pred_pts_pix = project_to_img(source_points, Hs_b, pred_matrix).numpy() # [2, N]

    # 2. 重塑为二维网格结构 [2, 8, 8]
    # 假设 source_points 是 meshgrid 生成的 (row-major 或 col-major)
    # AffineLoss 中通常是 meshgrid(y, x, indexing='ij') -> stack(x, y)
    # 这种情况下，展平时先遍历列(W)，再遍历行(H)。即 reshape(H, W)
    try:
        gt_grid = gt_pts_pix.reshape(2, grid_side_len, grid_side_len)     # [2, H_grid, W_grid]
        pred_grid = pred_pts_pix.reshape(2, grid_side_len, grid_side_len)
    except ValueError:
        # 如果点数不对，无法重塑，回退到仅显示散点
        print(f"Error: Grid size mismatch. Expected {grid_side_len**2} points, got {gt_pts_pix.shape[1]}")
        return None, None

    # --- 绘图初始化 ---
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    # 上部分给全局图，下部分给 5 个局部图
    gs = gridspec.GridSpec(2, 5, height_ratios=[2.2, 1], hspace=0.35, wspace=0.3)

    # ==========================
    # 1. Global View (Top)
    # ==========================
    ax_global = fig.add_subplot(gs[0, :])
    ax_global.set_title(f"Global Deformation Field ({grid_side_len}x{grid_side_len} Grid) - Green: GT, Red: Pred", fontsize=14, fontweight='bold')
    
    # 绘制全图网格点
    ax_global.scatter(gt_pts_pix[0], gt_pts_pix[1], c='green', s=15, alpha=0.6, label='GT Grid')
    ax_global.scatter(pred_pts_pix[0], pred_pts_pix[1], c='red', s=15, alpha=0.6, label='Pred Grid')
    
    # 绘制全图误差场 (Quiver)
    # 箭头从 GT 指向 Pred
    ax_global.quiver(gt_pts_pix[0], gt_pts_pix[1], 
                     pred_pts_pix[0] - gt_pts_pix[0], pred_pts_pix[1] - gt_pts_pix[1],
                     angles='xy', scale_units='xy', scale=1, 
                     color='red', alpha=0.4, width=0.002, headwidth=3, label='Error Vector')

    # 强制锁定坐标系 (关键！)
    ax_global.set_xlim(0, W)
    ax_global.set_ylim(H, 0) # Y轴向下
    ax_global.grid(True, linestyle=':', alpha=0.3)
    ax_global.legend(loc='upper right')

    # ==========================
    # 2. Local Views (Bottom)
    # ==========================
    
    # 定义 5 个关键区域的切片索引 (row_slice, col_slice)
    # 选取 2x2 的块
    roi_configs = [
        ("Top-Left", slice(0, 2), slice(0, 2)),
        ("Top-Right", slice(0, 2), slice(grid_side_len-2, grid_side_len)),
        ("Center", slice(grid_side_len//2-1, grid_side_len//2+1), slice(grid_side_len//2-1, grid_side_len//2+1)),
        ("Bottom-Left", slice(grid_side_len-2, grid_side_len), slice(0, 2)),
        ("Bottom-Right", slice(grid_side_len-2, grid_side_len), slice(grid_side_len-2, grid_side_len))
    ]

    # 绘制顺序以形成闭合四边形: (0,0) -> (0,1) -> (1,1) -> (1,0) -> (0,0)
    # 对应的 flat index (在 2x2 块内): 0 -> 2 -> 3 -> 1 -> 0
    # 注意 numpy reshape 后的存储顺序
    poly_order = [0, 2, 3, 1, 0] 

    for i, (name, r_slice, c_slice) in enumerate(roi_configs):
        ax_local = fig.add_subplot(gs[1, i])
        
        # 提取局部 2x2 数据 [2, 2, 2] -> [2, 4]
        # x 坐标
        local_gt_x = gt_grid[0, r_slice, c_slice].flatten()
        local_gt_y = gt_grid[1, r_slice, c_slice].flatten()
        local_pred_x = pred_grid[0, r_slice, c_slice].flatten()
        local_pred_y = pred_grid[1, r_slice, c_slice].flatten()
        
        # 计算平均误差
        diff = np.sqrt((local_gt_x - local_pred_x)**2 + (local_gt_y - local_pred_y)**2)
        mean_err = np.mean(diff)
        
        # 绘制点
        ax_local.scatter(local_gt_x, local_gt_y, c='green', s=60, edgecolors='white', zorder=3)
        ax_local.scatter(local_pred_x, local_pred_y, c='red', s=60, edgecolors='white', zorder=3)
        
        # 绘制结构多边形 (Polygon) - 关键：体现局部几何畸变
        ax_local.plot(local_gt_x[poly_order], local_gt_y[poly_order], 'g-', linewidth=2, alpha=0.5)
        ax_local.plot(local_pred_x[poly_order], local_pred_y[poly_order], 'r--', linewidth=2, alpha=0.5)
        
        # 绘制对应点连线 (误差箭头)
        for k in range(4):
            ax_local.annotate("", 
                              xy=(local_pred_x[k], local_pred_y[k]), 
                              xytext=(local_gt_x[k], local_gt_y[k]),
                              arrowprops=dict(arrowstyle="->", color="orange", lw=1.5, alpha=0.8))

        # 自动调整视口范围 (以 GT 包围盒为中心并外扩)
        min_x, max_x = local_gt_x.min(), local_gt_x.max()
        min_y, max_y = local_gt_y.min(), local_gt_y.max()
        
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        span_x, span_y = max_x - min_x, max_y - min_y
        
        # 保证最小可视半径 (防止点重合时视口太小) 且 容纳误差
        margin = max(40, span_x * 1.2, span_y * 1.2, mean_err * 2.5) 
        
        ax_local.set_xlim(center_x - margin, center_x + margin)
        ax_local.set_ylim(center_y + margin, center_y - margin) # Y轴向下
        
        # 样式设置
        title_color = 'green' if mean_err < 1.0 else ('red' if mean_err > 20.0 else 'black')
        ax_local.set_title(f"{name}\nErr: {mean_err:.1f}px", fontsize=10, color=title_color, fontweight='bold')
        ax_local.grid(True, linestyle='--', alpha=0.5)
        ax_local.set_xticks([])
        ax_local.set_yticks([])
        
        # 在 Global View 上绘制对应的 ROI 框 (虚线矩形)
        rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y, 
                                 linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
        ax_global.add_patch(rect)
        # 标注 ROI 名字
        ax_global.text(min_x, min_y - 5, name, fontsize=8, color='blue', ha='left')

    plt.tight_layout()
    
    # 转换为图像
    img_out = fig_to_numpy(fig)
    plt.close(fig)
    
    return img_out, img_out