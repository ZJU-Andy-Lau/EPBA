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

def vis_affine_prediction(
    pred_matrix: torch.Tensor,  # [2, 3]
    gt_matrix: torch.Tensor,    # [2, 3]
    source_points: torch.Tensor,# [3, N] (Large Coords, Homo)
    Hs_b: torch.Tensor,         # [3, 3] (Proj Matrix to Img B)
    canvas_size: tuple = (512, 512),
    grid_density: int = 8       # Grid line density
):
    """
    可视化仿射变换预测结果。
    包含两个子图：
    1. 网格对齐 (Grid Alignment): 将 Source Points 的边界框变换并投影到 Image B，对比 GT(绿) 和 Pred(红)。
    2. 残差矢量场 (Residual Vector): 绘制重投影后的误差矢量箭头。
    """
    H, W = canvas_size
    
    # --- 数据准备 ---
    # 确保输入是 CPU Numpy
    pred_matrix = pred_matrix.detach().cpu()
    gt_matrix = gt_matrix.detach().cpu()
    source_points = source_points.detach().cpu() # [3, N]
    Hs_b = Hs_b.detach().cpu()

    # 1. 计算变换后的 Large Coords
    # M @ P -> [2, N]
    # 结果是物理坐标 (Large coords)
    pred_pts_large_2d = pred_matrix @ source_points
    gt_pts_large_2d = gt_matrix @ source_points
    
    # 2. 扩展为齐次坐标以便投影
    ones = torch.ones(1, source_points.shape[1])
    pred_pts_large_3d = torch.cat([pred_pts_large_2d, ones], dim=0) # [3, N]
    gt_pts_large_3d = torch.cat([gt_pts_large_2d, ones], dim=0)     # [3, N]

    # 3. 投影回 Image B 像素空间
    # p_img = H @ p_large (并做透视除法)
    def project_to_img(pts_large_3d, H_mat):
        pts_proj = H_mat @ pts_large_3d
        z = pts_proj[2:3, :] + 1e-7
        return pts_proj[:2, :] / z # [2, N]

    pred_pts_pix = project_to_img(pred_pts_large_3d, Hs_b).numpy() # [2, N] (y, x)
    gt_pts_pix = project_to_img(gt_pts_large_3d, Hs_b).numpy()     # [2, N] (y, x)

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # View 1: Residual Vector Field
    ax1 = axes[0]
    ax1.set_title("Residual Vector Field (Red=Pred, Green=GT)")
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0) # Y axis downwards
    
    # 绘制 GT 点 (绿色小点)
    ax1.scatter(gt_pts_pix[1], gt_pts_pix[0], c='g', s=2, alpha=0.5, label='GT')
    # 绘制 Pred 点 (红色小点)
    ax1.scatter(pred_pts_pix[1], pred_pts_pix[0], c='r', s=2, alpha=0.5, label='Pred')
    
    # 绘制误差箭头 (GT -> Pred)
    # 为了避免过于密集，进行下采样
    num_pts = gt_pts_pix.shape[1]
    step = max(1, num_pts // 200) # 最多画200个箭头
    for i in range(0, num_pts, step):
        # (x, y)
        start = (gt_pts_pix[1, i], gt_pts_pix[0, i])
        end = (pred_pts_pix[1, i], pred_pts_pix[0, i])
        ax1.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                  color='orange', alpha=0.6, head_width=5, length_includes_head=True)
    
    ax1.legend()
    
    # View 2: Grid Alignment (Bounding Box)
    # 绘制原始 source_points 的边界框经过变换后的样子
    ax2 = axes[1]
    ax2.set_title("Alignment Check (Green=GT, Red=Pred)")
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    
    # 计算边界框角点索引 (Min/Max X/Y)
    # 注意: source_points 是 [x, y, 1] (AffineLoss._create_grid)
    # grid_stride 导致点是网格状的。
    # 我们可以直接取四个角点 (TopLeft, TopRight, BottomLeft, BottomRight)
    # source_points 是 meshgrid 展平，顺序通常是 col-major 或 row-major
    # AffineLoss: stack([grid_x, grid_y...]) -> ij indexing -> y varies first? 
    # let's just pick min/max coordinates to be safe.
    src_x = source_points[0].numpy()
    src_y = source_points[1].numpy()
    
    # 找到四个角点的索引
    # TL: min_x, min_y; TR: max_x, min_y; ...
    # 由于仿射变换保持直线，画这四个点的连线即可模拟网格框
    corners_idx = []
    # 近似找角点
    for tx in [src_x.min(), src_x.max()]:
        for ty in [src_y.min(), src_y.max()]:
            # 找距离最近的点索引
            dist = (src_x - tx)**2 + (src_y - ty)**2
            corners_idx.append(np.argmin(dist))
            
    # 排序角点以绘制多边形 (TL -> TR -> BR -> BL)
    # 简单起见，分别绘制 GT 和 Pred 的所有点作为点云，或者仅连接角点
    # 这里绘制变换后的外框 (Polygon)
    
    def draw_poly(ax, pts_pix, color, label):
        # pts_pix: [2, N]
        # 提取角点
        poly_pts = pts_pix[:, corners_idx].T # [4, 2] (y, x)
        # 交换为 (x, y)
        poly_pts = poly_pts[:, [1, 0]]
        
        # 排序以形成闭合环 (TL, TR, BR, BL)
        # 简单的排序方法：按角度或坐标
        # 由于只有4个点，直接按 x, y 排序
        # 这种排序不一定构成凸包，但在矩形变换下通常是凸的
        center = poly_pts.mean(axis=0)
        angles = np.arctan2(poly_pts[:,1]-center[1], poly_pts[:,0]-center[0])
        sort_order = np.argsort(angles)
        poly_pts = poly_pts[sort_order]
        
        # 闭合
        poly_pts = np.vstack([poly_pts, poly_pts[0]])
        
        ax.plot(poly_pts[:, 0], poly_pts[:, 1], color=color, linewidth=2, label=label)
    
    draw_poly(ax2, gt_pts_pix, 'lime', 'GT Box')
    draw_poly(ax2, pred_pts_pix, 'red', 'Pred Box')
    
    # 同时也画点云作为背景参考
    ax2.scatter(gt_pts_pix[1, ::step], gt_pts_pix[0, ::step], c='g', s=1, alpha=0.2)
    ax2.scatter(pred_pts_pix[1, ::step], pred_pts_pix[0, ::step], c='r', s=1, alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    img = fig_to_numpy(fig)
    plt.close(fig)
    
    return img, img # 这里简单返回相同的图，或者您可以拆分返回