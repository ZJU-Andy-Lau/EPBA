import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Rectangle
from matplotlib.gridspec import GridSpec # [新增] 用于分块布局
import cv2
import io
from sklearn.decomposition import PCA

def fig_to_numpy(fig):
    """将 matplotlib figure 转换为 numpy array (H, W, 3) RGB"""
    io_buf = io.BytesIO()
    # [修改] 移除 bbox_inches='tight' 以保持固定尺寸控制，
    # 使用 pad_inches=0 减少白边，dpi控制分辨率
    fig.savefig(io_buf, format='png', dpi=100, pad_inches=0.1)
    io_buf.seek(0)
    img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
    io_buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.close(fig)
    return img

def denormalize_image(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """反归一化图像 Tensor -> HWC Uint8 Numpy"""
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor.copy()

    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)

def rc2xy_mat(M_rc):
    """将 (Row, Col) 仿射矩阵转换为 (X, Y)"""
    M_xy = M_rc.copy()
    # 交换行
    M_xy[...,[0, 1], :] = M_xy[...,[1, 0], :]
    # 交换列
    M_xy[...,:, [0, 1]] = M_xy[...,:, [1, 0]]
    return M_xy

def concat_with_padding(img_list, pad_width=10, color=(255, 255, 255)):
    """水平拼接图像，中间添加空隙"""
    H = max(img.shape[0] for img in img_list)
    padded_list = []
    for i, img in enumerate(img_list):
        # Resize if height mismatch
        if img.shape[0] != H:
            W_new = int(img.shape[1] * H / img.shape[0])
            img = cv2.resize(img, (W_new, H))
        
        padded_list.append(img)
        if i < len(img_list) - 1:
            pad = np.full((H, pad_width, 3), color, dtype=np.uint8)
            padded_list.append(pad)
            
    return np.hstack(padded_list)

def make_checkerboard(img1, img2, num_tiles=8):
    """生成棋盘格"""
    H, W = img1.shape[:2]
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (W, H))
    
    h_step = H // num_tiles
    w_step = W // num_tiles
    mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(num_tiles):
        for x in range(num_tiles):
            if (x + y) % 2 == 0:
                y_s, y_e = y*h_step, min((y+1)*h_step, H)
                x_s, x_e = x*w_step, min((x+1)*w_step, W)
                mask[y_s:y_e, x_s:x_e] = 1
                
    mask = mask[..., None]
    return (img1 * mask + img2 * (1 - mask)).astype(np.uint8)

# --- Panel A: Registration Quality (Strip View) ---

def warp_image_by_global_affine(img_src, H_src_rc, H_dst_rc, M_global_rc, target_wh):
    """
    将 img_src (对应 H_src) 通过全局仿射 M_global 变换到 img_dst (对应 H_dst) 的视角下。
    变换链: Small_Src -> Large_Src -> Large_Dst -> Small_Dst
    """
    # 1. 准备矩阵
    # 确保 M_global 是 3x3
    if M_global_rc.shape == (2, 3):
        M_global_3x3 = np.eye(3)
        M_global_3x3[:2, :] = M_global_rc
    else:
        M_global_3x3 = M_global_rc
        
    # 计算 H_src 的逆 (Small -> Large)
    try:
        H_src_inv = np.linalg.inv(H_src_rc)
    except np.linalg.LinAlgError:
        H_src_inv = np.linalg.pinv(H_src_rc)
        
    # 2. 计算总变换矩阵 T_total (RC坐标系)
    T_rc = H_dst_rc @ M_global_3x3 @ H_src_inv
    
    # 3. 转换为 XY 坐标系 (OpenCV 使用 XY)
    T_xy = rc2xy_mat(T_rc)
    
    # 4. 执行变换
    warped_img = cv2.warpPerspective(img_src, T_xy, target_wh, flags=cv2.INTER_LINEAR)
    
    return warped_img

def vis_registration_strip(img_ref, img_target, title="Registration"):
    """
    生成单个条带: [Ref (Warped)] --gap-- [Target] --gap-- [Checkerboard]
    """
    checker = make_checkerboard(img_ref, img_target)
    
    # 添加文字标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    img_ref_labeled = img_ref.copy()
    cv2.putText(img_ref_labeled, "Image A (Warped)", (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    img_target_labeled = img_target.copy()
    cv2.putText(img_target_labeled, "Image B (Reference)", (10, 30), font, font_scale, (0, 255, 0), thickness)
    
    checker_labeled = checker.copy()
    cv2.putText(checker_labeled, f"{title}", (10, 30), font, font_scale, (0, 255, 255), thickness)
    
    strip = concat_with_padding([img_ref_labeled, img_target_labeled, checker_labeled], pad_width=10)
    return strip

# --- Panel B: Sparse Feature Matching ---

def vis_sparse_match(img_a, img_b, feat_a, feat_b, conf_a, num_points=10):
    H, W = img_a.shape[:2]
    C, Hf, Wf = feat_a.shape
    sy, sx = H / Hf, W / Wf
    
    # 1. 基于置信度采样点 (Conf > 0.5)
    if conf_a.shape != (Hf, Wf):
        conf_resized = cv2.resize(conf_a, (Wf, Hf))
    else:
        conf_resized = conf_a
        
    valid_y, valid_x = np.where(conf_resized > 0.5)
    
    if len(valid_y) == 0:
        valid_y, valid_x = np.where(conf_resized > 0.0)
        
    if len(valid_y) > num_points:
        indices = np.random.choice(len(valid_y), num_points, replace=False)
        sample_y, sample_x = valid_y[indices], valid_x[indices]
    else:
        sample_y, sample_x = valid_y, valid_x
        
    # 2. 相似度搜索
    fb_flat = feat_b.reshape(C, -1) # (C, N)
    fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)
    
    matches = []
    
    for y, x in zip(sample_y, sample_x):
        query = feat_a[:, y, x]
        query = query / (np.linalg.norm(query) + 1e-8)
        
        sim = query @ fb_norm
        idx = np.argmax(sim)
        y_match, x_match = idx // Wf, idx % Wf
        
        pt_a = (int(x * sx), int(y * sy))
        pt_b = (int(x_match * sx), int(y_match * sy))
        
        color = tuple(np.random.randint(0, 255, 3).tolist())
        matches.append((pt_a, pt_b, color))
        
    # 3. 绘图
    gap = 20
    H_max = max(img_a.shape[0], img_b.shape[0])
    canvas = np.full((H_max, W * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[:H, :W, :] = img_a
    canvas[:H, W+gap:, :] = img_b
    
    offset_x = W + gap
    
    for pt_a, pt_b, color in matches:
        pt_b_shifted = (pt_b[0] + offset_x, pt_b[1])
        cv2.circle(canvas, pt_a, 3, color, -1)
        cv2.circle(canvas, pt_b_shifted, 3, color, -1)
        cv2.line(canvas, pt_a, pt_b_shifted, color, 1, cv2.LINE_AA)
        
    return canvas

# --- Panel C: Confidence & Pyramid ---

def vis_confidence_overlay(img, conf_map):
    """
    面板 C1: 置信度热力图叠加 (红绿)
    [修改] 提高置信度色彩比重
    """
    H, W = img.shape[:2]
    
    # 1. 上采样
    conf_up = cv2.resize(conf_map, (W, H), interpolation=cv2.INTER_LINEAR)
    conf_up = np.clip(conf_up, 0, 1)
    
    # 2. 生成红绿热力图 (RdYlGn)
    cmap = plt.get_cmap('RdYlGn')
    heatmap = cmap(conf_up)[:, :, :3] # (H, W, 3) float 0-1
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 3. 叠加 [修改] 权重调整: img 0.3, heatmap 0.7 (原 0.6/0.4)
    overlay = cv2.addWeighted(img, 0.3, heatmap, 0.7, 0)
    return overlay

def vis_pyramid_response(feat_a, feat_b, level_num = 2):
    """
    面板 C2: 金字塔响应可视化
    [修改] 逻辑与 CostVolume 对齐:
    1. 计算 Full 相似度 (1x)
    2. 对相似度图进行 Avg Pooling 得到下采样层级
    """
    C, Hf, Wf = feat_a.shape
    # 选取中心点作为 Query
    cy, cx = Hf // 2, Wf // 2
    query = feat_a[:, cy, cx] # (C,)
    query = query / (np.linalg.norm(query) + 1e-8)
    
    # 1. 计算 Level 0 全分辨率相似度
    fb_flat = feat_b.reshape(C, -1)
    fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)
    
    sim_full = query @ fb_norm # (Hf*Wf,)
    sim_full = sim_full.reshape(1, 1, Hf, Wf) # (1, 1, H, W) for pooling
    sim_full_tensor = torch.from_numpy(sim_full)
    
    # 2. 模拟 Cost Volume 的金字塔构建 (Avg Pooling)
    # 假设有 3 层: Stride 1, 2, 4
    levels = range(level_num) 
    heatmaps = []
    
    for lvl in levels:
        if lvl == 0:
            sim_lvl = sim_full_tensor
            stride = 1
        else:
            stride = 2 ** lvl
            # CostVolume 使用 AvgPool2d
            sim_lvl = torch.nn.functional.avg_pool2d(sim_full_tensor, kernel_size=stride, stride=stride)
            
        sim_np = sim_lvl.squeeze().numpy() # (H_lvl, W_lvl)
        
        # 归一化显示
        sim_norm = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-8)
        sim_img = (sim_norm * 255).astype(np.uint8)
        sim_img = cv2.applyColorMap(sim_img, cv2.COLORMAP_JET)
        
        # 统一 resize 到 128x128 以便显示 (Nearest 保持像素感，或者 Linear 平滑)
        # 为了展示下采样效果，使用 Nearest 放大回看比较清晰
        sim_img_disp = cv2.resize(sim_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # 添加边框和文字
        sim_img_disp = cv2.copyMakeBorder(sim_img_disp, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(sim_img_disp, f"Lvl {lvl} (x{stride})", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        heatmaps.append(sim_img_disp)
        
    # 生成 Colorbar (保持不变)
    colorbar = np.zeros((148, 30, 3), dtype=np.uint8)
    for i in range(128):
        val = int(i * 255 / 128)
        color = cv2.applyColorMap(np.array([[[val]]], dtype=np.uint8), cv2.COLORMAP_JET)[0,0]
        colorbar[148-20-i-1, :, :] = color
    cv2.putText(colorbar, "Hi", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    cv2.putText(colorbar, "Lo", (2, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    heatmaps.append(colorbar)
    return concat_with_padding(heatmaps, pad_width=10, color=(0,0,0))

# --- Panel D: Iterative Trajectory ---

def vis_trajectory_fixed_size(pred_affines_list, gt_matrix_rc, H_a_rc, H, W):
    """
    面板 D: 迭代轨迹可视化 (三列布局: 主图 | 放大图 | 图例)
    [修改] 解决遮挡问题，采用独立区域布局
    """
    # 1. 设置宽幅画布
    # 3 个区域，比例大概 1.5 : 1 : 0.5
    fig = plt.figure(figsize=(10, 4), dpi=100)
    gs = GridSpec(1, 3, width_ratios=[2, 1.5, 0.8], figure=fig)
    
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_legend = fig.add_subplot(gs[0, 2])
    
    # 2. 数据准备 (同前)
    pt_small_rc = np.array([[H/2], [W/2], [1.0]]) 
    try:
        H_a_inv = np.linalg.inv(H_a_rc)
    except np.linalg.LinAlgError:
        H_a_inv = np.linalg.pinv(H_a_rc)
    pt_large_a = H_a_inv @ pt_small_rc
    pt_large_a = pt_large_a / (pt_large_a[2] + 1e-8)
    
    start_pt = pt_large_a[:2]
    path_points = [start_pt.flatten()]
    for mat in pred_affines_list:
        p_pred = mat @ pt_large_a
        path_points.append(p_pred.flatten())
    path_points = np.array(path_points)
    target_pt = (gt_matrix_rc @ pt_large_a).flatten()
    
    # 坐标转换 RC -> XY
    path_xy = path_points[:, ::-1]
    target_xy = target_pt[::-1]
    start_xy = path_xy[0]
    final_xy = path_xy[-1]
    
    # 3. 区域 1: 主轨迹图 (ax_main)
    ax_main.plot(path_xy[:, 0], path_xy[:, 1], 'b.-', alpha=0.6, linewidth=1.5, label='Trajectory')
    ax_main.scatter(start_xy[0], start_xy[1], c='gray', marker='o', s=60, label='Start')
    ax_main.scatter(final_xy[0], final_xy[1], c='red', marker='*', s=150, label='Pred')
    ax_main.scatter(target_xy[0], target_xy[1], c='green', marker='x', s=100, label='GT')
    
    ax_main.set_title("Optimization Trajectory")
    ax_main.grid(True, linestyle='--', alpha=0.3)
    
    # 自动范围
    all_pts = np.vstack([path_xy, target_xy[None, :]])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = np.maximum(max_xy - min_xy, 10.0)
    margin = span * 0.2
    ax_main.set_xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
    ax_main.set_ylim(max_xy[1] + margin[1], min_xy[1] - margin[1])
    ax_main.set_aspect('equal')
    
    # 4. 区域 2: 独立放大图 (ax_zoom)
    # 仅绘制 Target 和 Final Pred
    ax_zoom.scatter(target_xy[0], target_xy[1], c='green', marker='x', s=150)
    ax_zoom.scatter(final_xy[0], final_xy[1], c='red', marker='*', s=200)
    ax_zoom.plot([target_xy[0], final_xy[0]], [target_xy[1], final_xy[1]], 'k--', alpha=0.5)
    
    error = np.linalg.norm(target_xy - final_xy)
    mid_pt = (target_xy + final_xy) / 2
    ax_zoom.text(mid_pt[0], mid_pt[1], f"{error:.2f}px", fontsize=12, ha='center', va='bottom')
    
    # 设置非常紧凑的范围
    sub_pts = np.vstack([target_xy, final_xy])
    s_min = sub_pts.min(axis=0)
    s_max = sub_pts.max(axis=0)
    s_span = np.maximum(s_max - s_min, 0.5) # 最小0.5px
    s_margin = s_span * 0.5 
    ax_zoom.set_xlim(s_min[0] - s_margin[0], s_max[0] + s_margin[0])
    ax_zoom.set_ylim(s_max[1] + s_margin[1], s_min[1] - s_margin[1])
    ax_zoom.set_title("Zoom: Final Error")
    ax_zoom.grid(True, alpha=0.2)
    
    # 5. 区域 3: 图例和信息 (ax_legend)
    ax_legend.axis('off')
    # 手动绘制图例元素
    handles = [
        plt.Line2D([0], [0], color='b', marker='.', linestyle='-'),
        plt.Line2D([0], [0], color='gray', marker='o', linestyle=''),
        plt.Line2D([0], [0], color='red', marker='*', linestyle=''),
        plt.Line2D([0], [0], color='green', marker='x', linestyle=''),
    ]
    labels = ['Path', 'Start', 'Pred', 'GT']
    ax_legend.legend(handles, labels, loc='center', fontsize='medium')
    
    # 6. 转换输出
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    # [修正] 使用新的 buffer_rgba 接口并修复可能的属性错误
    img_arr = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    img_arr = img_arr[:, :, :3]
    
    plt.close(fig)
    return img_arr

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
    
# --- Legacy Helpers (Keep) ---
def feats_pca(feats:np.ndarray):
    if feats.ndim == 3:
        feats = feats[None]
    B,H,W,C = feats.shape
    feats = feats.reshape(-1,C)
    pca = PCA(n_components=3)
    feats = pca.fit_transform(feats)
    feats = 255. * (feats - feats.min()) / (feats.max() - feats.min())
    feats = feats.reshape(B,H,W,3).astype(np.uint8)
    if B == 1:
        feats = feats.squeeze(0)
    return feats

def vis_conf(conf:np.ndarray,img:np.ndarray,ds,div = .5,output_path = None):
    from utils import get_coord_mat
    points = (get_coord_mat(conf.shape[0],conf.shape[1]) * ds + ds * .5).reshape(-1,2)
    scores = conf.reshape(-1)
    canvas_cont = img.copy()
    canvas_div = img.copy()

    def score_to_color_cont(score):
        red = int((1 - score) * 255)
        green = int(score * 255)
        return (red , green, 0)
    
    def score_to_color_div(score,div):
        if score >= div:
            return (0,255,0)
        else:
            return (255,0,0)
    
    for p,score in zip(points,scores):
        p = p.astype(int)
        if 0 <= p[0] < img.shape[0] and 0 <= p[1] < img.shape[1]:
            color_cont = score_to_color_cont(score)
            color_div = score_to_color_div(score,div)
            cv2.circle(canvas_cont,(p[1],p[0]),radius=1,color=color_cont,thickness=-1)
            cv2.circle(canvas_div,(p[1],p[0]),radius=1,color=color_div,thickness=-1)
    
    return canvas_cont, canvas_div

def create_checkerboard(img1: np.ndarray, 
                        img2: np.ndarray, 
                        output_path: str, 
                        block_size: int = 32):
 
    H, W = img1.shape[:2]
    checkerboard_img = np.zeros_like(img1)

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            i_block = (i // block_size) % 2
            j_block = (j // block_size) % 2
            
            if i_block == j_block:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    img1[i:min(i+block_size, H), j:min(j+block_size, W)]
            else:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    img2[i:min(i+block_size, H), j:min(j+block_size, W)]
    
    if checkerboard_img.ndim == 3 and checkerboard_img.shape[2] == 3:
        checkerboard_img_bgr = cv2.cvtColor(checkerboard_img, cv2.COLOR_RGB2BGR)
    else:
        checkerboard_img_bgr = checkerboard_img

    cv2.imwrite(output_path, checkerboard_img_bgr)

def validate_affine_solver(coords_src, coords_dst, merged_affine, num_samples=6, dpi=120):
    """
    可视化验证局部到全局的仿射变换求解结果。
    该函数不直接显示图像，而是返回两张渲染好的图像数组。

    Args:
        coords_src (torch.Tensor): (B, N, 2), 窗口网格点的大图原始坐标
        coords_dst (torch.Tensor): (B, N, 2), 窗口网格点经过局部变换后的目标坐标
        merged_affine (torch.Tensor): (2, 3), 求解得到的全局仿射变换矩阵
        num_samples (int): 在特写图中展示的窗口数量
        dpi (int): 绘图清晰度，影响返回图片的像素尺寸

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - img_local_shifts: 记录了局部偏移的可视化结果 (H, W, 3) uint8 数组
            - img_global_error: 记录了全局配准精度的可视化结果 (H, W, 3) uint8 数组
    """

    # =========================================================
    # 定义嵌套函数：用于绘制单个图表并转化为 Numpy 数组
    # =========================================================
    def _render_plot_to_array(title, pts_a, pts_b, color_a, label_a, color_b, label_b, draw_arrows, calc_error=False):
        B, N, _ = pts_a.shape
        # 随机或固定采样窗口索引
        sample_indices = np.linspace(0, B-1, num_samples, dtype=int)
        
        # 创建画布
        fig = plt.figure(figsize=(16, 8), dpi=dpi)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 使用 GridSpec 布局: 左边 1 列 (概览), 右边多列 (特写)
        cols_detail = 3 # 特写图的列数
        rows_detail = int(np.ceil(num_samples / cols_detail))
        gs = GridSpec(rows_detail, cols_detail + 2, figure=fig) # +2 给概览图留空间

        # --- 1. 绘制全局概览图 (占据左侧 2/5 的空间) ---
        ax_global = fig.add_subplot(gs[:, :2])
        ax_global.set_title("Global Overview (Window Distribution)")
        
        # 绘制所有窗口的中心点
        all_centers = pts_a.mean(axis=1) # (B, 2)
        ax_global.scatter(all_centers[:, 0], all_centers[:, 1], c='lightgray', s=10, alpha=0.5, label='All Windows')
        
        # 高亮选中的样本窗口
        sample_centers = all_centers[sample_indices]
        ax_global.scatter(sample_centers[:, 0], sample_centers[:, 1], c='orange', s=50, edgecolors='black', label='Selected Samples', zorder=5)
        
        # 为选中的窗口添加编号
        for i, idx in enumerate(sample_indices):
            cx, cy = all_centers[idx]
            ax_global.text(cx, cy, str(i+1), fontsize=12, fontweight='bold', color='black')

        ax_global.legend()
        ax_global.set_aspect('equal')
        ax_global.set_xlabel("Global X (px)")
        ax_global.set_ylabel("Global Y (px)")
        ax_global.invert_yaxis() # 图像坐标系 Y 轴通常向下
        ax_global.grid(True, linestyle='--', alpha=0.3)

        # --- 2. 绘制局部特写子图 ---
        for i, idx in enumerate(sample_indices):
            # 计算子图位置
            r = i // cols_detail
            c = i % cols_detail + 2 # +2 是因为前两列给了概览图
            
            ax = fig.add_subplot(gs[r, c])
            
            # 获取当前窗口的点
            pa = pts_a[idx] # (N, 2)
            pb = pts_b[idx] # (N, 2)
            
            # 计算当前窗口的局部范围，用于设置坐标轴
            all_local_pts = np.vstack([pa, pb])
            min_x, max_x = all_local_pts[:, 0].min(), all_local_pts[:, 0].max()
            min_y, max_y = all_local_pts[:, 1].min(), all_local_pts[:, 1].max()
            margin = max(1.0, (max_x - min_x) * 0.2) # 留白
            
            # 绘制点 A
            ax.scatter(pa[:, 0], pa[:, 1], c=color_a, marker='o', s=20, alpha=0.7, label=label_a if i==0 else "")
            # 绘制点 B
            ax.scatter(pb[:, 0], pb[:, 1], c=color_b, marker='x', s=20, alpha=0.7, label=label_b if i==0 else "")
            
            # 绘制连接线或箭头
            if draw_arrows:
                # 绘制箭头: A -> B
                ax.quiver(pa[:, 0], pa[:, 1], pb[:, 0]-pa[:, 0], pb[:, 1]-pa[:, 1], 
                          angles='xy', scale_units='xy', scale=1, color='gray', alpha=0.5, width=0.005)
            else:
                # 绘制线段: A - B (表示误差)
                for j in range(len(pa)):
                    ax.plot([pa[j, 0], pb[j, 0]], [pa[j, 1], pb[j, 1]], color='black', alpha=0.3, linewidth=1)

            ax.set_title(f"Sample {i+1} (ID: {idx})", fontsize=9)
            
            # --- 新增功能：计算并显示误差 ---
            if calc_error:
                # 计算两组点之间的欧式距离 (N,)
                errors = np.linalg.norm(pa - pb, axis=1)
                mean_err = np.mean(errors)
                # 在左上角显示误差文本
                ax.text(0.05, 0.95, f"Mean Err:\n{mean_err:.2f} px", 
                        transform=ax.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            ax.invert_yaxis()
            
            if i == 0:
                ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        
        # --- 3. 将 Figure 渲染为 Numpy Array ---
        fig.canvas.draw()
        
        # 兼容不同版本的 Matplotlib 获取 buffer
        try:
            # 尝试直接获取 RGB buffer (Matplotlib < 3.8)
            buf = fig.canvas.tostring_rgb()
            w, h = fig.canvas.get_width_height()
            img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        except AttributeError:
            # 兼容 Matplotlib >= 3.8 或其他后端
            # buffer_rgba 返回 (H, W, 4)
            buf = fig.canvas.buffer_rgba()
            img_arr = np.asarray(buf)[:, :, :3] # 取 RGB 通道，丢弃 Alpha
            
        # 确保是深拷贝，因为关闭 fig 后 buffer 可能失效
        img_arr = img_arr.copy()
        
        plt.close(fig) # 极其重要：关闭 figure 释放内存
        return img_arr

    # =========================================================
    # 主逻辑开始
    # =========================================================
    
    # 1. 数据预处理：转为 Numpy
    if isinstance(coords_src, torch.Tensor):
        coords_src = coords_src.detach().cpu().numpy()
        coords_dst = coords_dst.detach().cpu().numpy()
        merged_affine = merged_affine.detach().cpu().numpy()

    B, N, _ = coords_src.shape
    
    # 2. 计算 src 通过全局仿射变换后的坐标 (coords_src_af)
    src_flat = coords_src.reshape(-1, 2)
    rot = merged_affine[:, :2]
    trans = merged_affine[:, 2]
    
    src_af_flat = (src_flat @ rot.T) + trans
    coords_src_af = src_af_flat.reshape(B, N, 2)

    # 3. 生成第一张图：局部偏移 (Local Shifts)
    img_local_shifts = _render_plot_to_array(
        title="Validation 1: Local Shifts (Source -> Destination)",
        pts_a=coords_src, 
        pts_b=coords_dst, 
        color_a='blue', label_a='Src (Start)',
        color_b='red',  label_b='Dst (Target)',
        draw_arrows=True,
        calc_error=True 
    )

    # 4. 生成第二张图：全局配准精度 (Global Registration Error)
    img_global_error = _render_plot_to_array(
        title="Validation 2: Global Registration Error (Global Transformed -> Local Target)",
        pts_a=coords_src_af, 
        pts_b=coords_dst, 
        color_a='green', label_a='Global_AF (Fit)',
        color_b='red',   label_b='Dst (Target)',
        draw_arrows=False,
        calc_error=True # 第二张图开启误差计算显示
    )
    
    return img_local_shifts, img_global_error

def vis_windows_distribution(quad_A, rects_Rs, dpi=200):
    """
    绘制矩形 Rs 相对于四边形 A 的位置关系图，并将 A 的质心置于原点。
    返回绘制结果的 RGB 图片数组。

    参数:
    quad_A: numpy array, shape (4, 2)
        四边形 A 的四个顶点坐标 (x, y)。
    rects_Rs: numpy array, shape (N, 2, 2)
        N 个矩形的坐标。
        rects_Rs[i, 0, :] 是左上角 (x, y)，
        rects_Rs[i, 1, :] 是右下角 (x, y)。
    dpi: int
        输出图片的 DPI (每英寸点数)，控制分辨率。默认为 100。

    返回:
    image_array: numpy array, shape (H, W, 3)
        绘制好的图片数据 (RGB格式)。
    """
    
    # 1. 计算四边形 A 的质心 (Centroid)
    centroid = np.mean(quad_A, axis=0)
    # print(f"计算得到的质心坐标 (原坐标系): {centroid}")

    # 2. 坐标平移 (Broadcasting)
    # 将 A 和 Rs 的所有点都减去质心坐标，使得 A 的质心位于 (0,0)
    A_centered = quad_A - centroid
    Rs_centered = rects_Rs - centroid

    # 3. 开始绘图
    # 注意：这里创建 figure 但不立即显示
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    # --- 绘制四边形 A ---
    poly_patch = Polygon(
        A_centered, 
        closed=True, 
        edgecolor='blue', 
        facecolor='skyblue', 
        alpha=0.6, 
        linewidth=2, 
        label='Quadrilateral A (Centered)'
    )
    ax.add_patch(poly_patch)
    

    # --- 绘制 N 个矩形 Rs ---
    for i in range(len(Rs_centered)):
        top_left = Rs_centered[i, 0]      # [x_tl, y_tl]
        bottom_right = Rs_centered[i, 1]  # [x_br, y_br]
        
        x_tl, y_tl = top_left
        x_br, y_br = bottom_right

        anchor_x = x_tl
        anchor_y = y_br
        width = x_br - x_tl
        height = y_tl - y_br 

        rect_patch = Rectangle(
            (anchor_x, anchor_y), 
            width, 
            height,
            linewidth=1,
            edgecolor='red',
            facecolor='none', 
            linestyle='--'
        )
        ax.add_patch(rect_patch)
        
        # 在矩形中心标上序号
        rect_center_x = anchor_x + width / 2
        rect_center_y = anchor_y + height / 2
        ax.text(rect_center_x, rect_center_y, str(i), color='red', fontsize=8, ha='center', va='center')

    # --- 设置图形属性 ---
    
    # 收集所有点以自动调整坐标轴范围
    all_x = list(A_centered[:, 0])
    all_y = list(A_centered[:, 1])
    all_x.extend(Rs_centered[:, :, 0].flatten())
    all_y.extend(Rs_centered[:, :, 1].flatten())
    
    pad = 5.0
    min_x, max_x = min(all_x) - pad, max(all_x) + pad
    min_y, max_y = min(all_y) - pad, max(all_y) + pad
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # 关键：设置横纵坐标比例一致
    ax.set_aspect('equal')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(loc='upper right')
    
    plt.title(f"Relative Position Visualization (N={len(rects_Rs)} Rectangles)")
    plt.xlabel("Relative X")
    plt.ylabel("Relative Y")
    plt.tight_layout()

    # --- 4. 转换为 Numpy 数组 ---
    
    # 绘制画布内容
    fig.canvas.draw()
    
    # 获取图像尺寸
    w, h = fig.canvas.get_width_height()
    
    # 从缓冲区读取 RGB 数据
    # tostring_rgb() 在较新版本可能是 tobytes() 或 buffer_rgba()，但 tostring_rgb 兼容性较好
    try:
        buf = fig.canvas.tostring_rgb()
    except AttributeError:
        # 兼容不同版本的 Matplotlib
        buf = fig.canvas.buffer_rgba()
        
    # 将字符串/字节流转换为 uint8 数组
    image_array = np.frombuffer(buf, dtype=np.uint8)
    
    # 如果使用的是 buffer_rgba，可能会得到 RGBA，需要根据长度 reshape
    if len(image_array) == w * h * 4:
        image_array = image_array.reshape((h, w, 4))
        image_array = image_array[:, :, :3] # 只保留 RGB
    else:
        image_array = image_array.reshape((h, w, 3))
    
    # 关闭图形以释放内存，避免在循环调用时内存泄漏
    plt.close(fig)
    
    return image_array