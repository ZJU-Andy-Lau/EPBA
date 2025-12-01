import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cv2
import io
from sklearn.decomposition import PCA

def fig_to_numpy(fig):
    """将 matplotlib figure 转换为 numpy array (H, W, 3) RGB"""
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1)
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
    M_xy[...,[0, 1], :] = M_xy[...,[1, 0], :]
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

def vis_registration_strip(img_ref, img_target, title="Registration"):
    """
    生成单个条带: [Ref] --gap-- [Target] --gap-- [Checkerboard]
    """
    checker = make_checkerboard(img_ref, img_target)
    
    # 添加文字标签
    img_ref_labeled = img_ref.copy()
    cv2.putText(img_ref_labeled, "Image A (Warped)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    img_target_labeled = img_target.copy()
    cv2.putText(img_target_labeled, "Image B (Reference)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    checker_labeled = checker.copy()
    cv2.putText(checker_labeled, f"Checkerboard: {title}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    strip = concat_with_padding([img_ref_labeled, img_target_labeled, checker_labeled], pad_width=20)
    return strip

# --- Panel B: Sparse Feature Matching ---

def vis_sparse_match(img_a, img_b, feat_a, feat_b, conf_a, num_points=10):
    """
    面板 B: 稀疏特征匹配连线 (基于置信度采样 + 相似度搜索)
    """
    H, W = img_a.shape[:2]
    C, Hf, Wf = feat_a.shape
    sy, sx = H / Hf, W / Wf
    
    # 1. 基于置信度采样点 (Conf > 0.5)
    # conf_a: (H/16, W/16) or (H, W) depending on layer, assume (Hf, Wf)
    if conf_a.shape != (Hf, Wf):
        conf_resized = cv2.resize(conf_a, (Wf, Hf))
    else:
        conf_resized = conf_a
        
    valid_y, valid_x = np.where(conf_resized > 0.5)
    
    if len(valid_y) == 0:
        # Fallback if no high conf points
        valid_y, valid_x = np.where(conf_resized > 0.0)
        
    if len(valid_y) > num_points:
        indices = np.random.choice(len(valid_y), num_points, replace=False)
        sample_y, sample_x = valid_y[indices], valid_x[indices]
    else:
        sample_y, sample_x = valid_y, valid_x
        
    # 2. 相似度搜索
    fb_flat = feat_b.reshape(C, -1) # (C, N)
    fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)
    
    matches = [] # list of ((ax, ay), (bx, by), color)
    
    for y, x in zip(sample_y, sample_x):
        # Query
        query = feat_a[:, y, x]
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Search
        sim = query @ fb_norm
        idx = np.argmax(sim)
        y_match, x_match = idx // Wf, idx % Wf
        
        # Coords mapping to image
        pt_a = (int(x * sx), int(y * sy))
        pt_b = (int(x_match * sx), int(y_match * sy))
        
        # Random Color
        color = tuple(np.random.randint(0, 255, 3).tolist())
        matches.append((pt_a, pt_b, color))
        
    # 3. 绘图 (带空隙)
    gap = 20
    H_max = max(img_a.shape[0], img_b.shape[0])
    canvas = np.full((H_max, W * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[:H, :W, :] = img_a
    canvas[:H, W+gap:, :] = img_b
    
    offset_x = W + gap
    
    for pt_a, pt_b, color in matches:
        pt_b_shifted = (pt_b[0] + offset_x, pt_b[1])
        
        # 画小点 (r=3)
        cv2.circle(canvas, pt_a, 3, color, -1)
        cv2.circle(canvas, pt_b_shifted, 3, color, -1)
        
        # 画细线 (thickness=1)
        cv2.line(canvas, pt_a, pt_b_shifted, color, 1, cv2.LINE_AA)
        
    return canvas

# --- Panel C: Confidence & Pyramid ---

def vis_confidence_overlay(img, conf_map):
    """
    面板 C1: 置信度热力图叠加 (红绿)
    """
    H, W = img.shape[:2]
    
    # 1. 上采样
    conf_up = cv2.resize(conf_map, (W, H), interpolation=cv2.INTER_LINEAR)
    conf_up = np.clip(conf_up, 0, 1)
    
    # 2. 生成红绿热力图 (RdYlGn)
    # matplotlib cm: value 0->Red, 1->Green
    cmap = plt.get_cmap('RdYlGn')
    heatmap = cmap(conf_up)[:, :, :3] # (H, W, 3) float 0-1
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 3. 叠加 (0.6 img + 0.4 heatmap)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

def vis_pyramid_response(feat_a, feat_b):
    """
    面板 C2: 金字塔响应可视化
    """
    C, Hf, Wf = feat_a.shape
    # 选取中心点作为 Query
    cy, cx = Hf // 2, Wf // 2
    query = feat_a[:, cy, cx]
    query = query / (np.linalg.norm(query) + 1e-8)
    
    # 构建多层特征 (模拟金字塔)
    # 这里直接对 feat_b 进行不同尺度的池化来模拟
    scales = [1, 0.5, 0.25] # Level 0, 1, 2
    heatmaps = []
    
    for s in scales:
        if s == 1:
            fb_curr = feat_b
        else:
            # Average pooling for downsampling feature
            fb_curr = torch.tensor(feat_b).unsqueeze(0) # (1, C, H, W)
            fb_curr = torch.nn.functional.interpolate(fb_curr, scale_factor=s, mode='area')
            fb_curr = fb_curr.squeeze(0).numpy()
            
        c_curr, h_curr, w_curr = fb_curr.shape
        fb_flat = fb_curr.reshape(C, -1)
        fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)
        
        sim = query @ fb_norm
        sim_map = sim.reshape(h_curr, w_curr)
        
        # 归一化显示
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
        sim_img = (sim_map * 255).astype(np.uint8)
        sim_img = cv2.applyColorMap(sim_img, cv2.COLORMAP_JET)
        
        # 统一 resize 到 128x128 以便显示
        sim_img_disp = cv2.resize(sim_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # 添加边框和文字
        sim_img_disp = cv2.copyMakeBorder(sim_img_disp, 20, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255,255,255))
        cv2.putText(sim_img_disp, f"Scale {s}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        heatmaps.append(sim_img_disp)
        
    # 生成 Colorbar
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

def vis_trajectory_zoomed(pred_affines_list, gt_matrix_rc, H, W):
    """
    面板 D: 迭代轨迹 (自动放大聚焦)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 使用中心点作为观察对象
    pt_homo_rc = np.array([[H/2], [W/2], [1.0]]) # (3, 1)
    
    # 1. 计算所有关键点位置 (Row, Col)
    # Target
    target_rc = gt_matrix_rc @ pt_homo_rc # (2, 1)
    
    # Trajectory
    path_rc = []
    for mat in pred_affines_list:
        p = mat @ pt_homo_rc
        path_rc.append(p)
    path_rc = np.array(path_rc).squeeze(-1) # (Steps, 2)
    
    # 转换到 XY (Col, Row) 用于绘图
    # path_xy: (Steps, 2) -> [col, row]
    path_xy = path_rc[:, ::-1] 
    target_xy = target_rc.squeeze(-1)[::-1]
    start_xy = path_xy[0]
    end_xy = path_xy[-1]
    
    # 2. 计算 Bounding Box 并添加 Margin
    all_pts = np.vstack([path_xy, target_xy[None, :]])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    span = max_xy - min_xy
    margin = np.maximum(span * 0.3, 5.0) # 至少留 5px 边距
    
    # 设置视图范围 (Zoom In)
    ax.set_xlim(min_xy[0] - margin[0], max_xy[0] + margin[0])
    ax.set_ylim(max_xy[1] + margin[1], min_xy[1] - margin[1]) # Y轴反转
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 3. 绘图
    # Target
    ax.scatter(target_xy[0], target_xy[1], c='g', marker='x', s=150, linewidth=2, label='Target (GT)', zorder=10)
    
    # Start
    ax.scatter(start_xy[0], start_xy[1], c='gray', marker='o', s=80, label='Start', zorder=5)
    
    # Path (Line)
    ax.plot(path_xy[:, 0], path_xy[:, 1], 'b-', alpha=0.5, linewidth=1.5)
    
    # Path (Points) - 标注每个点
    ax.scatter(path_xy[:, 0], path_xy[:, 1], c='b', marker='.', s=30, zorder=6)
    
    # End
    ax.scatter(end_xy[0], end_xy[1], c='r', marker='*', s=200, label='Prediction', zorder=10)
    
    ax.legend(loc='best', fontsize='small')
    ax.set_title("Iterative Trajectory (Zoomed)")
    
    return fig_to_numpy(fig)
    
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
    # 保留原有的点阵可视化逻辑，用于 Debug_Basic
    from .utils import get_coord_mat
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
        # 确保点在图像范围内
        if 0 <= p[0] < img.shape[0] and 0 <= p[1] < img.shape[1]:
            color_cont = score_to_color_cont(score)
            color_div = score_to_color_div(score,div)
            cv2.circle(canvas_cont,(p[1],p[0]),radius=1,color=color_cont,thickness=-1)
            cv2.circle(canvas_div,(p[1],p[0]),radius=1,color=color_div,thickness=-1)
    
    return canvas_cont, canvas_div

def vis_pyramid_correlation(corr_simi, corr_offset, norm_factor, num_levels=4, radius=4):
    # 占位，保留旧接口
    return {}