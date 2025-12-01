import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
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
    """
    反归一化图像 Tensor -> HWC Uint8 Numpy
    img_tensor: (C, H, W) or (H, W, C) torch tensor or numpy array
    """
    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.detach().cpu().numpy()
    else:
        img = img_tensor.copy()

    # 如果是 CHW，转为 HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    # 确保是连续内存，防止 cv2 报错
    img = np.ascontiguousarray(img)
    return img

def rc2xy_mat(M_rc):
    """
    将 (Row, Col) 坐标系的仿射矩阵转换为 (X, Y) 坐标系，用于 cv2.warpAffine。
    M_rc: [[a, b, ty], [c, d, tx]]  (Row_out = a*Row + b*Col + ty)
    M_xy: [[d, c, tx], [b, a, ty]]  (X_out   = d*X   + c*Y   + tx)
    """
    M_xy = M_rc.copy()
    # 交换行 (y <-> x output)
    M_xy[...,[0, 1], :] = M_xy[...,[1, 0], :]
    # 交换列 (y <-> x input)
    M_xy[...,:, [0, 1]] = M_xy[...,:, [1, 0]]
    return M_xy

def make_checkerboard(img1, img2, num_tiles=8):
    """
    生成两张图片的棋盘格混合视图。
    img1, img2: (H, W, 3) uint8 numpy arrays
    """
    H, W = img1.shape[:2]
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (W, H))
        
    h_step = H // num_tiles
    w_step = W // num_tiles
    
    mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(num_tiles):
        for x in range(num_tiles):
            if (x + y) % 2 == 0:
                y_start, y_end = y*h_step, min((y+1)*h_step, H)
                x_start, x_end = x*w_step, min((x+1)*w_step, W)
                mask[y_start:y_end, x_start:x_end] = 1
                
    mask = mask[..., None]
    checkerboard = img1 * mask + img2 * (1 - mask)
    return checkerboard.astype(np.uint8)

# --- Panel A: Registration Quality Panorama ---

def vis_registration_panorama(img_a, img_b, gt_matrix_rc, pred_matrix_rc):
    """
    面板 A: 配准质量三阶段全景 (三棋盘格)
    输入矩阵均为 RC 坐标系，内部会转为 XY 供 cv2 使用。
    """
    H, W = img_a.shape[:2]
    
    # 转换为 XY 坐标系矩阵
    gt_xy = rc2xy_mat(gt_matrix_rc)
    pred_xy = rc2xy_mat(pred_matrix_rc)
    
    # 1. Origin Alignment (原始数据验证)
    # 计算 GT 逆矩阵
    gt_3x3 = np.vstack([gt_xy, [0, 0, 1]])
    gt_inv = np.linalg.inv(gt_3x3)[:2, :]
    # 将 B 变换回 A (B -> A)
    img_b_rec = cv2.warpAffine(img_b, gt_inv, (W, H), flags=cv2.INTER_LINEAR)
    view1 = make_checkerboard(img_a, img_b_rec)
    cv2.putText(view1, "1. Origin Check (GT Inv)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Training Input (初始误差)
    view2 = make_checkerboard(img_a, img_b)
    cv2.putText(view2, "2. Input (Initial Error)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 3. Prediction Result (模型纠正)
    # 将 A 变换向 B (A -> B)
    img_a_corr = cv2.warpAffine(img_a, pred_xy, (W, H), flags=cv2.INTER_LINEAR)
    view3 = make_checkerboard(img_a_corr, img_b)
    cv2.putText(view3, "3. Prediction (Corrected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    panorama = np.hstack([view1, view2, view3])
    return panorama

# --- Panel B: Feature Matching ---

def vis_feature_matching(img_a, img_b, feat_a, feat_b, gt_matrix_rc, grid_size=8):
    """
    面板 B(2): 基于特征相似度搜索的真实匹配连线
    feat_a, feat_b: (C, H, W) numpy arrays
    """
    H, W = img_a.shape[:2]
    C, Hf, Wf = feat_a.shape
    
    # 缩放比例
    sy, sx = H / Hf, W / Wf
    
    # 1. 在 Feature A 上生成采样点 (RC 格式)
    step_y, step_x = Hf // grid_size, Wf // grid_size
    yy, xx = np.meshgrid(
        np.arange(step_y//2, Hf, step_y),
        np.arange(step_x//2, Wf, step_x),
        indexing='ij'
    )
    pts_a_feat_rc = np.stack([yy.ravel(), xx.ravel()], axis=1) # (N, 2) -> (row, col)
    
    # 展平 Feat B 用于搜索
    fb_flat = feat_b.reshape(C, -1) # (C, Hf*Wf)
    # 归一化特征以计算余弦相似度
    fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)

    # 准备 GT 验证数据 (使用 RC 矩阵)
    # P_B_gt(RC) = M_gt(RC) @ P_A(RC)_homo
    pts_a_img_rc = pts_a_feat_rc * np.array([sy, sx]) # (row, col)
    pts_a_homo = np.hstack([pts_a_img_rc, np.ones((len(pts_a_img_rc), 1))])
    pts_b_gt_rc = (gt_matrix_rc @ pts_a_homo.T).T # (N, 2) row, col
    
    matches_pred_rc = []
    colors = []
    
    for i, (r, c) in enumerate(pts_a_feat_rc):
        # 2. 提取 Query
        query = feat_a[:, r, c]
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # 3. 全局搜索 Argmax
        sim = query @ fb_norm # (Hf*Wf,)
        idx = np.argmax(sim)
        
        r_match, c_match = idx // Wf, idx % Wf
        
        # 转回图像坐标
        p_pred_rc = np.array([r_match, c_match]) * np.array([sy, sx])
        matches_pred_rc.append(p_pred_rc)
        
        # 4. 验证误差 (Euclidean distance in pixels)
        # dist = sqrt( (r1-r2)^2 + (c1-c2)^2 )
        p_gt_rc = pts_b_gt_rc[i]
        dist = np.linalg.norm(p_pred_rc - p_gt_rc)
        
        # 阈值: 5% 图像宽度
        if dist < W * 0.05:
            colors.append((0, 255, 0)) # Green
        else:
            colors.append((255, 0, 0)) # Red
            
    # 绘图 (OpenCV 使用 XY 坐标: pt=(col, row))
    canvas = np.hstack([img_a, img_b])
    offset_x = W
    
    for i in range(len(pts_a_img_rc)):
        # A 点
        pt_a_xy = (int(pts_a_img_rc[i][1]), int(pts_a_img_rc[i][0]))
        # B 点 (Pred)
        pt_b_xy = (int(matches_pred_rc[i][1]) + offset_x, int(matches_pred_rc[i][0]))
        
        col = colors[i]
        cv2.circle(canvas, pt_a_xy, 4, col, -1)
        cv2.circle(canvas, pt_b_xy, 4, col, -1)
        cv2.line(canvas, pt_a_xy, pt_b_xy, col, 1)
        
    cv2.putText(canvas, "Feature Similarity Argmax Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return canvas

# --- Panel C: Pyramid Response ---

def vis_pyramid_response(feat_a, feat_b, grid_size=3):
    """
    面板 C(1): 相关性金字塔响应 (Heatmap)
    """
    C, Hf, Wf = feat_a.shape
    
    # 选取几个特征点 (垂直分布)
    step = Hf // (grid_size + 1)
    queries = []
    for i in range(1, grid_size+1):
        queries.append((i*step, Hf//2)) # (row, col)
        
    fb_flat = feat_b.reshape(C, -1)
    fb_norm = fb_flat / (np.linalg.norm(fb_flat, axis=0, keepdims=True) + 1e-8)
    
    heatmaps = []
    for r, c in queries:
        query = feat_a[:, r, c]
        query = query / (np.linalg.norm(query) + 1e-8)
        
        sim = query @ fb_norm
        sim_map = sim.reshape(Hf, Wf)
        
        # 归一化到 0-255
        sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
        sim_map_uint8 = (sim_map * 255).astype(np.uint8)
        sim_map_uint8 = cv2.resize(sim_map_uint8, (Wf*8, Hf*8), interpolation=cv2.INTER_NEAREST) # 放大以便观察
        
        heatmap = cv2.applyColorMap(sim_map_uint8, cv2.COLORMAP_JET)
        
        # 标记 Query 在 A 中的相对位置 (作为参考)
        # 注意 cv2 坐标是 (x, y) -> (col, row)
        marker_x = int(c * 8)
        marker_y = int(r * 8)
        cv2.drawMarker(heatmap, (marker_x, marker_y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        
        heatmaps.append(heatmap)
        
    return np.hstack(heatmaps)

# --- Panel D: Iterative Trajectory ---

def vis_iterative_trajectory(pred_affines_list, gt_matrix_rc, H, W):
    """
    面板 D: 迭代轨迹追踪
    pred_affines_list: (Steps, 2, 3) RC matrix
    gt_matrix_rc: (2, 3) RC matrix
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 定义网格点 (4x4) (Row, Col)
    grid_num = 4
    rows = np.linspace(H*0.25, H*0.75, grid_num)
    cols = np.linspace(W*0.25, W*0.75, grid_num)
    grid_r, grid_c = np.meshgrid(rows, cols, indexing='ij')
    
    ones = np.ones_like(grid_r)
    pts_homo_rc = np.stack([grid_r.ravel(), grid_c.ravel(), ones.ravel()], axis=0) # (3, N)
    
    # 1. 计算 Target (GT 变换后)
    # pts_target_rc = M_gt @ pts
    pts_target_rc = gt_matrix_rc @ pts_homo_rc # (2, N) [row, col]
    
    # 2. 计算 Trajectory
    paths_rc = []
    for mat in pred_affines_list:
        pts_pred = mat @ pts_homo_rc
        paths_rc.append(pts_pred)
    paths_rc = np.array(paths_rc) # (Steps, 2, N)
    
    # 绘图时转换为 XY (matplotlib: x=col, y=row)
    # path: [row, col] -> plot(col, row)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0) # Y轴向下
    ax.set_aspect('equal')
    ax.set_title("Iterative Trajectory (Green: GT, Red: Pred)")
    
    for i in range(pts_homo_rc.shape[1]):
        # Target (GT)
        tgt_r, tgt_c = pts_target_rc[0, i], pts_target_rc[1, i]
        ax.scatter(tgt_c, tgt_r, c='g', marker='x', s=60, zorder=5)
        
        # Start
        start_r, start_c = paths_rc[0, 0, i], paths_rc[0, 1, i]
        ax.scatter(start_c, start_r, c='gray', marker='o', s=30, alpha=0.5)
        
        # Path
        path_r = paths_rc[:, 0, i]
        path_c = paths_rc[:, 1, i]
        ax.plot(path_c, path_r, 'b-', alpha=0.4, linewidth=1)
        
        # Final Pred
        end_r, end_c = paths_rc[-1, 0, i], paths_rc[-1, 1, i]
        ax.scatter(end_c, end_r, c='r', marker='.', s=80, zorder=4)
        
        # Error Vector (End -> Target)
        ax.arrow(end_c, end_r, 
                 tgt_c - end_c, tgt_r - end_r,
                 color='orange', width=0.5, head_width=5, alpha=0.6, length_includes_head=True)

    plt.tight_layout()
    img = fig_to_numpy(fig)
    return img

# 保留原有的辅助函数 (兼容旧代码)
def vis_pyramid_correlation(corr_simi, corr_offset, norm_factor, num_levels=4, radius=4):
    # (略，保持原样，此处省略以节省空间，实际文件需包含)
    pass