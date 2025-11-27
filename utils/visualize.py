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

def vis_corr_heatmap(corr_simi, title="Correlation Heatmap"):
    """
    可视化相关性热图
    Args:
        corr_simi: [B, C, H, W] 相关性特征
    """
    # 取第一个样本，中心点位置的特征
    B, C, H, W = corr_simi.shape
    
    # 假设 corr_radius = 4, 则 diameter = 9, level = 4
    # C = 4 * 9 * 9 = 324
    # 我们只取第一层金字塔 (Level 0) 的热图
    radius = 4
    diameter = 2 * radius + 1
    level0_dim = diameter ** 2
    
    if C < level0_dim:
        # 如果通道数不对，可能配置不同，做个简单的 reshape 尝试或跳过
        return np.zeros((256, 256, 3), dtype=np.uint8)

    # 取图像中心点 (H//2, W//2) 的相关性向量
    cy, cx = H // 2, W // 2
    feat_vec = corr_simi[0, :level0_dim, cy, cx] # (81,)
    
    # 重塑为 9x9 热图
    heatmap = feat_vec.view(diameter, diameter).detach().cpu().numpy()
    
    # 归一化绘制
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(heatmap, cmap='jet')
    ax.set_title(f"{title} @ ({cy},{cx})")
    plt.colorbar(im, ax=ax)
    
    img = fig_to_numpy(fig)
    plt.close(fig)
    return img

def vis_flow_quiver(bg_img, flow, stride=32, title="Offset Flow"):
    """
    可视化稀疏光流场
    Args:
        bg_img: [3, H, W] Tensor, 背景图
        flow: [2, H, W] Tensor, 光流/偏移量 (像素单位)
    """
    # 转换图像
    img_np = bg_img[0].permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # 提取流
    flow_np = flow[0].detach().cpu().numpy() # (2, H, W)
    u = flow_np[1, ::stride, ::stride] # x方向偏移
    v = flow_np[0, ::stride, ::stride] # y方向偏移
    
    H, W = img_np.shape[:2]
    x = np.arange(0, W, stride)
    y = np.arange(0, H, stride)
    X, Y = np.meshgrid(x, y)
    
    # 修正尺寸匹配 (meshgrid 和 slice 可能有一点出入)
    ny, nx = u.shape
    X = X[:ny, :nx]
    Y = Y[:ny, :nx]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img_np)
    # Quiver: X, Y, U, V. 注意 y轴方向，图像坐标系y向下，但plot通常y向上
    # 这里直接画，方向需要注意
    ax.quiver(X, Y, u, v, color='r', angles='xy', scale_units='xy', scale=1, width=0.005)
    ax.set_title(title)
    ax.axis('off')
    
    img = fig_to_numpy(fig)
    plt.close(fig)
    return img

def vis_reprojection(img_a, img_b, coords_a_in_b, title="Reprojection Check"):
    """
    重投影验证：在图A画网格点，在图B画投影点
    Args:
        img_a: [3, H, W]
        img_b: [3, H, W]
        coords_a_in_b: [H, W, 2] (row, col) A中每个像素在B中的位置
    """
    img_a_np = img_a[0].permute(1, 2, 0).detach().cpu().numpy()
    img_b_np = img_b[0].permute(1, 2, 0).detach().cpu().numpy()
    
    # 归一化
    img_a_np = (img_a_np - img_a_np.min()) / (img_a_np.max() - img_a_np.min())
    img_b_np = (img_b_np - img_b_np.min()) / (img_b_np.max() - img_b_np.min())
    
    H, W = img_a_np.shape[:2]
    
    # 稀疏采样点
    step = 64
    y_range = np.arange(step//2, H, step)
    x_range = np.arange(step//2, W, step)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(img_a_np)
    ax1.set_title("Image A (Source Grid)")
    ax2.imshow(img_b_np)
    ax2.set_title("Image B (Projected Points)")
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(y_range) * len(x_range)))
    idx = 0
    
    coords = coords_a_in_b[0].detach().cpu().numpy() # (H, W, 2)
    
    for y in y_range:
        for x in x_range:
            color = colors[idx]
            # 画在 A 上
            ax1.plot(x, y, marker='o', color=color, markersize=5)
            
            # 画在 B 上 (查找坐标)
            # coords是 (row, col) -> (y, x)
            target_y, target_x = coords[y, x]
            
            # 只有在图像范围内的才画
            if 0 <= target_x < W and 0 <= target_y < H:
                ax2.plot(target_x, target_y, marker='x', color=color, markersize=5)
            
            idx += 1
            
    img = fig_to_numpy(fig)
    plt.close(fig)
    return img

def vis_grid_evolution(M_history, M_gt, H=512, W=512, title="Affine Grid Evolution"):
    """
    可视化仿射变换迭代过程
    Args:
        M_history: List of [2, 3] tensors (prediction at each step)
        M_gt: [2, 3] tensor (Ground Truth)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0) # Image coordinates
    ax.set_aspect('equal')
    
    # 定义基准网格 (一个方框和对角线)
    box = torch.tensor([
        [100, 100, 1], [W-100, 100, 1], [W-100, H-100, 1], [100, H-100, 1], [100, 100, 1]
    ], dtype=torch.float32).T # (3, 5)
    
    # 辅助函数：变换并绘图
    def plot_transformed_box(M, color, label, linestyle='-'):
        # M: (2, 3)
        # box: (3, 5)
        # out: (2, 5)
        M_cpu = M.detach().cpu()
        box_trans = M_cpu @ box
        ax.plot(box_trans[0, :], box_trans[1, :], color=color, label=label, linestyle=linestyle, linewidth=2)
    
    # 1. 绘制 GT (绿色虚线)
    plot_transformed_box(M_gt, 'green', 'Ground Truth', '--')
    
    # 2. 绘制 迭代过程 (红色，透明度渐变)
    steps = len(M_history)
    for i, M in enumerate(M_history):
        alpha = (i + 1) / steps
        label = 'Prediction' if i == steps - 1 else None
        plot_transformed_box(M, (1, 0, 0, alpha), label)
        
    ax.legend()
    ax.set_title(title)
    
    img = fig_to_numpy(fig)
    plt.close(fig)
    return img