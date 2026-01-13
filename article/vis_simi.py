import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import logging
from model.encoder import Encoder
from infer.utils import extract_features
from shared.utils import load_config
import os
from torchvision import transforms
from scipy.special import softmax
# --- 全局常量 ---
RESIZE_W, RESIZE_H = 1024, 1024 # (W, H) 格式

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@torch.no_grad()
def get_feature(encoder:Encoder,img1_np: np.ndarray, img2_np: np.ndarray):
    """
    用户实现的特征提取函数。
    
    注意：您需要用您的PyTorch模型替换此处的占位符逻辑。
    
    参数:
    - img1_np, img2_np: (1024, 1024, 3) 的 BGR numpy 数组
    - s: 预期的下采样率 (例如 8, 16, 32)
    
    返回:
    - (feat1, feat2): 两个 (h, w, D) 的 numpy 特征数组，
      其中 h = 1024 // s, w = 1024 // s。
      返回的特征必须是 float32 类型。
    """

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                        ])
    img_tensor1 = transform(img1_np)[None].cuda()
    img_tensor2 = transform(img2_np)[None].cuda()

    feat1,_ = encoder(img_tensor1)
    feat2,_ = encoder(img_tensor2)
    feat1 = feat1[0].permute(1,2,0).cpu().numpy()
    feat2 = feat2[0].permute(1,2,0).cpu().numpy()


    return feat1, feat2

def find_min_dist_pair(feat1: np.ndarray, feat2: np.ndarray, coords1: np.ndarray, coords2: np.ndarray ):
    """
    在GPU上高效计算最佳匹配对及其在原图上的最小距离。
    
    参数:
    - feat1, feat2: (h, w, D) 特征图
    - feat_orig_coords: (h, w, 2) 坐标查找表 [v_f, u_f] -> (x_o, y_o)
    
    返回:
    - min_dist: 最小的原图坐标距离
    - target_pt_f1_uv: feat1 上的目标点坐标 (u_f, v_f)
    - target_pt_f2_uv: feat2 上的目标点坐标 (u_f, v_f)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"[find_min_dist_pair] 使用设备: {device}")

    h, w, D = feat1.shape
    
    # (h, w, D) -> (h*w, D)
    f1_flat = torch.from_numpy(feat1).to(device).reshape(-1, D)
    f2_flat = torch.from_numpy(feat2).to(device).reshape(-1, D)
    
    # L2 归一化 (用于余弦相似度)
    # 添加 epsilon 防止除以零
    f1_norm = f1_flat / (torch.norm(f1_flat, dim=1, keepdim=True) + 1e-8)
    f2_norm = f2_flat / (torch.norm(f2_flat, dim=1, keepdim=True) + 1e-8)
    
    # (N, D) @ (D, M) -> (N, M)
    # N = h*w (feat2), M = h*w (feat1)
    logging.debug("[find_min_dist_pair] 正在计算相似度矩阵...")
    sim_matrix = f2_norm @ f1_norm.T
    
    # 步骤 5: 找到 feat2 中每个特征在 feat1 中的最佳匹配
    # best_indices_f1 存储了 feat1_flat 的索引
    _, best_indices_f1 = torch.max(sim_matrix, dim=1) # shape: (h*w,)
    
    # 步骤 5: 计算原图坐标距离
    # (h, w, 2) -> (h*w, 2)
    coords1_flat = torch.from_numpy(coords1).to(device).reshape(-1, 2)[best_indices_f1]
    coords2_flat = torch.from_numpy(coords2).to(device).reshape(-1, 2)
    
    # 计算欧氏距离的平方 (更快)，或者直接计算范数
    distances = torch.norm(coords1_flat - coords2_flat, dim=1) # shape: (h*w,)
    
    # 找到距离最小的那个
    min_dist, min_idx_f2_flat = torch.min(distances, dim=0)
    
    # 步骤 5: 记录这对点
    idx_f2_flat = min_idx_f2_flat.item()
    idx_f1_flat = best_indices_f1[idx_f2_flat].item()
    
    # 将一维索引转回二维 (v, u) 坐标 [y, x]
    target_pt_f2_vu = (idx_f2_flat // w, idx_f2_flat % w)
    target_pt_f1_vu = (idx_f1_flat // w, idx_f1_flat % w)
    
    # 转换为 (u, v) 坐标 [x, y] 格式，方便 matplotlib 绘图
    target_pt_f2_uv = (target_pt_f2_vu[1], target_pt_f2_vu[0])
    target_pt_f1_uv = (target_pt_f1_vu[1], target_pt_f1_vu[0])
    
    logging.debug(f"[find_min_dist_pair] 最小距离: {min_dist.item():.4f}")
    
    return min_dist.item(), target_pt_f1_uv, target_pt_f2_uv


def plot_similarity_map(data: dict, save_file: Path):
    """(图3) 绘制feat1与feat2目标点的相似度图"""
    logging.debug(f"绘制图3: {save_file}")
    feat1 = data['feat1']
    feat2 = data['feat2']
    target_pt_f1_uv = data['target_pt_f1'] # (u_f, v_f)
    target_pt_f2_uv = data['target_pt_f2'] # (u_f, v_f)
    
    h, w, D = feat1.shape
    
    # 目标点在 feat2 中的特征向量
    target_vec_f2 = feat2[target_pt_f2_uv[1], target_pt_f2_uv[0], :] # (D,)
    
    # 计算它与 feat1 中所有特征的余弦相似度
    f1_flat = feat1.reshape(h * w, D)
    
    # 归一化
    target_vec_norm = target_vec_f2 / (np.linalg.norm(target_vec_f2) + 1e-8)
    f1_flat_norm = f1_flat / (np.linalg.norm(f1_flat, axis=1, keepdims=True) + 1e-8)
    
    # (h*w, D) @ (D,) -> (h*w,)
    sims = f1_flat_norm @ target_vec_norm
    sim_map = sims.reshape(h, w)
    # max_sim = sim_map.max()
    # min_sim = 2 * np.median(sim_map) - max_sim
    # sim_map = np.clip((sim_map - min_sim) / (max_sim - min_sim),a_min=0.,a_max=1.)
    sim_map = softmax(sim_map)
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    
    # (3) 绘制蓝到黄的热力图
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=300)
    ax.imshow(sim_map, cmap='viridis', vmin=0.0, vmax=1.0) 
    ax.axis('off')
    
    plt.savefig(str(save_file).replace('.png','_1.png'), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

    target_vec_f1 = feat1[target_pt_f1_uv[1], target_pt_f1_uv[0], :] # (D,)
    
    # 计算它与 feat1 中所有特征的余弦相似度
    f2_flat = feat2.reshape(h * w, D)
    
    # 归一化
    target_vec_norm = target_vec_f1 / (np.linalg.norm(target_vec_f1) + 1e-8)
    f2_flat_norm = f2_flat / (np.linalg.norm(f2_flat, axis=1, keepdims=True) + 1e-8)
    
    # (h*w, D) @ (D,) -> (h*w,)
    sims = f2_flat_norm @ target_vec_norm
    sim_map = sims.reshape(h, w)
    # max_sim = sim_map.max()
    # min_sim = 2 * np.median(sim_map) - max_sim
    # sim_map = np.clip((sim_map - min_sim) / (max_sim - min_sim),a_min=0.,a_max=1.)
    sim_map = softmax(sim_map)
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    
    # (3) 绘制蓝到黄的热力图
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=300)
    ax.imshow(sim_map, cmap='viridis', vmin=0.0, vmax=1.0) 
    ax.axis('off')
    
    plt.savefig(str(save_file).replace('.png','_2.png'), bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

def get_random_affine_transform(orig_w, orig_h, min_area_ratio=0.5, jitter_ratio=0.1):
    """
    生成一个随机仿射变换矩阵 M (2x3).
    M 将 (resize_space, 1024x1024) 上的点映射到 (original_space) 上的点。
    满足边界和面积约束。

    返回:
    - M: (2, 3) numpy 数组
    """
    
    # 1. 定义源点 (resize 空间的3个角点)
    src_pts = np.float32([ [0, 0], [RESIZE_W, 0], [0, RESIZE_H] ])
    
    # 2. 定义约束
    min_area = min_area_ratio * (orig_w * orig_h)
    
    # 启发式：基础矩形的最小尺寸
    min_crop_w = int(orig_w * np.sqrt(min_area_ratio))
    min_crop_h = int(orig_h * np.sqrt(min_area_ratio))

    if min_crop_w >= orig_w or min_crop_h >= orig_h:
         min_crop_w = max(1, orig_w) # 确保不为0
         min_crop_h = max(1, orig_h)
         logging.warning("图像尺寸太小，无法进行 1/2 裁切，将尝试使用完整图像。")

    max_attempts = 100 # 防止死循环
    for _ in range(max_attempts):
        # 3. 生成一个基础矩形 (x1, y1, w, h)
        if min_crop_w == orig_w:
            crop_w = orig_w
            x1 = 0
        else:
            # 确保 randint 范围有效
            max_w = max(min_crop_w, orig_w)
            crop_w = np.random.randint(min_crop_w, max_w + 1)
            x1 = np.random.randint(0, orig_w - crop_w + 1) if orig_w > crop_w else 0

        if min_crop_h == orig_h:
            crop_h = orig_h
            y1 = 0
        else:
            max_h = max(min_crop_h, orig_h)
            crop_h = np.random.randint(min_crop_h, max_h + 1)
            y1 = np.random.randint(0, orig_h - crop_h + 1) if orig_h > crop_h else 0

        # 4. 定义此矩形的3个角点
        rect_tl = (x1, y1)
        rect_tr = (x1 + crop_w, y1)
        rect_bl = (x1, y1 + crop_h)
        
        # 5. 添加 Jitter 来生成目标平行四边形
        max_jitter_x = crop_w * jitter_ratio
        max_jitter_y = crop_h * jitter_ratio

        def get_jitter():
            return np.random.uniform(-max_jitter_x, max_jitter_x), \
                   np.random.uniform(-max_jitter_y, max_jitter_y)
        
        j_tl_x, j_tl_y = get_jitter()
        j_tr_x, j_tr_y = get_jitter()
        j_bl_x, j_bl_y = get_jitter()

        # 目标点 (在原图空间中)
        O_tl = np.float32([rect_tl[0] + j_tl_x, rect_tl[1] + j_tl_y])
        O_tr = np.float32([rect_tr[0] + j_tr_x, rect_tr[1] + j_tr_y])
        O_bl = np.float32([rect_bl[0] + j_bl_x, rect_bl[1] + j_bl_y])
        
        # 6. 计算第4个点 (平行四边形)
        O_br = O_tr + O_bl - O_tl
        
        dst_pts = np.float32([O_tl, O_tr, O_bl])
        all_dst_pts = [O_tl, O_tr, O_bl, O_br]

        # 7. 检查约束
        # 约束1: 边界 (所有4个角点)
        all_in_bounds = True
        for pt in all_dst_pts:
            if not (0 <= pt[0] < orig_w and 0 <= pt[1] < orig_h):
                all_in_bounds = False
                break
        
        if not all_in_bounds:
            continue # 边界检查失败, 重试

        # 约束2: 面积 (平行四边形面积 = 叉乘的模)
        v_tr = O_tr - O_tl
        v_bl = O_bl - O_tl
        area = abs(v_tr[0] * v_bl[1] - v_tr[1] * v_bl[0])
        
        if area < min_area:
            continue # 面积检查失败, 重试
            
        # 8. 成功. 计算 M 并返回
        M = cv2.getAffineTransform(dst_pts, src_pts)
        logging.debug(f"成功生成仿射变换 M: {M.ravel()}")
        return M
    
    # 9. 如果循环100次都失败了 (保险措施)
    logging.error(f"无法在 {max_attempts} 次尝试中生成有效的随机仿射变换。")
    logging.error("将返回一个居中的、无抖动(Jitter)的矩形裁切变换。")
    crop_w, crop_h = min_crop_w, min_crop_h
    x1 = (orig_w - crop_w) // 2
    y1 = (orig_h - crop_h) // 2
    
    O_tl = np.float32([x1, y1])
    O_tr = np.float32([x1 + crop_w, y1])
    O_bl = np.float32([x1, y1 + crop_h])
    dst_pts = np.float32([O_tl, O_tr, O_bl])
    return cv2.getAffineTransform(src_pts, dst_pts)

def plot_correspondence(data: dict, save_file: Path):
    """(图1) 绘制对应关系图"""
    logging.debug(f"绘制图1: {save_file}")
    resize1 = data['resize1']
    resize2 = data['resize2']
    target_pt_f1_uv = data['target_pt_f1'] # (u_f, v_f)
    target_pt_f2_uv = data['target_pt_f2'] # (u_f, v_f)
    s = data['s']
    
    H_r, W_r = resize1.shape[:2] # (1024, 1024)

    u_r1 = (target_pt_f1_uv[0] + 0.5) * s - 0.5
    v_r1 = (target_pt_f1_uv[1] + 0.5) * s - 0.5
    
    u_r2 = (target_pt_f2_uv[0] + 0.5) * s - 0.5
    v_r2 = (target_pt_f2_uv[1] + 0.5) * s - 0.5

    plt.imshow(cv2.cvtColor(resize1, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.plot(u_r1,v_r1, 'r+', markersize=12, markeredgewidth=2)
    plt.savefig(str(save_file).replace('.png','_1.png'),bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    plt.imshow(cv2.cvtColor(resize2, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.plot(u_r2,v_r2, 'r+', markersize=12, markeredgewidth=2)
    plt.savefig(str(save_file).replace('.png','_2.png'),bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    # 步骤 1: Argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--img1_path", type=str, help="输入图片1的路径")
    parser.add_argument("--img2_path", type=str, help="输入图片2的路径")
    parser.add_argument("--output_dir", type=str, help="输出结果的目录")
    parser.add_argument("--n_results", type=int, default=100, help="随机裁切的总次数 (N)")
    parser.add_argument("--downsample_s", type=int, default=16, help="特征提取器的下采样率 (s)")
    parser.add_argument('--dino_path',type=str,default='./weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
    parser.add_argument('--adapter_path',type=str)
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--device',type=str,default='cuda')
    
    args = parser.parse_args()

    # 步骤 2: 加载图像
    logging.info(f"加载图片 1: {args.img1_path}")
    img1_orig = cv2.imread(args.img1_path)
    if img1_orig is None:
        logging.error(f"无法读取图片 1: {args.img1_path}")
        return

    logging.info(f"加载图片 2: {args.img2_path}")
    img2_orig = cv2.imread(args.img2_path)
    if img2_orig is None:
        logging.error(f"无法读取图片 2: {args.img2_path}")
        return

    if img1_orig.shape != img2_orig.shape:
        logging.warning(f"警告: 两张图片尺寸不同! "
                        f"{img1_orig.shape} vs {img2_orig.shape}. "
                        "假设它们仍然是配准的。")
    
    orig_h, orig_w = img1_orig.shape[:2]
    logging.info(f"原图尺寸 (H, W): ({orig_h}, {orig_w})")

    model_configs = load_config(args.model_config_path)
    encoder = Encoder(dino_weight_path=args.dino_path,
                      embed_dim=model_configs['encoder']['embed_dim'],
                      ctx_dim=model_configs['encoder']['ctx_dim'])
    encoder.load_adapter(args.adapter_path)
      
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_count = 0
    count = 0   
    
    while True:
        total_count += 1
        logging.info(f"--- 裁切 {total_count} ---")
        
        # 3a. [NEW] 生成随机仿射变换矩阵 M
        # M maps from (resize_space) -> (original_space)
        M = get_random_affine_transform(
            orig_w, orig_h, 
            min_area_ratio=0.5, 
            jitter_ratio=0.1
        )
        
        # 3b. [NEW] 应用仿射变换 (Warp)
        # 使用黑色填充边界，因为我们假定变换都在图像内部
        resize1 = cv2.resize(img1_orig,(1024,1024))
        resize2 = cv2.warpAffine(
            img2_orig, M, (RESIZE_W, RESIZE_H), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0)
        )

        input_1 = cv2.cvtColor(resize1,cv2.COLOR_RGB2GRAY)
        input_2 = cv2.cvtColor(resize2,cv2.COLOR_RGB2GRAY)
        input_1 = np.stack([input_1] * 3,axis=-1)[None]
        input_2 = np.stack([input_2] * 3,axis=-1)[None]
        
        # 3c. [REMOVED] 旧的矩形裁切和尺度计算逻辑
        
        # 3d. [UPDATED] 计算特征图坐标 -> 原图坐标的映射
        s = args.downsample_s
        h_f, w_f = RESIZE_H // s, RESIZE_W // s
        
        if h_f == 0 or w_f == 0:
            logging.error(f"下采样率 s={s} 太大，导致特征图尺寸为零。请减小 s。")
            continue

        # 创建特征图网格 (v_f, u_f)
        v_f, u_f = np.mgrid[0:h_f, 0:w_f]

        # 核心数学：将特征坐标 (u_f, v_f) 映射回原图 (x_o, y_o)
        
        # 步骤 1: 特征图 (f) -> Resize图 (r) (像素中心)
        u_r = (u_f.astype(np.float32) + 0.5) * s - 0.5
        v_r = (v_f.astype(np.float32) + 0.5) * s - 0.5

        # 步骤 2: Resize图 (r) -> 原图 (o) (使用仿射矩阵 M)
        # (x_o) = M[0,0] * u_r + M[0,1] * v_r + M[0,2]
        # (y_o) = M[1,0] * u_r + M[1,1] * v_r + M[1,2]
        m11, m12, m13 = M[0]
        m21, m22, m23 = M[1]
        
        x_o_feat = m11 * u_r + m12 * v_r + m13
        y_o_feat = m21 * u_r + m22 * v_r + m23
        
        coords_1 = np.stack([u_r,v_r],axis=-1)
        coords_2 = np.stack([x_o_feat,y_o_feat],axis=-1)

        # 3e. 特征提取
        feats1, feats2 = extract_features(encoder,input_1,input_2,device=args.device)
        match_feat_1,_,_ = feats1
        match_feat_2,_,_ = feats2
        match_feat_1 = match_feat_1.squeeze().permute(1,2,0).cpu().numpy()
        match_feat_2 = match_feat_2.squeeze().permute(1,2,0).cpu().numpy()
        
        # 3f. [关键] 匹配与距离计算
        min_dist, target_pt_f1_uv, target_pt_f2_uv = find_min_dist_pair(
            match_feat_1, match_feat_2, coords_1,coords_2
        )

        e = max(((target_pt_f1_uv[0] - w_f * 0.5) ** 2 + (target_pt_f1_uv[1] - h_f * 0.5) ** 2) ** 0.5,
                ((target_pt_f2_uv[0] - w_f * 0.5) ** 2 + (target_pt_f2_uv[1] - h_f * 0.5) ** 2) ** 0.5)
        
        if min_dist > 0.5 or e > ((h_f * 0.25) ** 2 + (w_f * 0.25) ** 2) ** 0.5:
            continue
        
        # 3g. 存储结果
        plot_data = {
            "resize1": resize1,
            "resize2": resize2,
            "feat1": match_feat_1,
            "feat2": match_feat_2,
            "target_pt_f1": target_pt_f1_uv, # (u_f, v_f)
            "target_pt_f2": target_pt_f2_uv, # (u_f, v_f)
            "s": s
        }

        
        
        count += 1
        
        save_path = output_dir / f"{count}_dist_{min_dist:.4f}"
        save_path.mkdir(exist_ok=True)
        
        logging.info(f"  保存 第 {count} (dist={min_dist:.4f}) 到 {save_path.name}")
        
        try:
            plot_correspondence(plot_data, save_path / "pos.png")
            plot_similarity_map(plot_data, save_path / "similarity_map.png")
            cv2.imwrite(save_path / "img_1.png",resize1)
            cv2.imwrite(save_path / "img_2.png",resize2)
        except Exception as e:
            logging.error(f"为 No.{count} 绘图时发生错误: {e}")

        if count >= args.n_results:
            break
        


    logging.info("所有任务完成。")

if __name__ == "__main__":
    main()

