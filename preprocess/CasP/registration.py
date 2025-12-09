import warnings
warnings.filterwarnings("ignore")
import os
import glob
import argparse
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import List, Tuple, Dict, Any
import h5py
from tqdm import tqdm

# 引入 CasP 工程中的模块
# 确保在工程根目录下运行，以便 python 能找到 src 包
from src.models.nets import CasP
from preprocess.conf.model import ConfHead
from shared.visualize import make_checkerboard

# ==========================================
# 1. 置信度预测模型 (用户自定义部分)
# ==========================================
class ConfPred:
    def __init__(self, dino_weight_path: str, conf_head_path: str, device='cuda'):
        """
        初始化置信度预测模型。
        """
        print("[ConfPred] Initializing Confidence Prediction Model...")
        self.conf_head = ConfHead(dino_weight_path)
        self.conf_head.load_head(conf_head_path)
        self.conf_head = self.conf_head.to(device)

    def pred(self, img: np.ndarray) -> np.ndarray:
        """
        对输入图像预测置信度图。
        
        Args:
            img: 输入图像，numpy array, 形状 (H, W, 3), RGB顺序
        
        Returns:
            conf_map: 置信度图，numpy array, 形状 (H, W), 值域 [0, 1]
        """
        H, W = img.shape[:2]
        mid_row, mid_col = H // 2, W // 2
        imgs_quater = np.stack([
            img[:mid_row, :mid_col],  # tl
            img[:mid_row, mid_col:],  # tr
            img[mid_row:, :mid_col],  # bl
            img[mid_row:, mid_col:],  # br
        ], axis=0)
        
        confs = self.conf_head.pred(imgs_quater)
        
        conf_full = np.zeros((H, W), dtype=np.float32)
        conf_full[:mid_row, :mid_col] = confs[0]
        conf_full[:mid_row, mid_col:] = confs[1]
        conf_full[mid_row:, :mid_col] = confs[2]
        conf_full[mid_row:, mid_col:] = confs[3]

        return conf_full


# ==========================================
# 2. 图像配准器核心逻辑
# ==========================================
class ImageRegistrar:
    def __init__(self, args):
        self.args = args
        self.device = args.device if torch.cuda.is_available() else 'cpu'
        print(f"[Registrar] Using device: {self.device}")

        self.transform_type = args.transform_type
        print(f"[Registrar] Transformation Type: {self.transform_type}")

        # 加载 CasP 模型
        self.matcher = self._load_casp_model(args.config_path, args.casp_weights)
        
        # 初始化置信度预测模型
        self.conf_predictor = ConfPred(args.dino_weights, args.conf_head_weights, args.device)

        # 定义尺寸参数
        self.original_h, self.original_w = 3000, 3000
        self.resample_h, self.resample_w = 3456, 3456
        
        # 分块参数: 3456 / 3 = 1152
        self.grid_rows = 3
        self.grid_cols = 3
        self.block_h = 1152
        self.block_w = 1152

    def _load_casp_model(self, config_path: str, ckpt_path: str) -> torch.nn.Module:
        """加载配置并初始化 CasP 模型"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # 自动下载权重逻辑 (如果不存在)
        if not os.path.exists(ckpt_path):
            print(f"[Registrar] Weights not found at {ckpt_path}, attempting download...")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            url = "https://huggingface.co/pq-chen/CasP/resolve/main/casp_outdoor.pth"
            try:
                import requests
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(ckpt_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("[Registrar] Download complete.")
            except Exception as e:
                raise RuntimeError(f"Failed to download weights: {e}")

        # 加载配置
        config = OmegaConf.load(config_path).config
        # 可以在此覆盖默认阈值
        config.threshold = 0.2 
        
        # 初始化模型
        matcher = CasP(config)
        state_dict = torch.load(ckpt_path, map_location='cpu')
        matcher.load_state_dict(state_dict)
        matcher.eval().to(self.device)
        return matcher

    def preprocess_image(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        处理输入的RGB图像，重采样为 3456*3456，并计算置信度。
        
        Args:
            img: 输入图像 numpy array, (H, W, 3), RGB格式
        
        Returns:
            img_orig: 原始 3000*3000 图像
            img_resampled: 3456*3456 图像
            conf_map: 3456*3456 置信度图
        """
        # 假设输入已经是 RGB 格式的 numpy 数组
        img_orig = img

        # 确保原始尺寸一致 (容错处理)
        if img_orig.shape[0] != self.original_h or img_orig.shape[1] != self.original_w:
            print(f"  [Warn] Image size {img_orig.shape[:2]} != (3000, 3000). Resizing original.")
            img_orig = cv2.resize(img_orig, (self.original_w, self.original_h))

        # 1. 重采样为 3456*3456
        img_resampled = cv2.resize(img_orig, (self.resample_w, self.resample_h), interpolation=cv2.INTER_LINEAR)

        # 2. 预测置信度 (输入重采样后的图)
        conf_map = self.conf_predictor.pred(img_resampled)
        
        # 确保置信度图尺寸正确
        if conf_map.shape[:2] != (self.resample_h, self.resample_w):
             conf_map = cv2.resize(conf_map, (self.resample_w, self.resample_h), interpolation=cv2.INTER_NEAREST)
        
        if img_orig.ndim == 2:
            img_orig = img_orig[:, :, None]
            img_resampled = img_resampled[:, :, None]

        return img_orig, img_resampled, conf_map

    def match_blocks(self, img0: np.ndarray, img1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对两张 3456*3456 的图像进行分块 CasP 匹配。
        """
        all_pts0 = []
        all_pts1 = []

        # 遍历 3x3 网格
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                # 计算块的边界
                y_start = r * self.block_h
                y_end = y_start + self.block_h
                x_start = c * self.block_w
                x_end = x_start + self.block_w

                # 裁剪块
                crop0 = img0[y_start:y_end, x_start:x_end]
                crop1 = img1[y_start:y_end, x_start:x_end]

                # 转换为 Tensor 并归一化 (CasP 期望 0-1 的 float tensor, shape [1, C, H, W])
                tensor0 = torch.from_numpy(crop0.transpose(2, 0, 1)).float()[None].to(self.device) / 255.0
                tensor1 = torch.from_numpy(crop1.transpose(2, 0, 1)).float()[None].to(self.device) / 255.0

                data = {'image0': tensor0, 'image1': tensor1}

                # CasP 推理
                with torch.no_grad():
                    results = self.matcher(data)

                # 获取匹配点 (局部坐标)
                pts0_local = results['points0'].cpu().numpy()
                pts1_local = results['points1'].cpu().numpy()

                if len(pts0_local) > 0:
                    # 将局部坐标转换为 3456 尺度下的全局坐标
                    # points 格式为 [x, y]
                    offset = np.array([x_start, y_start])
                    pts0_global = pts0_local + offset
                    pts1_global = pts1_local + offset

                    all_pts0.append(pts0_global)
                    all_pts1.append(pts1_global)

        if not all_pts0:
            return np.array([]), np.array([])

        return np.concatenate(all_pts0, axis=0), np.concatenate(all_pts1, axis=0)

    def process_registration(self, imgs: List[np.ndarray], output_dir: str, key: str):
        """执行完整的 N 张图像配准流程"""
        if len(imgs) < 2:
            print("Need at least 2 images to run registration.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # ---------------------------
        # Step 1: 处理基准图像 (img_0)
        # ---------------------------
        ref_img = imgs[0]
        print(f"Processing Reference Image (index 0)")
        ref_orig, ref_resampled, ref_conf = self.preprocess_image(ref_img)
        
        # 保存基准图像作为参考
        # cv2.imwrite(os.path.join(output_dir, "registered_0.jpg"), cv2.cvtColor(ref_orig, cv2.COLOR_RGB2BGR))

        warped_imgs = []

        # ---------------------------
        # Step 2: 遍历 img_1 到 img_N
        # ---------------------------

        check_idx = np.random.randint(1,len(imgs))

        for i in range(1, len(imgs)):
            tgt_img = imgs[i]
            print(f"\nRegistering Image {i}/{len(imgs)-1}...")

            # 1. 预处理
            tgt_orig, tgt_resampled, tgt_conf = self.preprocess_image(tgt_img)

            # 2. 分块 CasP 匹配 (在 3456 尺度下)
            print("  > Running 3x3 block matching...")
            pts0_3456, pts1_3456 = self.match_blocks(ref_resampled, tgt_resampled)
            print(f"  > Found {len(pts0_3456)} raw matches.")

            if len(pts0_3456) == 0:
                print("  [Error] No matches found. Skipping.")
                continue

            # 3. 根据置信度筛选 (阈值 > 0.5)
            # pts 是 [x, y], conf_map 是 [h, w] -> [y, x]
            valid_mask = []
            h_conf, w_conf = ref_conf.shape
            
            for k in range(len(pts0_3456)):
                # 坐标取整并边界限制
                x0, y0 = np.clip(pts0_3456[k], [0, 0], [w_conf-1, h_conf-1]).astype(int)
                x1, y1 = np.clip(pts1_3456[k], [0, 0], [w_conf-1, h_conf-1]).astype(int)
                
                # 检查两个对应点的置信度是否都 > 0.5
                if ref_conf[y0, x0] > self.args.conf_threshold and tgt_conf[y1, x1] > self.args.conf_threshold:
                    valid_mask.append(True)
                else:
                    valid_mask.append(False)
            
            pts0_filtered = pts0_3456[valid_mask]
            pts1_filtered = pts1_3456[valid_mask]
            print(f"  > {len(pts0_filtered)} matches remaining after confidence filtering.")

            if len(pts0_filtered) < 4:
                print("  [Error] Not enough matches to estimate transformation. Skipping.")
                continue

            # 4. 坐标缩放回 3000*3000
            scale_x = self.original_w / self.resample_w  # 3000 / 3456
            scale_y = self.original_h / self.resample_h  # 3000 / 3456
            
            pts0_3000 = pts0_filtered * np.array([scale_x, scale_y])
            pts1_3000 = pts1_filtered * np.array([scale_x, scale_y])

            # 5. 估计变换矩阵 (Homography) 并变换
            # 我们需要将 img_i (pts1) 变换到 img_0 (pts0)
            if self.transform_type == 'H':
                # 单应变换 (3x3 Matrix)
                H, mask = cv2.findHomography(pts1_3000, pts0_3000, cv2.RANSAC, ransacReprojThreshold=self.args.ransac_threshold)
                if H is not None:
                    warped_img = cv2.warpPerspective(tgt_orig, H, (self.original_w, self.original_h))
                    
            elif self.transform_type == 'A':
                # 仿射变换 (2x3 Matrix)
                H, mask = cv2.estimateAffine2D(pts1_3000, pts0_3000, method=cv2.RANSAC, ransacReprojThreshold=self.args.ransac_threshold)
                if H is not None:
                    warped_img = cv2.warpAffine(tgt_orig, H, (self.original_w, self.original_h))
                
            warped_imgs.append(warped_img)

            # # ---------------------------
            # # [新增功能] 1. 保存变换后的图像
            # # ---------------------------
            # out_filename = f"registered_{i}.jpg"
            # out_path = os.path.join(output_dir, out_filename)
            # cv2.imwrite(out_path, warped_img)
            # print(f"  > Saved registered image: {out_path}")

            # # ---------------------------
            # # [新增功能] 2. 保存变换矩阵 H
            # # ---------------------------
            # matrix_filename = f"transform_matrix_{i}.txt"
            # matrix_path = os.path.join(output_dir, matrix_filename)
            # # 使用 fmt='%.8f' 保证精度
            # np.savetxt(matrix_path, H, fmt='%.3e', header=f"Homography Matrix for Image {i} -> Ref Image", footer="")
            # print(f"  > Saved transformation matrix: {matrix_path}")

            # ---------------------------
            # [新增功能] 3. 生成并保存棋盘格对比图
            # ---------------------------
            if i == check_idx:
                ref = ref_orig[:,:,0]            
                checkerboard_img = make_checkerboard(ref, warped_img, 15)
                checkerboard_filename = f"{key}_{i}.jpg"
                checkerboard_path = os.path.join(output_dir, checkerboard_filename)
                cv2.imwrite(checkerboard_path, checkerboard_img)
            # print(f"  > Saved checkerboard overlay: {checkerboard_path}")

        return warped_imgs
    
# ==========================================
# 3. 主程序入口
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="CasP Image Registration Pipeline")
    parser.add_argument("--dataset_path", type=str, required=True)
    # parser.add_argument("--key", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="tmp/output_registration", help="结果输出路径")
    parser.add_argument("--config_path", type=str, default="preprocess/CasP/configs/model/net/casp.yaml", help="CasP 配置文件路径")
    parser.add_argument("--casp_weights", type=str, default="preprocess/CasP/weights/casp_outdoor.pth", help="CasP 权重文件路径")
    parser.add_argument("--dino_weights", type=str, default="weights")
    parser.add_argument("--conf_head_weights", type=str, default="weights")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--transform_type", type=str, default='H', choices=['H', 'A'],
                        help="Transformation model to use: 'homography' (3x3) or 'affine' (2x3)")
    parser.add_argument("--conf_threshold",type=float,default=0.5)
    parser.add_argument("--ransac_threshold",type=int,default=1)

    args = parser.parse_args()

    database = h5py.File(args.dataset_path, 'r+')
    # key = args.key
    # if key is None:
    #     key = str(np.random.randint(0, len(database.keys())))
    
    args.output_path = os.path.join(args.output_path,f"{args.transform_type}_{args.conf_threshold}_{args.ransac_threshold}")
    
    for key in tqdm(database.keys()):
        img_num = len(database[key]['images'])
        imgs = [database[key]['images'][f"{i}"][:] for i in range(img_num)]

        # 初始化配准器并运行
        registrar = ImageRegistrar(args)
        warped_imgs = registrar.process_registration(imgs, args.output_path, key)

        for i in range(1,img_num):
            database[key]['images'][str(i)][:] = warped_imgs[i]

if __name__ == "__main__":
    main()