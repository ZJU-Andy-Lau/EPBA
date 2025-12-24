import argparse
import os
import sys
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def apply_colormap_to_parallax(parallax_map:np.ndarray, cmap_name='RdYlGn_r'):
    """
    将视差图转换为伪彩色热力图
    Args:
        parallax_map: (H, W) float32
        cmap_name: matplotlib colormap name. 
                   'RdYlGn_r' means Green (Low) -> Red (High)
    Returns:
        heatmap: (H, W, 3) uint8 BGR (for OpenCV)
    """
    # 1. 获取有效值范围
    # 使用 nanmin/nanmax，如果全为 nan 则处理
    if np.all(np.isnan(parallax_map)):
        return np.zeros((*parallax_map.shape, 3), dtype=np.uint8)
    
    vmin = parallax_map.min()
    med = np.median(parallax_map)
    vmax = 2 * med - vmin
    
    # 2. 归一化到 [0, 1]
    if vmax - vmin < 1e-6:
        norm_map = np.zeros_like(parallax_map)
    else:
        norm_map = np.clip((parallax_map - vmin) / (vmax - vmin),a_max=1.0,a_min=0.0)
    
    # 3. 应用 Colormap
    cmap = plt.get_cmap(cmap_name)
    # cmap 返回的是 (N, 4) RGBA float [0,1]
    colored_map = cmap(norm_map) 
    
    # 4. 转换为 OpenCV 格式 (0-255, RGB -> BGR)
    # colored_map[..., :3] 取 RGB 通道，忽略 Alpha
    heatmap = (colored_map[..., :3] * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
    return heatmap

def apply_binary_colormap_to_parallax(parallax_map,left,right):
    """
    将视差图转换为二值可视化图
    Args:
        parallax_map: (H, W) float32
    Returns:
        heatmap: (H, W, 3) uint8 BGR (for OpenCV)
    """
    if np.all(np.isnan(parallax_map)):
        return np.zeros((*parallax_map.shape, 3), dtype=np.uint8)
    
    parallax_map[np.isnan(parallax_map)] == np.nanmax(parallax_map)
    parallax_map = np.clip(parallax_map,a_max=2 * right,a_min=0.)

    mid = (left + right) * 0.5
    a = np.log(9) / ((right - left) * 0.5)
    conf_map = 1. / (1. + np.exp(a * (parallax_map - mid)))
    heatmap = np.zeros((*parallax_map.shape, 3), dtype=np.uint8)
    heatmap[:,:,1] = (conf_map * 255.).astype(np.uint8)
    heatmap[:,:,2] = ((1. - conf_map) * 255.).astype(np.uint8)
    
    return heatmap

class ParallaxVisualizer:
    def __init__(self, args):
        self.dataset_path = args.dataset_path
        self.dataset_key = args.dataset_key
        self.output_dir = args.output_dir
        self.k = args.window_size
        self.alpha = args.alpha
        self.left = args.left
        self.right = args.right
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_dataset(self):
        """主处理循环"""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"H5 file not found: {self.dataset_path}")

        with h5py.File(self.dataset_path, 'r') as f:
            if self.dataset_key not in f:
                raise KeyError(f"Dataset key '{self.dataset_key}' not found in H5 file.")
            
            group = f[self.dataset_key]
            if 'images' not in group or 'parallax' not in group:
                raise ValueError(f"Dataset '{self.dataset_key}' missing 'images' or 'parallax' groups.")
            
            img_ids = list(group['images'].keys())
            print(f"Found {len(img_ids)} images in {self.dataset_key}. Starting visualization...")

            for img_id in tqdm(img_ids):
                # 读取原始图像
                img_data = group['images'][img_id][:]
                # 读取视差图
                if img_id not in group['parallax']:
                    print(f"Warning: No parallax map found for {img_id}, skipping.")
                    continue
                parallax_data = group['parallax'][img_id][:]
                
                # 处理单张图像
                vis_result = self.visualize_single_image(img_data, parallax_data)
                
                # 保存结果
                out_name = f"{self.dataset_key.replace('/', '_')}_{img_id}_vis.jpg"
                out_path = os.path.join(self.output_dir, out_name)
                cv2.imwrite(out_path, vis_result)

    def visualize_single_image(self, img, parallax):
        """
        对单张图像进行处理：下采样 -> 填充 -> 上采样 -> 叠加
        """
        H, W = parallax.shape
        K = self.k

        # --- 步骤 1: 填充图像以适应窗口大小 ---
        pad_h = (K - H % K) % K
        pad_w = (K - W % K) % K
        
        # 使用 NaN 填充边界
        parallax_padded = np.pad(parallax, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.nan)
        H_pad, W_pad = parallax_padded.shape
        
        # --- 步骤 2: 下采样 (Reshape & Median) ---
        # 变换形状为 (H_grid, K, W_grid, K)
        # 这样 axis 1 和 3 分别对应窗口内的高度和宽度方向
        reshaped = parallax_padded.reshape(H_pad // K, K, W_pad // K, K)
        
        # 计算中位数，忽略 NaN
        # axis=(1, 3) 聚合每个窗口
        downsampled = np.nanmedian(reshaped, axis=(1, 3))
        
        # --- 步骤 3: 填充 NaN 值 ---
        # 需求：将所有为 nan 的 patch 赋值为下采样视差图中非 nan 的最大值
        if np.all(np.isnan(downsampled)):
            # 如果整张图都是 NaN，给一个默认值 0
            max_val = 0.0
        else:
            max_val = np.nanmax(downsampled)
            
        downsampled = np.where(np.isnan(downsampled), max_val, downsampled)
        
        # --- 步骤 4: 上采样 (Upsampling) ---
        # 恢复到 padding 后的大小
        # 使用 INTER_NEAREST 保持格子状，或者 INTER_LINEAR 平滑
        # 这里使用 INTER_LINEAR 配合后面热力图看起来更自然，如果需要严格的块状显示可改为 INTER_NEAREST
        upsampled_padded = cv2.resize(downsampled, (W_pad, H_pad), interpolation=cv2.INTER_CUBIC)
        
        # 裁剪回原始尺寸
        upsampled = upsampled_padded[:H, :W]
        
        # --- 步骤 5: 生成热力图并叠加 ---
        # heatmap = apply_colormap_to_parallax(upsampled, cmap_name='RdYlGn_r')
        heatmap = apply_binary_colormap_to_parallax(upsampled,self.left,self.right)
        
        # 准备底图 (转为 BGR)
        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 假设输入是 RGB
            
        # 叠加: result = img * (1-alpha) + heatmap * alpha
        overlay = cv2.addWeighted(img_bgr, 1 - self.alpha, heatmap, self.alpha, 0)
        
        return overlay

def main():
    parser = argparse.ArgumentParser(description="Visualize Parallax Maps from HDF5")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument("--dataset_key", type=str, required=True, help="The dataset key to visualize (e.g., 'area1/2020')")
    parser.add_argument("--output_dir", type=str, default="vis_results", help="Directory to save visualization results")
    parser.add_argument("--window_size", type=int, default=50, help="Sliding window size K (stride is also K)")
    parser.add_argument("--alpha", type=float, default=0.4, help="Opacity of the parallax overlay (0.0 to 1.0)")
    parser.add_argument("--left",type=float,default=2.)
    parser.add_argument("--right",type=float,default=8.)
    
    args = parser.parse_args()
    
    visualizer = ParallaxVisualizer(args)
    try:
        visualizer.process_dataset()
        print(f"Visualization complete. Results saved to {args.output_dir}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()