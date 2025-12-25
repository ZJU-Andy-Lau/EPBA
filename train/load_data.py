import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset,DataLoader,Sampler
import os
import cv2
from tqdm import tqdm,trange
from shared.utils import get_coord_mat,bilinear_interpolate
from torchvision import transforms
import random
import math
import time
import h5py
import json
import torch.distributed as dist
from typing import List, Tuple

   
def residual_average(arr:np.ndarray, a:int) -> np.ndarray:
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., np.newaxis]
    H, W, C = arr.shape
    new_H = ((H + a - 1) // a) * a
    new_W = ((W + a - 1) // a) * a
    padded = np.pad(arr, ((0, new_H - H), (0, new_W - W), (0, 0)),constant_values=np.nan)
    reshaped = padded.reshape(new_H//a, a, new_W//a, a, C)
    output = np.nanmedian(reshaped,axis=(1,3))
    if is_2d:
        output = output.squeeze(axis=-1)
    output[np.isnan(output)] = -1.
    return output

def downsample(arr,ds):
    if ds <= 0:
        return arr
    H,W = arr.shape[:2]
    lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
    samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
    sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
    arr_ds = bilinear_interpolate(arr,sample_idxs)
    arr_ds = arr_ds.reshape(len(lines),len(samps),-1).squeeze()
    return arr_ds

def xy2rc_mat(M: np.ndarray) -> np.ndarray:
    """
    将基于 (x, y) 坐标系的变换矩阵转换为基于 (row, col) 坐标系的矩阵。
    原理: P_rc = S @ P_xy, 其中 S 是交换前两维的置换矩阵。
    M_rc = S @ M_xy @ S^-1
    这等价于交换 M 的前两行，并交换 M 的前两列。
    
    Args:
        M: shape (..., 3, 3) or (..., 2, 3)
    """
    M_new = M.copy()
    # 交换第0行和第1行
    M_new[..., [0, 1], :] = M_new[..., [1, 0], :]
    # 交换第0列和第1列
    M_new[..., :, [0, 1]] = M_new[..., :, [1, 0]]
    return M_new

def generate_affine_matrices(image_size, size_range, output_size=(512, 512), k = 1):
    """
    生成 k 对具有较大重叠但变换参数独立的单应性裁切矩阵。
    
    参数:
        image_size (tuple): 输入图像的尺寸 (H, W)
        size_range (tuple): 裁切窗口大小范围 (min_size, max_size)
        output_size (tuple): 输出图的大小 (W, H)
        k (int): 需要生成的窗口对数量
        
    返回:
        H_as (np.ndarray): (k, 3, 3) A -> Output (XY Coordinates)
        H_bs (np.ndarray): (k, 3, 3) B -> Output (XY Coordinates)
        M_a_b (np.ndarray): (2, 3) A -> B 全局仿射 (XY Coordinates)
    """
    
    H_img, W_img = image_size
    dst_w, dst_h = output_size
    min_s, max_s = size_range
    
    # ---------------------------------------------------------
    # 1. 生成全局仿射变换 M_a_b (A -> B)
    # ---------------------------------------------------------
    delta = np.random.uniform(-5e-5, 5e-5, size=(2, 2))
    M_linear = np.eye(2) + delta
    
    # 全局平移范围限制，避免两图完全不重叠
    t_limit = min_s / 4.0
    t_vec = np.random.uniform(-t_limit, t_limit, size=(2,))
    
    M_a_b = np.hstack([M_linear, t_vec.reshape(2, 1)])
    
    H_as = []
    H_bs = []
    
    # ---------------------------------------------------------
    # 2. 循环生成 k 对解耦的裁切矩阵
    # ---------------------------------------------------------
    for _ in range(k):
        # --- A. 基准形状生成 ---
        crop_s = np.random.randint(min_s, max_s + 1)
        theta = np.random.uniform(0, 2 * np.pi)
        
        half_s = crop_s / 2.0
        # 顺时针顺序：左上，右上，右下，左下
        corners_base = np.array([
            [-half_s, -half_s],
            [ half_s, -half_s],
            [ half_s,  half_s],
            [-half_s,  half_s]
        ]).T # (2, 4)
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners_rot = R @ corners_base # (2, 4)
        
        # --- B. 独立扰动 (Independent Jitter) ---
        # 允许角点有边长 20% 的独立抖动
        perspective_rho = crop_s * 0.1
        
        # A 的独立形状 (局部坐标)
        noise_A = np.random.uniform(-perspective_rho, perspective_rho, size=(2, 4))
        corners_A_local = corners_rot + noise_A
        
        # B 的独立形状 (局部坐标)
        # 关键点：这里使用独立的随机噪声，不再依赖 A
        noise_B = np.random.uniform(-perspective_rho, perspective_rho, size=(2, 4))
        corners_B_local = corners_rot + noise_B
        
        # 为了计算 B 的边界，我们需要将 B 的局部形状变换到 B 的像素坐标系的比例尺下
        # (即应用 M_linear，但不加平移，平移在最后算)
        corners_B_local_transformed = M_linear @ corners_B_local
        
        # --- C. 计算安全边界 (Robust Margin Calculation) ---
        # 允许中心点有额外的随机偏移 (例如边长的 10%)
        # 这保证了 A 和 B 不是完全同心，而是有错位
        shift_limit = crop_s * 0.1
        
        # 1. A 的安全边界 (仅需包含形状 A)
        max_A_x = np.max(np.abs(corners_A_local[0, :]))
        max_A_y = np.max(np.abs(corners_A_local[1, :]))
        margin_A_x = int(np.ceil(max_A_x))
        margin_A_y = int(np.ceil(max_A_y))
        
        # 2. B 的安全边界 (包含形状 B + 中心偏移余量)
        # 我们必须预留 shift_limit 的空间，这样稍后随机偏移时才不会出界
        max_B_x = np.max(np.abs(corners_B_local_transformed[0, :])) + shift_limit
        max_B_y = np.max(np.abs(corners_B_local_transformed[1, :])) + shift_limit
        margin_B_x = int(np.ceil(max_B_x))
        margin_B_y = int(np.ceil(max_B_y))
        
        # 3. 计算 A 的中心点 (cx, cy) 的有效采样区域
        # 这里的逻辑是：找出 A 的中心点范围，使得：
        #   (1) A 在图 A 内
        #   (2) 对应的 B (经过 M_ab 变换且加上最大 shift 后) 在图 B 内
        
        # A 的自身限制
        valid_A_x_min = margin_A_x
        valid_A_x_max = W_img - margin_A_x
        valid_A_y_min = margin_A_y
        valid_A_y_max = H_img - margin_A_y
        
        # B 的限制投射回 A 的坐标系 (近似)
        # B_center = A_center + t_vec + shift
        # 所以 A_center = B_center - t_vec - shift
        # 这是一个保守估计
        t_x, t_y = t_vec
        
        valid_B_proj_x_min = margin_B_x - t_x
        valid_B_proj_x_max = W_img - margin_B_x - t_x
        valid_B_proj_y_min = margin_B_y - t_y
        valid_B_proj_y_max = H_img - margin_B_y - t_y
        
        final_x_min = int(np.ceil(max(valid_A_x_min, valid_B_proj_x_min)))
        final_x_max = int(np.floor(min(valid_A_x_max, valid_B_proj_x_max)))
        final_y_min = int(np.ceil(max(valid_A_y_min, valid_B_proj_y_min)))
        final_y_max = int(np.floor(min(valid_A_y_max, valid_B_proj_y_max)))
        
        # 检查是否有解 (通常 min_s 只要不是大得离谱，都有解)
        if final_x_max <= final_x_min or final_y_max <= final_y_min:
             # 回退机制：如果实在找不到，就取图像中心，虽然理论上前面参数设置合理不会进这里
             cx = W_img // 2
             cy = H_img // 2
        else:
             cx = np.random.randint(final_x_min, final_x_max)
             cy = np.random.randint(final_y_min, final_y_max)
             
        center_A = np.array([cx, cy])
        
        # --- D. 生成最终坐标 ---
        
        # 1. 计算 A 的全局角点
        pts_A_global = corners_A_local + center_A[:, None] # (2, 4)
        
        # 2. 计算 B 的全局角点
        # 先计算理论上的对应中心 (完美对齐的情况)
        center_B_theoretical = M_a_b @ np.array([cx, cy, 1.0])
        
        # 生成随机中心偏移 (在预留的 shift_limit 范围内)
        shift_x = np.random.uniform(-shift_limit, shift_limit)
        shift_y = np.random.uniform(-shift_limit, shift_limit)
        center_B_actual = center_B_theoretical + np.array([shift_x, shift_y])
        
        # 加上 B 独立的局部形状 (已变换过线性部分)
        pts_B_global = corners_B_local_transformed + center_B_actual[:, None]
        
        # --- E. 构建 Homography ---
        src_pts_A = pts_A_global.T.astype(np.float32)
        src_pts_B = pts_B_global.T.astype(np.float32)
        
        dst_pts = np.array([
            [0, 0],
            [dst_w, 0],
            [dst_w, dst_h],
            [0, dst_h]
        ], dtype=np.float32)
        
        H_a = cv2.getPerspectiveTransform(src_pts_A, dst_pts)
        H_b = cv2.getPerspectiveTransform(src_pts_B, dst_pts)
        
        H_as.append(H_a)
        H_bs.append(H_b)
        
    H_as = np.stack(H_as, axis=0)
    H_bs = np.stack(H_bs, axis=0)
    
    return H_as, H_bs, M_a_b


def process_image(
    img1_full: np.ndarray,
    img2_full: np.ndarray,
    residual1_full: np.ndarray,
    residual2_full: np.ndarray,
    K: int,
    backup_num:int = 5,
    min_crop_side: int = 256,
    output_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes full-size images to generate K pairs of overlapping crops,
    with a chance of applying rotation. This version is efficient and correct.
    """
    H, W, _ = img1_full.shape
    
    imgs1 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)
    imgs2 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)
    residual1 = np.zeros((K, output_size, output_size), dtype=np.float32)
    residual2 = np.zeros((K, output_size, output_size), dtype=np.float32)
    # print(H,W,min_crop_side,int(H * 0.5))

    # 生成 XY 坐标系下的矩阵
    H_as_xy, H_bs_xy, M_a_b_xy = generate_affine_matrices((H,W),(min_crop_side,int(H * 0.5)),(output_size,output_size),K * backup_num)

    # OpenCV 使用 (x, y) 坐标系，所以这里先用 XY 矩阵进行 warp
    img2_full_aff = cv2.warpAffine(img2_full, M_a_b_xy, (W, H), flags=cv2.INTER_LINEAR)
    residual2_full_aff = cv2.warpAffine(residual2_full, M_a_b_xy, (W, H), flags=cv2.INTER_NEAREST, borderValue=np.nan)
    H_as = []
    H_bs = []

    for k in range(K):
        dsize = (output_size, output_size)
        i = 0
        while True:
            residual_test = cv2.warpPerspective(residual2_full_aff, H_bs_xy[i * K + k], dsize, flags=cv2.INTER_NEAREST, borderValue=np.nan)
            residual_test = residual_average(residual_test,16)
            mask_1 = residual_test < 3.
            mask_2 = (residual_test > 5.) | np.isnan(residual_test)
            total_num = len(residual_test.ravel())
            if mask_1.sum() > total_num * 0.2 and mask_2.sum() < total_num * 0.7:
                break
            if i + 1 >= backup_num:
                break
            i += 1
        imgs1[k] = cv2.warpPerspective(img1_full, H_as_xy[i * K + k], dsize, flags=cv2.INTER_LINEAR)
        imgs2[k] = cv2.warpPerspective(img2_full_aff, H_bs_xy[i * K + k], dsize, flags=cv2.INTER_LINEAR)
        residual1[k] = cv2.warpPerspective(residual1_full, H_as_xy[i * K + k], dsize, flags=cv2.INTER_NEAREST, borderValue=np.nan)
        residual2[k] = cv2.warpPerspective(residual2_full_aff, H_bs_xy[i * K + k], dsize, flags=cv2.INTER_NEAREST, borderValue=np.nan)
        H_as.append(H_as_xy[i * K + k])
        H_bs.append(H_bs_xy[i * K + k])
    

    # [关键修改] 将生成的 XY 坐标系矩阵转换为 Row-Col 坐标系矩阵
    # 以供后续 PyTorch 模型训练使用
    H_as_rc = xy2rc_mat(torch.stack(H_as,dim=0))
    H_bs_rc = xy2rc_mat(torch.stack(H_bs,dim=0))
    M_a_b_rc = xy2rc_mat(M_a_b_xy)

    return imgs1, imgs2, residual1, residual2, H_as_rc, H_bs_rc, M_a_b_rc

class TrainDataset(Dataset):
    def __init__(self,root,
                 dataset_idxs = None,
                 batch_size = 1,
                 downsample=16,
                 input_size = 512,
                 norm_coefs = {
                     'mean':(0.485, 0.456, 0.406),
                     'std':(0.229, 0.224, 0.225)
                 },
                 mode='train'):
        super().__init__()
        self.root = root
        if mode == 'train':
            self.database = h5py.File(os.path.join(self.root,'train_data.h5'),'r')
        elif mode == 'test':
            self.database = h5py.File(os.path.join(self.root,'test_data.h5'),'r')
        else:
            raise ValueError("mode should be either train or test")

        if dataset_idxs is None:
            self.dataset_num = len(self.database.keys())
            dataset_idxs = range(self.dataset_num)
        else:
            self.dataset_num = len(dataset_idxs)

        self.database_keys = [list(self.database.keys())[i] for i in dataset_idxs]
        self.DOWNSAMPLE=downsample
        self.img_size = self.database[self.database_keys[0]]['images']['0'][:].shape[0]
        self.input_size = input_size
        self.batch_size = batch_size
        self.red_mids = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # for key in self.database_keys:
        #     img_num = len(self.database[key]['residuals'])
        #     res = np.concatenate([self.database[key]['residuals'][f"residual_{i}"][:].reshape(-1) for i in range(img_num)])
        #     self.red_mids.append(np.nanmedian(res))


        self.distort_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.1)],p=.7),
            transforms.RandomInvert(p=.2),
            transforms.Normalize(norm_coefs['mean'], norm_coefs['std']) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ])
        
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_coefs['mean'], norm_coefs['std']) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ])
        

    def get_train_images(self):
        return [np.stack([self.database[key]['images']['0'][:]] * 3,axis=-1) for key in self.database_keys]
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, index):
        seed = int(index + self.epoch * 10000)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        key = self.database_keys[index]
        img_num = len(self.database[key]['images'])
        idx1,idx2 = np.random.choice(img_num,2)
        if idx1 == idx2:
            idx2 = (idx1 + 1) % img_num

        image_1_full = self.database[key]['images'][f"{idx1}"][:]
        image_2_full = self.database[key]['images'][f"{idx2}"][:]
        residual_1_full = self.database[key]['parallax'][f"{idx1}"][:]
        residual_2_full = self.database[key]['parallax'][f"{idx2}"][:]

        H,W = image_1_full.shape[:2]

        dsus_ratio = np.random.rand()
        if dsus_ratio > 0.5:
            dsus_ratio = (dsus_ratio - 0.5) * 7 + 1.
            img_1_ds = cv2.resize(image_1_full,(int(W / dsus_ratio), int(H / dsus_ratio)),interpolation=cv2.INTER_LINEAR)
            image_1_full = cv2.resize(img_1_ds,(W,H),cv2.INTER_LINEAR)


        image_1_full = np.stack([image_1_full] * 3,axis=-1)
        image_2_full = np.stack([image_2_full] * 3,axis=-1)

        imgs1, imgs2, residual1, residual2, H_as, H_bs, M_a_b = \
            process_image(img1_full=image_1_full,
                          img2_full=image_2_full,
                          residual1_full=residual_1_full,
                          residual2_full=residual_2_full,
                          K=self.batch_size,
                          backup_num=5,
                          min_crop_side=256,
                          output_size=self.input_size,
                          )

        imgs1_train = torch.stack([self.distort_transform(img) for img in imgs1],dim=0)
        imgs2_train = torch.stack([self.distort_transform(img) for img in imgs2],dim=0)
        imgs1_label = torch.stack([self.norm_transform(img) for img in imgs1],dim=0)
        imgs2_label = torch.stack([self.norm_transform(img) for img in imgs2],dim=0)

        residual1 = np.stack([residual_average(residual,self.DOWNSAMPLE) for residual in residual1],axis=0)
        residual2 = np.stack([residual_average(residual,self.DOWNSAMPLE) for residual in residual2],axis=0)
        residual1[np.isnan(residual1)] = -1
        residual2[np.isnan(residual2)] = -1
        residual1 = torch.from_numpy(residual1).unsqueeze(1) # B,1,h,w
        residual2 = torch.from_numpy(residual2).unsqueeze(1)

        H_as, H_bs, M_a_b = torch.from_numpy(H_as), torch.from_numpy(H_bs), torch.from_numpy(M_a_b)
        
        return imgs1_train,imgs2_train,imgs1_label,imgs2_label,residual1,residual2,H_as, H_bs, M_a_b


class ImageSampler(Sampler):
    """
    为所有rank提供相同的大图索引序列。
    确保在每个iteration，所有GPU都在处理同一张大图。
    """
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        indices = list(range(n))
        
        if self.shuffle:
            # 使用epoch作为种子，确保每个epoch的shuffle顺序不同，
            # 但在所有进程中给定epoch的shuffle顺序是相同的。
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
            
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch