import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from typing import List, Dict
from tqdm import tqdm

class GlobalAffineSolver:
    def __init__(self, images: List, device: str = 'cuda', 
                 anchor_indices: List[int] = None, 
                 grid_size: int = 5, diff_eps: float = 1.0, lambda_anchor: float = 1e5):
        """
        基于RPC投影一致性的全局仿射变换求解器 (支持多Anchor)。
        
        Args:
            images: 包含所有 RSImage 对象的列表，用于访问 RPC 和 DEM。
            device: 输出 Tensor 所在的设备。
            anchor_indices: 基准影像索引列表。列表中的所有影像其仿射变换将被强约束为单位阵。
                            如果为 None，默认使用 [0]。
            grid_size: 在每张影像上划分的锚点网格大小 (grid_size x grid_size)。
            diff_eps: 计算雅可比矩阵时的差分步长（像素单位）。
            lambda_anchor: 基准影像约束的权重。
        """
        self.images = images
        self.device = device
        
        # 处理多 Anchor 逻辑
        if anchor_indices is None:
            self.anchor_indices = [0]
        else:
            self.anchor_indices = anchor_indices
            
        self.grid_size = grid_size
        self.diff_eps = diff_eps
        self.lambda_anchor = lambda_anchor

    def _get_anchors(self, H: int, W: int) -> np.ndarray:
        """生成均匀分布的像素坐标锚点 (N_anchors, 2) [x(samp), y(line)]"""
        x = np.linspace(0, W - 1, self.grid_size)
        y = np.linspace(0, H - 1, self.grid_size)
        xx, yy = np.meshgrid(x, y)
        anchors = np.stack([xx.flatten(), yy.flatten()], axis=-1)
        return anchors

    def _project_batch(self, rpc_src, rpc_dst, pts_uv: np.ndarray, dem_src: np.ndarray) -> np.ndarray:
        """
        批量投影：Src Image -> Object Space -> Dst Image
        Args:
            pts_uv: (N, 2) 像素坐标 [samp, line]
            dem_src: 源影像的DEM数据 numpy array
        Returns:
            pts_uv_dst: (N, 2) 目标影像上的像素坐标 [samp, line]
        """
        # 1. 插值获取高程 (Bilinear Interpolation)
        # RSImage.dem 是 (H, W) 格式，pts_uv 是 (x/samp, y/line)
        h, w = dem_src.shape
        x = pts_uv[:, 0]
        y = pts_uv[:, 1]
        
        # 简单的边界处理
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        
        wx = x - x0
        wy = y - y0
        
        # 获取四个点的高程
        h00 = dem_src[y0, x0]
        h10 = dem_src[y0, x1]
        h01 = dem_src[y1, x0]
        h11 = dem_src[y1, x1]
        
        # 双线性插值
        heights = (1 - wy) * ((1 - wx) * h00 + wx * h10) + wy * ((1 - wx) * h01 + wx * h11)
        
        # 2. RPC 正反投影
        # RPC库通常接受 (samp, line, h)
        lats, lons = rpc_src.RPC_PHOTO2OBJ(x, y, heights, 'numpy')
        samps_dst, lines_dst = rpc_dst.RPC_OBJ2PHOTO(lats, lons, heights, 'numpy')
        
        return np.stack([samps_dst, lines_dst], axis=-1)

    def _compute_jacobian(self, rpc_src, rpc_dst, pts_uv: np.ndarray, dem_src: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵 J = d(Project(p)) / dp
        Returns:
            J: (N, 2, 2) 矩阵
               [[dx'/dx, dx'/dy],
                [dy'/dx, dy'/dy]]
        """
        N = pts_uv.shape[0]
        eps = self.diff_eps
        
        # 构造扰动点
        pts_x_plus = pts_uv + np.array([eps, 0])
        pts_y_plus = pts_uv + np.array([0, eps])
        
        # 投影
        # 中心点投影
        p_center = self._project_batch(rpc_src, rpc_dst, pts_uv, dem_src)
        p_x_plus = self._project_batch(rpc_src, rpc_dst, pts_x_plus, dem_src)
        p_y_plus = self._project_batch(rpc_src, rpc_dst, pts_y_plus, dem_src)
        
        # 差分
        dp_dx = (p_x_plus - p_center) / eps
        dp_dy = (p_y_plus - p_center) / eps
        
        # 堆叠为 (N, 2, 2)
        # dp_dx = [du'/dx, dv'/dx], dp_dy = [du'/dy, dv'/dy]
        # J = [dp_dx^T, dp_dy^T]^T
        J = np.stack([dp_dx, dp_dy], axis=-1) 
        
        return J

    def _apply_affine_np(self, M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        应用仿射变换 M (2x3) 到点 pts (Nx2)
        pts: [x, y]
        """
        # M: [[m0, m1, m2], [m3, m4, m5]]
        # x' = m0*x + m1*y + m2
        N = pts.shape[0]
        pts_homo = np.concatenate([pts, np.ones((N, 1))], axis=1) # (N, 3)
        return (M @ pts_homo.T).T # (N, 2)

    def solve(self, pair_results: List[Dict]) -> torch.Tensor:
        """
        求解全局仿射变换。
        Args:
            pair_results: 列表，每个元素为 {id_a: M_ab, id_b: M_ba}，其中 M 为 torch.Tensor (2,3)
        Returns:
            Ms: (num_images, 2, 3) 优化后的仿射变换矩阵
        """
        print(f"Constructing Global RPC Constraint System for {len(self.images)} images...")
        
        num_images = len(self.images)
        num_params_per_img = 6
        num_vars = num_images * num_params_per_img
        
        row_list = []
        col_list = []
        val_list = []
        rhs_list = []
        
        curr_row = 0
        
        for pair in tqdm(pair_results, desc="Building Equations"):
            ids = list(pair.keys())
            id_i, id_j = ids[0], ids[1]
            
            # 获取网络预测的相对仿射 M_ij (Tensor -> Numpy)
            M_ij = pair[id_i].detach().cpu().numpy().astype(np.float64) # 2x3
            
            # 准备数据
            img_i = self.images[id_i]
            img_j = self.images[id_j]
            
            # 生成锚点 P_i (在影像 i 上)
            P_i = self._get_anchors(img_i.H, img_i.W) # (K, 2)
            
            # 1. 计算 "理想目标点" \hat{P}_j
            # 先对 P_i 应用网络预测的 M_ij
            P_i_prime = self._apply_affine_np(M_ij, P_i)
            # 再通过 RPC 投影到 j
            P_j_hat = self._project_batch(img_i.rpc, img_j.rpc, P_i_prime, img_i.dem)
            
            # 2. 计算 "原始投影点" P_j_raw (无矫正)
            P_j_raw = self._project_batch(img_i.rpc, img_j.rpc, P_i, img_i.dem)
            
            # 3. 计算雅可比 J_ij at P_i
            J = self._compute_jacobian(img_i.rpc, img_j.rpc, P_i, img_i.dem) # (K, 2, 2)
            
            # 4. 构建方程
            # 线性化方程: M_j * P_j_hat - J * M_i * P_i = P_j_raw - J * P_i
            
            K = P_i.shape[0]
            
            # 计算 RHS 常数项: (K, 2)
            # J @ P_i needs careful shape: (K,2,2) @ (K,2,1) -> (K,2,1)
            J_Pi = (J @ P_i[..., None]).squeeze(-1)
            RHS = P_j_raw - J_Pi # (K, 2)
            
            for k in range(K):
                # 对每个锚点，有两个方程 (x方向, y方向)
                
                # --- 方程 X ---
                # 系数 M_j (Target): [x_hat, y_hat, 1, 0, 0, 0]
                x_hat, y_hat = P_j_hat[k]
                base_idx_j = id_j * 6
                
                row_list.extend([curr_row] * 3)
                col_list.extend([base_idx_j, base_idx_j + 1, base_idx_j + 2])
                val_list.extend([x_hat, y_hat, 1.0])
                
                # 系数 M_i (Source): - [J00*x, J00*y, J00, J01*x, J01*y, J01]
                # P_i' = M_i * P_i = [m0x+m1y+m2, m3x+m4y+m5]
                # J * P_i' 的 x 分量 = J00 * (m0x+...) + J01 * (m3x+...)
                x, y = P_i[k]
                j00, j01 = J[k, 0, 0], J[k, 0, 1]
                base_idx_i = id_i * 6
                
                row_list.extend([curr_row] * 6)
                col_list.extend([base_idx_i + c for c in range(6)])
                # 注意负号 (移项后)
                val_list.extend([
                    -j00 * x, -j00 * y, -j00,  # 对应 m0, m1, m2
                    -j01 * x, -j01 * y, -j01   # 对应 m3, m4, m5
                ])
                
                rhs_list.append(RHS[k, 0])
                curr_row += 1
                
                # --- 方程 Y ---
                # 系数 M_j (Target): [0, 0, 0, x_hat, y_hat, 1]
                row_list.extend([curr_row] * 3)
                col_list.extend([base_idx_j + 3, base_idx_j + 4, base_idx_j + 5])
                val_list.extend([x_hat, y_hat, 1.0])
                
                # 系数 M_i (Source): - [J10*x, J10*y, J10, J11*x, J11*y, J11]
                # J * P_i' 的 y 分量 = J10 * (m0x+...) + J11 * (m3x+...)
                j10, j11 = J[k, 1, 0], J[k, 1, 1]
                
                row_list.extend([curr_row] * 6)
                col_list.extend([base_idx_i + c for c in range(6)])
                val_list.extend([
                    -j10 * x, -j10 * y, -j10, 
                    -j11 * x, -j11 * y, -j11
                ])
                
                rhs_list.append(RHS[k, 1])
                curr_row += 1

        # 5. 添加 Anchor 约束 (支持多张 Anchor)
        # m0=1, m1=0, m2=0, m3=0, m4=1, m5=0
        target_vals = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        for anchor_idx in self.anchor_indices:
            # 安全检查
            if anchor_idx < 0 or anchor_idx >= num_images:
                print(f"Warning: Anchor index {anchor_idx} is out of bounds (0-{num_images-1}), skipping.")
                continue
                
            base_idx_anchor = anchor_idx * 6
            for i in range(6):
                row_list.append(curr_row)
                col_list.append(base_idx_anchor + i)
                val_list.append(self.lambda_anchor) # 强约束权重
                rhs_list.append(target_vals[i] * self.lambda_anchor)
                curr_row += 1
            
        # 6. 构建稀疏矩阵
        A = sp.coo_matrix((val_list, (row_list, col_list)), shape=(curr_row, num_vars))
        b = np.array(rhs_list)
        
        print(f"Solving sparse system shape: {A.shape}...")
        
        # 7. 求解
        result = lsqr(A, b, show=False)
        x = result[0]
        
        # 8. 格式化输出
        Ms_np = x.reshape(num_images, 2, 3)
        
        # 转换为 Tensor
        Ms = torch.from_numpy(Ms_np).to(device=self.device, dtype=torch.float32)
        
        print("Global affine solving finished.")
        return Ms