import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from typing import List, Dict, Set
from tqdm import tqdm

class GlobalAffineSolver:
    def __init__(self, images: List, device: str = 'cuda', 
                 anchor_indices: List[int] = None, 
                 grid_size: int = 5, diff_eps: float = 1.0):
        """
        基于RPC投影一致性的全局仿射变换求解器 (硬约束版)。
        
        Args:
            images: 包含所有 RSImage 对象的列表，用于访问 RPC 和 DEM。
            device: 输出 Tensor 所在的设备。
            anchor_indices: 基准影像索引列表。这些影像的 M 将被严格固定为单位阵，不作为未知数求解。
                            如果为 None，默认使用 [0]。
            grid_size: 在每张影像上划分的锚点网格大小 (grid_size x grid_size)。
            diff_eps: 计算雅可比矩阵时的差分步长（像素单位）。
        """
        self.images = images
        self.device = device
        
        # 处理多 Anchor 逻辑
        if anchor_indices is None:
            self.anchor_indices = [0]
        else:
            self.anchor_indices = anchor_indices
            
        # 将 list 转为 set 以便快速查找
        self.anchor_set = set(self.anchor_indices)
            
        self.grid_size = grid_size
        self.diff_eps = diff_eps

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
        """
        h, w = dem_src.shape
        x = pts_uv[:, 0]
        y = pts_uv[:, 1]
        
        # 边界处理
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        
        wx = x - x0
        wy = y - y0
        
        # DEM 双线性插值
        h00 = dem_src[y0, x0]
        h10 = dem_src[y0, x1]
        h01 = dem_src[y1, x0]
        h11 = dem_src[y1, x1]
        heights = (1 - wy) * ((1 - wx) * h00 + wx * h10) + wy * ((1 - wx) * h01 + wx * h11)
        
        # RPC 正反投影
        lats, lons = rpc_src.RPC_PHOTO2OBJ(x, y, heights, 'numpy')
        samps_dst, lines_dst = rpc_dst.RPC_OBJ2PHOTO(lats, lons, heights, 'numpy')
        
        return np.stack([samps_dst, lines_dst], axis=-1)

    def _compute_jacobian(self, rpc_src, rpc_dst, pts_uv: np.ndarray, dem_src: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵 J = d(Project(p)) / dp
        """
        N = pts_uv.shape[0]
        eps = self.diff_eps
        
        # 构造扰动点
        pts_x_plus = pts_uv + np.array([eps, 0])
        pts_y_plus = pts_uv + np.array([0, eps])
        
        # 投影
        p_center = self._project_batch(rpc_src, rpc_dst, pts_uv, dem_src)
        p_x_plus = self._project_batch(rpc_src, rpc_dst, pts_x_plus, dem_src)
        p_y_plus = self._project_batch(rpc_src, rpc_dst, pts_y_plus, dem_src)
        
        # 差分
        dp_dx = (p_x_plus - p_center) / eps
        dp_dy = (p_y_plus - p_center) / eps
        
        # J = [dp_dx^T, dp_dy^T]^T -> (N, 2, 2)
        J = np.stack([dp_dx, dp_dy], axis=-1) 
        
        return J

    def _apply_affine_np(self, M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        应用仿射变换 M (2x3) 到点 pts (Nx2)
        """
        N = pts.shape[0]
        pts_homo = np.concatenate([pts, np.ones((N, 1))], axis=1) # (N, 3)
        return (M @ pts_homo.T).T # (N, 2)

    def solve(self, pair_results: List[Dict]) -> torch.Tensor:
        """
        求解全局仿射变换 (硬约束消元法)。
        """
        print(f"Constructing Global RPC Constraint System (Hard Constraints) for {len(self.images)} images...")
        
        num_images = len(self.images)
        
        # 1. 建立变量索引映射
        # 只有非 Anchor 影像才有对应的未知数 (每张图 6 个参数)
        id_to_var_idx = {}
        curr_var_idx = 0
        
        for i in range(num_images):
            if i in self.anchor_set:
                id_to_var_idx[i] = None # Anchor 没有变量
            else:
                id_to_var_idx[i] = curr_var_idx
                curr_var_idx += 6
        
        num_vars = curr_var_idx
        print(f"Total variables to solve: {num_vars} (Fixed Anchors: {len(self.anchor_set)})")
        
        # 稀疏矩阵构建列表
        row_list = []
        col_list = []
        val_list = []
        rhs_list = []
        
        curr_row = 0
        
        for pair in tqdm(pair_results, desc="Building Equations"):
            ids = list(pair.keys())
            id_i, id_j = ids[0], ids[1]
            
            # 如果两个都是 Anchor，则该边不产生方程，直接跳过 (或者仅用于验证)
            if id_i in self.anchor_set and id_j in self.anchor_set:
                continue

            # 获取网络预测的 M_ij
            M_ij = pair[id_i].detach().cpu().numpy().astype(np.float64) # 2x3
            
            img_i = self.images[id_i]
            img_j = self.images[id_j]
            
            # 生成锚点 P_i
            P_i = self._get_anchors(img_i.H, img_i.W) # (K, 2)
            K = P_i.shape[0]
            
            # 计算中间量
            P_i_prime = self._apply_affine_np(M_ij, P_i)
            P_j_hat = self._project_batch(img_i.rpc, img_j.rpc, P_i_prime, img_i.dem) # Target
            P_j_raw = self._project_batch(img_i.rpc, img_j.rpc, P_i, img_i.dem) # Raw Source Proj
            J = self._compute_jacobian(img_i.rpc, img_j.rpc, P_i, img_i.dem) # (K, 2, 2)
            
            # 基础 RHS = P_j_raw - J * P_i
            J_Pi = (J @ P_i[..., None]).squeeze(-1) # (K, 2)
            RHS_base = P_j_raw - J_Pi # (K, 2)
            
            for k in range(K):
                # 对每个点产生 x, y 两个方程
                
                # --- 方程 X ---
                rhs_val_x = RHS_base[k, 0]
                
                # 1. 处理 M_j (Target) 的系数: [x_hat, y_hat, 1, 0, 0, 0]
                x_hat, y_hat = P_j_hat[k]
                
                if id_j in self.anchor_set:
                    # j 是 Anchor: M_j = I
                    # M_j * P_j_hat = P_j_hat
                    # 将这一项移到 RHS 右边: RHS_new = RHS_base - P_j_hat
                    rhs_val_x -= x_hat
                else:
                    # j 是 Free: 添加变量系数
                    base_idx = id_to_var_idx[id_j]
                    row_list.extend([curr_row] * 3)
                    col_list.extend([base_idx, base_idx + 1, base_idx + 2])
                    val_list.extend([x_hat, y_hat, 1.0])
                
                # 2. 处理 M_i (Source) 的系数: -J * P_i'
                # x分量系数: -[J00*x, J00*y, J00, J01*x, J01*y, J01]
                x, y = P_i[k]
                j00, j01 = J[k, 0, 0], J[k, 0, 1]
                
                if id_i in self.anchor_set:
                    # i 是 Anchor: M_i = I
                    # 项 -J * M_i * P_i 变为 -J * P_i
                    # 将这一项移到 RHS 右边: RHS_new = RHS - (-J * P_i) = RHS + J * P_i
                    # 这一项是 J @ P_i 的 x 分量
                    rhs_val_x += (j00 * x + j01 * y) # 注意：这里实际上加的是 J*(M_i*P_i)，当M_i=I时就是J*P_i
                    # 在计算 RHS_base 时我们减去了 J*P_i，现在又加回来
                    # 数学上: RHS = (P_j_raw - J*P_i) - (-J*P_i) = P_j_raw
                    # 为了代码逻辑一致性，这里保持 "移项" 的操作
                else:
                    # i 是 Free: 添加变量系数
                    base_idx = id_to_var_idx[id_i]
                    row_list.extend([curr_row] * 6)
                    col_list.extend([base_idx + c for c in range(6)])
                    val_list.extend([
                        -j00 * x, -j00 * y, -j00, 
                        -j01 * x, -j01 * y, -j01
                    ])
                
                rhs_list.append(rhs_val_x)
                curr_row += 1
                
                # --- 方程 Y ---
                rhs_val_y = RHS_base[k, 1]
                
                # 1. M_j (Target): [0, 0, 0, x_hat, y_hat, 1]
                if id_j in self.anchor_set:
                    # 移项 - y_hat
                    rhs_val_y -= y_hat
                else:
                    base_idx = id_to_var_idx[id_j]
                    row_list.extend([curr_row] * 3)
                    col_list.extend([base_idx + 3, base_idx + 4, base_idx + 5])
                    val_list.extend([x_hat, y_hat, 1.0])
                
                # 2. M_i (Source): -[J10*x, J10*y, J10, J11*x, J11*y, J11]
                j10, j11 = J[k, 1, 0], J[k, 1, 1]
                
                if id_i in self.anchor_set:
                    # 移项 + (J10*x + J11*y)
                    rhs_val_y += (j10 * x + j11 * y)
                else:
                    base_idx = id_to_var_idx[id_i]
                    row_list.extend([curr_row] * 6)
                    col_list.extend([base_idx + c for c in range(6)])
                    val_list.extend([
                        -j10 * x, -j10 * y, -j10, 
                        -j11 * x, -j11 * y, -j11
                    ])
                
                rhs_list.append(rhs_val_y)
                curr_row += 1

        # 2. 构建与求解稀疏系统
        A = sp.coo_matrix((val_list, (row_list, col_list)), shape=(curr_row, num_vars))
        b = np.array(rhs_list)
        
        print(f"Solving sparse system shape: {A.shape}...")
        
        if num_vars > 0:
            result = lsqr(A, b, show=False)
            x = result[0]
        else:
            x = np.array([])
        
        # 3. 重构结果矩阵 Ms
        Ms_np = np.zeros((num_images, 2, 3), dtype=np.float32)
        
        # 填充单位阵到所有位置作为默认值 (覆盖了 Anchor 的情况)
        for i in range(num_images):
            Ms_np[i] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            
        # 填充求解出的 Free 图像参数
        for i in range(num_images):
            if i not in self.anchor_set:
                var_idx = id_to_var_idx[i]
                # 取出6个参数
                params = x[var_idx : var_idx+6]
                Ms_np[i] = params.reshape(2, 3)
        
        # 转换为 Tensor
        Ms = torch.from_numpy(Ms_np).to(device=self.device, dtype=torch.float32)
        
        print("Global affine solving finished.")
        return Ms