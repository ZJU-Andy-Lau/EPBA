import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from typing import List, Dict, Set
from tqdm import tqdm

class GlobalAffineSolver:
    def __init__(self, images: List, device: str = 'cuda', 
                 anchor_indices: List[int] = None, 
                 grid_size: int = 5, diff_eps: float = 1.0,
                 max_iter: int = 10, converge_tol: float = 1e-4):
        """
        基于RPC投影一致性的全局仿射变换求解器 (迭代硬约束版)。
        
        Args:
            images: 包含所有 RSImage 对象的列表。
            device: 输出 Tensor 所在的设备。
            anchor_indices: 基准影像索引列表，M固定为单位阵。
            grid_size: 锚点网格大小。
            diff_eps: 雅可比差分步长。
            max_iter: 最大迭代次数。
            converge_tol: 收敛阈值 (参数变化量的最大绝对值)。
        """
        self.images = images
        self.device = device
        
        if anchor_indices is None:
            self.anchor_indices = [0]
        else:
            self.anchor_indices = anchor_indices
            
        self.anchor_set = set(self.anchor_indices)
        self.grid_size = grid_size
        self.diff_eps = diff_eps
        self.max_iter = max_iter
        self.converge_tol = converge_tol

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
        
        # 边界处理，防止插值越界
        x = np.clip(x, 0, w - 1.001)
        y = np.clip(y, 0, h - 1.001)
        
        x0 = np.floor(x).astype(int)
        x1 = x0 + 1
        y0 = np.floor(y).astype(int)
        y1 = y0 + 1
        
        wx = x - x0
        wy = y - y0
        
        h00 = dem_src[y0, x0]
        h10 = dem_src[y0, x1]
        h01 = dem_src[y1, x0]
        h11 = dem_src[y1, x1]
        heights = (1 - wy) * ((1 - wx) * h00 + wx * h10) + wy * ((1 - wx) * h01 + wx * h11)
        
        lats, lons = rpc_src.RPC_PHOTO2OBJ(x, y, heights, 'numpy')
        samps_dst, lines_dst = rpc_dst.RPC_OBJ2PHOTO(lats, lons, heights, 'numpy')
        
        return np.stack([samps_dst, lines_dst], axis=-1)

    def _compute_jacobian(self, rpc_src, rpc_dst, pts_uv: np.ndarray, dem_src: np.ndarray) -> np.ndarray:
        """
        计算雅可比矩阵 J = d(Project(p)) / dp
        """
        eps = self.diff_eps
        
        # 构造扰动点
        pts_x_plus = pts_uv + np.array([eps, 0])
        pts_y_plus = pts_uv + np.array([0, eps])
        
        p_center = self._project_batch(rpc_src, rpc_dst, pts_uv, dem_src)
        p_x_plus = self._project_batch(rpc_src, rpc_dst, pts_x_plus, dem_src)
        p_y_plus = self._project_batch(rpc_src, rpc_dst, pts_y_plus, dem_src)
        
        dp_dx = (p_x_plus - p_center) / eps
        dp_dy = (p_y_plus - p_center) / eps
        
        J = np.stack([dp_dx, dp_dy], axis=-1) # (N, 2, 2)
        return J

    def _apply_affine_np(self, M: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        应用仿射变换 M (2x3) 到点 pts (Nx2)
        """
        N = pts.shape[0]
        pts_homo = np.concatenate([pts, np.ones((N, 1))], axis=1)
        return (M @ pts_homo.T).T

    def solve(self, pair_results: List[Dict]) -> torch.Tensor:
        """
        迭代求解全局仿射变换。
        """
        print(f"Constructing Iterative Global RPC Solver for {len(self.images)} images...")
        num_images = len(self.images)
        
        # 1. 建立变量索引映射
        id_to_var_idx = {}
        curr_var_idx = 0
        for i in range(num_images):
            if i in self.anchor_set:
                id_to_var_idx[i] = None
            else:
                id_to_var_idx[i] = curr_var_idx
                curr_var_idx += 6
        num_vars = curr_var_idx
        
        # 2. 初始化所有 Ms 为单位阵
        Ms_curr = np.zeros((num_images, 2, 3), dtype=np.float64)
        for i in range(num_images):
            Ms_curr[i] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            
        # 3. 迭代循环
        for iter_k in range(self.max_iter):
            print(f"--- Iteration {iter_k + 1}/{self.max_iter} ---")
            
            row_list = []
            col_list = []
            val_list = []
            rhs_list = []
            curr_row = 0
            
            for pair in tqdm(pair_results, desc="Building Equations"):
                ids = list(pair.keys())
                id_i, id_j = ids[0], ids[1]
                
                if id_i in self.anchor_set and id_j in self.anchor_set:
                    continue

                M_ij_net = pair[id_i].detach().cpu().numpy().astype(np.float64)
                img_i = self.images[id_i]
                img_j = self.images[id_j]
                
                # 原始锚点 (Grid on I)
                P_i_raw = self._get_anchors(img_i.H, img_i.W)
                K = P_i_raw.shape[0]
                
                # 计算 "理想目标点" \hat{P}_j (这是常数，不随迭代变化)
                # Target = RPC_ij( Net_Predict(P_i_raw) )
                P_i_net_prime = self._apply_affine_np(M_ij_net, P_i_raw)
                P_j_hat = self._project_batch(img_i.rpc, img_j.rpc, P_i_net_prime, img_i.dem)
                
                # --- 迭代更新部分 ---
                # 计算当前线性化点 P_curr = M_i^{(k)} * P_i_raw
                # 注意：如果 i 是 Anchor，Ms_curr[i] 始终为 I，所以 P_curr = P_i_raw
                P_i_curr = self._apply_affine_np(Ms_curr[id_i], P_i_raw)
                
                # 在当前位置计算雅可比 J 和 投影值
                J = self._compute_jacobian(img_i.rpc, img_j.rpc, P_i_curr, img_i.dem)
                P_j_proj_curr = self._project_batch(img_i.rpc, img_j.rpc, P_i_curr, img_i.dem)
                
                # 构建方程: M_j * P_j_hat - J * M_i * P_i_raw = RHS
                # RHS = P_j_proj_curr - J * P_i_curr
                
                J_Pi_curr = (J @ P_i_curr[..., None]).squeeze(-1)
                RHS_base = P_j_proj_curr - J_Pi_curr
                
                for k in range(K):
                    # --- 方程 X ---
                    rhs_val_x = RHS_base[k, 0]
                    
                    # 1. M_j (Target) 系数: P_j_hat
                    x_hat, y_hat = P_j_hat[k]
                    if id_j in self.anchor_set:
                        # M_j = I => 移项 -x_hat
                        rhs_val_x -= x_hat
                    else:
                        base_idx = id_to_var_idx[id_j]
                        row_list.extend([curr_row] * 3)
                        col_list.extend([base_idx, base_idx + 1, base_idx + 2])
                        val_list.extend([x_hat, y_hat, 1.0])
                    
                    # 2. M_i (Source) 系数: -J * P_i_raw
                    # 注意这里乘的是原始 P_i_raw，因为未知数 M_i 是作用在原始坐标上的
                    x_raw, y_raw = P_i_raw[k]
                    j00, j01 = J[k, 0, 0], J[k, 0, 1]
                    
                    if id_i in self.anchor_set:
                        # M_i = I => 移项 + (J * P_i_raw).x
                        rhs_val_x += (j00 * x_raw + j01 * y_raw)
                    else:
                        base_idx = id_to_var_idx[id_i]
                        row_list.extend([curr_row] * 6)
                        col_list.extend([base_idx + c for c in range(6)])
                        val_list.extend([
                            -j00 * x_raw, -j00 * y_raw, -j00,
                            -j01 * x_raw, -j01 * y_raw, -j01
                        ])
                    
                    rhs_list.append(rhs_val_x)
                    curr_row += 1
                    
                    # --- 方程 Y ---
                    rhs_val_y = RHS_base[k, 1]
                    
                    # 1. M_j
                    if id_j in self.anchor_set:
                        rhs_val_y -= y_hat
                    else:
                        base_idx = id_to_var_idx[id_j]
                        row_list.extend([curr_row] * 3)
                        col_list.extend([base_idx + 3, base_idx + 4, base_idx + 5])
                        val_list.extend([x_hat, y_hat, 1.0])
                        
                    # 2. M_i
                    j10, j11 = J[k, 1, 0], J[k, 1, 1]
                    
                    if id_i in self.anchor_set:
                        rhs_val_y += (j10 * x_raw + j11 * y_raw)
                    else:
                        base_idx = id_to_var_idx[id_i]
                        row_list.extend([curr_row] * 6)
                        col_list.extend([base_idx + c for c in range(6)])
                        val_list.extend([
                            -j10 * x_raw, -j10 * y_raw, -j10,
                            -j11 * x_raw, -j11 * y_raw, -j11
                        ])
                        
                    rhs_list.append(rhs_val_y)
                    curr_row += 1

            # 求解
            if num_vars > 0:
                A = sp.coo_matrix((val_list, (row_list, col_list)), shape=(curr_row, num_vars))
                b = np.array(rhs_list)
                
                # 使用 lsqr 求解
                result = lsqr(A, b, show=False)
                x = result[0]
            else:
                x = np.array([])
            
            # 构建新一轮的解
            Ms_next = np.zeros_like(Ms_curr)
            for i in range(num_images):
                if i in self.anchor_set:
                    Ms_next[i] = np.eye(2, 3) # Anchor 保持为 I
                else:
                    var_idx = id_to_var_idx[i]
                    Ms_next[i] = x[var_idx : var_idx+6].reshape(2, 3)
            
            # 收敛检查
            diff = np.max(np.abs(Ms_next - Ms_curr))
            print(f"Max parameter update: {diff:.6f}")
            
            Ms_curr = Ms_next
            
            if diff < self.converge_tol:
                print(f"Converged at iteration {iter_k + 1}")
                break
        
        Ms = torch.from_numpy(Ms_curr).to(device=self.device, dtype=torch.float32)
        print("Global affine solving finished.")
        return Ms