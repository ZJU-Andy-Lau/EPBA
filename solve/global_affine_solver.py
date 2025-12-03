# global_affine_solver.py
import torch
from typing import Optional, Sequence, Tuple


class GlobalAffineSolver:
    """
    GlobalAffineSolver: 在图结构上进行全局仿射平差的求解器。
    """

    def __init__(
        self,
        num_nodes: int,
        lambda_anchor: float = 1e8,
        lambda_A: float = 0.0,
        lambda_t: float = 0.0,
        sigma_A: float = 1e-4,
        sigma_t: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            num_nodes: 节点数量 N。
            lambda_anchor: anchor 约束的权重（越大表示越“硬”的锚点）。
            lambda_A: 对 A 部分（线性 4 个参数）的先验正则权重。
            lambda_t: 对 t 部分（平移 2 个参数）的先验正则权重。
            sigma_A: 边观测中线性部分噪声的标准差（用于构造权重）。
            sigma_t: 边观测中平移部分噪声的标准差（用于构造权重）。
            device: PyTorch 的设备（CPU/GPU）。
            dtype: PyTorch 的数据类型（默认为 float32）。
        """
        self.num_nodes = num_nodes
        self.lambda_anchor = float(lambda_anchor)
        self.lambda_A = float(lambda_A)
        self.lambda_t = float(lambda_t)
        self.sigma_A = float(sigma_A)
        self.sigma_t = float(sigma_t)
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype

    # ---------- 工具函数：仿射矩阵和参数向量的互相转换 ----------

    @staticmethod
    def _flatten_affine(M: torch.Tensor) -> torch.Tensor:
        """
        将 (N, 3, 3) 或 (N, 2, 3) 的仿射矩阵转换为 (N, 6) 参数向量：
            [a11, a12, a21, a22, tx, ty]
        """
        if M.ndim != 3:
            raise ValueError("M must have shape (N, 3, 3) or (N, 2, 3)")
        if M.shape[1:] == (3, 3):
            A = M[:, :2, :2]
            t = M[:, :2, 2]
        elif M.shape[1:] == (2, 3):
            A = M[:, :, :2]
            t = M[:, :, 2]
        else:
            raise ValueError("M must have shape (N, 3, 3) or (N, 2, 3)")

        a11 = A[:, 0, 0]
        a12 = A[:, 0, 1]
        a21 = A[:, 1, 0]
        a22 = A[:, 1, 1]
        tx = t[:, 0]
        ty = t[:, 1]
        return torch.stack([a11, a12, a21, a22, tx, ty], dim=-1)

    @staticmethod
    def _unflatten_affine(x: torch.Tensor) -> torch.Tensor:
        """
        将 (N, 6) 参数向量 [a11,a12,a21,a22,tx,ty] 转换为 (N, 3, 3) 仿射矩阵。
        """
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError("x must have shape (N, 6)")
        a11, a12, a21, a22, tx, ty = x.unbind(dim=-1)

        A = torch.stack(
            [
                torch.stack([a11, a12], dim=-1),
                torch.stack([a21, a22], dim=-1),
            ],
            dim=1,
        )
        t = torch.stack([tx, ty], dim=-1).unsqueeze(-1)

        N = x.shape[0]
        M = torch.zeros((N, 3, 3), dtype=x.dtype, device=x.device)
        M[:, :2, :2] = A
        M[:, :2, 2:] = t
        M[:, 2, 2] = 1.0
        return M

    # ---------- 工具函数：构建单条边的 Ci, Cj ----------

    @staticmethod
    def _build_C_matrices(A_ij: torch.Tensor, t_ij: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对一条边 (i -> j)，给定观测的 A_ij (2x2), t_ij (2,)
        构建 6x6 的 Ci, Cj，使得

            A_i - A_j A_ij = 0
            t_i - (A_j t_ij + t_j) = 0

        可以写成：
            r = Ci x_i + Cj x_j,  r ∈ R^6

        x_i = [a11,a12,a21,a22,tx,ty]^T。
        """
        if A_ij.shape != (2, 2):
            raise ValueError("A_ij must have shape (2,2)")
        if t_ij.shape != (2,):
            raise ValueError("t_ij must have shape (2,)")

        r11 = A_ij[0, 0]
        r12 = A_ij[0, 1]
        r21 = A_ij[1, 0]
        r22 = A_ij[1, 1]
        ux = t_ij[0]
        uy = t_ij[1]

        dtype = A_ij.dtype
        device = A_ij.device

        Ci = torch.eye(6, dtype=dtype, device=device)
        Cj = torch.zeros((6, 6), dtype=dtype, device=device)

        # 线性部分：
        # Row 0: a_i11 - (a_j11*r11 + a_j12*r21)
        Cj[0, 0] = -r11
        Cj[0, 1] = -r21

        # Row 1: a_i12 - (a_j11*r12 + a_j12*r22)
        Cj[1, 0] = -r12
        Cj[1, 1] = -r22

        # Row 2: a_i21 - (a_j21*r11 + a_j22*r21)
        Cj[2, 2] = -r11
        Cj[2, 3] = -r21

        # Row 3: a_i22 - (a_j21*r12 + a_j22*r22)
        Cj[3, 2] = -r12
        Cj[3, 3] = -r22

        # 平移部分：
        # Row 4: t_ix - (a_j11*ux + a_j12*uy + t_jx)
        Cj[4, 0] = -ux
        Cj[4, 1] = -uy
        Cj[4, 4] = -1.0

        # Row 5: t_iy - (a_j21*ux + a_j22*uy + t_jy)
        Cj[5, 2] = -ux
        Cj[5, 3] = -uy
        Cj[5, 5] = -1.0

        return Ci, Cj

    # ---------- 主求解函数 ----------

    def solve(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        A_ij: torch.Tensor,
        t_ij: torch.Tensor,
        w_ij: Optional[torch.Tensor] = None,
        anchor_indices: Optional[Sequence[int]] = None,
        anchor_M: Optional[torch.Tensor] = None,
        prior_M: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        求解每个节点的最优仿射矩阵 M_i^*。

        Args:
            edge_src: (E,) long，边的起点节点索引 i。
            edge_dst: (E,) long，边的终点节点索引 j。
            A_ij: (E,2,2)，每条边观测仿射的线性部分。
            t_ij: (E,2)，每条边观测仿射的平移部分。
            w_ij: (E,)，每条边的权重（可选，默认全为 1）。
            anchor_indices: 锚点节点列表（可选），用于固定部分节点。
            anchor_M: (A,3,3) 或 (A,2,3)，锚点节点的目标仿射矩阵。
            prior_M: (N,3,3) 或 (N,2,3)，每个节点的先验仿射（用于 A/t 正则）。

        Returns:
            M_star: (N,3,3)，每个节点的最优仿射估计。
        """
        device = self.device
        dtype = self.dtype

        E = edge_src.shape[0]
        if edge_dst.shape[0] != E or A_ij.shape[0] != E or t_ij.shape[0] != E:
            raise ValueError("Edge arrays must have the same length E")
        if A_ij.shape[1:] != (2, 2) or t_ij.shape[1:] != (2,):
            raise ValueError("A_ij must be (E,2,2), t_ij must be (E,2)")

        if w_ij is None:
            w_ij = torch.ones(E, dtype=dtype, device=device)
        else:
            w_ij = w_ij.to(device=device, dtype=dtype)

        edge_src = edge_src.to(device=device, dtype=torch.long)
        edge_dst = edge_dst.to(device=device, dtype=torch.long)
        A_ij = A_ij.to(device=device, dtype=dtype)
        t_ij = t_ij.to(device=device, dtype=dtype)

        N = self.num_nodes
        num_unknowns = 6 * N

        H = torch.zeros((num_unknowns, num_unknowns), dtype=dtype, device=device)
        g = torch.zeros((num_unknowns,), dtype=dtype, device=device)

        # 根据 sigma_A, sigma_t 构造残差行的缩放因子
        if self.sigma_A <= 0 or self.sigma_t <= 0:
            raise ValueError("sigma_A and sigma_t must be positive")
        scale_A = 1.0 / self.sigma_A
        scale_t = 1.0 / self.sigma_t

        # ---------- 1) 边约束：A 和 t 使用不同噪声尺度 ----------

        for e in range(E):
            i = int(edge_src[e])
            j = int(edge_dst[e])
            w = float(w_ij[e])

            Ai_j = A_ij[e]
            ti_j = t_ij[e]

            Ci, Cj = self._build_C_matrices(Ai_j, ti_j)

            # 残差行缩放：前 4 行是 A，后 2 行是 t
            row_scales = torch.ones((6,), dtype=dtype, device=device)
            row_scales[0:4] *= scale_A
            row_scales[4:6] *= scale_t

            Ci = row_scales.view(-1, 1) * Ci
            Cj = row_scales.view(-1, 1) * Cj

            i_start = 6 * i
            j_start = 6 * j

            H_ii = w * (Ci.T @ Ci)
            H_jj = w * (Cj.T @ Cj)
            H_ij = w * (Ci.T @ Cj)
            H_ji = w * (Cj.T @ Ci)

            H[i_start:i_start+6, i_start:i_start+6] += H_ii
            H[j_start:j_start+6, j_start:j_start+6] += H_jj
            H[i_start:i_start+6, j_start:j_start+6] += H_ij
            H[j_start:j_start+6, i_start:i_start+6] += H_ji
            # 此处目标为 0，所以 g 不变

        # ---------- 2) Anchor 约束：固定部分节点到指定仿射 ----------

        if anchor_indices is not None and anchor_M is not None:
            anchor_indices = list(anchor_indices)
            anchor_M = anchor_M.to(device=device, dtype=dtype)
            if anchor_M.shape[0] != len(anchor_indices):
                raise ValueError("anchor_M must have same length as anchor_indices")
            anchor_x = self._flatten_affine(anchor_M)
            for idx_a, k in enumerate(anchor_indices):
                if k < 0 or k >= N:
                    raise ValueError(f"Anchor index {k} out of range [0, {N})")
                x_k0 = anchor_x[idx_a]
                k_start = 6 * k
                lam = self.lambda_anchor
                H[k_start:k_start+6, k_start:k_start+6] += lam * torch.eye(6, dtype=dtype, device=device)
                g[k_start:k_start+6] += lam * x_k0

        # ---------- 3) A / t 的先验正则 ----------

        if prior_M is None:
            prior_M = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
        else:
            prior_M = prior_M.to(device=device, dtype=dtype)
        prior_x = self._flatten_affine(prior_M)

        for i in range(N):
            i_start = 6 * i
            x0 = prior_x[i]

            # A 的先验（前 4 个分量）
            if self.lambda_A > 0.0:
                lamA = self.lambda_A
                H[i_start + 0, i_start + 0] += lamA
                H[i_start + 1, i_start + 1] += lamA
                H[i_start + 2, i_start + 2] += lamA
                H[i_start + 3, i_start + 3] += lamA
                g[i_start + 0] += lamA * x0[0]
                g[i_start + 1] += lamA * x0[1]
                g[i_start + 2] += lamA * x0[2]
                g[i_start + 3] += lamA * x0[3]

            # t 的先验（后 2 个分量，可选）
            if self.lambda_t > 0.0:
                lamT = self.lambda_t
                H[i_start + 4, i_start + 4] += lamT
                H[i_start + 5, i_start + 5] += lamT
                g[i_start + 4] += lamT * x0[4]
                g[i_start + 5] += lamT * x0[5]

        # ---------- 4) 解线性方程组 H x = g ----------

        try:
            x = torch.linalg.solve(H, g)
        except RuntimeError:
            # 若数值上接近奇异，则加入微小阻尼
            eps = 1e-8
            H_damped = H + eps * torch.eye(num_unknowns, dtype=dtype, device=device)
            x = torch.linalg.solve(H_damped, g)

        x = x.view(N, 6)
        M_star = self._unflatten_affine(x)
        return M_star
