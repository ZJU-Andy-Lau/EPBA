import numpy as np
import torch
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from shared.rpc import RPCModelParameterTorch,project_linesamp
from shared.utils import sample_points_in_overlap
from infer.rs_image import RSImage
from infer.monitor import StatusReporter
from infer.utils import create_grid_img

from copy import deepcopy
import os
import cv2



def affine_apply(A: np.ndarray, pts_line_samp: np.ndarray) -> np.ndarray:
    """
    A: (2,3), pts: (N,2) in (line, samp)
    return: (N,2) in (line, samp)
    """
    pts = np.asarray(pts_line_samp, dtype=np.float64)
    X = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return X @ A.T


def affine_inv(A: np.ndarray) -> np.ndarray:
    """
    Invert 2x3 affine in (line,samp):
      y = R x + t
      x = R^{-1}(y - t)
    """
    R = A[:, :2]
    t = A[:, 2]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    return np.concatenate([R_inv, t_inv.reshape(2, 1)], axis=1)


def affine_apply_torch(A: torch.Tensor, pts_line_samp: torch.Tensor) -> torch.Tensor:
    pts = pts_line_samp.to(dtype=torch.float64)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float64, device=pts.device)
    X = torch.cat([pts, ones], dim=1)
    return X @ A.t()


def affine_inv_torch(A: torch.Tensor) -> torch.Tensor:
    R = A[:, :2]
    t = A[:, 2]
    R_inv = torch.linalg.inv(R)
    t_inv = -R_inv @ t
    return torch.cat([R_inv, t_inv.reshape(2, 1)], dim=1)


def pack_A(A: np.ndarray) -> np.ndarray:
    """(2,3) -> (6,)"""
    return np.asarray(A, dtype=np.float64).reshape(-1)


def unpack_A(p: np.ndarray) -> np.ndarray:
    """(6,) -> (2,3)"""
    return np.asarray(p, dtype=np.float64).reshape(2, 3)


def xy_to_line_samp(pts_samp_line: np.ndarray) -> np.ndarray:
    """
    user pts: (samp,line) => internal: (line,samp)
    """
    pts = np.asarray(pts_samp_line, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be (N,2), got {pts.shape}")
    return pts[:, [1, 0]]


def line_samp_to_xy(pts_line_samp: np.ndarray) -> np.ndarray:
    """
    internal: (line,samp) => rpc input: (samp,line)
    """
    pts = np.asarray(pts_line_samp, dtype=np.float64)
    return pts[:, [1, 0]]


def line_samp_to_xy_torch(pts_line_samp: torch.Tensor) -> torch.Tensor:
    return pts_line_samp[:, [1, 0]]


@dataclass
class PBASolveReport:
    iters: int
    rms_per_iter: List[float]


class PBAAffineSolver:
    def __init__(
        self,
        images:list[RSImage],
        results,
        fixed_id: int,
        sample_points_num: int = 256,
        device: str = "cuda",
        reporter: StatusReporter = None,
        output_path = None
    ):
        self.fixed_id = fixed_id
        self.device = device
        self.reporter = reporter
        if not output_path is None:
            self.output_path = os.path.join(output_path,'pba_solver_output')
            os.makedirs(self.output_path,exist_ok=True)

        self.ties,self.rpcs = self._process_data(images,results)
        self.M = len(self.rpcs)
        if not (0 <= fixed_id < self.M):
            raise ValueError("fixed_id out of range")

        # RPC 放到指定 device；并清空其内部 adjust（本类不使用它）
        for rpc in self.rpcs:
            rpc.to_gpu(device)
            rpc.Clear_Adjust()

        # 初始化 A: obs -> orig
        self.A = np.zeros((self.M, 2, 3), dtype=np.float64)
        self.A[:] = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)
        self.A[self.fixed_id] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0]], dtype=np.float64)

        # 变量影像集合（剔除 fixed）
        self.var_ids = [m for m in range(self.M) if m != self.fixed_id]
        self.id2pos = {m: k for k, m in enumerate(self.var_ids)}  # image -> block index
        self._prepare_tie_tensors()

    def _prepare_tie_tensors(self):
        self.tie_tensors = []
        self.neighbors = {m: set() for m in range(self.M)}
        for d in self.ties:
            i = int(d["i"])
            j = int(d["j"])
            self.neighbors[i].add(j)
            self.neighbors[j].add(i)
            tie = {
                "i": i,
                "j": j,
                "pts_i": torch.tensor(d["pts_i"], dtype=torch.float64, device=self.device),
                "pts_j": torch.tensor(d["pts_j"], dtype=torch.float64, device=self.device),
                "heights": torch.tensor(d["heights"], dtype=torch.float64, device=self.device),
            }
            self.tie_tensors.append(tie)

    def _process_data_old(self,images:list[RSImage],results,sample_points_num = 256):
        ties = []
        rpcs = [image.rpc for image in images]
        for match in results:
            i,j = match.keys()
            image_i:RSImage = images[i]
            image_j:RSImage = images[j]
            rpc_i_adj = deepcopy(image_i.rpc)
            rpc_i_adj.Update_Adjust(match[i])
            rpc_j_adj = deepcopy(image_j.rpc)
            rpc_j_adj.Update_Adjust(match[j])
            sampled_xys = sample_points_in_overlap(image_i.corner_xys,image_j.corner_xys,K=sample_points_num)
            linesamp_i_1 = image_i.xy_to_sampline(sampled_xys)[:,[1,0]]
            heights = image_i.dem[linesamp_i_1[:,0].astype(int),linesamp_i_1[:,1].astype(int)]
            linesamp_j = np.stack(project_linesamp(rpc_i_adj,image_j.rpc,linesamp_i_1[:,0],linesamp_i_1[:,1],heights,output_type='numpy'),axis=-1)
            linesamp_i_2 = np.stack(project_linesamp(rpc_j_adj,image_i.rpc,linesamp_j[:,0],linesamp_j[:,1],heights,output_type='numpy'),axis=-1)
            linesamp_i = (linesamp_i_1 + linesamp_i_2) * 0.5
            self.reporter.log(f"{i}-{j} dis: \t {np.linalg.norm(linesamp_i_1 - linesamp_i_2,axis=-1).mean()}")
            
            tie = {
                'i':i,
                'j':j,
                'pts_i':linesamp_i,
                'pts_j':linesamp_j,
                'heights':heights
            }
            ties.append(tie)
        return ties,rpcs

    def _process_data(self,images:list[RSImage],results,sample_points_num = 256):
        ties = []
        rpcs = [image.rpc for image in images]
        for match in results:
            i,j = int(match['i']),int(match['j'])
            image_i = images[i]
            image_j = images[j]
            rpc_i_adj = deepcopy(image_i.rpc)
            rpc_i_adj.Update_Adjust(match['M'])
            sampled_xys = sample_points_in_overlap(image_i.corner_xys,image_j.corner_xys,K=sample_points_num)
            linesamp_i = image_i.xy_to_sampline(sampled_xys)[:,[1,0]]
            heights = image_i.dem[linesamp_i[:,0].astype(int),linesamp_i[:,1].astype(int)]
            linesamp_j = np.stack(project_linesamp(rpc_i_adj,image_j.rpc,linesamp_i[:,0],linesamp_i[:,1],heights,output_type='numpy'),axis=-1)

            tie = {
                'i':i,
                'j':j,
                'pts_i':linesamp_i,
                'pts_j':linesamp_j,
                'heights':heights
            }
            ties.append(tie)
        return ties,rpcs    

    def _predict_obs_from_i_to_j(
        self,
        i: int,
        j: int,
        obs_i_ls: np.ndarray,  # (N,2) line,samp (observed)
        h: np.ndarray,
        A_i: np.ndarray,
        A_j: np.ndarray,
    ) -> np.ndarray:
        """
        使用 i 的观测点 (obs_i) + A_i 得到 orig_i，
        再通过 rpc_i PHOTO2OBJ 得到 (lat,lon)，
        再通过 rpc_j OBJ2PHOTO 得到 orig_j_pred，
        最后通过 A_j^{-1} 映射回 j 的观测坐标系，得到 obs_j_pred。
        """
        # obs -> orig (image i)
        orig_i = affine_apply(A_i, obs_i_ls)

        # orig_i -> obj (lat, lon)
        samp_i, line_i = line_samp_to_xy(orig_i).T
        lat, lon = self.rpcs[i].RPC_PHOTO2OBJ(samp_i, line_i, h, output_type="numpy")

        # obj -> orig_j_pred
        samp_j0, line_j0 = self.rpcs[j].RPC_OBJ2PHOTO(lat, lon, h, output_type="numpy")
        orig_j_pred = np.stack([np.asarray(line_j0), np.asarray(samp_j0)], axis=1)

        # orig -> obs (image j)
        A_j_inv = affine_inv(A_j)
        obs_j_pred = affine_apply(A_j_inv, orig_j_pred)
        return obs_j_pred

    def _predict_obs_from_i_to_j_torch(
        self,
        i: int,
        j: int,
        obs_i_ls: torch.Tensor,
        h: torch.Tensor,
        A_i: torch.Tensor,
        A_j: torch.Tensor,
    ) -> torch.Tensor:
        orig_i = affine_apply_torch(A_i, obs_i_ls)
        samp_i, line_i = line_samp_to_xy_torch(orig_i).t()
        lat, lon = self.rpcs[i].RPC_PHOTO2OBJ(samp_i, line_i, h, output_type="tensor")
        samp_j0, line_j0 = self.rpcs[j].RPC_OBJ2PHOTO(lat, lon, h, output_type="tensor")
        orig_j_pred = torch.stack([line_j0, samp_j0], dim=1)
        A_j_inv = affine_inv_torch(A_j)
        obs_j_pred = affine_apply_torch(A_j_inv, orig_j_pred)
        return obs_j_pred

    def residual_vector(self, A_all: np.ndarray) -> np.ndarray:
        """
        拼接所有 tie 的双向残差，返回 1D 向量：
          [r_line0, r_samp0, r_line1, r_samp1, ...]
        """
        res_list = []

        for d in self.ties:
            i = int(d["i"]); j = int(d["j"])
            # pts_i_ls = xy_to_line_samp(d["pts_i"])
            # pts_j_ls = xy_to_line_samp(d["pts_j"])
            pts_i_ls = d["pts_i"]
            pts_j_ls = d["pts_j"]
            h = np.asarray(d["heights"], dtype=np.float64).reshape(-1)

            if pts_i_ls.shape[0] != pts_j_ls.shape[0] or pts_i_ls.shape[0] != h.shape[0]:
                raise ValueError("pts_i, pts_j, heights must have same N")

            A_i = A_all[i]
            A_j = A_all[j]

            pred_j = self._predict_obs_from_i_to_j(i, j, pts_i_ls, h, A_i, A_j)
            pred_i = self._predict_obs_from_i_to_j(j, i, pts_j_ls, h, A_j, A_i)

            rj = (pred_j - pts_j_ls).reshape(-1)
            ri = (pred_i - pts_i_ls).reshape(-1)
            res_list.append(rj)
            res_list.append(ri)

        if len(res_list) == 0:
            return np.zeros((0,), dtype=np.float64)
        return np.concatenate(res_list, axis=0)

    def residual_vector_torch(self, A_all: torch.Tensor) -> torch.Tensor:
        res_list = []
        for d in self.tie_tensors:
            i = d["i"]
            j = d["j"]
            pts_i_ls = d["pts_i"]
            pts_j_ls = d["pts_j"]
            h = d["heights"].reshape(-1)

            A_i = A_all[i]
            A_j = A_all[j]

            pred_j = self._predict_obs_from_i_to_j_torch(i, j, pts_i_ls, h, A_i, A_j)
            pred_i = self._predict_obs_from_i_to_j_torch(j, i, pts_j_ls, h, A_j, A_i)

            rj = (pred_j - pts_j_ls).reshape(-1)
            ri = (pred_i - pts_i_ls).reshape(-1)
            res_list.append(rj)
            res_list.append(ri)

        if len(res_list) == 0:
            return torch.zeros((0,), dtype=torch.float64, device=A_all.device)
        return torch.cat(res_list, dim=0)

    def rms(self, A_all: np.ndarray) -> float:
        r = self.residual_vector(A_all)
        if r.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(r**2)))

    def solve(
        self,
        max_iters: int = 15,
        tol: float = 1e-6,
        damping: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Gauss-Newton（自动微分雅可比），返回：
          - A_est: (M,2,3)
        """
        A = torch.tensor(self.A, dtype=torch.float64, device=self.device)
        rms_hist = []
        prev = None

        n_params = 6 * (self.M - 1)

        for it in range(1, max_iters + 1):
            r0 = self.residual_vector_torch(A)
            rms0 = float(torch.sqrt(torch.mean(r0**2)).item()) if r0.numel() else 0.0
            rms_hist.append(rms0)

            if verbose:
                if self.reporter:
                    self.reporter.log(f"[iter {it:02d}] RMS = {rms0:.6f} px, residual_dim={r0.numel()}, params={n_params}")

            if prev is not None and abs(prev - rms0) < tol:
                return A.detach().cpu()
            prev = rms0

            if r0.numel() == 0:
                return A.detach().cpu()

            # Normal equations: H dx = -g
            H = torch.zeros((n_params, n_params), dtype=torch.float64, device=self.device)
            g = torch.zeros((n_params,), dtype=torch.float64, device=self.device)

            J_cache = {}

            for m in self.var_ids:
                pos = self.id2pos[m]
                base = A[m].detach().reshape(-1)

                def residual_for_param(p):
                    A_tmp = A.detach().clone()
                    A_tmp[m] = p.reshape(2, 3)
                    return self.residual_vector_torch(A_tmp)

                try:
                    Jm = torch.autograd.functional.jacobian(residual_for_param, base, vectorize=True)
                except TypeError:
                    Jm = torch.autograd.functional.jacobian(residual_for_param, base)

                J_cache[m] = Jm

                idx = slice(6 * pos, 6 * pos + 6)
                H[idx, idx] += Jm.T @ Jm
                g[idx] += Jm.T @ r0

            # Cross terms
            for a in range(len(self.var_ids)):
                ma = self.var_ids[a]
                Ja = J_cache[ma]
                ia = self.id2pos[ma]
                Ia = slice(6 * ia, 6 * ia + 6)
                for mb in self.neighbors.get(ma, []):
                    if mb <= ma or mb == self.fixed_id:
                        continue
                    if mb not in J_cache:
                        continue
                    Jb = J_cache[mb]
                    ib = self.id2pos[mb]
                    Ib = slice(6 * ib, 6 * ib + 6)
                    Hab = Ja.T @ Jb
                    H[Ia, Ib] += Hab
                    H[Ib, Ia] += Hab.T

            # Levenberg damping
            H += damping * torch.eye(n_params, dtype=torch.float64, device=self.device)

            dx = -torch.linalg.solve(H, g)

            # Update blocks
            for m in self.var_ids:
                pos = self.id2pos[m]
                delta = dx[6 * pos: 6 * pos + 6]
                A[m] = (A[m].reshape(-1) + delta).reshape(2, 3)

            # Keep fixed as identity
            A[self.fixed_id] = torch.tensor([[1.0, 0.0, 0.0],
                                             [0.0, 1.0, 0.0]], dtype=torch.float64, device=self.device)

        return A.detach().cpu()
