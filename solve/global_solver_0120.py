import numpy as np
import torch
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from shared.rpc import RPCModelParameterTorch,project_linesamp
from shared.utils import sample_points_in_overlap,get_overlap_area,project_mercator,mercator2lonlat
from infer.rs_image import RSImage
from infer.monitor import StatusReporter

from copy import deepcopy
import os


def affine_apply(A: np.ndarray, pts_line_samp: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_line_samp, dtype=np.float64)
    X = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return X @ A.T


def affine_inv(A: np.ndarray) -> np.ndarray:
    R = A[:, :2]
    t = A[:, 2]
    R_inv = np.linalg.inv(R)
    t_inv = -R_inv @ t
    return np.concatenate([R_inv, t_inv.reshape(2, 1)], axis=1)


def pack_A(A: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float64).reshape(-1)


def unpack_A(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(2, 3)


def xy_to_line_samp(pts_samp_line: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_samp_line, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"pts must be (N,2), got {pts.shape}")
    return pts[:, [1, 0]]


def line_samp_to_xy(pts_line_samp: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_line_samp, dtype=np.float64)
    return pts[:, [1, 0]]


def project_mercator_numpy(latlon: np.ndarray) -> np.ndarray:
    latlon_t = torch.from_numpy(latlon.astype(np.float64))
    proj = project_mercator(latlon_t).cpu().numpy()
    return proj


def mercator2lonlat_numpy(coord: np.ndarray) -> np.ndarray:
    coord_t = torch.from_numpy(coord.astype(np.float64))
    lonlat = mercator2lonlat(coord_t).cpu().numpy()
    return lonlat


@dataclass
class PBASolveReport:
    iters: int
    rms_per_iter: List[float]


class PBAAffineSolver:
    def __init__(
        self,
        images:list[RSImage],
        results,
        fixed_id: int = None,
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

        results = self._get_matches_overlap_area(images,results)
        if fixed_id is None or fixed_id < 0 or fixed_id >= len(images):
            self.fixed_id = self._get_fixed_id(images,results)
            reporter.log(f"Set fixed id from {fixed_id} to {self.fixed_id}")
        self.ties,self.rpcs = self._process_data(images,results,sample_points_num)
        self.M = len(self.rpcs)

        for rpc in self.rpcs:
            rpc.to_gpu(device)
            rpc.Clear_Adjust()

        self.A = np.zeros((self.M, 2, 3), dtype=np.float64)
        self.A[:] = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0]], dtype=np.float64)
        self.A[self.fixed_id] = np.array([[1.0, 0.0, 0.0],
                                          [0.0, 1.0, 0.0]], dtype=np.float64)

        self.var_ids = [m for m in range(self.M) if m != self.fixed_id]
        self.id2pos = {m: k for k, m in enumerate(self.var_ids)}
        self._prepare_tie_tensors()

    def _get_matches_overlap_area(self,images:list[RSImage],results):
        for match in results:
            i,j = int(match['i']),int(match['j'])
            image_i = images[i]
            image_j = images[j]
            overlap_area = get_overlap_area(image_i.corner_xys,image_j.corner_xys)
            match['overlap_area'] = overlap_area
        return results

    def _get_fixed_id(self,images:list[RSImage],results):
        overlap_area_sum = np.zeros((len(images),))
        for match in results:
            i,j = int(match['i']),int(match['j'])
            overlap_area = match['overlap_area']
            overlap_area_sum[i] += overlap_area
            overlap_area_sum[j] += overlap_area
        return int(np.argmax(overlap_area_sum))

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

    def _process_data(self,images:list[RSImage],results,sample_points_num = 256):
        ties = []
        rpcs = [image.rpc for image in images]
        min_area = 1e11
        for match in results:
            if match['overlap_area'] < min_area:
                min_area = match['overlap_area']
        for match in results:
            i,j = int(match['i']),int(match['j'])
            image_i = images[i]
            image_j = images[j]
            overlap_area = match['overlap_area']
            K = int(sample_points_num * (overlap_area / min_area))
            rpc_i_adj = deepcopy(image_i.rpc)
            rpc_i_adj.Update_Adjust(match['M'])
            sampled_xys = sample_points_in_overlap(image_i.corner_xys,image_j.corner_xys,K=K)
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

    def _predict_obs_from_point(self, image_idx: int, point_xy: np.ndarray, h: float, A: np.ndarray) -> np.ndarray:
        lonlat = mercator2lonlat_numpy(point_xy.reshape(1, 2))
        lat = lonlat[0, 0]
        lon = lonlat[0, 1]
        samp, line = self.rpcs[image_idx].RPC_OBJ2PHOTO(lat, lon, h, output_type="numpy")
        orig = np.stack([np.asarray(line), np.asarray(samp)], axis=0).reshape(1, 2)
        obs_pred = affine_apply(affine_inv(A), orig)
        return obs_pred.reshape(2)

    def _point_base_xy(self, image_idx: int, obs_ls: np.ndarray, h: float, A: np.ndarray) -> np.ndarray:
        orig = affine_apply(A, obs_ls.reshape(1, 2))
        samp, line = line_samp_to_xy(orig).reshape(-1)
        lat, lon = self.rpcs[image_idx].RPC_PHOTO2OBJ(samp, line, h, output_type="numpy")
        latlon = np.stack([np.asarray(lat), np.asarray(lon)], axis=-1)
        xy = project_mercator_numpy(latlon.reshape(1,2))
        return xy.reshape(2)

    def _residual_point(self, i: int, j: int, obs_i: np.ndarray, obs_j: np.ndarray, point_xy: np.ndarray, h: float, A_i: np.ndarray, A_j: np.ndarray) -> np.ndarray:
        pred_i = self._predict_obs_from_point(i, point_xy, h, A_i)
        pred_j = self._predict_obs_from_point(j, point_xy, h, A_j)
        ri = pred_i - obs_i
        rj = pred_j - obs_j
        return np.concatenate([ri, rj], axis=0)

    def _affine_eps(self) -> np.ndarray:
        return np.array([1e-6, 1e-6, 1e-3, 1e-6, 1e-6, 1e-3], dtype=np.float64)

    def _xy_eps(self) -> float:
        return 1e-3

    def _jacobian_point(self, i: int, j: int, obs_i: np.ndarray, obs_j: np.ndarray, point_xy: np.ndarray, h: float, A_i: np.ndarray, A_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r0 = self._residual_point(i, j, obs_i, obs_j, point_xy, h, A_i, A_j)
        Ji = np.zeros((4, 6), dtype=np.float64)
        Jj = np.zeros((4, 6), dtype=np.float64)
        Jx = np.zeros((4, 2), dtype=np.float64)

        eps_aff = self._affine_eps()
        for k in range(6):
            delta = eps_aff[k]
            if i != self.fixed_id:
                Ai_p = A_i.copy().reshape(-1)
                Ai_m = A_i.copy().reshape(-1)
                Ai_p[k] += delta
                Ai_m[k] -= delta
                rp = self._residual_point(i, j, obs_i, obs_j, point_xy, h, unpack_A(Ai_p), A_j)
                rm = self._residual_point(i, j, obs_i, obs_j, point_xy, h, unpack_A(Ai_m), A_j)
                Ji[:, k] = (rp - rm) / (2.0 * delta)
            if j != self.fixed_id:
                Aj_p = A_j.copy().reshape(-1)
                Aj_m = A_j.copy().reshape(-1)
                Aj_p[k] += delta
                Aj_m[k] -= delta
                rp = self._residual_point(i, j, obs_i, obs_j, point_xy, h, A_i, unpack_A(Aj_p))
                rm = self._residual_point(i, j, obs_i, obs_j, point_xy, h, A_i, unpack_A(Aj_m))
                Jj[:, k] = (rp - rm) / (2.0 * delta)

        eps_xy = self._xy_eps()
        for k in range(2):
            pt_p = point_xy.copy()
            pt_m = point_xy.copy()
            pt_p[k] += eps_xy
            pt_m[k] -= eps_xy
            rp = self._residual_point(i, j, obs_i, obs_j, pt_p, h, A_i, A_j)
            rm = self._residual_point(i, j, obs_i, obs_j, pt_m, h, A_i, A_j)
            Jx[:, k] = (rp - rm) / (2.0 * eps_xy)

        return r0, Ji, Jj, Jx

    def rms(self, A_all: np.ndarray) -> float:
        res = []
        for d in self.ties:
            i = int(d["i"])
            j = int(d["j"])
            pts_i = d["pts_i"]
            pts_j = d["pts_j"]
            h = np.asarray(d["heights"], dtype=np.float64).reshape(-1)
            A_i = A_all[i]
            A_j = A_all[j]
            for k in range(pts_i.shape[0]):
                obs_i = pts_i[k].astype(np.float64)
                obs_j = pts_j[k].astype(np.float64)
                point_xy = self._point_base_xy(i, obs_i, h[k], A_i)
                r = self._residual_point(i, j, obs_i, obs_j, point_xy, h[k], A_i, A_j)
                res.append(r)
        if len(res) == 0:
            return 0.0
        r_all = np.concatenate(res, axis=0)
        return float(np.sqrt(np.mean(r_all ** 2)))

    def solve(
        self,
        max_iters: int = 15,
        tol: float = 1e-6,
        damping: float = 1e-6,
        verbose: bool = True,
    ):
        A = self.A.copy()
        rms_hist = []
        prev = None
        n_params = 6 * (self.M - 1)

        for it in range(1, max_iters + 1):
            H = np.zeros((n_params, n_params), dtype=np.float64)
            b = np.zeros((n_params,), dtype=np.float64)
            sum_sq = 0.0
            count = 0

            for d in self.ties:
                i = int(d["i"])
                j = int(d["j"])
                pts_i = d["pts_i"]
                pts_j = d["pts_j"]
                h = np.asarray(d["heights"], dtype=np.float64).reshape(-1)
                A_i = A[i]
                A_j = A[j]

                for k in range(pts_i.shape[0]):
                    obs_i = pts_i[k].astype(np.float64)
                    obs_j = pts_j[k].astype(np.float64)
                    point_xy = self._point_base_xy(i, obs_i, h[k], A_i)
                    r0, Ji, Jj, Jx = self._jacobian_point(i, j, obs_i, obs_j, point_xy, h[k], A_i, A_j)
                    l = -r0
                    sum_sq += float(np.sum(r0 ** 2))
                    count += int(r0.size)

                    Js_blocks = []
                    idx_blocks = []
                    if i != self.fixed_id:
                        Js_blocks.append(Ji)
                        idx_blocks.append(self.id2pos[i])
                    if j != self.fixed_id and j != i:
                        Js_blocks.append(Jj)
                        idx_blocks.append(self.id2pos[j])

                    if len(Js_blocks) == 0:
                        continue

                    Js = np.concatenate(Js_blocks, axis=1)
                    H_ss = Js.T @ Js
                    H_sx = Js.T @ Jx
                    H_xs = Jx.T @ Js
                    H_xx = Jx.T @ Jx
                    b_s = Js.T @ l
                    b_x = Jx.T @ l

                    H_xx_inv = np.linalg.inv(H_xx)
                    H_schur = H_ss - H_sx @ H_xx_inv @ H_xs
                    b_schur = b_s - H_sx @ H_xx_inv @ b_x

                    for a, pos_a in enumerate(idx_blocks):
                        Ia = slice(6 * pos_a, 6 * pos_a + 6)
                        for b_idx, pos_b in enumerate(idx_blocks):
                            Ib = slice(6 * pos_b, 6 * pos_b + 6)
                            H[Ia, Ib] += H_schur[6 * a:6 * a + 6, 6 * b_idx:6 * b_idx + 6]
                        b[Ia] += b_schur[6 * a:6 * a + 6]

            rms0 = float(math.sqrt(sum_sq / count)) if count else 0.0
            rms_hist.append(rms0)

            if verbose:
                if self.reporter:
                    self.reporter.log(f"[iter {it:02d}] RMS = {rms0:.6f} px, residual_dim={count}, params={n_params}")

            if prev is not None and abs(prev - rms0) < tol:
                return torch.from_numpy(A)
            prev = rms0

            if count == 0:
                return torch.from_numpy(A)

            H += damping * np.eye(n_params, dtype=np.float64)
            dx = np.linalg.solve(H, b)

            for m in self.var_ids:
                pos = self.id2pos[m]
                delta = dx[6 * pos: 6 * pos + 6]
                A[m] = (A[m].reshape(-1) + delta).reshape(2, 3)

            A[self.fixed_id] = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]], dtype=np.float64)

        return torch.from_numpy(A)
