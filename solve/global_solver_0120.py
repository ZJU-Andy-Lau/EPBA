import numpy as np
import torch
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from shared.rpc import RPCModelParameterTorch,project_linesamp
from infer.monitor import StatusReporter

from copy import deepcopy
import os

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None


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
    r = 6378137.0
    lat = np.asarray(latlon[:, 0], dtype=np.float64)
    lon = np.asarray(latlon[:, 1], dtype=np.float64)
    lon_rad = lon * np.pi / 180.0
    lat_rad = lat * np.pi / 180.0
    x = r * lon_rad
    y = r * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))
    return np.stack([y, x], axis=-1)


def mercator2lonlat_numpy(coord: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    r = 6378137.0
    lon = (180.0 * coord[:, 1]) / (np.pi * r)
    lat = (2.0 * np.arctan(np.exp(coord[:, 0] / r)) - np.pi * 0.5) * 180.0 / np.pi
    return np.stack([lat, lon], axis=-1)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    area = polygon_area(poly)
    if abs(area) < 1e-12:
        return np.mean(poly, axis=0)
    x = poly[:, 0]
    y = poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=np.float64)


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if polygon_area(poly) < 0:
        return poly[::-1]
    return poly


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return p2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float64)


def polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.copy()
    clip = ensure_ccw(clip)
    for i in range(len(clip)):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % len(clip)]
        input_list = output
        if len(input_list) == 0:
            break
        output = []
        s = input_list[-1]
        for e in input_list:
            cross_e = (cp2[0] - cp1[0]) * (e[1] - cp1[1]) - (cp2[1] - cp1[1]) * (e[0] - cp1[0])
            cross_s = (cp2[0] - cp1[0]) * (s[1] - cp1[1]) - (cp2[1] - cp1[1]) * (s[0] - cp1[0])
            inside_e = cross_e >= 0
            inside_s = cross_s >= 0
            if inside_e:
                if not inside_s:
                    output.append(line_intersection(s, e, cp1, cp2))
                output.append(e)
            elif inside_s:
                output.append(line_intersection(s, e, cp1, cp2))
            s = e
        output = np.asarray(output, dtype=np.float64)
    return output


def get_overlap_area(corners_a, corners_b):
    corners_a = np.asarray(corners_a, dtype=float)
    corners_b = np.asarray(corners_b, dtype=float)
    if corners_a.ndim == 3:
        corners_a = corners_a[0]
    if corners_b.ndim == 3:
        corners_b = corners_b[0]
    if corners_a.shape != (4, 2) or corners_b.shape != (4, 2):
        raise ValueError(f"corners must be (4,2). got A={corners_a.shape}, B={corners_b.shape}")
    inter = polygon_clip(ensure_ccw(corners_a), ensure_ccw(corners_b))
    area = abs(polygon_area(inter))
    return area


def sample_points_in_overlap(corners_a, corners_b, K, shrink=0.9, seed=None, max_iter_factor=200):
    rng = np.random.default_rng(seed)
    corners_a = np.asarray(corners_a, dtype=float)
    corners_b = np.asarray(corners_b, dtype=float)
    if corners_a.ndim == 3:
        corners_a = corners_a[0]
    if corners_b.ndim == 3:
        corners_b = corners_b[0]
    if corners_a.shape != (4, 2) or corners_b.shape != (4, 2):
        raise ValueError(f"corners must be (4,2). got A={corners_a.shape}, B={corners_b.shape}")
    inter = polygon_clip(ensure_ccw(corners_a), ensure_ccw(corners_b))
    if inter.size == 0:
        raise ValueError("no overlap")
    c = polygon_centroid(inter)
    inter = c + shrink * (inter - c)
    if abs(polygon_area(inter)) < 1e-12:
        raise ValueError("overlap area too small")
    v0 = inter[0]
    triangles = []
    areas = []
    for i in range(1, len(inter) - 1):
        tri = np.stack([v0, inter[i], inter[i + 1]], axis=0)
        a = abs(polygon_area(tri))
        if a > 1e-12:
            triangles.append(tri)
            areas.append(a)
    if len(triangles) == 0:
        raise ValueError("overlap area too small")
    areas = np.asarray(areas, dtype=np.float64)
    probs = areas / np.sum(areas)
    pts = []
    for _ in range(K):
        idx = rng.choice(len(triangles), p=probs)
        tri = triangles[idx]
        r1 = rng.random()
        r2 = rng.random()
        sqrt_r1 = math.sqrt(r1)
        a = 1.0 - sqrt_r1
        b = sqrt_r1 * (1.0 - r2)
        c = sqrt_r1 * r2
        pt = a * tri[0] + b * tri[1] + c * tri[2]
        pts.append(pt)
    return np.asarray(pts, dtype=float)


@dataclass
class PBASolveReport:
    iters: int
    rms_per_iter: List[float]


class PBAAffineSolver:
    def __init__(
        self,
        images:list,
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

    def _get_matches_overlap_area(self,images:list,results):
        for match in results:
            i,j = int(match['i']),int(match['j'])
            image_i = images[i]
            image_j = images[j]
            overlap_area = get_overlap_area(image_i.corner_xys,image_j.corner_xys)
            match['overlap_area'] = overlap_area
        return results

    def _get_fixed_id(self,images:list,results):
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

    def _process_data(self,images:list,results,sample_points_num = 256):
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

    def _affine_pred_and_jacobian_batch(self, A: np.ndarray, p_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = A[:, :2]
        t = A[:, 2]
        R_inv = np.linalg.inv(R)
        y = p_orig - t
        v = (R_inv @ y.T).T
        r0 = R_inv[:, 0]
        r1 = R_inv[:, 1]
        v0 = v[:, 0]
        v1 = v[:, 1]
        J = np.zeros((p_orig.shape[0], 2, 6), dtype=np.float64)
        J[:, :, 0] = (-v0[:, None] * r0[None, :])
        J[:, :, 1] = (-v1[:, None] * r0[None, :])
        J[:, :, 2] = (-r0[None, :])
        J[:, :, 3] = (-v0[:, None] * r1[None, :])
        J[:, :, 4] = (-v1[:, None] * r1[None, :])
        J[:, :, 5] = (-r1[None, :])
        return v, J

    def _predict_from_points(self, image_idx: int, point_xy: np.ndarray, h: np.ndarray, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lonlat = mercator2lonlat_numpy(point_xy)
        lat = lonlat[:, 0]
        lon = lonlat[:, 1]
        samp, line = self.rpcs[image_idx].RPC_OBJ2PHOTO(lat, lon, h, output_type="numpy")
        p_orig = np.stack([np.asarray(line), np.asarray(samp)], axis=1)
        pred, J = self._affine_pred_and_jacobian_batch(A, p_orig)
        return p_orig, pred, J

    def _point_base_xy_batch(self, image_idx: int, obs_ls: np.ndarray, h: np.ndarray, A: np.ndarray) -> np.ndarray:
        orig = affine_apply(A, obs_ls)
        samp, line = line_samp_to_xy(orig).T
        lat, lon = self.rpcs[image_idx].RPC_PHOTO2OBJ(samp, line, h, output_type="numpy")
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        latlon = np.stack([lat, lon], axis=-1)
        if latlon.ndim == 1:
            latlon = latlon[None, :]
        xy = project_mercator_numpy(latlon)
        return xy

    def _residual_and_affine_jacobian_batch(self, i: int, j: int, obs_i: np.ndarray, obs_j: np.ndarray, point_xy: np.ndarray, h: np.ndarray, A_i: np.ndarray, A_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, pred_i, Ji2 = self._predict_from_points(i, point_xy, h, A_i)
        _, pred_j, Jj2 = self._predict_from_points(j, point_xy, h, A_j)
        ri = pred_i - obs_i
        rj = pred_j - obs_j
        r = np.concatenate([ri, rj], axis=1)
        Ji = np.zeros((obs_i.shape[0], 4, 6), dtype=np.float64)
        Jj = np.zeros((obs_i.shape[0], 4, 6), dtype=np.float64)
        Ji[:, 0:2, :] = Ji2
        Jj[:, 2:4, :] = Jj2
        return r, Ji, Jj

    def _residual_only_batch(self, i: int, j: int, obs_i: np.ndarray, obs_j: np.ndarray, point_xy: np.ndarray, h: np.ndarray, A_i: np.ndarray, A_j: np.ndarray) -> np.ndarray:
        _, pred_i, _ = self._predict_from_points(i, point_xy, h, A_i)
        _, pred_j, _ = self._predict_from_points(j, point_xy, h, A_j)
        ri = pred_i - obs_i
        rj = pred_j - obs_j
        return np.concatenate([ri, rj], axis=1)

    def _xy_eps(self) -> float:
        return 1e-3

    def _jacobian_point_batch(self, i: int, j: int, obs_i: np.ndarray, obs_j: np.ndarray, point_xy: np.ndarray, h: np.ndarray, A_i: np.ndarray, A_j: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r0, Ji, Jj = self._residual_and_affine_jacobian_batch(i, j, obs_i, obs_j, point_xy, h, A_i, A_j)
        Jx = np.zeros((obs_i.shape[0], 4, 2), dtype=np.float64)
        eps_xy = self._xy_eps()
        pt_p = point_xy.copy()
        pt_m = point_xy.copy()
        pt_p[:, 0] += eps_xy
        pt_m[:, 0] -= eps_xy
        rp = self._residual_only_batch(i, j, obs_i, obs_j, pt_p, h, A_i, A_j)
        rm = self._residual_only_batch(i, j, obs_i, obs_j, pt_m, h, A_i, A_j)
        Jx[:, :, 0] = (rp - rm) / (2.0 * eps_xy)
        pt_p = point_xy.copy()
        pt_m = point_xy.copy()
        pt_p[:, 1] += eps_xy
        pt_m[:, 1] -= eps_xy
        rp = self._residual_only_batch(i, j, obs_i, obs_j, pt_p, h, A_i, A_j)
        rm = self._residual_only_batch(i, j, obs_i, obs_j, pt_m, h, A_i, A_j)
        Jx[:, :, 1] = (rp - rm) / (2.0 * eps_xy)
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
            point_xy = self._point_base_xy_batch(i, pts_i, h, A_i)
            r = self._residual_only_batch(i, j, pts_i, pts_j, point_xy, h, A_i, A_j)
            res.append(r.reshape(-1))
        if len(res) == 0:
            return 0.0
        r_all = np.concatenate(res, axis=0)
        return float(np.sqrt(np.mean(r_all ** 2)))

    def _add_block(self, blocks: Dict[Tuple[int, int], np.ndarray], a: int, b: int, value: np.ndarray):
        key = (a, b)
        if key in blocks:
            blocks[key] += value
        else:
            blocks[key] = value.copy()

    def _build_sparse(self, blocks: Dict[Tuple[int, int], np.ndarray], n_blocks: int, damping: float) -> Any:
        n_params = n_blocks * 6
        if sp is None:
            H = np.zeros((n_params, n_params), dtype=np.float64)
            for (a, b_idx), blk in blocks.items():
                ia = slice(6 * a, 6 * a + 6)
                ib = slice(6 * b_idx, 6 * b_idx + 6)
                H[ia, ib] += blk
            for a in range(n_blocks):
                ia = slice(6 * a, 6 * a + 6)
                H[ia, ia] += damping * np.eye(6, dtype=np.float64)
            return H
        H = sp.lil_matrix((n_params, n_params), dtype=np.float64)
        for (a, b_idx), blk in blocks.items():
            ia = slice(6 * a, 6 * a + 6)
            ib = slice(6 * b_idx, 6 * b_idx + 6)
            H[ia, ib] += blk
        for a in range(n_blocks):
            ia = slice(6 * a, 6 * a + 6)
            H[ia, ia] += damping * np.eye(6, dtype=np.float64)
        return H.tocsr()

    def _solve_system(self, H: Any, b: np.ndarray) -> np.ndarray:
        if sp is None or spla is None or not sp.issparse(H):
            return np.linalg.solve(H, b)
        return spla.spsolve(H, b)

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
        n_blocks = len(self.var_ids)
        n_params = 6 * n_blocks

        for it in range(1, max_iters + 1):
            blocks: Dict[Tuple[int, int], np.ndarray] = {}
            b = np.zeros((n_params,), dtype=np.float64)
            sum_sq = 0.0
            count = 0

            for d in self.ties:
                i = int(d["i"])
                j = int(d["j"])
                pts_i = d["pts_i"].astype(np.float64)
                pts_j = d["pts_j"].astype(np.float64)
                h = np.asarray(d["heights"], dtype=np.float64).reshape(-1)
                A_i = A[i]
                A_j = A[j]

                point_xy = self._point_base_xy_batch(i, pts_i, h, A_i)
                r0, Ji, Jj, Jx = self._jacobian_point_batch(i, j, pts_i, pts_j, point_xy, h, A_i, A_j)
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

                Js = np.concatenate(Js_blocks, axis=2)
                H_ss = np.einsum('nki,nkj->nij', Js, Js)
                H_sx = np.einsum('nki,nkj->nij', Js, Jx)
                H_xs = np.einsum('nki,nkj->nij', Jx, Js)
                H_xx = np.einsum('nki,nkj->nij', Jx, Jx)
                b_s = np.einsum('nki,nk->ni', Js, l)
                b_x = np.einsum('nki,nk->ni', Jx, l)

                H_xx = H_xx + np.eye(2, dtype=np.float64)[None, :, :] * 1e-12
                H_xx_inv = np.linalg.inv(H_xx)
                H_schur = H_ss - np.einsum('nij,njk,nkl->nil', H_sx, H_xx_inv, H_xs)
                b_schur = b_s - np.einsum('nij,njk,nk->ni', H_sx, H_xx_inv, b_x)

                H_schur_sum = np.sum(H_schur, axis=0)
                b_schur_sum = np.sum(b_schur, axis=0)

                for a, pos_a in enumerate(idx_blocks):
                    ia = slice(6 * pos_a, 6 * pos_a + 6)
                    for b_idx, pos_b in enumerate(idx_blocks):
                        blk = H_schur_sum[6 * a:6 * a + 6, 6 * b_idx:6 * b_idx + 6]
                        self._add_block(blocks, pos_a, pos_b, blk)
                    b[ia] += b_schur_sum[6 * a:6 * a + 6]

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

            H = self._build_sparse(blocks, n_blocks, damping)
            dx = self._solve_system(H, b)

            for m in self.var_ids:
                pos = self.id2pos[m]
                delta = dx[6 * pos: 6 * pos + 6]
                A[m] = (A[m].reshape(-1) + delta).reshape(2, 3)

            A[self.fixed_id] = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 1.0, 0.0]], dtype=np.float64)

        return torch.from_numpy(A)
