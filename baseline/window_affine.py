import sys
import os

sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")

import argparse
import itertools
import traceback
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
from shapely.geometry import Polygon, box

from shared.utils import str2bool, get_current_time, load_model_state_dict, load_config, project_mercator
from infer.utils import is_overlap, find_intersection, get_report_dict, partition_pairs, apply_H
from infer.rs_image import RSImage, RSImageMeta
from infer.monitor import StatusMonitor, StatusReporter
from infer.pair import Solver, default_configs
from model.encoder import Encoder
from model.predictor import Predictor
from baseline.matchers import build_matcher
from shared.rpc import project_linesamp


@dataclass
class WindowSample:
    pair_i: int
    pair_j: int
    tie_idx: int
    sample_idx: int
    tie_xy: np.ndarray
    diag_xy: np.ndarray


@dataclass
class WindowAffineResult:
    root: str
    pair_i: int
    pair_j: int
    tie_idx: int
    sample_idx: int
    method: str
    status: str
    error_pix: float
    match_points: int
    affine: Optional[np.ndarray] = None


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_roots(args) -> List[str]:
    roots = []
    if args.roots is not None and len(args.roots.strip()) > 0:
        roots.extend([x.strip() for x in args.roots.split(',') if len(x.strip()) > 0])
    if args.root is not None and len(args.root.strip()) > 0:
        roots.append(args.root.strip())
    dedup = []
    seen = set()
    for root in roots:
        if root not in seen:
            dedup.append(root)
            seen.add(root)
    return dedup


def load_images_meta(root: str, args, reporter) -> List[RSImageMeta]:
    reporter.update(current_step=f"Loading Meta [{os.path.basename(root)}]")
    base_path = os.path.join(root, 'adjust_images')
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if args.select_imgs != '-1':
        select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    else:
        select_img_idxs = range(len(img_folders))
    img_folders = [img_folders[i] for i in select_img_idxs]
    metas = []
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        metas.append(RSImageMeta(args, img_path, idx, args.device))
    return metas


def load_images(args, metas: List[RSImageMeta], reporter) -> List[RSImage]:
    reporter.update(current_step="Loading Images")
    images = [RSImage(meta, device=args.device) for meta in metas]
    return images


def get_pairs(metas: List[RSImageMeta], min_window_size: int) -> List[Tuple[int, int]]:
    pair_idxs = []
    for i, j in itertools.combinations(range(len(metas)), 2):
        if is_overlap(metas[i], metas[j], min_window_size ** 2):
            pair_idxs.append((i, j))
    return pair_idxs


def load_epba_models(args):
    model_configs = load_config(args.model_config_path)
    encoder = Encoder(
        dino_weight_path=args.dino_path,
        embed_dim=model_configs['encoder']['embed_dim'],
        ctx_dim=model_configs['encoder']['ctx_dim'],
        use_adapter=args.use_adapter,
        use_conf=args.use_conf,
    )
    encoder.load_adapter(args.adapter_path)

    predictor = Predictor(
        corr_levels=model_configs['predictor']['corr_levels'],
        corr_radius=model_configs['predictor']['corr_radius'],
        context_dim=model_configs['predictor']['ctx_dim'],
        hidden_dim=model_configs['predictor']['hidden_dim'],
        use_mtf=args.use_mtf,
    )
    load_model_state_dict(predictor, args.predictor_path)

    if args.predictor_iter_num is None:
        args.predictor_iter_num = model_configs['predictor']['iter_num']

    if torch.cuda.is_available():
        encoder = encoder.to(args.device).eval().half()
        predictor = predictor.to(args.device).eval()
    else:
        encoder = encoder.to(args.device).eval()
        predictor = predictor.to(args.device).eval()
    return encoder, predictor


class TiePointWindowSolver(Solver):
    def __init__(self, rs_image_a: RSImage, rs_image_b: RSImage, configs: dict, device: str = 'cuda', reporter=None):
        cfg = {**default_configs, **configs}
        super().__init__(rs_image_a=rs_image_a, rs_image_b=rs_image_b, configs=cfg, device=device, reporter=reporter)

    def init_window_pairs_from_samples(self, samples: List[WindowSample]):
        if len(samples) == 0:
            self.window_pairs = []
            self.window_pairs_num = 0
            self.window_size = -1
            return
        diags = np.stack([s.diag_xy for s in samples], axis=0).astype(np.float64)
        self.build_window_pairs(diags)


def convert_tiepoint_to_xy(image: RSImage, line: int, samp: int, height: float) -> np.ndarray:
    lat, lon = image.rpc.RPC_PHOTO2OBJ(np.array([samp], dtype=np.float64), np.array([line], dtype=np.float64), np.array([height], dtype=np.float64), 'numpy')
    latlon = torch.from_numpy(np.stack([lat, lon], axis=-1))
    yx = project_mercator(latlon).cpu().numpy()[0]
    return yx


def generate_samples_for_pair(
    image_a: RSImage,
    image_b: RSImage,
    pair_i: int,
    pair_j: int,
    k_samples: int,
    window_size: float,
    max_trials: int,
    random_state: np.random.RandomState,
    reporter,
) -> List[WindowSample]:
    if image_a.tie_points is None or image_b.tie_points is None:
        return []

    tie_num = min(len(image_a.tie_points), len(image_b.tie_points))
    if tie_num == 0:
        return []

    corners_a = image_a.corner_xys
    corners_b = image_b.corner_xys
    polygon_corners = find_intersection(np.stack([corners_a, corners_b], axis=0))
    if polygon_corners.shape[0] < 3:
        return []

    overlap_poly = Polygon(polygon_corners)
    if not overlap_poly.is_valid:
        overlap_poly = overlap_poly.buffer(0)
    if overlap_poly.is_empty:
        return []

    samples: List[WindowSample] = []
    half = 0.5 * float(window_size)
    bounds = overlap_poly.bounds

    for tie_idx in range(tie_num):
        line_a, samp_a = image_a.tie_points[tie_idx]
        line_b, samp_b = image_b.tie_points[tie_idx]
        if line_a < 0 or line_a >= image_a.H or samp_a < 0 or samp_a >= image_a.W:
            continue
        if line_b < 0 or line_b >= image_b.H or samp_b < 0 or samp_b >= image_b.W:
            continue

        height_a = float(image_a.dem[line_a, samp_a])
        if not np.isfinite(height_a):
            continue

        tie_xy = convert_tiepoint_to_xy(image_a, line_a, samp_a, height_a)
        tie_x = float(tie_xy[0])
        tie_y = float(tie_xy[1])

        tie_samples = 0
        attempts = 0

        while tie_samples < k_samples and attempts < max_trials:
            attempts += 1
            dx = random_state.uniform(-half, half)
            dy = random_state.uniform(-half, half)
            cx = tie_x + dx
            cy = tie_y + dy
            tlx = cx - half
            tly = cy - half
            brx = cx + half
            bry = cy + half

            if tlx < bounds[0] or brx > bounds[2] or tly < bounds[1] or bry > bounds[3]:
                continue

            if not (tlx <= tie_x <= brx and tly <= tie_y <= bry):
                continue

            win_poly = box(tlx, tly, brx, bry)
            if not overlap_poly.contains(win_poly):
                continue

            diag = np.array([[tlx, tly], [brx, bry]], dtype=np.float64)
            samples.append(
                WindowSample(
                    pair_i=pair_i,
                    pair_j=pair_j,
                    tie_idx=tie_idx,
                    sample_idx=tie_samples,
                    tie_xy=tie_xy.astype(np.float64),
                    diag_xy=diag,
                )
            )
            tie_samples += 1

        if tie_samples < k_samples:
            for _ in range(k_samples - tie_samples):
                cx = np.clip(tie_x, bounds[0] + half, bounds[2] - half)
                cy = np.clip(tie_y, bounds[1] + half, bounds[3] - half)
                tlx = cx - half
                tly = cy - half
                brx = cx + half
                bry = cy + half
                win_poly = box(tlx, tly, brx, bry)
                if overlap_poly.contains(win_poly):
                    diag = np.array([[tlx, tly], [brx, bry]], dtype=np.float64)
                    samples.append(
                        WindowSample(
                            pair_i=pair_i,
                            pair_j=pair_j,
                            tie_idx=tie_idx,
                            sample_idx=tie_samples,
                            tie_xy=tie_xy.astype(np.float64),
                            diag_xy=diag,
                        )
                    )
                    tie_samples += 1
                else:
                    break

        if reporter is not None and tie_idx % 100 == 0:
            reporter.log(f"pair {pair_i}-{pair_j} tie {tie_idx + 1}/{tie_num} sampled")

    return samples


def match_and_estimate_baseline(
    args,
    matcher,
    image_a: RSImage,
    image_b: RSImage,
    img_a: np.ndarray,
    img_b: np.ndarray,
    H_a: np.ndarray,
    H_b: np.ndarray,
    device: str,
) -> Tuple[Optional[np.ndarray], int]:
    match_res = matcher.match(img_a, img_b)
    pts_a_crop = np.asarray(match_res.pts0)
    pts_b_crop = np.asarray(match_res.pts1)
    if pts_a_crop.shape[0] < args.min_match_points or pts_b_crop.shape[0] < args.min_match_points:
        return None, 0

    pts_a_rc = pts_a_crop[:, [1, 0]].astype(np.float32)
    pts_b_rc = pts_b_crop[:, [1, 0]].astype(np.float32)

    pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(device)
    pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(device)

    H_a_inv = torch.from_numpy(np.linalg.inv(H_a)).unsqueeze(0).to(device=device, dtype=torch.float32)
    H_b_inv = torch.from_numpy(np.linalg.inv(H_b)).unsqueeze(0).to(device=device, dtype=torch.float32)

    pts_a_global = apply_H(pts_a_rc_t, H_a_inv, device).squeeze(0).cpu().numpy()
    pts_b_global = apply_H(pts_b_rc_t, H_b_inv, device).squeeze(0).cpu().numpy()

    sampline_b = pts_b_global[:, [1, 0]]
    heights_b = image_b.dem_interp(sampline_b)
    valid = np.isfinite(heights_b)
    if valid.sum() < args.min_match_points:
        return None, int(valid.sum())

    pts_a_global = pts_a_global[valid]
    pts_b_global = pts_b_global[valid]
    heights_b = heights_b[valid]

    lines_proj_a, samps_proj_a = project_linesamp(
        image_b.rpc,
        image_a.rpc,
        pts_b_global[:, 0],
        pts_b_global[:, 1],
        heights_b,
        'numpy',
    )
    pts_b_to_a = np.stack([lines_proj_a, samps_proj_a], axis=-1).astype(np.float32)
    pts_a_obs = pts_a_global.astype(np.float32)

    M, inliers = cv2.estimateAffine2D(
        pts_a_obs,
        pts_b_to_a,
        ransacReprojThreshold=args.affine_ransac_thresh,
        maxIters=args.affine_ransac_max_iters,
        confidence=args.affine_ransac_conf,
        refineIters=args.affine_ransac_refine_iters,
    )
    if M is None:
        return None, pts_a_obs.shape[0]

    return M.astype(np.float32), pts_a_obs.shape[0]


def estimate_epba_for_subset(
    solver: TiePointWindowSolver,
    subset_indices: List[int],
    encoder: Encoder,
    predictor: Predictor,
    args,
) -> Dict[int, Optional[np.ndarray]]:
    if len(subset_indices) == 0:
        return {}

    sub_pairs = [solver.window_pairs[i] for i in subset_indices]
    old_pairs = solver.window_pairs
    solver.window_pairs = sub_pairs

    try:
        if len(sub_pairs) == 0:
            return {idx: None for idx in subset_indices}

        if args.model_use_quadsplit:
            window_size = abs(float(sub_pairs[0].diag[1, 0] - sub_pairs[0].diag[0, 0]))
            if window_size >= args.model_min_window_size_for_quadsplit:
                solver.quadsplit_windows()

        Hs_a, _ = solver.collect_Hs(to_tensor=True)
        preds, scores = solver.get_window_affines(encoder, predictor)

        affines: Dict[int, Optional[np.ndarray]] = {}
        for local_idx, global_idx in enumerate(subset_indices):
            affine_local = preds[local_idx].unsqueeze(0)
            H_local = Hs_a[local_idx].unsqueeze(0)
            score_local = scores[local_idx].unsqueeze(0)
            merged = solver.merge_affines(affine_local, H_local, score_local)
            affines[global_idx] = merged.detach().cpu().numpy().astype(np.float32)
        return affines
    finally:
        solver.window_pairs = old_pairs


def validate_tiepoint_error(
    image_a: RSImage,
    image_b: RSImage,
    tie_idx: int,
    affine: np.ndarray,
) -> Optional[float]:
    if image_a.tie_points is None or image_b.tie_points is None:
        return None
    if tie_idx >= len(image_a.tie_points) or tie_idx >= len(image_b.tie_points):
        return None

    line_a, samp_a = image_a.tie_points[tie_idx]
    line_b, samp_b = image_b.tie_points[tie_idx]

    if line_a < 0 or line_a >= image_a.H or samp_a < 0 or samp_a >= image_a.W:
        return None
    if line_b < 0 or line_b >= image_b.H or samp_b < 0 or samp_b >= image_b.W:
        return None

    height_a = float(image_a.dem[line_a, samp_a])
    if not np.isfinite(height_a):
        return None

    src = np.array([line_a, samp_a, 1.0], dtype=np.float64)
    dst_a = affine.astype(np.float64) @ src

    lat_a, lon_a = image_a.rpc.RPC_PHOTO2OBJ(
        np.array([dst_a[1]], dtype=np.float64),
        np.array([dst_a[0]], dtype=np.float64),
        np.array([height_a], dtype=np.float64),
        'numpy',
    )
    samp_b_proj, line_b_proj = image_b.rpc.RPC_OBJ2PHOTO(
        lat_a,
        lon_a,
        np.array([height_a], dtype=np.float64),
        'numpy',
    )
    pred = np.array([line_b_proj[0], samp_b_proj[0]], dtype=np.float64)
    tgt = np.array([line_b, samp_b], dtype=np.float64)
    return float(np.linalg.norm(pred - tgt))


def process_pair(
    args,
    root: str,
    pair_id: Tuple[int, int],
    images_by_id: Dict[int, RSImage],
    matcher,
    encoder: Optional[Encoder],
    predictor: Optional[Predictor],
    reporter,
    rng: np.random.RandomState,
) -> List[WindowAffineResult]:
    i, j = pair_id
    image_a = images_by_id[i]
    image_b = images_by_id[j]

    solver_cfg = {
        'output_path': os.path.join(args.output_path, f"{os.path.basename(root)}_pair_{i}_{j}"),
        'max_window_num': -1,
        'min_window_size': args.window_size,
        'max_window_size': args.window_size,
        'min_area_ratio': args.min_cover_area_ratio,
        'quad_split_times': args.quad_split_times,
        'iter_num': args.predictor_iter_num if args.predictor_iter_num is not None else 10,
        'match': None,
    }
    solver = TiePointWindowSolver(image_a, image_b, configs=solver_cfg, device=args.device, reporter=reporter)

    samples = generate_samples_for_pair(
        image_a=image_a,
        image_b=image_b,
        pair_i=i,
        pair_j=j,
        k_samples=args.k_samples,
        window_size=args.window_size,
        max_trials=args.max_sample_trials,
        random_state=rng,
        reporter=reporter,
    )
    if len(samples) == 0:
        return []

    solver.init_window_pairs_from_samples(samples)
    if len(solver.window_pairs) == 0:
        return []

    imgs_a, imgs_b = solver.collect_imgs()
    Hs_a, Hs_b = solver.collect_Hs(to_tensor=False)

    results: List[WindowAffineResult] = []

    if args.estimate_mode in ['baseline', 'both']:
        reporter.update(current_step="Baseline Estimation")
        for idx, sample in enumerate(samples):
            if idx >= len(imgs_a):
                break
            affine, nmatch = match_and_estimate_baseline(
                args=args,
                matcher=matcher,
                image_a=image_a,
                image_b=image_b,
                img_a=imgs_a[idx],
                img_b=imgs_b[idx],
                H_a=Hs_a[idx],
                H_b=Hs_b[idx],
                device=args.device,
            )
            if affine is None:
                results.append(
                    WindowAffineResult(
                        root=root,
                        pair_i=i,
                        pair_j=j,
                        tie_idx=sample.tie_idx,
                        sample_idx=sample.sample_idx,
                        method='baseline',
                        status='failed',
                        error_pix=float('nan'),
                        match_points=nmatch,
                        affine=None,
                    )
                )
                continue

            err = validate_tiepoint_error(image_a, image_b, sample.tie_idx, affine)
            if err is None:
                err = float('nan')
                status = 'invalid'
            else:
                status = 'ok'
            results.append(
                WindowAffineResult(
                    root=root,
                    pair_i=i,
                    pair_j=j,
                    tie_idx=sample.tie_idx,
                    sample_idx=sample.sample_idx,
                    method='baseline',
                    status=status,
                    error_pix=float(err),
                    match_points=nmatch,
                    affine=affine,
                )
            )

    if args.estimate_mode in ['model', 'both']:
        if encoder is None or predictor is None:
            raise RuntimeError("Model mode requires encoder and predictor.")

        reporter.update(current_step="Model Estimation")
        all_indices = list(range(len(samples)))
        batches = [all_indices[s:s + args.model_batch_size] for s in range(0, len(all_indices), args.model_batch_size)]

        for subset in batches:
            affine_map = estimate_epba_for_subset(
                solver=solver,
                subset_indices=subset,
                encoder=encoder,
                predictor=predictor,
                args=args,
            )
            for idx in subset:
                sample = samples[idx]
                affine = affine_map.get(idx, None)
                if affine is None:
                    results.append(
                        WindowAffineResult(
                            root=root,
                            pair_i=i,
                            pair_j=j,
                            tie_idx=sample.tie_idx,
                            sample_idx=sample.sample_idx,
                            method='model',
                            status='failed',
                            error_pix=float('nan'),
                            match_points=0,
                            affine=None,
                        )
                    )
                    continue

                err = validate_tiepoint_error(image_a, image_b, sample.tie_idx, affine)
                if err is None:
                    err = float('nan')
                    status = 'invalid'
                else:
                    status = 'ok'
                results.append(
                    WindowAffineResult(
                        root=root,
                        pair_i=i,
                        pair_j=j,
                        tie_idx=sample.tie_idx,
                        sample_idx=sample.sample_idx,
                        method='model',
                        status=status,
                        error_pix=float(err),
                        match_points=0,
                        affine=affine,
                    )
                )

    return results


def summarize_results(results: List[WindowAffineResult]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if len(results) == 0:
        return summary

    keys = sorted(set((r.root, r.method) for r in results))
    for root, method in keys:
        vals = np.array([
            r.error_pix for r in results
            if r.root == root and r.method == method and np.isfinite(r.error_pix)
        ], dtype=np.float64)
        if vals.size == 0:
            summary[f"{root}::{method}"] = {
                'count': 0,
                'mean': float('nan'),
                'median': float('nan'),
                '<1pix_percent': float('nan'),
                '<3pix_percent': float('nan'),
                '<5pix_percent': float('nan'),
            }
        else:
            rep = get_report_dict(vals)
            summary[f"{root}::{method}"] = {
                'count': rep['count'],
                'mean': rep['mean'],
                'median': rep['median'],
                '<1pix_percent': rep['<1pix_percent'],
                '<3pix_percent': rep['<3pix_percent'],
                '<5pix_percent': rep['<5pix_percent'],
            }

    methods = sorted(set(r.method for r in results))
    for method in methods:
        vals = np.array([
            r.error_pix for r in results
            if r.method == method and np.isfinite(r.error_pix)
        ], dtype=np.float64)
        if vals.size == 0:
            summary[f"ALL::{method}"] = {
                'count': 0,
                'mean': float('nan'),
                'median': float('nan'),
                '<1pix_percent': float('nan'),
                '<3pix_percent': float('nan'),
                '<5pix_percent': float('nan'),
            }
        else:
            rep = get_report_dict(vals)
            summary[f"ALL::{method}"] = {
                'count': rep['count'],
                'mean': rep['mean'],
                'median': rep['median'],
                '<1pix_percent': rep['<1pix_percent'],
                '<3pix_percent': rep['<3pix_percent'],
                '<5pix_percent': rep['<5pix_percent'],
            }

    return summary


def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
    else:
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        args.device = "cpu"

    experiment_id_clean = str(args.experiment_id).replace(":", "_").replace(" ", "_")
    monitor = None
    if rank == 0:
        monitor = StatusMonitor(world_size, experiment_id_clean)
        monitor.start()
    reporter = StatusReporter(rank, world_size, experiment_id_clean, monitor)

    matcher = None
    if args.estimate_mode in ['baseline', 'both']:
        matcher = build_matcher(args)

    encoder = None
    predictor = None
    if args.estimate_mode in ['model', 'both']:
        encoder, predictor = load_epba_models(args)

    try:
        init_random_seed(args.random_seed + rank)
        rng = np.random.RandomState(args.random_seed + rank)

        roots = parse_roots(args)
        if len(roots) == 0:
            raise ValueError("No valid roots provided by --root/--roots")

        all_local_results: List[WindowAffineResult] = []

        for root_idx, root in enumerate(roots):
            reporter.update(current_task=f"Root {root_idx + 1}/{len(roots)}", current_step=f"Preparing [{os.path.basename(root)}]")
            metas = []
            pairs_ids_all = []
            pairs_ids_chunks = None

            if rank == 0:
                metas = load_images_meta(root, args, reporter)
                pairs_ids_all = get_pairs(metas, args.window_size)
                pairs_ids_chunks = partition_pairs(pairs_ids_all, world_size)

            container = [metas, pairs_ids_all]
            dist.broadcast_object_list(container, src=0)
            metas = container[0]
            pairs_ids_all = container[1]

            recv = [None]
            dist.scatter_object_list(recv, pairs_ids_chunks if rank == 0 else None, src=0)
            pairs_ids = recv[0]

            total_pairs = len(pairs_ids)
            reporter.update(progress=f"0/{total_pairs}", current_step=f"Processing [{os.path.basename(root)}]")

            if total_pairs > 0:
                image_ids = sorted(set(x for t in pairs_ids for x in t))
                images = load_images(args, [metas[i] for i in image_ids], reporter)
                images_by_id = {img.id: img for img in images}

                for pidx, pair_id in enumerate(pairs_ids):
                    reporter.update(progress=f"{pidx + 1}/{total_pairs}", current_task=f"{os.path.basename(root)} {pair_id[0]}=>{pair_id[1]}")
                    pair_results = process_pair(
                        args=args,
                        root=root,
                        pair_id=pair_id,
                        images_by_id=images_by_id,
                        matcher=matcher,
                        encoder=encoder,
                        predictor=predictor,
                        reporter=reporter,
                        rng=rng,
                    )
                    all_local_results.extend(pair_results)

                for img in images:
                    del img

        reporter.update(current_step="Gathering Results")
        if rank == 0:
            gathered = [None for _ in range(world_size)]
        else:
            gathered = None

        dist.barrier()
        dist.gather_object(all_local_results, gathered if rank == 0 else None, dst=0)

        if rank == 0:
            all_results: List[WindowAffineResult] = []
            for part in gathered:
                if part is not None:
                    all_results.extend(part)

            summary = summarize_results(all_results)
            for key in sorted(summary.keys()):
                rep = summary[key]
                reporter.log(f"[{key}] count={rep['count']} mean={rep['mean']:.6f} median={rep['median']:.6f} <1={rep['<1pix_percent']:.2f}% <3={rep['<3pix_percent']:.2f}% <5={rep['<5pix_percent']:.2f}%")

            os.makedirs(args.output_path, exist_ok=True)
            out_path = os.path.join(args.output_path, "window_affine_results.csv")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write("root,pair_i,pair_j,tie_idx,sample_idx,method,status,error_pix,match_points\n")
                for r in all_results:
                    f.write(
                        f"{r.root},{r.pair_i},{r.pair_j},{r.tie_idx},{r.sample_idx},{r.method},{r.status},{r.error_pix},{r.match_points}\n"
                    )

    except Exception as e:
        if reporter is not None:
            reporter.update(current_task="ERROR", error=traceback.format_exc())
        raise e
    finally:
        if monitor is not None:
            monitor.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--roots', type=str, default=None)
    parser.add_argument('--select_imgs', type=str, default='-1')

    parser.add_argument('--estimate_mode', type=str, default='both', choices=['baseline', 'model', 'both'])

    parser.add_argument('--k_samples', type=int, default=8)
    parser.add_argument('--window_size', type=int, default=500)
    parser.add_argument('--max_sample_trials', type=int, default=100)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)

    parser.add_argument('--matcher', type=str, default='loftr', choices=['sift', 'loftr', 'superglue', 'aspanformer', 'roma'])
    parser.add_argument('--loftr_weight_path', type=str, default=None)
    parser.add_argument('--superpoint_weight_path', type=str, default=None)
    parser.add_argument('--superglue_weight_path', type=str, default=None)
    parser.add_argument('--aspanformer_config_path', type=str, default=None)
    parser.add_argument('--aspanformer_weight_path', type=str, default=None)
    parser.add_argument('--roma_weight_path', type=str, default=None)
    parser.add_argument('--roma_dinov2_weight_path', type=str, default=None)
    parser.add_argument('--roma_variant', type=str, default='outdoor')

    parser.add_argument('--sift_ratio_thresh', type=float, default=0.75)
    parser.add_argument('--fm_ransac_thresh', type=float, default=3.0)
    parser.add_argument('--fm_confidence', type=float, default=0.99)
    parser.add_argument('--min_match_points', type=int, default=12)

    parser.add_argument('--affine_ransac_thresh', type=float, default=3.0)
    parser.add_argument('--affine_ransac_max_iters', type=int, default=2000)
    parser.add_argument('--affine_ransac_conf', type=float, default=0.99)
    parser.add_argument('--affine_ransac_refine_iters', type=int, default=10)

    parser.add_argument('--dino_path', type=str, default='weights')
    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')
    parser.add_argument('--predictor_path', type=str, default='weights/predictor.pth')
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--predictor_iter_num', type=int, default=None)
    parser.add_argument('--use_adapter', type=str2bool, default=True)
    parser.add_argument('--use_conf', type=str2bool, default=True)
    parser.add_argument('--use_mtf', type=str2bool, default=True)

    parser.add_argument('--model_batch_size', type=int, default=64)
    parser.add_argument('--model_use_quadsplit', type=str2bool, default=True)
    parser.add_argument('--quad_split_times', type=int, default=1)
    parser.add_argument('--model_min_window_size_for_quadsplit', type=int, default=500)

    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--usgs_dem', type=str2bool, default=False)

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()

    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]', get_current_time())

    args.output_path = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)
