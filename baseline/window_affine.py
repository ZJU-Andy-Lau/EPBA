import argparse
import itertools
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
from shapely.geometry import Point, Polygon

from baseline.matchers import build_matcher
from infer.rs_image import RSImage, RSImageMeta
from infer.utils import (
    apply_H,
    apply_M,
    extract_features,
    find_intersection,
    get_coord_mat,
    get_report_dict,
    quadsplit_diags,
    solve_weighted_affine,
)
from model.encoder import Encoder
from model.predictor import Predictor
from shared.utils import load_config, load_model_state_dict, project_mercator
from solve.solve_windows import WindowSolver


@dataclass
class WindowData:
    image: RSImage
    imgs: np.ndarray
    dems: np.ndarray
    Hs: np.ndarray


def init_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images_meta(root: str, args) -> List[RSImageMeta]:
    base_path = os.path.join(root, "adjust_images")
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if args.select_imgs != "-1":
        select_img_idxs = [int(i) for i in args.select_imgs.split(",")]
        img_folders = [img_folders[i] for i in select_img_idxs]
    metas = []
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        metas.append(RSImageMeta(args, img_path, idx, args.device))
    return metas


def load_images(metas: List[RSImageMeta], device: str) -> List[RSImage]:
    return [RSImage(meta, device=device) for meta in metas]


def compute_common_polygon(images: List[RSImage]) -> Polygon:
    corners = [image.corner_xys for image in images]
    intersection = find_intersection(np.stack(corners, axis=0))
    return Polygon(intersection)


def tiepoint_to_object_xy(image: RSImage, tie_point: np.ndarray) -> np.ndarray:
    line, samp = int(tie_point[0]), int(tie_point[1])
    height = image.dem[line, samp]
    lat, lon = image.rpc.RPC_PHOTO2OBJ(samp, line, height, "tensor")
    latlon = torch.stack([lat, lon], dim=-1).to(torch.float64)
    xy = project_mercator(latlon[None]).cpu().numpy()[0]
    return xy


def sample_windows(
    tie_xy: np.ndarray,
    polygon: Polygon,
    window_size: float,
    k: int,
    max_attempts: int,
) -> List[np.ndarray]:
    windows = []
    half = window_size / 2.0
    if polygon.is_empty:
        return windows
    for _ in range(max_attempts):
        if len(windows) >= k:
            break
        minx, miny, maxx, maxy = polygon.bounds
        cx = random.uniform(max(minx, tie_xy[0] - half), min(maxx, tie_xy[0] + half))
        cy = random.uniform(max(miny, tie_xy[1] - half), min(maxy, tie_xy[1] + half))
        diag = np.array([[cx - half, cy - half], [cx + half, cy + half]], dtype=np.float64)
        corners = np.array(
            [
                [diag[0, 0], diag[0, 1]],
                [diag[1, 0], diag[0, 1]],
                [diag[1, 0], diag[1, 1]],
                [diag[0, 0], diag[1, 1]],
            ]
        )
        if all(polygon.contains(Point(pt[0], pt[1])) for pt in corners):
            if abs(cx - tie_xy[0]) <= half and abs(cy - tie_xy[1]) <= half:
                windows.append(diag)
    if len(windows) < k:
        cx, cy = tie_xy
        diag = np.array([[cx - half, cy - half], [cx + half, cy + half]], dtype=np.float64)
        corners = np.array(
            [
                [diag[0, 0], diag[0, 1]],
                [diag[1, 0], diag[0, 1]],
                [diag[1, 0], diag[1, 1]],
                [diag[0, 0], diag[1, 1]],
            ]
        )
        if all(polygon.contains(Point(pt[0], pt[1])) for pt in corners):
            windows.append(diag)
    return windows[:k]


def warp_windows_for_image(image: RSImage, window_diags: List[np.ndarray], output_size: int, rpc=None) -> WindowData:
    diags = np.stack(window_diags, axis=0)
    corners_linesamps = image.convert_diags_to_corners(diags, rpc=rpc)
    imgs, dems, Hs = image.crop_windows(corners_linesamps, output_size=(output_size, output_size))
    return WindowData(image=image, imgs=imgs, dems=dems, Hs=Hs)


def load_model(args) -> Tuple[Encoder, Predictor]:
    model_configs = load_config(args.model_config_path)
    encoder = Encoder(
        dino_weight_path=args.dino_path,
        embed_dim=model_configs["encoder"]["embed_dim"],
        ctx_dim=model_configs["encoder"]["ctx_dim"],
        use_adapter=args.use_adapter,
        use_conf=args.use_conf,
    )
    encoder.load_adapter(args.adapter_path)
    predictor = Predictor(
        corr_levels=model_configs["predictor"]["corr_levels"],
        corr_radius=model_configs["predictor"]["corr_radius"],
        context_dim=model_configs["predictor"]["ctx_dim"],
        hidden_dim=model_configs["predictor"]["hidden_dim"],
        use_mtf=args.use_mtf,
    )
    load_model_state_dict(predictor, args.predictor_path)
    encoder = encoder.to(args.device).eval().half()
    predictor = predictor.to(args.device).eval()
    return encoder, predictor


def merge_affines(affines: torch.Tensor, Hs: torch.Tensor, scores: torch.Tensor, device: str) -> torch.Tensor:
    mat_size = 4
    coords_mat = get_coord_mat(mat_size, mat_size, Hs.shape[0], 16, device)
    coords_mat_flat = coords_mat.flatten(1, 2)
    coords_src = apply_H(coords_mat_flat, torch.linalg.inv(Hs), device)
    coords_dst = apply_M(coords_src, affines, device)
    coords_src_flat = coords_src.reshape(-1, 2)
    coords_dst_flat = coords_dst.reshape(-1, 2)
    scores_norm = scores / scores.mean()
    scores_norm = scores_norm.unsqueeze(-1).expand(-1, mat_size**2).reshape(-1)
    merged_affine = solve_weighted_affine(coords_src_flat, coords_dst_flat, scores_norm)
    return merged_affine


def estimate_affine_model(
    encoder: Encoder,
    predictor: Predictor,
    image_a: RSImage,
    image_b: RSImage,
    diag_a: np.ndarray,
    diag_b: np.ndarray,
    device: str,
    output_resolution: int,
    min_window_size: float,
    quad_split_times: int,
    predictor_iter_num: Optional[int],
) -> Optional[torch.Tensor]:
    rpc_a = deepcopy(image_a.rpc)
    rpc_b = deepcopy(image_b.rpc)
    current_diags_a = np.stack([diag_a], axis=0)
    current_diags_b = np.stack([diag_b], axis=0)
    current_size = float(abs(current_diags_a[0, 1, 0] - current_diags_a[0, 0, 0]))
    last_affine = None
    while current_size >= min_window_size:
        window_a = warp_windows_for_image(
            image_a,
            list(current_diags_a),
            output_resolution,
            rpc=rpc_a,
        )
        window_b = warp_windows_for_image(
            image_b,
            list(current_diags_b),
            output_resolution,
            rpc=rpc_b,
        )
        imgs_a = window_a.imgs
        imgs_b = window_b.imgs
        dems_a = torch.from_numpy(window_a.dems).to(device=device, dtype=torch.float32)
        dems_b = torch.from_numpy(window_b.dems).to(device=device, dtype=torch.float32)
        Hs_a = torch.from_numpy(window_a.Hs).to(device=device, dtype=torch.float32)
        Hs_b = torch.from_numpy(window_b.Hs).to(device=device, dtype=torch.float32)
        feats_a, feats_b = extract_features(encoder, imgs_a, imgs_b, device=device)
        solver = WindowSolver(
            imgs_a.shape[0],
            imgs_a.shape[1],
            imgs_a.shape[2],
            predictor=predictor,
            feats_a=feats_a,
            feats_b=feats_b,
            H_as=Hs_a,
            H_bs=Hs_b,
            rpc_a=rpc_a,
            rpc_b=rpc_b,
            height_a=dems_a,
            height_b=dems_b,
            test_imgs_a=imgs_a,
            test_imgs_b=imgs_b,
            predictor_max_iter=predictor_iter_num,
        )
        preds = solver.solve(flag="ab", final_only=True, return_vis=False)
        _, _, confs_a = feats_a
        _, _, confs_b = feats_b
        scores_a = confs_a.reshape(confs_a.shape[0], -1).mean(dim=1)
        scores_b = confs_b.reshape(confs_b.shape[0], -1).mean(dim=1)
        scores = torch.sqrt(scores_a * scores_b)
        last_affine = merge_affines(preds, Hs_a, scores, device)
        rpc_a.Update_Adjust(last_affine)
        current_diags_a = quadsplit_diags(current_diags_a)
        current_diags_b = quadsplit_diags(current_diags_b)
        if quad_split_times > 1:
            for _ in range(quad_split_times - 1):
                current_diags_a = quadsplit_diags(current_diags_a)
                current_diags_b = quadsplit_diags(current_diags_b)
        current_size = float(abs(current_diags_a[0, 1, 0] - current_diags_a[0, 0, 0]))
    return last_affine


def estimate_affine_baseline(
    matcher,
    window_a: WindowData,
    window_b: WindowData,
    idx_a: int,
    idx_b: int,
    device: str,
) -> Optional[torch.Tensor]:
    img_a = window_a.imgs[idx_a]
    img_b = window_b.imgs[idx_b]
    match_result = matcher.match(img_a, img_b)
    pts_a = match_result.pts0
    pts_b = match_result.pts1
    if pts_a.shape[0] < 3:
        return None
    pts_a_rc = pts_a[:, [1, 0]]
    pts_b_rc = pts_b[:, [1, 0]]
    pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(device)
    pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(device)
    H_a_inv = torch.from_numpy(np.linalg.inv(window_a.Hs[idx_a])).unsqueeze(0).to(device, dtype=torch.float32)
    H_b_inv = torch.from_numpy(np.linalg.inv(window_b.Hs[idx_b])).unsqueeze(0).to(device, dtype=torch.float32)
    pts_a_global = apply_H(pts_a_rc_t, H_a_inv, device).squeeze(0).cpu().numpy()
    pts_b_global = apply_H(pts_b_rc_t, H_b_inv, device).squeeze(0).cpu().numpy()
    lines_b = np.clip(np.round(pts_b_global[:, 0]).astype(int), 0, window_b.image.H - 1)
    samps_b = np.clip(np.round(pts_b_global[:, 1]).astype(int), 0, window_b.image.W - 1)
    heights = window_b.image.dem[lines_b, samps_b]
    lat, lon = window_b.image.rpc.RPC_PHOTO2OBJ(samps_b, lines_b, heights, "numpy")
    samps_a, lines_a = window_a.image.rpc.RPC_OBJ2PHOTO(lat, lon, heights, "numpy")
    pts_b_proj = np.stack([lines_a, samps_a], axis=-1)
    src = torch.from_numpy(pts_a_global).to(device=device, dtype=torch.float32)
    dst = torch.from_numpy(pts_b_proj).to(device=device, dtype=torch.float32)
    scores = torch.ones((src.shape[0],), device=device, dtype=torch.float32)
    return solve_weighted_affine(src, dst, scores)


def evaluate_tiepoint_error(
    image_a: RSImage,
    image_b: RSImage,
    tie_point_a: np.ndarray,
    tie_point_b: np.ndarray,
    affine: torch.Tensor,
    device: str,
) -> float:
    line_a, samp_a = float(tie_point_a[0]), float(tie_point_a[1])
    coords = torch.tensor([[line_a, samp_a]], device=device, dtype=torch.float32)
    affine = affine.to(device=device, dtype=torch.float32)
    adjusted = apply_M(coords.unsqueeze(0), affine.unsqueeze(0), device).squeeze(0)[0]
    line_adj, samp_adj = adjusted[0].item(), adjusted[1].item()
    line_adj_i = int(np.clip(round(line_adj), 0, image_a.H - 1))
    samp_adj_i = int(np.clip(round(samp_adj), 0, image_a.W - 1))
    height = image_a.dem[line_adj_i, samp_adj_i]
    lat, lon = image_a.rpc.RPC_PHOTO2OBJ(samp_adj, line_adj, height, "numpy")
    samp_b, line_b = image_b.rpc.RPC_OBJ2PHOTO(lat, lon, height, "numpy")
    line_b_gt, samp_b_gt = float(tie_point_b[0]), float(tie_point_b[1])
    return float(np.linalg.norm(np.array([line_b, samp_b]) - np.array([line_b_gt, samp_b_gt])))


def process_root(root: str, args) -> Dict[str, List[float]]:
    metas = load_images_meta(root, args)
    images = load_images(metas, args.device)
    if any(image.tie_points is None for image in images):
        raise ValueError(f"Missing tie points in root: {root}")
    lengths = [image.tie_points.shape[0] for image in images]
    if len(set(lengths)) != 1:
        raise ValueError(f"Tie point counts mismatch in root: {root}")
    polygon = compute_common_polygon(images)
    results: Dict[str, List[float]] = {"model": []}
    matchers = {}
    for matcher_name in args.baseline_matchers:
        matcher_args = argparse.Namespace(**vars(args))
        matcher_args.matcher = matcher_name
        matchers[matcher_name] = build_matcher(matcher_args)
        results[f"baseline_{matcher_name}"] = []
    encoder, predictor = load_model(args)
    num_tiepoints = lengths[0]
    for tie_idx in range(num_tiepoints):
        tie_xy = tiepoint_to_object_xy(images[0], images[0].tie_points[tie_idx])
        window_diags = sample_windows(
            tie_xy,
            polygon,
            args.window_size,
            args.window_num,
            args.max_sampling_attempts,
        )
        if len(window_diags) < 2:
            continue
        window_data = [warp_windows_for_image(image, window_diags, args.output_resolution) for image in images]
        for img_idx_a, img_idx_b in itertools.combinations(range(len(images)), 2):
            image_a = images[img_idx_a]
            image_b = images[img_idx_b]
            tie_point_a = image_a.tie_points[tie_idx]
            tie_point_b = image_b.tie_points[tie_idx]
            for idx_a, idx_b in itertools.combinations(range(len(window_diags)), 2):
                print("Processing RAE")
                model_affine = estimate_affine_model(
                    encoder,
                    predictor,
                    image_a,
                    image_b,
                    window_diags[idx_a],
                    window_diags[idx_b],
                    args.device,
                    args.output_resolution,
                    args.min_window_size,
                    args.quad_split_times,
                    args.predictor_iter_num,
                )
                if model_affine is not None:
                    err = evaluate_tiepoint_error(
                        image_a,
                        image_b,
                        tie_point_a,
                        tie_point_b,
                        model_affine,
                        args.device,
                    )
                    results["model"].append(err)
                for matcher_name, matcher in matchers.items():
                    print(f"Processing {matcher_name}")
                    affine = estimate_affine_baseline(
                        matcher,
                        window_data[img_idx_a],
                        window_data[img_idx_b],
                        idx_a,
                        idx_b,
                        args.device,
                    )
                    if affine is None:
                        continue
                    err = evaluate_tiepoint_error(
                        image_a,
                        image_b,
                        tie_point_a,
                        tie_point_b,
                        affine,
                        args.device,
                    )
                    results[f"baseline_{matcher_name}"].append(err)
    return results


def print_report(results: Dict[str, List[float]], title: str) -> None:
    print(f"\n--- {title} ---")
    for key, values in results.items():
        if len(values) == 0:
            print(f"{key}: no valid samples")
            continue
        report = get_report_dict(np.array(values))
        print(
            f"{key}: count={report['count']} mean={report['mean']:.4f} median={report['median']:.4f} "
            f"rmse={report['rmse']:.4f} max={report['max']:.4f} <1pix={report['<1pix_percent']:.2f}% "
            f"<3pix={report['<3pix_percent']:.2f}% <5pix={report['<5pix_percent']:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=str, required=True)
    parser.add_argument("--select_imgs", type=str, default="-1")
    parser.add_argument("--window_size", type=float, required=True)
    parser.add_argument("--min_window_size", type=float, default=500)
    parser.add_argument("--window_num", type=int, required=True)
    parser.add_argument("--output_resolution", type=int, default=512)
    parser.add_argument("--max_sampling_attempts", type=int, default=200)
    parser.add_argument("--quad_split_times", type=int, default=1)
    parser.add_argument("--baseline_matchers", type=str, default="loftr")
    parser.add_argument("--dino_path", type=str, default="weights")
    parser.add_argument("--adapter_path", type=str, default="weights/adapter.pth")
    parser.add_argument("--predictor_path", type=str, default="weights/predictor.pth")
    parser.add_argument("--model_config_path", type=str, default="configs/model_config.yaml")
    parser.add_argument("--predictor_iter_num", type=int, default=None)
    parser.add_argument("--use_adapter", type=bool, default=True)
    parser.add_argument("--use_conf", type=bool, default=True)
    parser.add_argument("--use_mtf", type=bool, default=True)
    parser.add_argument("--loftr_weight_path", type=str, default=None)
    parser.add_argument("--sift_ratio_thresh", type=float, default=0.75)
    parser.add_argument("--fm_ransac_thresh", type=float, default=3.0)
    parser.add_argument("--fm_confidence", type=float, default=0.99)
    parser.add_argument("--superpoint_weight_path", type=str, default=None)
    parser.add_argument("--superglue_weight_path", type=str, default=None)
    parser.add_argument("--aspanformer_config_path", type=str, default=None)
    parser.add_argument("--aspanformer_weight_path", type=str, default=None)
    parser.add_argument("--roma_variant", type=str, default="outdoor")
    parser.add_argument("--roma_weight_path", type=str, default=None)
    parser.add_argument("--roma_dinov2_weight_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.baseline_matchers = [m.strip() for m in args.baseline_matchers.split(",") if m.strip()]
    init_random_seed(args.random_seed)

    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    all_results: Dict[str, List[float]] = {}
    for root in roots:
        print(f"ROOT = {root}")
        root_results = process_root(root, args)
        print_report(root_results, f"Root {root}")
        for key, values in root_results.items():
            all_results.setdefault(key, []).extend(values)
    print_report(all_results, "All Roots")


if __name__ == "__main__":
    main()
