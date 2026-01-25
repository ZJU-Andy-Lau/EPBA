import sys
import os

sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
from typing import List, Tuple
import csv
import time

import cv2
import torch
from torchvision.transforms import v2

from shared.utils import str2bool, get_current_time, load_config
from infer.utils import find_intersection, find_squares, apply_H
from infer.rs_image import RSImage, RSImageMeta
from baseline.matchers import build_matcher
from model.encoder import Encoder
from tqdm import tqdm

def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_meta(args, root: str, image_id: int) -> RSImageMeta:
    return RSImageMeta(args, root, image_id, args.device)


def get_overlap_windows(args, image_a: RSImage, image_b: RSImage):
    corners_a = image_a.corner_xys
    corners_b = image_b.corner_xys
    polygon_corners = find_intersection(np.stack([corners_a, corners_b], axis=0))
    window_diags = find_squares(polygon_corners, args.max_window_size, args.min_window_size, args.min_cover_area_ratio)
    if len(window_diags) == 0:
        return None
    if args.max_window_num > 0 and len(window_diags) > args.max_window_num:
        idxs = np.random.choice(range(len(window_diags)), args.max_window_num, replace=False)
        window_diags = window_diags[idxs]
    corners_linesamps_a = image_a.convert_diags_to_corners(window_diags)
    corners_linesamps_b = image_b.convert_diags_to_corners(window_diags, image_b.rpc)
    imgs_a, _, Hs_a = image_a.crop_windows(corners_linesamps_a, output_size=(args.crop_size, args.crop_size))
    imgs_b, _, Hs_b = image_b.crop_windows(corners_linesamps_b, output_size=(args.crop_size, args.crop_size))
    return imgs_a, imgs_b, Hs_a, Hs_b


def load_encoder(args):
    model_configs = load_config(args.model_config_path)
    encoder = Encoder(
        dino_weight_path=args.dino_path,
        embed_dim=model_configs['encoder']['embed_dim'],
        ctx_dim=model_configs['encoder']['ctx_dim'],
        use_adapter=args.use_adapter,
        use_conf=args.use_conf,
    )
    encoder.load_adapter(args.adapter_path)
    if torch.cuda.is_available():
        encoder = encoder.to(args.device).eval().half()
    else:
        encoder = encoder.to(args.device).eval()
    return encoder


def extract_conf_maps(encoder: Encoder, img_a: np.ndarray, img_b: np.ndarray, device: str):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input_a = torch.from_numpy(img_a).permute(2, 0, 1).contiguous().unsqueeze(0).to(device, non_blocking=True)
    input_b = torch.from_numpy(img_b).permute(2, 0, 1).contiguous().unsqueeze(0).to(device, non_blocking=True)
    input_a = transform(input_a)
    input_b = transform(input_b)
    if torch.cuda.is_available():
        input_a = input_a.half()
        input_b = input_b.half()
        ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
    else:
        ctx = torch.autocast(device_type="cpu", dtype=torch.float32)
    with ctx:
        feats_a, feats_b = encoder(input_a, input_b)
    conf_a = feats_a[2].detach().to(torch.float32)
    conf_b = feats_b[2].detach().to(torch.float32)
    return conf_a, conf_b


def make_conf_overlay(image: np.ndarray, conf_map: np.ndarray, alpha: float):
    h, w = image.shape[:2]
    conf_resized = cv2.resize(conf_map, (w, h), interpolation=cv2.INTER_LINEAR)
    conf_norm = np.clip(conf_resized, 0.0, 1.0)
    color = np.zeros((h, w, 3), dtype=np.float32)
    color[..., 1] = conf_norm * 255.0
    color[..., 2] = (1.0 - conf_norm) * 255.0
    overlay = image.astype(np.float32) * (1.0 - alpha) + color * alpha
    return overlay.astype(np.uint8), conf_norm


def draw_points(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], radius: int = 2):
    if points is None or len(points) == 0:
        return image
    out = image.copy()
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1)
    return out


def window_to_global(coords_xy: np.ndarray, H_inv: np.ndarray, device: str):
    if len(coords_xy) == 0:
        return coords_xy
    coords_rc = coords_xy[:, [1, 0]]
    coords_rc_t = torch.from_numpy(coords_rc).unsqueeze(0).to(device, dtype=torch.float32)
    H_inv_t = torch.from_numpy(H_inv).unsqueeze(0).to(device, dtype=torch.float32)
    coords_global_rc = apply_H(coords_rc_t, H_inv_t, device).squeeze(0).cpu().numpy()
    return coords_global_rc[:, [1, 0]]


def global_to_window(coords_xy: np.ndarray, H: np.ndarray, device: str):
    if len(coords_xy) == 0:
        return coords_xy
    coords_rc = coords_xy[:, [1, 0]]
    coords_rc_t = torch.from_numpy(coords_rc).unsqueeze(0).to(device, dtype=torch.float32)
    H_t = torch.from_numpy(H).unsqueeze(0).to(device, dtype=torch.float32)
    coords_win_rc = apply_H(coords_rc_t, H_t, device).squeeze(0).cpu().numpy()
    return coords_win_rc[:, [1, 0]]


def project_b_inliers_to_a(image_a: RSImage, image_b: RSImage, b_points_global_xy: np.ndarray):
    if len(b_points_global_xy) == 0:
        return np.empty((0, 2))
    samps = b_points_global_xy[:, 0]
    lines = b_points_global_xy[:, 1]
    lines_i = np.clip(np.round(lines).astype(int), 0, image_b.H - 1)
    samps_i = np.clip(np.round(samps).astype(int), 0, image_b.W - 1)
    heights = image_b.dem[lines_i, samps_i]
    lats, lons = image_b.rpc.RPC_PHOTO2OBJ(samps, lines, heights, 'numpy')
    samps_a, lines_a = image_a.rpc.RPC_OBJ2PHOTO(lats, lons, heights, 'numpy')
    return np.stack([samps_a, lines_a], axis=-1)


def estimate_affine(src_xy: np.ndarray, dst_xy: np.ndarray):
    if len(src_xy) < 3:
        return None
    M, _ = cv2.estimateAffine2D(src_xy, dst_xy, ransacReprojThreshold=1e5)
    if M is not None:
        return M
    A = np.concatenate([src_xy, np.ones((src_xy.shape[0], 1))], axis=1)
    bx = dst_xy[:, 0]
    by = dst_xy[:, 1]
    params_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    return np.stack([params_x, params_y], axis=0)


def apply_affine(points_xy: np.ndarray, M: np.ndarray):
    if M is None or len(points_xy) == 0:
        return np.empty((0, 2))
    pts_h = np.concatenate([points_xy, np.ones((points_xy.shape[0], 1))], axis=1)
    return (M @ pts_h.T).T


def compute_residuals(points_a: np.ndarray, points_b: np.ndarray):
    if len(points_a) == 0 or len(points_b) == 0:
        return np.array([])
    return np.linalg.norm(points_a - points_b, axis=1)


def write_stats(path: str, low_ratio: float, high_ratio: float, residuals: np.ndarray):
    mean = float(np.mean(residuals)) if residuals.size > 0 else 0.0
    median = float(np.median(residuals)) if residuals.size > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size > 0 else 0.0
    with open(path, "w") as f:
        f.write(f"low_conf_ratio: {low_ratio:.6f}\n")
        f.write(f"high_conf_ratio: {high_ratio:.6f}\n")
        f.write(f"residual_mean: {mean:.6f}\n")
        f.write(f"residual_median: {median:.6f}\n")
        f.write(f"residual_rmse: {rmse:.6f}\n")


def sample_conf_values(conf_norm: np.ndarray, points_xy: np.ndarray):
    if len(points_xy) == 0:
        return np.array([])
    h, w = conf_norm.shape[:2]
    xs = np.clip(np.round(points_xy[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points_xy[:, 1]).astype(int), 0, h - 1)
    return conf_norm[ys, xs]


def main(args):
    init_random_seed(args.random_seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    image_a = RSImage(load_image_meta(args, args.view_a, 0), device=args.device)
    image_b = RSImage(load_image_meta(args, args.view_b, 1), device=args.device)

    matcher = build_matcher(args)
    encoder = load_encoder(args)

    window_data = get_overlap_windows(args, image_a, image_b)
    if window_data is None:
        return
    imgs_a, imgs_b, Hs_a, Hs_b = window_data
    total_windows = len(imgs_a)
    if total_windows == 0:
        return
    if args.num_windows > 0 and args.num_windows < total_windows:
        window_indices = np.random.choice(range(total_windows), args.num_windows, replace=False)
    else:
        window_indices = range(total_windows)

    os.makedirs(args.output_path, exist_ok=True)
    csv_rows = []
    good_results = []

    for w_idx, idx in enumerate(tqdm(window_indices)):
        img_a_win = imgs_a[idx]
        img_b_win = imgs_b[idx]
        H_a = Hs_a[idx]
        H_b = Hs_b[idx]
        H_a_inv = np.linalg.inv(H_a)
        H_b_inv = np.linalg.inv(H_b)

        match_result = matcher.match(img_a_win, img_b_win)
        pts_a = match_result.pts0
        pts_b = match_result.pts1
        if pts_a.shape[0] < 4:
            continue

        conf_a, conf_b = extract_conf_maps(encoder, img_a_win, img_b_win, args.device)
        overlay_a, conf_a_norm = make_conf_overlay(img_a_win, conf_a.squeeze(0).squeeze(0).cpu().numpy(), args.conf_alpha)
        overlay_b, conf_b_norm = make_conf_overlay(img_b_win, conf_b.squeeze(0).squeeze(0).cpu().numpy(), args.conf_alpha)

        window_dir = os.path.join(args.output_path, f"window_{w_idx:03d}")
        os.makedirs(window_dir, exist_ok=True)

        for threshold in range(1, 10, 2):
            H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, float(threshold))
            if mask is None:
                inliers = np.zeros((pts_a.shape[0],), dtype=bool)
            else:
                inliers = mask.ravel().astype(bool)
            pts_a_in = pts_a[inliers]
            pts_b_in = pts_b[inliers]
            pts_a_out = pts_a[~inliers]
            pts_b_out = pts_b[~inliers]

            overlay_a_pts = draw_points(overlay_a, pts_a_in, (0, 255, 0), args.point_radius)
            overlay_a_pts = draw_points(overlay_a_pts, pts_a_out, (0, 0, 255), args.point_radius)
            overlay_b_pts = draw_points(overlay_b, pts_b_in, (0, 255, 0), args.point_radius)
            overlay_b_pts = draw_points(overlay_b_pts, pts_b_out, (0, 0, 255), args.point_radius)

            pts_b_global = window_to_global(pts_b_in, H_b_inv, args.device)
            pts_a_proj_global = project_b_inliers_to_a(image_a, image_b, pts_b_global)
            pts_a_proj_win = global_to_window(pts_a_proj_global, H_a, args.device)

            pts_a_in_win = pts_a_in
            affine_M = estimate_affine(pts_a_in_win, pts_a_proj_win)
            pts_a_warp = apply_affine(pts_a_in_win, affine_M)
            residuals = compute_residuals(pts_a_warp, pts_a_proj_win)

            conf_vals = sample_conf_values(conf_a_norm, pts_a_in_win)
            low_mask = conf_vals <= args.conf_thresh
            high_mask = conf_vals > args.conf_thresh
            low_ratio = float(low_mask.sum() / max(len(conf_vals), 1))
            high_ratio = float(high_mask.sum() / max(len(conf_vals), 1))

            out_dir = os.path.join(window_dir, f"threshold_{threshold}")
            os.makedirs(out_dir, exist_ok=True)

            cv2.imwrite(os.path.join(out_dir, "overlay_A.png"), overlay_a_pts)
            cv2.imwrite(os.path.join(out_dir, "overlay_B.png"), overlay_b_pts)

            img_a_inliers = draw_points(img_a_win, pts_a_in_win, (0, 255, 255), args.point_radius)
            img_a_inliers = draw_points(img_a_inliers, pts_a_proj_win, (255, 255, 0), args.point_radius)
            cv2.imwrite(os.path.join(out_dir, "proj_inliers_A.png"), img_a_inliers)

            if affine_M is not None:
                warped_a = cv2.warpAffine(img_a_win, affine_M, (img_a_win.shape[1], img_a_win.shape[0]))
            else:
                warped_a = img_a_win.copy()
            warped_a = draw_points(warped_a, pts_a_warp, (0, 255, 255), args.point_radius)
            warped_a = draw_points(warped_a, pts_a_proj_win, (255, 255, 0), args.point_radius)
            cv2.imwrite(os.path.join(out_dir, "warp_residual_A.png"), warped_a)

            write_stats(os.path.join(out_dir, "stats.txt"), low_ratio, high_ratio, residuals)

            csv_rows.append({
                "window_index": w_idx,
                "threshold": threshold,
                "num_matches": int(pts_a.shape[0]),
                "num_inliers": int(pts_a_in.shape[0]),
                "low_conf_ratio": low_ratio,
                "high_conf_ratio": high_ratio,
                "residual_mean": float(np.mean(residuals)) if residuals.size > 0 else 0.0,
                "residual_median": float(np.median(residuals)) if residuals.size > 0 else 0.0,
                "residual_rmse": float(np.sqrt(np.mean(residuals ** 2))) if residuals.size > 0 else 0.0,
            })
            if low_ratio > 0.6 and np.mean(residuals) > 5. :
                good_results.append({
                    'window_index':w_idx,
                    'threshold':threshold,
                    'low_conf_ratio':low_ratio,
                    'residual_mean':np.mean(residuals)
                })

    csv_path = os.path.join(args.output_path, "summary.csv")
    fieldnames = [
        "window_index",
        "threshold",
        "num_matches",
        "num_inliers",
        "low_conf_ratio",
        "high_conf_ratio",
        "residual_mean",
        "residual_median",
        "residual_rmse",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    for result in good_results:
        print(result)
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--view_a', type=str, required=True)
    parser.add_argument('--view_b', type=str, required=True)

    parser.add_argument('--matcher', type=str, default='loftr', choices=['loftr'])
    parser.add_argument('--loftr_weight_path', type=str, default=None)
    parser.add_argument('--fm_ransac_thresh', type=float, default=3.0)
    parser.add_argument('--fm_confidence', type=float, default=0.99)

    parser.add_argument('--max_window_size', type=int, default=2000)
    parser.add_argument('--min_window_size', type=int, default=500)
    parser.add_argument('--max_window_num', type=int, default=1024)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--num_windows', type=int, default=10)

    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--conf_alpha', type=float, default=0.7)

    parser.add_argument('--dino_path', type=str, default='weights')
    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--use_adapter', type=str2bool, default=True)
    parser.add_argument('--use_conf', type=str2bool, default=True)

    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--usgs_dem', type=str2bool, default=False)
    parser.add_argument('--point_radius', type=int, default=2)

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()

    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]', get_current_time())

    args.output_path = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)
