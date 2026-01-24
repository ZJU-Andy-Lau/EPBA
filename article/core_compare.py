import argparse
import csv
import json
import os
import random
import math
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

from shared.utils import load_model_state_dict, load_config
from shared.visualize import make_checkerboard
from infer.utils import extract_features, get_coord_mat, apply_H, apply_M, solve_weighted_affine
from model.encoder import Encoder
from model.predictor import Predictor
from solve.solve_windows import WindowSolver


def rc_to_xy_matrix(M_rc: np.ndarray) -> np.ndarray:
    M_xy = M_rc.copy()
    M_xy[[0, 1], :] = M_xy[[1, 0], :]
    M_xy[:, [0, 1]] = M_xy[:, [1, 0]]
    return M_xy


def xy_to_rc_matrix(M_xy: np.ndarray) -> np.ndarray:
    M_rc = M_xy.copy()
    M_rc[[0, 1], :] = M_rc[[1, 0], :]
    M_rc[:, [0, 1]] = M_rc[:, [1, 0]]
    return M_rc


def invert_affine_rc(M_rc: np.ndarray) -> np.ndarray:
    M_h = np.eye(3, dtype=np.float64)
    M_h[:2, :] = M_rc
    M_inv = np.linalg.inv(M_h)
    return M_inv[:2, :]


def apply_affine_rc(M_rc: np.ndarray, pts_rc: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_rc, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    return pts_h @ M_rc.T


def merge_affine_rc(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_h = np.eye(3, dtype=np.float64)
    B_h = np.eye(3, dtype=np.float64)
    A_h[:2, :] = A
    B_h[:2, :] = B
    C_h = B_h @ A_h
    return C_h[:2, :]


def random_affine_rc(rng: np.random.Generator, max_translate=50.0, max_rotate_deg=10.0, max_scale=0.1, max_shear=0.05) -> np.ndarray:
    tx = rng.uniform(-max_translate, max_translate)
    ty = rng.uniform(-max_translate, max_translate)
    theta = np.deg2rad(rng.uniform(-max_rotate_deg, max_rotate_deg))
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale)
    shx = rng.uniform(-max_shear, max_shear)
    shy = rng.uniform(-max_shear, max_shear)

    cos_t = math.cos(theta) * scale
    sin_t = math.sin(theta) * scale

    A = np.array([
        [cos_t + shx * sin_t, -sin_t + shx * cos_t, ty],
        [shy * cos_t + sin_t, -shy * sin_t + cos_t, tx],
    ], dtype=np.float64)

    return A


def build_grid_rc(h: int, w: int, grid_size: int = 32) -> np.ndarray:
    rows = np.linspace(0, h - 1, grid_size)
    cols = np.linspace(0, w - 1, grid_size)
    grid = np.stack(np.meshgrid(rows, cols, indexing='ij'), axis=-1)
    return grid.reshape(-1, 2)


def compute_transform_error(M_pred_rc: np.ndarray, M_true_rc: np.ndarray, h: int, w: int, grid_size: int = 32) -> dict:
    pts = build_grid_rc(h, w, grid_size)
    pred = apply_affine_rc(M_pred_rc, pts)
    true = apply_affine_rc(M_true_rc, pts)
    d = np.linalg.norm(pred - true, axis=1)
    return {
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "max": float(np.max(d)),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
    }


def load_image_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(1024,1024))
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_image_gray(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(path, img)


def estimate_affine_loftr(loftr_model, img_a: np.ndarray, img_b: np.ndarray, device: str, ransac_threshold:float) -> np.ndarray:
    img0 = torch.from_numpy(img_a).float().to(device) / 255.0
    img1 = torch.from_numpy(img_b).float().to(device) / 255.0
    batch = {"image0": img0.unsqueeze(0).unsqueeze(0), "image1": img1.unsqueeze(0).unsqueeze(0)}
    with torch.no_grad():
        correspondences = loftr_model(batch)
    mkpts0 = correspondences["keypoints0"].cpu().numpy()
    mkpts1 = correspondences["keypoints1"].cpu().numpy()
    if mkpts0.shape[0] < 10:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    M_xy, _ = cv2.estimateAffine2D(mkpts0, mkpts1, ransacReprojThreshold=ransac_threshold)
    if M_xy is None:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    return M_xy.astype(np.float64)

@torch.no_grad()
def predict_affine_project(encoder: Encoder, predictor: Predictor, img_a: np.ndarray, img_b: np.ndarray, device: str, predictor_iter_num: int, min_window: int) -> np.ndarray:
    h, w = img_a.shape[:2]
    M_current = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    window_size = h // 2
    while window_size >= min_window:
        M_current_xy = rc_to_xy_matrix(M_current)
        img_a_warped = cv2.warpAffine(img_a, M_current_xy, (w, h), flags=cv2.INTER_LINEAR)
        tiles_a = []
        tiles_b = []
        Hs_a = []
        Hs_b = []
        for r in range(0, h, window_size):
            for c in range(0, w, window_size):
                tile_a = img_a_warped[r:r + window_size, c:c + window_size]
                tile_b = img_b[r:r + window_size, c:c + window_size]
                if tile_a.shape[0] != window_size or tile_a.shape[1] != window_size:
                    continue
                if window_size != 512:
                    tile_a = cv2.resize(tile_a, (512, 512), interpolation=cv2.INTER_LINEAR)
                    tile_b = cv2.resize(tile_b, (512, 512), interpolation=cv2.INTER_LINEAR)
                tiles_a.append(np.stack([tile_a] * 3, axis=-1))
                tiles_b.append(np.stack([tile_b] * 3, axis=-1))
                scale = float(window_size) / 512.0
                H = np.array([[1.0 / scale, 0.0, -float(r) / scale], [0.0, 1.0 / scale, -float(c) / scale], [0.0, 0.0, 1.0]], dtype=np.float32)
                Hs_a.append(H)
                Hs_b.append(H)
        imgs_a = np.stack(tiles_a, axis=0)
        imgs_b = np.stack(tiles_b, axis=0)
        feats_a, feats_b = extract_features(encoder, imgs_a, imgs_b, device=device)
        B, H, W = imgs_a.shape[:3]
        H_as = torch.from_numpy(np.stack(Hs_a, axis=0)).to(device=device, dtype=torch.float32)
        H_bs = torch.from_numpy(np.stack(Hs_b, axis=0)).to(device=device, dtype=torch.float32)
        solver = WindowSolver(B, H, W, predictor, feats_a, feats_b, H_as, H_bs, rpc_a=None, rpc_b=None, height_a=None, height_b=None, predictor_max_iter=predictor_iter_num)
        preds = solver.solve(flag="ab", final_only=True)
        coords_mat = get_coord_mat(32, 32, B, 16, device=device)
        coords_mat_flat = coords_mat.flatten(1, 2)
        coords_src = apply_H(coords=coords_mat_flat, Hs=torch.linalg.inv(H_as), device=device)
        coords_dst = apply_M(coords=coords_src, Ms=preds, device=device)
        coords_src_flat = coords_src.reshape(-1, 2)
        coords_dst_flat = coords_dst.reshape(-1, 2)
        scores = torch.ones((B,), dtype=coords_src.dtype, device=device)
        scores_norm = scores / scores.mean()
        scores_norm = scores_norm.unsqueeze(-1).expand(-1, 1024).reshape(-1)
        merged_affine = solve_weighted_affine(coords_src_flat, coords_dst_flat, scores_norm)
        M_delta = merged_affine.detach().cpu().numpy()
        M_current = merge_affine_rc(M_current, M_delta)
        window_size = window_size // 2
    return M_current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs_core_compare")
    parser.add_argument("--dino_path", type=str, default="weights")
    parser.add_argument("--adapter_path", type=str, default="weights/adapter.pth")
    parser.add_argument("--predictor_path", type=str, default="weights/predictor.pth")
    parser.add_argument("--model_config_path", type=str, default="configs/model_config.yaml")
    parser.add_argument("--predictor_iter_num", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loftr_weight_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_window", type=int, default=128)
    parser.add_argument('--ransac_threshold',type=float, default=3.)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_configs = load_config(args.model_config_path)
    encoder = Encoder(dino_weight_path=args.dino_path, embed_dim=model_configs['encoder']['embed_dim'], ctx_dim=model_configs['encoder']['ctx_dim'], use_adapter=True, use_conf=True)
    encoder.load_adapter(args.adapter_path)
    predictor = Predictor(corr_levels=model_configs['predictor']['corr_levels'], corr_radius=model_configs['predictor']['corr_radius'], context_dim=model_configs['predictor']['ctx_dim'], hidden_dim=model_configs['predictor']['hidden_dim'], use_mtf=True)
    load_model_state_dict(predictor, args.predictor_path)

    predictor_iter_num = args.predictor_iter_num
    if predictor_iter_num is None:
        predictor_iter_num = model_configs['predictor']['iter_num']

    encoder = encoder.to(args.device).eval()
    predictor = predictor.to(args.device).eval()

    try:
        from kornia.feature import LoFTR
    except Exception as exc:
        raise RuntimeError("kornia is required for LoFTR") from exc

    if args.loftr_weight_path is not None:
        loftr_model = LoFTR(pretrained=None)
        checkpoint = torch.load(args.loftr_weight_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        loftr_model.load_state_dict(state_dict)
    else:
        loftr_model = LoFTR(pretrained="outdoor")
    loftr_model = loftr_model.to(args.device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_rows = []

    a_paths = sorted(glob(os.path.join(args.input_dir, "*_a.png")))
    for a_path in tqdm(a_paths):
        name = os.path.basename(a_path).replace("_a.png", "")
        b_path = os.path.join(args.input_dir, f"{name}_b.png")
        if not os.path.exists(b_path):
            continue

        img_a = load_image_gray(a_path)
        img_b = load_image_gray(b_path)
        if img_a.shape[:2] != (1024, 1024) or img_b.shape[:2] != (1024, 1024):
            raise ValueError(f"{name} size mismatch")

        rng = np.random.default_rng(args.seed + hash(name) % 100000)
        M_gt_rc = random_affine_rc(rng)
        M_gt_xy = rc_to_xy_matrix(M_gt_rc)
        img_a_warp = cv2.warpAffine(img_a, M_gt_xy, (img_a.shape[1], img_a.shape[0]), flags=cv2.INTER_LINEAR)
        M_gt_inv_rc = invert_affine_rc(M_gt_rc)

        M_pred_project_rc = predict_affine_project(encoder, predictor, img_a_warp, img_b, args.device, predictor_iter_num, args.min_window)
        M_pred_loftr_xy = estimate_affine_loftr(loftr_model, img_a_warp, img_b, args.device, args.ransac_threshold)
        M_pred_loftr_rc = xy_to_rc_matrix(M_pred_loftr_xy)

        error_project = compute_transform_error(M_pred_project_rc, M_gt_inv_rc, img_a.shape[0], img_a.shape[1])
        error_loftr = compute_transform_error(M_pred_loftr_rc, M_gt_inv_rc, img_a.shape[0], img_a.shape[1])

        out_dir = os.path.join(args.output_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        save_image_gray(os.path.join(out_dir, "a.png"), img_a)
        save_image_gray(os.path.join(out_dir, "b.png"), img_b)
        save_image_gray(os.path.join(out_dir, "a_warp.png"), img_a_warp)

        M_pred_project_xy = rc_to_xy_matrix(M_pred_project_rc)
        warped_project = cv2.warpAffine(img_a_warp, M_pred_project_xy, (img_b.shape[1], img_b.shape[0]), flags=cv2.INTER_LINEAR)
        checker_project = make_checkerboard(warped_project, img_b, num_tiles=16)
        save_image_gray(os.path.join(out_dir, "checker_project.png"), checker_project)

        M_pred_loftr_xy = rc_to_xy_matrix(M_pred_loftr_rc)
        warped_loftr = cv2.warpAffine(img_a_warp, M_pred_loftr_xy, (img_b.shape[1], img_b.shape[0]), flags=cv2.INTER_LINEAR)
        checker_loftr = make_checkerboard(warped_loftr, img_b, num_tiles=16)
        save_image_gray(os.path.join(out_dir, "checker_loftr.png"), checker_loftr)

        with open(os.path.join(out_dir, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump({
                "M_gt_rc": M_gt_rc.tolist(),
                "M_gt_inv_rc": M_gt_inv_rc.tolist(),
                "M_pred_project_rc": M_pred_project_rc.tolist(),
                "M_pred_loftr_rc": M_pred_loftr_rc.tolist(),
                "error_project": error_project,
                "error_loftr": error_loftr,
            }, f, ensure_ascii=False, indent=2)

        summary_rows.append([
            name,
            error_project["mean"],
            error_project["median"],
            error_project["max"],
            error_project["rmse"],
            error_loftr["mean"],
            error_loftr["median"],
            error_loftr["max"],
            error_loftr["rmse"],
        ])

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "name",
            "project_mean",
            "project_median",
            "project_max",
            "project_rmse",
            "loftr_mean",
            "loftr_median",
            "loftr_max",
            "loftr_rmse",
        ])
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
