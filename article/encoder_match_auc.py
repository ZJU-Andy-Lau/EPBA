import sys
import os

sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
import csv

import cv2
import torch
from torchvision.transforms import v2

from shared.utils import str2bool, get_current_time, load_config
from model.encoder import Encoder


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image(path: str, size: int):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.stack([img] * 3,axis=-1)
    return img


def apply_affine(img: np.ndarray, M: np.ndarray):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, M, (w, h))


def build_random_affine(seed: int, max_rotate: float, max_scale_delta: float, max_shift: float):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-max_rotate, max_rotate)
    scale = rng.uniform(1.0 - max_scale_delta, 1.0 + max_scale_delta)
    shift_x = rng.uniform(-max_shift, max_shift)
    shift_y = rng.uniform(-max_shift, max_shift)
    center = (0.5, 0.5)
    M = cv2.getRotationMatrix2D((center[0], center[1]), angle, scale)
    M[0, 2] += shift_x
    M[1, 2] += shift_y
    return M


def load_encoder(args):
    model_configs = load_config(args.model_config_path)
    encoder = Encoder(
        dino_weight_path=args.dino_path,
        embed_dim=model_configs['encoder']['embed_dim'],
        ctx_dim=model_configs['encoder']['ctx_dim'],
    )
    encoder.load_adapter(args.adapter_path)
    if torch.cuda.is_available():
        encoder = encoder.to(args.device).eval().half()
    else:
        encoder = encoder.to(args.device).eval()
    return encoder


def encode_features(encoder: Encoder, img_a: np.ndarray, img_b: np.ndarray, device: str):
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
    match_a = feats_a[0].detach().to(torch.float32)
    conf_a = feats_a[2].detach().to(torch.float32)
    match_b = feats_b[0].detach().to(torch.float32)
    conf_b = feats_b[2].detach().to(torch.float32)
    return match_a, conf_a, match_b, conf_b


def get_high_conf_coords(conf_map: np.ndarray, thresh: float):
    mask = conf_map > thresh
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=-1)
    return coords


def map_coords(coords: np.ndarray, src_size: int, feat_h: int, feat_w: int):
    if len(coords) == 0:
        return coords
    xs = np.clip(np.round(coords[:, 0] / float(src_size) * feat_w).astype(int), 0, feat_w - 1)
    ys = np.clip(np.round(coords[:, 1] / float(src_size) * feat_h).astype(int), 0, feat_h - 1)
    return np.stack([xs, ys], axis=-1)


def apply_affine_coords(coords: np.ndarray, M: np.ndarray):
    if len(coords) == 0:
        return coords
    pts_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], axis=1)
    warped = (M @ pts_h.T).T
    return warped


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray):
    if vec_a.ndim == 1:
        vec_a = vec_a[None]
    if vec_b.ndim == 1:
        vec_b = vec_b[None]
    a_norm = vec_a / np.linalg.norm(vec_a, axis=1, keepdims=True).clip(min=1e-12)
    b_norm = vec_b / np.linalg.norm(vec_b, axis=1, keepdims=True).clip(min=1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def compute_auc(pos_scores: np.ndarray, neg_scores: np.ndarray):
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels == 1]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main(args):
    init_random_seed(args.random_seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    img_a = load_image(args.view_a, args.image_size)
    img_b = load_image(args.view_b, args.image_size)

    if args.apply_warp:
        M = build_random_affine(args.random_seed, args.warp_rotate, args.warp_scale_delta, args.warp_shift)
        img_b = apply_affine(img_b, M)
    else:
        M = None

    encoder = load_encoder(args)
    match_a, conf_a, match_b, conf_b = encode_features(encoder, img_a, img_b, args.device)

    conf_a_np = conf_a.squeeze(0).squeeze(0).cpu().numpy()
    conf_b_np = conf_b.squeeze(0).squeeze(0).cpu().numpy()
    feat_h, feat_w = conf_a_np.shape[-2:]
    coords_feat = get_high_conf_coords(conf_a_np, args.conf_thresh)
    if coords_feat.shape[0] == 0:
        return
    coords_img = np.stack([
        coords_feat[:, 0] / float(feat_w) * args.image_size,
        coords_feat[:, 1] / float(feat_h) * args.image_size
    ], axis=-1)

    if args.max_points > 0 and coords_img.shape[0] > args.max_points:
        idxs = np.random.choice(coords_img.shape[0], args.max_points, replace=False)
        coords_img = coords_img[idxs]

    if M is not None:
        coords_img_b = apply_affine_coords(coords_img, M)
    else:
        coords_img_b = coords_img.copy()

    coords_feat_a = map_coords(coords_img, args.image_size, feat_h, feat_w)
    coords_feat_b = map_coords(coords_img_b, args.image_size, feat_h, feat_w)
    valid = (
        (coords_feat_b[:, 0] >= 0)
        & (coords_feat_b[:, 0] < feat_w)
        & (coords_feat_b[:, 1] >= 0)
        & (coords_feat_b[:, 1] < feat_h)
    )
    coords_feat_a = coords_feat_a[valid]
    coords_feat_b = coords_feat_b[valid]
    if coords_feat_a.shape[0] == 0:
        return

    feat_a_np = match_a.squeeze(0).cpu().numpy()
    feat_b_np = match_b.squeeze(0).cpu().numpy()
    vec_a = feat_a_np[:, coords_feat_a[:, 1], coords_feat_a[:, 0]].T
    vec_b = feat_b_np[:, coords_feat_b[:, 1], coords_feat_b[:, 0]].T
    pos_scores = cosine_similarity(vec_a, vec_b)

    neg_scores_list = []
    for i in range(vec_a.shape[0]):
        neg_indices = np.random.choice(vec_b.shape[0], args.neg_k, replace=vec_b.shape[0] < args.neg_k)
        neg_vecs = vec_b[neg_indices]
        neg_scores = cosine_similarity(vec_a[i], neg_vecs)
        neg_scores_list.append(neg_scores)
    neg_scores = np.concatenate(neg_scores_list) if neg_scores_list else np.array([])

    auc = compute_auc(pos_scores, neg_scores)

    print(f"AUC: {auc:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--view_a', type=str, required=True)
    parser.add_argument('--view_b', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--apply_warp', type=str2bool, default=False)
    parser.add_argument('--warp_rotate', type=float, default=3.0)
    parser.add_argument('--warp_scale_delta', type=float, default=0.02)
    parser.add_argument('--warp_shift', type=float, default=5.0)

    parser.add_argument('--conf_thresh', type=float, default=0.5)
    parser.add_argument('--neg_k', type=int, default=10)
    parser.add_argument('--max_points', type=int, default=5000)

    parser.add_argument('--dino_path', type=str, default='weights')
    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    main(args)
