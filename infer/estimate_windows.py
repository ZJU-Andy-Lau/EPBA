import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import random
import itertools
import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from shapely.geometry import Polygon, Point, box

from model.encoder import Encoder
from model.predictor import Predictor
from shared.utils import str2bool, get_current_time, load_model_state_dict, load_config, project_mercator
from shared.rpc import project_linesamp
from infer.utils import is_overlap
from infer.rs_image import RSImage, RSImageMeta
from infer.pair import Pair


def init_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images_meta(args) -> List[RSImageMeta]:
    base_path = os.path.join(args.root, 'adjust_images')
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


def load_images(args, metas: List[RSImageMeta]) -> List[RSImage]:
    return [RSImage(meta, device=args.device) for meta in metas]


def get_pairs(args, metas: List[RSImageMeta]) -> List[Tuple[int, int]]:
    pair_idxs = []
    min_area = args.window_size * args.window_size
    for i, j in itertools.combinations(range(len(metas)), 2):
        if is_overlap(metas[i], metas[j], min_area):
            pair_idxs.append((i, j))
    return pair_idxs


def load_models(args):
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
    encoder = encoder.to(args.device).eval().half()
    predictor = predictor.to(args.device).eval()
    return encoder, predictor


def tiepoints_to_xy(image: RSImage) -> np.ndarray:
    if image.tie_points is None or len(image.tie_points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    lines = image.tie_points[:, 0]
    samps = image.tie_points[:, 1]
    heights = image.dem[lines, samps]
    lats, lons = image.rpc.RPC_PHOTO2OBJ(samps, lines, heights, 'numpy')
    latlon = np.stack([lats, lons], axis=-1)
    yx = project_mercator(torch.as_tensor(latlon, dtype=torch.float64)).cpu().numpy()
    return yx[:, [1, 0]]


def sample_windows_for_pair(
    image_a: RSImage,
    image_b: RSImage,
    window_size: float,
    k_per_tiepoint: int,
    max_trials: int,
    rng: np.random.Generator,
):
    overlap_geom = Polygon(image_a.corner_xys).intersection(Polygon(image_b.corner_xys))
    if overlap_geom.is_empty:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if overlap_geom.geom_type == 'Polygon':
        overlap_poly = overlap_geom
    elif overlap_geom.geom_type == 'MultiPolygon':
        overlap_poly = max(overlap_geom.geoms, key=lambda g: g.area)
    else:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    if not overlap_poly.is_valid:
        overlap_poly = overlap_poly.buffer(0)
    if overlap_poly.is_empty:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)

    tie_xy = tiepoints_to_xy(image_a)
    if tie_xy.shape[0] == 0:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)

    diags = []
    tie_indices = []
    for idx, xy in enumerate(tie_xy):
        pt = Point(xy[0], xy[1])
        if not overlap_poly.buffer(1e-6).covers(pt):
            continue
        collected = 0
        attempts = 0
        while collected < k_per_tiepoint and attempts < max_trials:
            attempts += 1
            tlx = rng.uniform(xy[0] - window_size, xy[0])
            tly = rng.uniform(xy[1] - window_size, xy[1])
            candidate = box(tlx, tly, tlx + window_size, tly + window_size)
            if overlap_poly.covers(candidate):
                diags.append([[tlx, tly], [tlx + window_size, tly + window_size]])
                tie_indices.append(idx)
                collected += 1
    if len(diags) == 0:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.asarray(diags, dtype=np.float64), np.asarray(tie_indices, dtype=np.int64)


def build_solver_windows(pair: Pair, diags: np.ndarray, tie_indices: np.ndarray):
    solver = pair.solver_ab
    corners_a = solver.rs_image_a.convert_diags_to_corners(diags, solver.rpc_a)
    corners_b = solver.rs_image_b.convert_diags_to_corners(diags, solver.rpc_b)
    imgs_a, dems_a, Hs_a = solver.rs_image_a.crop_windows(corners_a)
    imgs_b, dems_b, Hs_b = solver.rs_image_b.crop_windows(corners_b)
    valid_mask = solver.get_valid_mask([imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b])
    if valid_mask.sum() == 0:
        solver.window_pairs = []
        return np.empty((0,), dtype=np.int64)
    imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b, diags, tie_indices = [
        x[valid_mask] for x in [imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b, diags, tie_indices]
    ]
    solver.window_pairs = solver.generate_window_pairs((imgs_a, dems_a, Hs_a), (imgs_b, dems_b, Hs_b), diags)
    return tie_indices


def calc_window_errors(pair: Pair, affines: torch.Tensor, tie_indices: np.ndarray) -> np.ndarray:
    if affines.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    affines_np = affines.detach().cpu().numpy()
    tie_a = pair.rs_image_a.tie_points
    heights_a = pair.rs_image_a.tie_points_heights
    errors = np.zeros((affines_np.shape[0],), dtype=np.float64)
    for i in range(affines_np.shape[0]):
        t_idx = int(tie_indices[i])
        line_a = float(tie_a[t_idx, 0])
        samp_a = float(tie_a[t_idx, 1])
        h = float(heights_a[t_idx])
        line_gt, samp_gt = project_linesamp(
            pair.rs_image_a.rpc,
            pair.rs_image_b.rpc,
            np.array([line_a], dtype=np.float64),
            np.array([samp_a], dtype=np.float64),
            np.array([h], dtype=np.float64),
            output_type='numpy',
        )
        src = np.array([line_a, samp_a, 1.0], dtype=np.float64)
        pred = affines_np[i] @ src
        errors[i] = np.sqrt((pred[0] - line_gt[0]) ** 2 + (pred[1] - samp_gt[0]) ** 2)
    return errors


def summarize_errors(errors: np.ndarray) -> Dict[str, float]:
    if errors.size == 0:
        return {
            'count': 0,
            'mean': float('nan'),
            'median': float('nan'),
            '<1pix_percent': float('nan'),
            '<3pix_percent': float('nan'),
            '<5pix_percent': float('nan'),
        }
    return {
        'count': int(errors.size),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        '<1pix_percent': float(100.0 * np.mean(errors < 1.0)),
        '<3pix_percent': float(100.0 * np.mean(errors < 3.0)),
        '<5pix_percent': float(100.0 * np.mean(errors < 5.0)),
    }


@torch.no_grad()
def main(args):
    init_random_seed(args.random_seed)
    os.makedirs(args.output_path, exist_ok=True)

    metas = load_images_meta(args)
    images = load_images(args, metas)
    pair_ids = get_pairs(args, metas)

    encoder, predictor = load_models(args)

    rng = np.random.default_rng(args.random_seed)
    all_errors = []
    pair_reports = []

    for i, j in pair_ids:
        image_a = images[i]
        image_b = images[j]
        if image_a.tie_points is None or image_b.tie_points is None:
            continue
        if image_a.tie_points.shape[0] == 0 or image_b.tie_points.shape[0] == 0:
            continue
        tie_num = min(image_a.tie_points.shape[0], image_b.tie_points.shape[0])
        if image_a.tie_points.shape[0] != tie_num:
            image_a.tie_points = image_a.tie_points[:tie_num]
            image_a.tie_points_heights = image_a.tie_points_heights[:tie_num]
        if image_b.tie_points.shape[0] != tie_num:
            image_b.tie_points = image_b.tie_points[:tie_num]
            image_b.tie_points_heights = image_b.tie_points_heights[:tie_num]

        pair_output = os.path.join(args.output_path, f'pair_{i}_{j}')
        configs = {
            'max_window_num': -1,
            'min_window_size': args.window_size,
            'max_window_size': args.window_size,
            'min_area_ratio': 0.0,
            'quad_split_times': 1,
            'iter_num': args.predictor_iter_num,
            'match': args.match,
            'output_path': pair_output,
        }
        pair = Pair(image_a, image_b, i, j, configs=configs, mutual=False, device=args.device)

        diags, tie_indices = sample_windows_for_pair(
            image_a,
            image_b,
            args.window_size,
            args.k_per_tiepoint,
            args.max_trials_per_tiepoint,
            rng,
        )
        if diags.shape[0] == 0:
            continue

        tie_indices_valid = build_solver_windows(pair, diags, tie_indices)
        if tie_indices_valid.shape[0] == 0:
            continue

        affines, _ = pair.solver_ab.get_window_affines(encoder, predictor)
        errors = calc_window_errors(pair, affines, tie_indices_valid)
        if errors.size == 0:
            continue
        all_errors.append(errors)
        report = summarize_errors(errors)
        report['pair'] = [int(i), int(j)]
        report['window_count'] = int(errors.size)
        pair_reports.append(report)

    if len(all_errors) > 0:
        all_errors = np.concatenate(all_errors, axis=0)
    else:
        all_errors = np.empty((0,), dtype=np.float64)

    global_report = summarize_errors(all_errors)

    output = {
        'experiment_id': args.experiment_id,
        'window_size': args.window_size,
        'k_per_tiepoint': args.k_per_tiepoint,
        'total_windows': int(all_errors.size),
        'global_report': global_report,
        'pair_reports': pair_reports,
    }

    with open(os.path.join(args.output_path, 'window_error_report.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print('--- Window Error Report (Summary) ---')
    print(f"Total windows checked: {global_report['count']}")
    print(f"Mean Error:   {global_report['mean']:.4f} pix")
    print(f"Median Error: {global_report['median']:.4f} pix")
    print(f"< 1.0 pix: {global_report['<1pix_percent']:.2f} %")
    print(f"< 3.0 pix: {global_report['<3pix_percent']:.2f} %")
    print(f"< 5.0 pix: {global_report['<5pix_percent']:.2f} %")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--select_imgs', type=str, default='-1')
    parser.add_argument('--dino_path', type=str, default='weights')
    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')
    parser.add_argument('--predictor_path', type=str, default='weights/predictor.pth')
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--predictor_iter_num', type=int, default=None)
    parser.add_argument('--use_adapter', type=str2bool, default=True)
    parser.add_argument('--use_conf', type=str2bool, default=True)
    parser.add_argument('--use_mtf', type=str2bool, default=True)
    parser.add_argument('--match', type=str, default=None)
    parser.add_argument('--window_size', type=float, default=500.0)
    parser.add_argument('--k_per_tiepoint', type=int, default=4)
    parser.add_argument('--max_trials_per_tiepoint', type=int, default=200)
    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--usgs_dem', type=str2bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]', get_current_time())
    args.output_path = os.path.join(args.output_path, args.experiment_id)

    main(args)
