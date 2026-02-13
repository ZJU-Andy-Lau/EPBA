import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import random
import itertools
import json
import time
from typing import List, Dict, Tuple
from copy import deepcopy

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
    min_area = args.min_window_size * args.min_window_size
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


def get_overlap_polygon(image_a: RSImage, image_b: RSImage):
    overlap_geom = Polygon(image_a.corner_xys).intersection(Polygon(image_b.corner_xys))
    if overlap_geom.is_empty:
        return None
    if overlap_geom.geom_type == 'Polygon':
        overlap_poly = overlap_geom
    elif overlap_geom.geom_type == 'MultiPolygon':
        overlap_poly = max(overlap_geom.geoms, key=lambda g: g.area)
    else:
        return None
    if not overlap_poly.is_valid:
        overlap_poly = overlap_poly.buffer(0)
    if overlap_poly.is_empty:
        return None
    return overlap_poly


def sample_initial_windows_for_pair(
    image_a: RSImage,
    image_b: RSImage,
    window_size: float,
    k_per_tiepoint: int,
    max_trials: int,
    rng: np.random.Generator,
):
    overlap_poly = get_overlap_polygon(image_a, image_b)
    if overlap_poly is None:
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


def build_solver_windows(
    pair: Pair,
    diags: np.ndarray,
    max_window_num: int,
    rng: np.random.Generator,
    scores: np.ndarray = None,
):
    solver = pair.solver_ab
    if diags.shape[0] == 0:
        solver.window_pairs = []
        solver.window_size = -1
        return False

    if max_window_num > 0 and diags.shape[0] > max_window_num:
        if scores is None:
            keep = rng.choice(np.arange(diags.shape[0]), size=max_window_num, replace=False)
        else:
            keep = np.argsort(-scores)[:max_window_num]
        diags = diags[keep]

    solver.window_size = float(np.abs(diags[0, 1, 0] - diags[0, 0, 0]))

    corners_a = solver.rs_image_a.convert_diags_to_corners(diags, solver.rpc_a)
    corners_b = solver.rs_image_b.convert_diags_to_corners(diags, solver.rpc_b)
    imgs_a, dems_a, Hs_a = solver.rs_image_a.crop_windows(corners_a)
    imgs_b, dems_b, Hs_b = solver.rs_image_b.crop_windows(corners_b)

    valid_mask = solver.get_valid_mask([imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b])
    if valid_mask.sum() == 0:
        solver.window_pairs = []
        solver.window_size = -1
        return False

    imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b, diags = [
        x[valid_mask] for x in [imgs_a, dems_a, Hs_a, imgs_b, dems_b, Hs_b, diags]
    ]
    solver.window_pairs = solver.generate_window_pairs((imgs_a, dems_a, Hs_a), (imgs_b, dems_b, Hs_b), diags)
    return len(solver.window_pairs) > 0


def get_split_diags_and_scores(pair: Pair):
    solver = pair.solver_ab
    if len(solver.window_pairs) == 0:
        return np.empty((0, 2, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    new_diags = []
    new_scores = []
    for window_pair in solver.window_pairs:
        diags_i, scores_i = window_pair.quadsplit(split_time=solver.configs['quad_split_times'],even_score=False)
        new_diags.append(diags_i)
        new_scores.append(scores_i)
        window_pair.clear()

    return np.concatenate(new_diags, axis=0), np.concatenate(new_scores, axis=0)


def estimate_initial_window_affine(
    image_a: RSImage,
    image_b: RSImage,
    init_diag: np.ndarray,
    encoder: Encoder,
    predictor: Predictor,
    args,
    rng: np.random.Generator,
    pair_idx: int,
    pair_total: int,
    window_global_idx: int,
    total_windows: int,
):
    configs = {
        'max_window_num': args.max_window_num,
        'min_window_size': args.min_window_size,
        'max_window_size': args.max_window_size,
        'min_area_ratio': args.min_cover_area_ratio,
        'quad_split_times': args.quad_split_times,
        'iter_num': args.predictor_iter_num,
        'match': args.match,
        'output_path': args.output_path,
    }
    pair = Pair(image_a, image_b, image_a.id, image_b.id, configs=configs, mutual=False, device=args.device)
    pair.solver_ab.rpc_a = deepcopy(image_a.rpc)
    pair.solver_ab.rpc_b = deepcopy(image_b.rpc)

    ok = build_solver_windows(
        pair=pair,
        diags=init_diag[None].astype(np.float64),
        max_window_num=args.max_window_num,
        rng=rng,
        scores=None,
    )
    if not ok:
        return None

    level_idx = 0
    while pair.solver_ab.window_size >= args.min_window_size and len(pair.solver_ab.window_pairs) > 0:
        t0 = time.perf_counter()
        Hs_a, _ = pair.solver_ab.collect_Hs(to_tensor=True)
        preds, scores = pair.solver_ab.get_window_affines(encoder, predictor)
        merged_affine = pair.solver_ab.merge_affines(preds, Hs_a, scores)
        pair.solver_ab.rpc_a.Update_Adjust(merged_affine)
        infer_time = time.perf_counter() - t0

        split_diags, split_scores = get_split_diags_and_scores(pair)
        split_num = int(split_diags.shape[0])
        if split_num == 0:
            print(f"[Pair {pair_idx}/{pair_total}] [Window {window_global_idx}/{total_windows}] level={level_idx} size={pair.solver_ab.window_size:.2f}m infer={infer_time:.3f}s split=0")
            break

        ok = build_solver_windows(
            pair=pair,
            diags=split_diags,
            max_window_num=args.max_window_num,
            rng=rng,
            scores=split_scores,
        )
        kept_num = len(pair.solver_ab.window_pairs)
        print(f"[Pair {pair_idx}/{pair_total}] [Window {window_global_idx}/{total_windows}] level={level_idx} size={pair.solver_ab.window_size:.2f}m infer={infer_time:.3f}s split={split_num} kept={kept_num}")
        if not ok:
            break
        level_idx += 1

    return pair.solver_ab.rpc_a.adjust_params.detach().cpu().numpy()


def calc_initial_window_error(
    image_a: RSImage,
    image_b: RSImage,
    affine: np.ndarray,
    tie_idx: int,
) -> float:
    line_a = float(image_a.tie_points[tie_idx, 0])
    samp_a = float(image_a.tie_points[tie_idx, 1])
    line_b = float(image_b.tie_points[tie_idx, 0])
    samp_b = float(image_b.tie_points[tie_idx, 1])
    h = float(image_a.tie_points_heights[tie_idx])

    line_gt, samp_gt = project_linesamp(
        image_b.rpc,
        image_a.rpc,
        np.array([line_b], dtype=np.float64),
        np.array([samp_b], dtype=np.float64),
        np.array([h], dtype=np.float64),
        output_type='numpy',
    )

    src = np.array([line_a, samp_a, 1.0], dtype=np.float64)
    pred = affine @ src
    return float(np.sqrt((pred[0] - line_gt[0]) ** 2 + (pred[1] - samp_gt[0]) ** 2))


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


def iter_batches(total: int, batch_size: int):
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        yield s, e


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

    total_pairs = len(pair_ids)
    start_all = time.perf_counter()

    for pair_k, (i, j) in enumerate(pair_ids, start=1):
        t_pair = time.perf_counter()
        image_a = images[i]
        image_b = images[j]
        if image_a.tie_points is None or image_b.tie_points is None:
            print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) skipped: tie points missing")
            continue
        if image_a.tie_points.shape[0] == 0 or image_b.tie_points.shape[0] == 0:
            print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) skipped: empty tie points")
            continue

        tie_num = min(image_a.tie_points.shape[0], image_b.tie_points.shape[0])
        if image_a.tie_points.shape[0] != tie_num:
            image_a.tie_points = image_a.tie_points[:tie_num]
            image_a.tie_points_heights = image_a.tie_points_heights[:tie_num]
        if image_b.tie_points.shape[0] != tie_num:
            image_b.tie_points = image_b.tie_points[:tie_num]
            image_b.tie_points_heights = image_b.tie_points_heights[:tie_num]

        init_diags, tie_indices = sample_initial_windows_for_pair(
            image_a=image_a,
            image_b=image_b,
            window_size=args.max_window_size,
            k_per_tiepoint=args.k_per_tiepoint,
            max_trials=args.max_trials_per_tiepoint,
            rng=rng,
        )
        if init_diags.shape[0] == 0:
            print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) skipped: no valid initial windows")
            continue

        total_windows = int(init_diags.shape[0])
        print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) initial windows={total_windows}, batch_size={args.batch_size}, max_window_num={args.max_window_num}")

        pair_errors = []
        valid_windows = 0
        global_window_idx = 0

        for bs, be in iter_batches(total_windows, args.batch_size):
            print(f"[Pair {pair_k}/{total_pairs}] processing window batch {bs+1}-{be}/{total_windows}")
            for local_idx in range(bs, be):
                global_window_idx += 1
                affine = estimate_initial_window_affine(
                    image_a=image_a,
                    image_b=image_b,
                    init_diag=init_diags[local_idx],
                    encoder=encoder,
                    predictor=predictor,
                    args=args,
                    rng=rng,
                    pair_idx=pair_k,
                    pair_total=total_pairs,
                    window_global_idx=global_window_idx,
                    total_windows=total_windows,
                )
                if affine is None:
                    continue
                err = calc_initial_window_error(
                    image_a=image_a,
                    image_b=image_b,
                    affine=affine,
                    tie_idx=int(tie_indices[local_idx]),
                )
                pair_errors.append(err)
                t_errors = np.array(pair_errors)
                print(f"[Pair {pair_k}/{total_pairs}] [Window {global_window_idx}/{total_windows}] mean:{t_errors.mean():.4f}px median:{np.median(t_errors):.4f}px")
                valid_windows += 1

        if len(pair_errors) == 0:
            print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) finished: no valid window results")
            continue

        pair_errors = np.asarray(pair_errors, dtype=np.float64)
        all_errors.append(pair_errors)

        report = summarize_errors(pair_errors)
        report['pair'] = [int(i), int(j)]
        report['window_count'] = int(valid_windows)
        pair_reports.append(report)

        pair_time = time.perf_counter() - t_pair
        print(f"[Pair {pair_k}/{total_pairs}] ({i},{j}) done in {pair_time:.2f}s | windows={valid_windows}/{total_windows} | mean={report['mean']:.4f} median={report['median']:.4f} <1={report['<1pix_percent']:.2f}% <3={report['<3pix_percent']:.2f}% <5={report['<5pix_percent']:.2f}%")

    if len(all_errors) > 0:
        all_errors = np.concatenate(all_errors, axis=0)
    else:
        all_errors = np.empty((0,), dtype=np.float64)

    global_report = summarize_errors(all_errors)

    output = {
        'experiment_id': args.experiment_id,
        'batch_size': args.batch_size,
        'max_window_size': args.max_window_size,
        'min_window_size': args.min_window_size,
        'max_window_num': args.max_window_num,
        'quad_split_times': args.quad_split_times,
        'k_per_tiepoint': args.k_per_tiepoint,
        'total_windows': int(all_errors.size),
        'global_report': global_report,
        'pair_reports': pair_reports,
    }

    with open(os.path.join(args.output_path, 'window_error_report.json'), 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_time = time.perf_counter() - start_all
    print('--- Initial Window Error Report (Summary) ---')
    print(f"Total runtime: {total_time:.2f} s")
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_window_size', type=float, default=8000.0)
    parser.add_argument('--min_window_size', type=float, default=500.0)
    parser.add_argument('--max_window_num', type=int, default=256)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    parser.add_argument('--quad_split_times', type=int, default=1)

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
