import sys
import os

sys.path.append(os.getcwd())

import warnings
warnings.filterwarnings("ignore")
import argparse
import numpy as np
import random
from typing import List, Tuple
import itertools
import traceback
import cv2
import importlib.util

import torch
import torch.distributed as dist

from shared.utils import str2bool, get_current_time
from infer.utils import is_overlap, get_report_dict, find_intersection, find_squares, apply_H, partition_pairs
from infer.rs_image import RSImage, RSImageMeta, vis_registration
from infer.validate import compute_multiview_pair_errors
from infer.monitor import StatusMonitor, StatusReporter
from solve.global_solver_0120 import PBAAffineSolver, get_overlap_area


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images_meta(args, reporter) -> List[RSImageMeta]:
    reporter.update(current_step="Loading Meta")
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


def load_images(args, metas: List[RSImageMeta], reporter) -> List[RSImage]:
    reporter.update(current_step="Loading Images")
    images = [RSImage(meta, device=args.device) for meta in metas]
    args.image_num = len(images)
    return images


def get_pairs(args, metas: List[RSImageMeta]):
    pair_idxs = []
    for i, j in itertools.combinations(range(len(metas)), 2):
        if is_overlap(metas[i], metas[j], args.min_window_size ** 2):
            pair_idxs.append((i, j))
    return pair_idxs


def run_sift_matching(img_a, img_b, ratio_thresh, fm_ransac_thresh, fm_confidence):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    gray_a = cv2.equalizeHist(gray_a)
    gray_b = cv2.equalizeHist(gray_b)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_a, None)
    kp2, des2 = sift.detectAndCompute(gray_b, None)
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    if len(good_matches) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    pts_a = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts_b = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    M, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, fm_ransac_thresh, fm_confidence)
    if M is None:
        return np.empty((0, 2)), np.empty((0, 2))
    mask = mask.ravel().astype(bool)
    return pts_a[mask], pts_b[mask]


def load_loftr_model(args, reporter):
    kornia_spec = importlib.util.find_spec("kornia.feature")
    if kornia_spec is None:
        reporter.log("kornia is required for LoFTR")
        raise RuntimeError("kornia is required for LoFTR")
    from kornia.feature import LoFTR
    reporter.update(current_step="Loading Model")
    if args.loftr_weight_path is not None:
        if not os.path.exists(args.loftr_weight_path):
            raise FileNotFoundError(f"LoFTR weights not found at: {args.loftr_weight_path}")
        loftr_model = LoFTR(pretrained=None)
        checkpoint = torch.load(args.loftr_weight_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        loftr_model.load_state_dict(state_dict)
        return loftr_model.to(args.device).eval()
    return LoFTR(pretrained='outdoor').to(args.device).eval()


@torch.no_grad()
def run_loftr_matching(loftr_model, img_a_np, img_b_np, device, fm_ransac_thresh, fm_confidence):
    img0 = cv2.cvtColor(img_a_np, cv2.COLOR_RGB2GRAY)
    img1 = cv2.cvtColor(img_b_np, cv2.COLOR_RGB2GRAY)
    img0 = torch.from_numpy(img0).float().to(device) / 255.0
    img1 = torch.from_numpy(img1).float().to(device) / 255.0
    batch = {'image0': img0.unsqueeze(0).unsqueeze(0), 'image1': img1.unsqueeze(0).unsqueeze(0)}
    correspondences = loftr_model(batch)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    if len(mkpts0) < 4:
        return np.empty((0, 2)), np.empty((0, 2))
    M, mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, fm_ransac_thresh, fm_confidence)
    if M is None:
        return np.empty((0, 2)), np.empty((0, 2))
    mask = mask.ravel().astype(bool)
    return mkpts0[mask], mkpts1[mask]


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


def build_match_points(args, image_a: RSImage, image_b: RSImage, matcher, reporter):
    window_data = get_overlap_windows(args, image_a, image_b)
    if window_data is None:
        return None
    imgs_a, imgs_b, Hs_a, Hs_b = window_data
    all_pts_a_global = []
    all_pts_b_global = []
    for idx in range(len(imgs_a)):
        if np.isnan(imgs_a[idx]).any() or np.isnan(imgs_b[idx]).any():
            continue
        if args.matcher == "sift":
            pts_a_crop, pts_b_crop = run_sift_matching(imgs_a[idx], imgs_b[idx], args.sift_ratio_thresh, args.fm_ransac_thresh, args.fm_confidence)
        else:
            pts_a_crop, pts_b_crop = run_loftr_matching(matcher, imgs_a[idx], imgs_b[idx], args.device, args.fm_ransac_thresh, args.fm_confidence)
        if len(pts_a_crop) == 0:
            continue
        pts_a_rc = pts_a_crop[:, [1, 0]]
        pts_b_rc = pts_b_crop[:, [1, 0]]
        pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(args.device)
        pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(args.device)
        H_a_inv = torch.from_numpy(np.linalg.inv(Hs_a[idx])).unsqueeze(0).to(args.device, dtype=torch.float32)
        H_b_inv = torch.from_numpy(np.linalg.inv(Hs_b[idx])).unsqueeze(0).to(args.device, dtype=torch.float32)
        pts_a_global = apply_H(pts_a_rc_t, H_a_inv, args.device).squeeze(0).cpu().numpy()
        pts_b_global = apply_H(pts_b_rc_t, H_b_inv, args.device).squeeze(0).cpu().numpy()
        all_pts_a_global.append(pts_a_global)
        all_pts_b_global.append(pts_b_global)
    if not all_pts_a_global:
        return None
    pts_a_obs = np.concatenate(all_pts_a_global, axis=0)
    pts_b_obs = np.concatenate(all_pts_b_global, axis=0)
    if pts_a_obs.shape[0] != pts_b_obs.shape[0]:
        min_len = min(pts_a_obs.shape[0], pts_b_obs.shape[0])
        pts_a_obs = pts_a_obs[:min_len]
        pts_b_obs = pts_b_obs[:min_len]
    if pts_a_obs.shape[0] > args.max_match_points:
        idxs = np.random.choice(range(pts_a_obs.shape[0]), args.max_match_points, replace=False)
        pts_a_obs = pts_a_obs[idxs]
        pts_b_obs = pts_b_obs[idxs]
    sampline = pts_a_obs[:, [1, 0]]
    heights = image_a.dem_interp(sampline)
    valid_mask = np.isfinite(heights)
    pts_a_obs = pts_a_obs[valid_mask]
    pts_b_obs = pts_b_obs[valid_mask]
    heights = heights[valid_mask]
    if pts_a_obs.shape[0] < args.min_match_points:
        return None
    return pts_a_obs, pts_b_obs, heights


class PBADirectTieSolver(PBAAffineSolver):
    def _get_matches_overlap_area(self, images: list, results):
        for match in results:
            i, j = int(match['i']), int(match['j'])
            image_i = images[i]
            image_j = images[j]
            overlap_area = get_overlap_area(image_i.corner_xys, image_j.corner_xys)
            match['overlap_area'] = overlap_area
        return results

    def _process_data(self, images: list, results, sample_points_num=256):
        ties = []
        rpcs = [image.rpc for image in images]
        for match in results:
            pts_i = np.asarray(match['pts_i'], dtype=np.float64)
            pts_j = np.asarray(match['pts_j'], dtype=np.float64)
            heights = np.asarray(match['heights'], dtype=np.float64).reshape(-1)
            if pts_i.ndim != 2 or pts_i.shape[1] != 2:
                continue
            if pts_j.ndim != 2 or pts_j.shape[1] != 2:
                continue
            if pts_i.shape[0] != pts_j.shape[0] or pts_i.shape[0] != heights.shape[0]:
                min_len = min(pts_i.shape[0], pts_j.shape[0], heights.shape[0])
                if min_len == 0:
                    continue
                pts_i = pts_i[:min_len]
                pts_j = pts_j[:min_len]
                heights = heights[:min_len]
            ties.append({
                'i': int(match['i']),
                'j': int(match['j']),
                'pts_i': pts_i,
                'pts_j': pts_j,
                'heights': heights
            })
        return ties, rpcs


def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
    else:
        dist.init_process_group(backend="gloo")
        args.device = "cpu"

    experiment_id_clean = str(args.experiment_id).replace(":", "_").replace(" ", "_")
    monitor = None
    if rank == 0:
        monitor = StatusMonitor(world_size, experiment_id_clean)
        monitor.start()
    reporter = StatusReporter(rank, world_size, experiment_id_clean, monitor)

    matcher = None
    if args.matcher == "loftr":
        matcher = load_loftr_model(args, reporter)

    try:
        init_random_seed(args.random_seed)

        metas = []
        if rank == 0:
            metas = load_images_meta(args, reporter)
            pairs_ids_all = get_pairs(args, metas)
            pairs_ids_chunks = partition_pairs(pairs_ids_all, world_size)

        reporter.update(current_step="Syncing Meta")
        broadcast_container = [metas]
        dist.broadcast_object_list(broadcast_container, src=0)
        metas = broadcast_container[0]

        scatter_recive = [None]
        dist.scatter_object_list(scatter_recive, pairs_ids_chunks if rank == 0 else None, src=0)
        pairs_ids = scatter_recive[0]

        local_results = []
        total_pairs = len(pairs_ids)
        reporter.update(current_task="Ready", progress=f"0/{total_pairs}", level="-", current_step="Ready")

        if len(pairs_ids) > 0:
            image_ids = sorted(set(x for t in pairs_ids for x in t))
            images = load_images(args, [metas[i] for i in image_ids], reporter)
            images_by_id = {image.id: image for image in images}
            for idx, (i, j) in enumerate(pairs_ids):
                reporter.update(progress=f"{idx + 1}/{total_pairs}")
                reporter.update(current_task=f"{i} => {j}")
                image_i = images_by_id[i]
                image_j = images_by_id[j]
                match_points = build_match_points(args, image_i, image_j, matcher, reporter)
                if match_points is None:
                    continue
                pts_i, pts_j, heights = match_points
                local_results.append({
                    'i': i,
                    'j': j,
                    'pts_i': pts_i,
                    'pts_j': pts_j,
                    'heights': heights
                })
            reporter.update(current_task="Finished", progress=f"{total_pairs}/{total_pairs}", level="-", current_step="Cleanup")
            for image in images:
                del image

        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results = [None for _ in range(world_size)]
        else:
            all_results = None
        dist.gather_object(local_results, all_results if rank == 0 else None, dst=0)

        if rank == 0:
            reporter.update(current_task="Global Solving", current_step="Global Optimization")
            all_results = [item for sublist in all_results for item in sublist]
            if len(all_results) == 0:
                reporter.log("No valid matches found for PBA")
                return
            image_ids = sorted(set(x for t in pairs_ids_all for x in t))
            images = load_images(args, [metas[i] for i in image_ids], reporter)
            solver = PBADirectTieSolver(
                images,
                all_results,
                fixed_id=args.fixed_id,
                device=args.device,
                reporter=reporter,
                output_path=args.output_path
            )
            Ms = solver.solve(max_iters=args.solver_max_iter)
            for i, image in enumerate(images):
                M = Ms[i]
                reporter.log(f"Affine Matrix of Image {image.id}\n{M}\n")
                image.rpc.Clear_Adjust()
                image.rpc.Update_Adjust(M)
                if args.output_rpc:
                    image.rpc.Merge_Adjust()
                    image.rpc.save_rpc_to_file(os.path.join(args.output_path, f"{image.root.replace('/', '_')}_rpc.txt"))

            all_distances = compute_multiview_pair_errors(images)
            report = get_report_dict(all_distances)
            reporter.log("\n" + "--- Global Error Report (Summary) ---")
            reporter.log(f"Total tie points checked: {report['count']}")
            reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
            reporter.log(f"Median Error: {report['median']:.4f} pix")
            reporter.log(f"Max Error:    {report['max']:.4f} pix")
            reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
            reporter.log(f"< 1.0 pix: {report['<1pix_percent']:.2f} %")
            reporter.log(f"< 3.0 pix: {report['<3pix_percent']:.2f} %")
            reporter.log(f"< 5.0 pix: {report['<5pix_percent']:.2f} %")

            reporter.update(current_step="Visualizing")
            for i, j in itertools.combinations(range(len(images)), 2):
                vis_registration(image_a=images[i], image_b=images[j], output_path=args.output_path, device=args.device)

    except Exception as e:
        error_msg = traceback.format_exc()
        if reporter:
            reporter.update(current_task="ERROR", error=error_msg)
        raise e
    finally:
        if monitor:
            monitor.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, help='path to dataset')
    parser.add_argument('--select_imgs', type=str, default='-1')

    parser.add_argument('--matcher', type=str, default='sift', choices=['sift', 'loftr'])
    parser.add_argument('--loftr_weight_path', type=str, default=None)

    parser.add_argument('--max_window_size', type=int, default=2000)
    parser.add_argument('--min_window_size', type=int, default=500)
    parser.add_argument('--max_window_num', type=int, default=1024)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    parser.add_argument('--crop_size', type=int, default=640)

    parser.add_argument('--sift_ratio_thresh', type=float, default=0.75)
    parser.add_argument('--fm_ransac_thresh', type=float, default=3.0)
    parser.add_argument('--fm_confidence', type=float, default=0.99)
    parser.add_argument('--min_match_points', type=int, default=30)
    parser.add_argument('--max_match_points', type=int, default=2000)

    parser.add_argument('--fixed_id', type=int, default=None)
    parser.add_argument('--solver_max_iter', type=int, default=15)

    parser.add_argument('--output_path', type=str, default='results')
    parser.add_argument('--experiment_id', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_rpc', type=str2bool, default=False)
    parser.add_argument('--usgs_dem', type=str2bool, default=False)

    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()

    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]', get_current_time())

    args.output_path = os.path.join(args.output_path, args.experiment_id)
    os.makedirs(args.output_path, exist_ok=True)

    main(args)
