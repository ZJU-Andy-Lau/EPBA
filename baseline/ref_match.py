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
import datetime

import torch
import torch.distributed as dist

from shared.utils import str2bool, get_current_time
from infer.utils import is_overlap, get_report_dict, find_intersection, find_squares, apply_H
from infer.rs_image import RSImage, RSImageMeta, vis_registration
from infer.validate import compute_multiview_pair_errors
from infer.monitor import StatusMonitor, StatusReporter


def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_images_meta(args, reporter) -> Tuple[List[RSImageMeta], List[RSImageMeta]]:
    reporter.update(current_step="Loading Meta")
    adjust_base_path = os.path.join(args.root, 'adjust_images')
    ref_base_path = os.path.join(args.root, 'ref_images')
    adjust_img_folders = sorted([d for d in os.listdir(adjust_base_path) if os.path.isdir(os.path.join(adjust_base_path, d))])
    if args.select_adjust_imgs != '-1':
        adjust_select_img_idxs = [int(i) for i in args.select_adjust_imgs.split(',')]
    else:
        adjust_select_img_idxs = range(len(adjust_img_folders))
    adjust_img_folders = [adjust_img_folders[i] for i in adjust_select_img_idxs]
    ref_img_folders = sorted([d for d in os.listdir(ref_base_path) if os.path.isdir(os.path.join(ref_base_path, d))])
    if args.select_ref_imgs != '-1':
        ref_select_img_idxs = [int(i) for i in args.select_ref_imgs.split(',')]
    else:
        ref_select_img_idxs = range(len(ref_img_folders))
    ref_img_folders = [ref_img_folders[i] for i in ref_select_img_idxs]
    adjust_metas = []
    ref_metas = []
    for idx, folder in enumerate(adjust_img_folders):
        img_path = os.path.join(adjust_base_path, folder)
        adjust_metas.append(RSImageMeta(args, img_path, idx, args.device))
    for idx, folder in enumerate(ref_img_folders):
        img_path = os.path.join(ref_base_path, folder)
        ref_metas.append(RSImageMeta(args, img_path, idx, args.device))
    return adjust_metas, ref_metas


def get_ref_list(args, adjust_meta: RSImageMeta, ref_metas: List[RSImageMeta], reporter) -> List[int]:
    reporter.update(current_step="Filtering Ref")
    ref_list = []
    for i in range(len(ref_metas)):
        if is_overlap(adjust_meta, ref_metas[i], args.min_window_size ** 2):
            ref_list.append(i)
    return ref_list


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


def build_match_points(args, image_a: RSImage, image_b: RSImage, matcher):
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
    if pts_a_obs.shape[0] < args.min_match_points:
        return None
    return pts_a_obs, pts_b_obs


def solve_pair_affine_from_matches(adjust_image: RSImage, ref_image: RSImage, pts_a_obs: np.ndarray, pts_b_obs: np.ndarray):
    heights = ref_image.dem_interp(pts_b_obs[:, ::-1])
    valid_mask = np.isfinite(heights)
    pts_a_obs = pts_a_obs[valid_mask]
    pts_b_obs = pts_b_obs[valid_mask]
    heights = heights[valid_mask]
    if pts_a_obs.shape[0] < 4:
        return None
    lons, lats = ref_image.rpc.RPC_PHOTO2OBJ(pts_b_obs[:, 1], pts_b_obs[:, 0], heights, 'numpy')
    samps_proj, lines_proj = adjust_image.rpc.RPC_OBJ2PHOTO(lons, lats, heights, 'numpy')
    pts_a_proj = np.stack([lines_proj, samps_proj], axis=-1)
    src_pts = pts_a_obs[:, ::-1].astype(np.float32)
    dst_pts = pts_a_proj[:, ::-1].astype(np.float32)
    affine_xy, _ = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if affine_xy is None:
        return None
    affine_rc = np.array([
        [affine_xy[1, 1], affine_xy[1, 0], affine_xy[1, 2]],
        [affine_xy[0, 1], affine_xy[0, 0], affine_xy[0, 2]]
    ], dtype=np.float32)
    return torch.from_numpy(affine_rc)


def main(args):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=120))
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

        ref_metas_all = []
        if rank == 0:
            adjust_metas_all, ref_metas_all = load_images_meta(args, reporter)
            adjust_metas_chunk = np.array_split(np.array(adjust_metas_all, dtype=object), world_size)

        reporter.update(current_step="Syncing Meta")
        scatter_adjust_metas = [None]
        dist.scatter_object_list(scatter_adjust_metas, adjust_metas_chunk if rank == 0 else None, src=0)
        adjust_metas: List[RSImageMeta] = scatter_adjust_metas[0]

        broadcast_container = [ref_metas_all]
        dist.broadcast_object_list(broadcast_container, src=0)
        ref_metas: List[RSImageMeta] = broadcast_container[0]

        dist.barrier()

        local_results = {}
        reporter.update(current_task="Ready", progress=f"-", level="-", current_step="Ready")

        if len(adjust_metas) > 0 and len(ref_metas) > 0:
            for adjust_idx, adjust_meta in enumerate(adjust_metas):
                reporter.update(progress=f"{adjust_idx}/{len(adjust_metas)}")
                ref_list = get_ref_list(args, adjust_meta, ref_metas, reporter)
                reporter.log(f"ref list for img_{adjust_meta.id} : {ref_list}")
                reporter.update(current_step="Loading Adjust Image")
                adjust_image = RSImage(adjust_meta, device=args.device)
                for ref_idx in ref_list:
                    reporter.update(current_step="Loading Ref Image")
                    reporter.update(current_task=f"{adjust_idx} => {ref_idx}")
                    ref_image = RSImage(ref_metas[ref_idx], device=args.device)
                    match_points = build_match_points(args, adjust_image, ref_image, matcher)
                    if match_points is not None:
                        pts_a_obs, pts_b_obs = match_points
                        affine = solve_pair_affine_from_matches(adjust_image, ref_image, pts_a_obs, pts_b_obs)
                        if affine is not None:
                            adjust_image.affine_list.append(affine)
                    del ref_image

                reporter.update(current_task="Baking RPC", level="-")
                if len(adjust_image.affine_list) > 0:
                    total_affine = adjust_image.merge_affines()
                    reporter.log(f"Affine Matrix of Image {adjust_image.id}\n{total_affine}\n")
                    adjust_image.rpc.Update_Adjust(total_affine)
                    local_results[adjust_image.id] = total_affine.detach().cpu()

                reporter.update(current_task="Check Error", level="-", current_step="-")
                if not adjust_image.tie_points is None:
                    for ref_idx in ref_list:
                        ref_image = RSImage(ref_metas[ref_idx], device=args.device)
                        ref_points = ref_image.get_ref_points()
                        dis = adjust_image.check_error(ref_points)
                        report = get_report_dict(dis)
                        reporter.log("\n" + f"--- Adj {adjust_image.id} => Ref {ref_image.id}  Error Report ---")
                        reporter.log(f"Total tie points checked: {report['count']}")
                        reporter.log(f"Mean Error:   {report['mean']:.4f} pix")
                        reporter.log(f"Median Error: {report['median']:.4f} pix")
                        reporter.log(f"Max Error:    {report['max']:.4f} pix")
                        reporter.log(f"RMSE:         {report['rmse']:.4f} pix")
                        reporter.log(f"< 1.0 pix: {report['<1pix_percent']:.2f} %")
                        reporter.log(f"< 3.0 pix: {report['<3pix_percent']:.2f} %")
                        reporter.log(f"< 5.0 pix: {report['<5pix_percent']:.2f} %")
                        del ref_image
                else:
                    for ref_idx in ref_list:
                        try:
                            ref_image = RSImage(ref_metas[ref_idx], device=args.device)
                            vis_registration(adjust_image, ref_image, os.path.join(args.output_path), device=args.device)
                            del ref_image
                        except:
                            reporter.log(f"{adjust_image.id} --- {ref_idx} vis registration error, pass")

                del adjust_image

            reporter.update(current_task="Finished", progress=f"{len(adjust_metas)}/{len(adjust_metas)}", level="-", current_step="Cleanup")

        reporter.update(current_step="Gathering Results")
        if rank == 0:
            all_results = [None for _ in range(world_size)]
        else:
            all_results = None

        dist.gather_object(local_results, all_results if rank == 0 else None, dst=0)
        dist.barrier()

        if rank == 0:
            all_results = {k: v for d in all_results if d for k, v in d.items()}
            reporter.update(current_step="Loading Images")
            images = [RSImage(meta, device=args.device) for meta in adjust_metas_all]
            for image in images:
                if image.id in all_results:
                    M = all_results[image.id]
                    image.rpc.Clear_Adjust()
                    image.rpc.Update_Adjust(M)
                    if args.output_rpc:
                        image.rpc.Merge_Adjust()
                        image.rpc.save_rpc_to_file(os.path.join(args.output_path, f"{image.root.replace('/', '_')}_rpc.txt"))

            if images and images[0].tie_points is not None:
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
    parser.add_argument('--select_adjust_imgs', type=str, default='-1')
    parser.add_argument('--select_ref_imgs', type=str, default='-1')

    parser.add_argument('--matcher', type=str, default='sift', choices=['sift', 'loftr'])
    parser.add_argument('--loftr_weight_path', type=str, default=None)

    parser.add_argument('--max_window_size', type=int, default=8000)
    parser.add_argument('--min_window_size', type=int, default=500)
    parser.add_argument('--max_window_num', type=int, default=64)
    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)
    parser.add_argument('--crop_size', type=int, default=640)

    parser.add_argument('--sift_ratio_thresh', type=float, default=0.75)
    parser.add_argument('--fm_ransac_thresh', type=float, default=3.0)
    parser.add_argument('--fm_confidence', type=float, default=0.99)
    parser.add_argument('--min_match_points', type=int, default=30)
    parser.add_argument('--max_match_points', type=int, default=2000)

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
