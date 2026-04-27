import argparse
import gc
import itertools
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf
from rasterio import Affine, features
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_geom
from shapely.geometry import Polygon, box, shape
from shapely.ops import unary_union
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASP_ROOT = PROJECT_ROOT / "preprocess" / "CasP"
if str(CASP_ROOT) not in sys.path:
    sys.path.insert(0, str(CASP_ROOT))

from src.models.nets import CasP  # noqa: E402


@dataclass
class ImageMeta:
    index: int
    path: Path
    name: str
    crs: CRS
    transform: Affine
    bounds: Tuple[float, float, float, float]
    width: int
    height: int
    nodata: Optional[float]
    footprint: Polygon


@dataclass
class WindowMeta:
    bounds: Tuple[float, float, float, float]
    polygon: Polygon
    cover_indices: List[int]


class TrainDatasetGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.input_dir = Path(args.input_dir).resolve()
        self.output_h5 = Path(args.output_h5).resolve()
        self.processing_crs: Optional[CRS] = None
        self.image_metas: List[ImageMeta] = []
        self.matcher = None

    def run(self) -> None:
        self._validate_args()
        image_paths = self._discover_tifs()
        self._load_image_metas(image_paths)
        valid_windows = self._generate_valid_windows()
        if not valid_windows:
            raise RuntimeError("No valid windows generated. Please relax overlap thresholds or check inputs.")

        next_group_id = self._get_next_group_id()
        first_new_group_id = next_group_id
        created_group_count = 0

        self.matcher, device = self._load_matcher()

        for win_idx, window in enumerate(tqdm(valid_windows, desc="Processing windows")):
            if self.args.max_groups is not None and created_group_count >= self.args.max_groups:
                break

            group_payload = self._build_group_payload(window, win_idx, device)
            if group_payload is None:
                continue

            group_id = str(next_group_id)
            if self._write_group(group_id, group_payload):
                next_group_id += 1
                created_group_count += 1

        self._cleanup_matcher()
        self._validate_new_groups(first_new_group_id, created_group_count)

        final_count = self._count_numeric_groups()
        print(
            f"Done. Appended {created_group_count} groups to {self.output_h5}. "
            f"Final numeric group count: {final_count}."
        )

    def _validate_new_groups(self, first_group_id: int, created_group_count: int) -> None:
        if created_group_count == 0:
            print("[Warn] No new groups were written.")
            return

        with h5py.File(self.output_h5, "r") as f:
            for gid in range(first_group_id, first_group_id + created_group_count):
                key = str(gid)
                if key not in f:
                    raise RuntimeError(f"Missing expected group after write: {key}")
                grp = f[key]
                if "images" not in grp or "parallax" not in grp:
                    raise RuntimeError(f"Group {key} missing images/parallax")

                img_keys = sorted(grp["images"].keys(), key=lambda x: int(x))
                par_keys = sorted(grp["parallax"].keys(), key=lambda x: int(x))
                if img_keys != par_keys:
                    raise RuntimeError(f"Group {key}: image/parallax keys mismatch")
                if len(img_keys) < self.args.min_cover:
                    raise RuntimeError(f"Group {key}: less than min_cover images")

                expected_keys = [str(i) for i in range(len(img_keys))]
                if img_keys != expected_keys:
                    raise RuntimeError(f"Group {key}: keys must be continuous from 0")

                for ds_key in img_keys:
                    img = grp["images"][ds_key]
                    para = grp["parallax"][ds_key]
                    if img.dtype != np.uint8 or img.ndim != 2:
                        raise RuntimeError(f"Group {key}/{ds_key}: image dtype/shape invalid")
                    if para.dtype != np.float32 or para.ndim != 2:
                        raise RuntimeError(f"Group {key}/{ds_key}: parallax dtype/shape invalid")
                    if img.shape != para.shape:
                        raise RuntimeError(f"Group {key}/{ds_key}: image/parallax shape mismatch")

    def _validate_args(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if self.args.window_size_m <= 0:
            raise ValueError("window_size_m must be positive")
        if self.args.output_size_px <= 0:
            raise ValueError("output_size_px must be positive")
        if self.args.grid_step_m <= 0:
            raise ValueError("grid_step_m must be positive")
        if self.args.min_cover < 2:
            raise ValueError("min_cover must be >= 2")
        if not Path(self.args.casp_config).exists():
            raise FileNotFoundError(f"CasP config not found: {self.args.casp_config}")
        if not Path(self.args.casp_weights).exists():
            raise FileNotFoundError(f"CasP weights not found: {self.args.casp_weights}")
        self.output_h5.parent.mkdir(parents=True, exist_ok=True)

    def _discover_tifs(self) -> List[Path]:
        files = [
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
        ]
        files.sort(key=lambda p: p.name)
        if not files:
            raise FileNotFoundError(f"No .tif/.tiff files found in {self.input_dir}")
        print(f"Found {len(files)} tif files.")
        return files

    def _select_processing_crs(self, first_crs: CRS, first_bounds: Tuple[float, float, float, float]) -> CRS:
        if self.args.processing_crs:
            return CRS.from_user_input(self.args.processing_crs)

        if first_crs.is_projected:
            return first_crs

        minx, miny, maxx, maxy = first_bounds
        center_lon = (minx + maxx) / 2.0
        center_lat = (miny + maxy) / 2.0
        zone = int((center_lon + 180) // 6) + 1
        epsg = 32600 + zone if center_lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg)

    def _load_image_metas(self, image_paths: Sequence[Path]) -> None:
        print("Loading image metadata and footprints...")
        with rasterio.open(image_paths[0]) as src0:
            if src0.crs is None:
                raise ValueError(f"Missing CRS: {image_paths[0]}")
            self.processing_crs = self._select_processing_crs(src0.crs, src0.bounds)

        for idx, path in enumerate(tqdm(image_paths, desc="Footprints")):
            with rasterio.open(path) as src:
                if src.crs is None:
                    raise ValueError(f"Missing CRS: {path}")
                footprint = self._extract_footprint(src)

                self.image_metas.append(
                    ImageMeta(
                        index=idx,
                        path=path,
                        name=path.name,
                        crs=src.crs,
                        transform=src.transform,
                        bounds=tuple(src.bounds),
                        width=src.width,
                        height=src.height,
                        nodata=src.nodata,
                        footprint=footprint,
                    )
                )

        print(f"Processing CRS: {self.processing_crs}")

    def _extract_footprint(self, src: rasterio.DatasetReader) -> Polygon:
        decimation = max(1, self.args.mask_decimation)
        out_h = max(1, src.height // decimation)
        out_w = max(1, src.width // decimation)

        mask = None
        try:
            mask = src.read_masks(self.args.band, out_shape=(out_h, out_w))
        except Exception:
            mask = None

        if mask is None or np.count_nonzero(mask) == 0:
            data = src.read(self.args.band, out_shape=(out_h, out_w))
            if src.nodata is not None:
                valid = data != src.nodata
            else:
                valid = data != 0
            mask = (valid.astype(np.uint8) * 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        down_tf = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        polygons = [
            shape(geom)
            for geom, val in features.shapes((mask > 0).astype(np.uint8), transform=down_tf)
            if val > 0
        ]
        if not polygons:
            raise ValueError(f"No valid footprint detected for {src.name}")

        geom = unary_union(polygons).buffer(0)
        if src.crs != self.processing_crs:
            geom = shape(transform_geom(src.crs, self.processing_crs, geom.__geo_interface__))

        simplify_tol = max(0.0, self.args.footprint_simplify_m)
        if simplify_tol > 0:
            geom = geom.simplify(simplify_tol, preserve_topology=True)

        geom = geom.buffer(0)
        if geom.is_empty:
            raise ValueError(f"Footprint empty after processing: {src.name}")
        return geom

    def _generate_valid_windows(self) -> List[WindowMeta]:
        print("Building candidate overlap region from pairwise intersections...")
        overlaps = []
        for i, j in itertools.combinations(range(len(self.image_metas)), 2):
            inter = self.image_metas[i].footprint.intersection(self.image_metas[j].footprint)
            if not inter.is_empty and inter.area > 0:
                overlaps.append(inter)

        if not overlaps:
            raise RuntimeError("No pairwise overlaps found among footprints.")

        candidate_region = unary_union(overlaps).buffer(0)
        if candidate_region.is_empty:
            raise RuntimeError("Pairwise overlap union is empty.")

        minx, miny, maxx, maxy = candidate_region.bounds
        windows: List[WindowMeta] = []
        win = self.args.window_size_m
        step = self.args.grid_step_m
        cover_thresh = self.args.coverage_threshold

        x_positions = np.arange(minx, maxx - win + 1e-6, step)
        y_positions = np.arange(maxy, miny + win - 1e-6, -step)

        for y_top in tqdm(y_positions, desc="Grid rows"):
            for x_left in x_positions:
                cell = box(x_left, y_top - win, x_left + win, y_top)
                if cell.intersection(candidate_region).is_empty:
                    continue
                cover_indices = []
                cell_area = cell.area
                for meta in self.image_metas:
                    ratio = meta.footprint.intersection(cell).area / cell_area
                    if ratio >= cover_thresh:
                        cover_indices.append(meta.index)
                if len(cover_indices) >= self.args.min_cover:
                    windows.append(
                        WindowMeta(
                            bounds=(x_left, y_top - win, x_left + win, y_top),
                            polygon=cell,
                            cover_indices=cover_indices,
                        )
                    )

        print(f"Generated {len(windows)} valid windows.")
        return windows

    def _load_matcher(self) -> Tuple[CasP, torch.device]:
        device = self._resolve_device(self.args.device)
        cfg = OmegaConf.load(self.args.casp_config).config
        cfg.threshold = self.args.casp_threshold

        matcher = CasP(cfg)
        state = torch.load(self.args.casp_weights, map_location=device)
        matcher.load_state_dict(state)
        matcher = matcher.eval().to(device)
        print(f"CasP loaded on {device}.")
        return matcher, device

    def _resolve_device(self, requested: str) -> torch.device:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            if self.args.allow_cpu_fallback:
                print("Requested CUDA but not available. Falling back to CPU.")
                return torch.device("cpu")
            raise RuntimeError("CUDA requested but not available.")
        return torch.device(requested)

    @staticmethod
    def _prep_tensor(img: np.ndarray, device: torch.device) -> torch.Tensor:
        if img.ndim == 2:
            img = img[:, :, None]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor.unsqueeze(0).to(device)

    def _get_match_windows(self, h: int, w: int) -> List[Tuple[int, int]]:
        ws = self.args.casp_window_size
        hs = [0] if h <= ws else np.unique(np.linspace(0, h - ws, math.ceil(h / ws)).astype(int)).tolist()
        ws_list = [0] if w <= ws else np.unique(np.linspace(0, w - ws, math.ceil(w / ws)).astype(int)).tolist()
        return [(top, left) for top in hs for left in ws_list]

    def _clear_matcher_cache(self) -> None:
        fine = getattr(self.matcher, "fine_reg_matching", None)
        if fine is None:
            return
        for attr in ["coords0", "coords1", "points", "four_point_disp"]:
            obj = getattr(fine, attr, None)
            if isinstance(obj, dict):
                obj.clear()

    def _crop_single_image(self, meta: ImageMeta, window: WindowMeta) -> Optional[np.ndarray]:
        left, bottom, right, top = window.bounds
        out_size = self.args.output_size_px
        res = self.args.window_size_m / out_size
        dst_transform = from_origin(left, top, res, res)

        with rasterio.open(meta.path) as src:
            dst = np.full((out_size, out_size), np.nan, dtype=np.float32)
            dst_mask = np.zeros((out_size, out_size), dtype=np.uint8)
            source_mask = src.read_masks(self.args.band)

            reproject(
                source=rasterio.band(src, self.args.band),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=self.processing_crs,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )

            reproject(
                source=source_mask,
                destination=dst_mask,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=self.processing_crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

        valid = (dst_mask > 0) & np.isfinite(dst)
        valid_ratio = float(np.count_nonzero(valid)) / valid.size
        if valid_ratio < self.args.min_valid_ratio:
            return None

        vals = dst[valid]
        p_low, p_high = np.percentile(vals, [self.args.norm_percentile_low, self.args.norm_percentile_high])
        if not np.isfinite(p_low) or not np.isfinite(p_high):
            return None
        if p_high <= p_low:
            p_low = float(np.nanmin(vals))
            p_high = float(np.nanmax(vals))
            if p_high <= p_low:
                return None

        scaled = (dst - p_low) / (p_high - p_low)
        scaled = np.clip(scaled, 0.0, 1.0)
        out = (scaled * 255.0).astype(np.uint8)
        out[~valid] = 0
        return out

    def _build_group_payload(
        self,
        window: WindowMeta,
        window_idx: int,
        device: torch.device,
    ) -> Optional[Dict]:
        images: List[np.ndarray] = []
        local_to_global: List[int] = []

        for global_idx in window.cover_indices:
            img = self._crop_single_image(self.image_metas[global_idx], window)
            if img is not None:
                local_to_global.append(global_idx)
                images.append(img)

        if len(images) < self.args.min_cover:
            print(f"[Warn] Skip window {window_idx}: too few valid crops ({len(images)}).")
            return None

        parallax, total_matches = self._compute_group_parallax(images, device)
        if total_matches < self.args.min_match_points_warning:
            print(f"[Warn] Window {window_idx} sparse matches: {total_matches}")

        return {
            "images": images,
            "parallax": parallax,
            "global_indices": local_to_global,
            "window_bounds": window.bounds,
        }

    def _compute_group_parallax(self, images: List[np.ndarray], device: torch.device) -> Tuple[List[np.ndarray], int]:
        h, w = images[0].shape
        accum: List[Dict[Tuple[int, int], List[float]]] = [defaultdict(list) for _ in images]
        window_coords = self._get_match_windows(h, w)
        total_matches = 0

        for i, j in itertools.combinations(range(len(images)), 2):
            pts0, pts1 = [], []
            for top, left in window_coords:
                ws = self.args.casp_window_size
                c0 = images[i][top: top + ws, left: left + ws]
                c1 = images[j][top: top + ws, left: left + ws]

                data = {
                    "image0": self._prep_tensor(c0, device),
                    "image1": self._prep_tensor(c1, device),
                }
                with torch.no_grad():
                    result = self.matcher(data)

                p0 = result["points0"].detach().cpu().numpy()
                p1 = result["points1"].detach().cpu().numpy()

                if len(p0) == 0:
                    continue

                p0[:, 0] += left
                p0[:, 1] += top
                p1[:, 0] += left
                p1[:, 1] += top
                pts0.append(p0)
                pts1.append(p1)

            if pts0:
                a = np.concatenate(pts0, axis=0)
                b = np.concatenate(pts1, axis=0)
                d = np.linalg.norm(a - b, axis=1)
                total_matches += len(d)

                a_rc = np.round(a).astype(int)
                b_rc = np.round(b).astype(int)
                valid_a = (a_rc[:, 0] >= 0) & (a_rc[:, 0] < w) & (a_rc[:, 1] >= 0) & (a_rc[:, 1] < h)
                valid_b = (b_rc[:, 0] >= 0) & (b_rc[:, 0] < w) & (b_rc[:, 1] >= 0) & (b_rc[:, 1] < h)

                for k in np.where(valid_a)[0]:
                    accum[i][(a_rc[k, 1], a_rc[k, 0])].append(float(d[k]))
                for k in np.where(valid_b)[0]:
                    accum[j][(b_rc[k, 1], b_rc[k, 0])].append(float(d[k]))

            self._clear_matcher_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        outputs = []
        for idx in range(len(images)):
            pm = np.full((h, w), np.nan, dtype=np.float32)
            for (y, x), values in accum[idx].items():
                pm[y, x] = np.float32(np.median(values))
            outputs.append(pm)

        return outputs, total_matches

    def _h5_compression_kwargs(self) -> Dict:
        if self.args.h5_compression == "none":
            return {}
        kwargs = {"compression": self.args.h5_compression}
        if self.args.h5_compression == "gzip" and self.args.h5_compression_opts is not None:
            kwargs["compression_opts"] = self.args.h5_compression_opts
        return kwargs

    def _write_group(self, group_id: str, payload: Dict) -> bool:
        images = payload["images"]
        parallax = payload["parallax"]
        if len(images) != len(parallax):
            print(f"[Warn] Skip group {group_id}: image/parallax count mismatch.")
            return False

        try:
            with h5py.File(self.output_h5, "a") as f:
                if group_id in f:
                    print(f"[Warn] Group {group_id} already exists, skipping.")
                    return False

                g = f.create_group(group_id)
                g_images = g.create_group("images")
                g_para = g.create_group("parallax")

                comp = self._h5_compression_kwargs()
                for i, (img, para) in enumerate(zip(images, parallax)):
                    if img.dtype != np.uint8 or img.ndim != 2:
                        raise ValueError(f"Invalid image format at {group_id}/{i}")
                    if para.dtype != np.float32 or para.ndim != 2:
                        raise ValueError(f"Invalid parallax format at {group_id}/{i}")
                    if img.shape != para.shape:
                        raise ValueError(f"Shape mismatch at {group_id}/{i}")

                    key = str(i)
                    g_images.create_dataset(key, data=img, **comp)
                    g_para.create_dataset(key, data=para, **comp)

                g.attrs["processing_crs"] = self.processing_crs.to_string()
                g.attrs["window_bounds"] = np.array(payload["window_bounds"], dtype=np.float64)
                g.attrs["window_size_m"] = float(self.args.window_size_m)
                g.attrs["output_size_px"] = int(self.args.output_size_px)
                g.attrs["resolution_m_per_px"] = float(self.args.window_size_m / self.args.output_size_px)
                g.attrs["global_indices"] = np.array(payload["global_indices"], dtype=np.int64)
                g.attrs["source_paths"] = np.array(
                    [str(self.image_metas[i].path) for i in payload["global_indices"]],
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                g.attrs["complete"] = True

            return True
        except Exception as exc:
            with h5py.File(self.output_h5, "a") as f:
                if group_id in f:
                    del f[group_id]
            print(f"[Warn] Failed writing group {group_id}: {exc}")
            return False

    def _get_next_group_id(self) -> int:
        if not self.output_h5.exists():
            return 0
        with h5py.File(self.output_h5, "a") as f:
            numeric_ids = []
            for k in f.keys():
                try:
                    numeric_ids.append(int(k))
                except ValueError:
                    print(f"[Warn] Non-numeric top-level key ignored: {k}")
                    continue
                grp = f[k]
                if "images" not in grp or "parallax" not in grp:
                    print(f"[Warn] Existing group {k} missing images/parallax")
        return (max(numeric_ids) + 1) if numeric_ids else 0

    def _count_numeric_groups(self) -> int:
        if not self.output_h5.exists():
            return 0
        with h5py.File(self.output_h5, "r") as f:
            cnt = 0
            for k in f.keys():
                try:
                    int(k)
                    cnt += 1
                except ValueError:
                    pass
            return cnt

    def _cleanup_matcher(self) -> None:
        self.matcher = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aligned training crops + parallax maps into appendable HDF5.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing tif/tiff files.")
    parser.add_argument("--output_h5", required=True, help="Target HDF5 path (append if exists).")
    parser.add_argument("--window_size_m", type=float, required=True, help="Crop window size in meters.")
    parser.add_argument("--output_size_px", type=int, required=True, help="Output crop size in pixels (square).")
    parser.add_argument("--grid_step_m", type=float, default=None, help="Grid stride in meters (default=window_size_m).")
    parser.add_argument("--min_cover", type=int, default=2, help="Minimum number of images covering each window.")
    parser.add_argument("--coverage_threshold", type=float, default=0.995, help="Area coverage ratio threshold per image.")
    parser.add_argument("--band", type=int, default=1, help="Band index to read from tif files (1-based).")
    parser.add_argument("--processing_crs", type=str, default=None, help="Override processing CRS (e.g., EPSG:32650).")
    parser.add_argument("--mask_decimation", type=int, default=10, help="Mask downsample factor for footprint extraction.")
    parser.add_argument("--footprint_simplify_m", type=float, default=0.5, help="Footprint simplify tolerance in processing CRS units.")

    parser.add_argument("--casp_config", type=str, default="preprocess/CasP/configs/model/net/casp.yaml", help="CasP config path.")
    parser.add_argument("--casp_weights", type=str, default="preprocess/CasP/weights/casp_outdoor.pth", help="CasP weights path.")
    parser.add_argument("--casp_window_size", type=int, default=1152, help="Matching tile size for CasP.")
    parser.add_argument("--casp_threshold", type=float, default=0.2, help="CasP matching threshold override.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Compute device, e.g. cuda:0 or cpu.")
    parser.add_argument("--allow_cpu_fallback", action="store_true", help="Fallback to CPU when CUDA unavailable.")
    parser.add_argument("--min_match_points_warning", type=int, default=100, help="Warn when group total matches < this value.")

    parser.add_argument("--norm_percentile_low", type=float, default=2.0, help="Lower percentile for uint8 normalization.")
    parser.add_argument("--norm_percentile_high", type=float, default=98.0, help="Upper percentile for uint8 normalization.")
    parser.add_argument("--min_valid_ratio", type=float, default=0.98, help="Min valid pixel ratio after crop reprojection.")

    parser.add_argument("--max_groups", type=int, default=None, help="Maximum number of NEW groups to append.")
    parser.add_argument("--h5_compression", type=str, default="gzip", choices=["none", "gzip", "lzf"], help="HDF5 compression type.")
    parser.add_argument("--h5_compression_opts", type=int, default=4, help="Compression level/opts when applicable.")

    args = parser.parse_args()
    if args.grid_step_m is None:
        args.grid_step_m = args.window_size_m
    return args


def main() -> None:
    args = parse_args()
    generator = TrainDatasetGenerator(args)
    generator.run()


if __name__ == "__main__":
    main()
