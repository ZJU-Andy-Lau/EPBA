from typing import Tuple

import cv2
import numpy as np


def get_fundamental_method(fm_method: str) -> int:
    method = (fm_method or "ransac").lower()
    if method == "ransac":
        return cv2.FM_RANSAC
    if method == "magsac":
        if not hasattr(cv2, "USAC_MAGSAC"):
            raise RuntimeError(
                "OpenCV does not expose cv2.USAC_MAGSAC in this build. "
                "Please upgrade OpenCV (opencv-contrib-python >= 4.5.x) or use fm_method='ransac'."
            )
        return cv2.USAC_MAGSAC
    raise ValueError(f"Unsupported fm_method: {fm_method}. Expected one of: ransac, magsac")


def filter_matches_by_fundamental(
    pts0: np.ndarray,
    pts1: np.ndarray,
    fm_method: str = "ransac",
    fm_ransac_thresh: float = 3.0,
    fm_confidence: float = 0.99,
    fm_max_iters: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    pts0 = np.asarray(pts0, dtype=np.float32)
    pts1 = np.asarray(pts1, dtype=np.float32)

    if pts0.ndim != 2 or pts1.ndim != 2 or pts0.shape[1] != 2 or pts1.shape[1] != 2:
        raise ValueError(f"pts0/pts1 must be (N,2), got {pts0.shape} and {pts1.shape}")
    if pts0.shape[0] != pts1.shape[0] or pts0.shape[0] < 8:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    method_flag = get_fundamental_method(fm_method)

    try:
        F, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=method_flag,
            ransacReprojThreshold=float(fm_ransac_thresh),
            confidence=float(fm_confidence),
            maxIters=int(fm_max_iters),
        )
    except TypeError:
        F, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=method_flag,
            ransacReprojThreshold=float(fm_ransac_thresh),
            confidence=float(fm_confidence),
        )

    if F is None or mask is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    inlier_mask = np.asarray(mask).reshape(-1).astype(bool)
    if inlier_mask.size != pts0.shape[0] or not np.any(inlier_mask):
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return pts0[inlier_mask], pts1[inlier_mask]
