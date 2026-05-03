import importlib.util
import time
from typing import Optional

import cv2
import numpy as np
import torch

from .base import BaseMatcher, MatchResult
from .fundamental import filter_matches_by_fundamental


class LoFTRMatcher(BaseMatcher):
    def __init__(
        self,
        weight_path: Optional[str] = None,
        fm_ransac_thresh: float = 3.0,
        fm_confidence: float = 0.99,
        fm_method: str = "ransac",
        fm_max_iters: int = 10000,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        if importlib.util.find_spec("kornia.feature") is None:
            raise RuntimeError("kornia.feature is required for LoFTR")
        from kornia.feature import LoFTR

        self.fm_ransac_thresh = fm_ransac_thresh
        self.fm_confidence = fm_confidence
        self.fm_method = fm_method
        self.fm_max_iters = fm_max_iters
        if weight_path is not None:
            if not weight_path:
                raise ValueError("LoFTR weight path is empty")
            loftr_model = LoFTR(pretrained=None)
            checkpoint = torch.load(weight_path, map_location="cpu")
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            loftr_model.load_state_dict(state_dict)
            self.model = loftr_model.to(self.device).eval()
        else:
            self.model = LoFTR(pretrained="outdoor").to(self.device).eval()

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        img0 = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        img0 = torch.from_numpy(img0).float().to(self.device) / 255.0
        img1 = torch.from_numpy(img1).float().to(self.device) / 255.0
        batch = {
            "image0": img0.unsqueeze(0).unsqueeze(0),
            "image1": img1.unsqueeze(0).unsqueeze(0),
        }
        correspondences = self.model(batch)
        mkpts0 = correspondences["keypoints0"].detach().cpu().numpy()
        mkpts1 = correspondences["keypoints1"].detach().cpu().numpy()
        mkpts0, mkpts1 = filter_matches_by_fundamental(
            mkpts0,
            mkpts1,
            fm_method=self.fm_method,
            fm_ransac_thresh=self.fm_ransac_thresh,
            fm_confidence=self.fm_confidence,
            fm_max_iters=self.fm_max_iters,
        )
        return MatchResult(
            mkpts0,
            mkpts1,
            match_time=time.perf_counter() - start,
        )
