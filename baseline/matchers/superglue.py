import os
import sys
import time
from typing import Optional

import numpy as np

from .base import BaseMatcher, MatchResult


class SuperPointSuperGlueMatcher(BaseMatcher):
    def __init__(
        self,
        superpoint_weight_path: Optional[str] = None,
        superglue_weight_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        repo_root = os.path.join(os.getcwd(), "third_party", "SuperGluePretrainedNetwork")
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from models.matching import Matching

        if not superpoint_weight_path:
            raise ValueError("SuperPoint weight path is required")
        if not superglue_weight_path:
            raise ValueError("SuperGlue weight path is required")
        if not os.path.exists(superpoint_weight_path):
            raise FileNotFoundError(f"SuperPoint weight path not found: {superpoint_weight_path}")
        if not os.path.exists(superglue_weight_path):
            raise FileNotFoundError(f"SuperGlue weight path not found: {superglue_weight_path}")

        superglue_variant = "indoor"
        lower_path = os.path.basename(superglue_weight_path).lower()
        if "outdoor" in lower_path:
            superglue_variant = "outdoor"

        self.matching = Matching({
            "superpoint": {
                "weight_path": superpoint_weight_path,
            },
            "superglue": {
                "weights": superglue_variant,
                "weight_path": superglue_weight_path,
            },
        }).to(self.device).eval()

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        image0 = self._to_gray_tensor(img_a, self.device)
        image1 = self._to_gray_tensor(img_b, self.device)
        pred = self.matching({"image0": image0, "image1": image1})
        kpts0 = pred["keypoints0"][0].detach().cpu().numpy()
        kpts1 = pred["keypoints1"][0].detach().cpu().numpy()
        matches0 = pred["matches0"][0].detach().cpu().numpy()
        valid = matches0 > -1
        if valid.sum() == 0:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        pts0 = kpts0[valid]
        pts1 = kpts1[matches0[valid]]
        return MatchResult(pts0, pts1, match_time=time.perf_counter() - start)
