import importlib.util
import time
from typing import Optional

import numpy as np
import torch

from .base import BaseMatcher, MatchResult


class SuperPointSuperGlueMatcher(BaseMatcher):
    def __init__(
        self,
        superpoint_weight_path: Optional[str] = None,
        superglue_weight_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        if importlib.util.find_spec("kornia.feature") is None:
            raise RuntimeError("kornia.feature is required for SuperPoint+SuperGlue")
        from kornia.feature import SuperPoint, SuperGlue

        self.superpoint = SuperPoint().to(self.device).eval()
        self.superglue = SuperGlue().to(self.device).eval()

        if superpoint_weight_path:
            sp_state = torch.load(superpoint_weight_path, map_location=self.device)
            if isinstance(sp_state, dict) and "state_dict" in sp_state:
                sp_state = sp_state["state_dict"]
            self.superpoint.load_state_dict(sp_state, strict=False)

        if superglue_weight_path:
            sg_state = torch.load(superglue_weight_path, map_location=self.device)
            if isinstance(sg_state, dict) and "state_dict" in sg_state:
                sg_state = sg_state["state_dict"]
            self.superglue.load_state_dict(sg_state, strict=False)

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        image0 = self._to_gray_tensor(img_a, self.device)
        image1 = self._to_gray_tensor(img_b, self.device)
        features0 = self.superpoint({"image": image0})
        features1 = self.superpoint({"image": image1})
        data = {
            "keypoints0": features0["keypoints"],
            "keypoints1": features1["keypoints"],
            "descriptors0": features0["descriptors"],
            "descriptors1": features1["descriptors"],
            "scores0": features0["scores"],
            "scores1": features1["scores"],
            "image0": image0,
            "image1": image1,
        }
        pred = self.superglue(data)
        matches0 = pred["matches0"][0].detach().cpu().numpy()
        valid = matches0 > -1
        if valid.sum() == 0:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        kpts0 = data["keypoints0"][0].detach().cpu().numpy()
        kpts1 = data["keypoints1"][0].detach().cpu().numpy()
        pts0 = kpts0[valid]
        pts1 = kpts1[matches0[valid]]
        return MatchResult(pts0, pts1, match_time=time.perf_counter() - start)
