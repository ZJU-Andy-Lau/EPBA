import importlib.util
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .base import BaseMatcher, MatchResult

from romatch import roma_outdoor, roma_indoor, tiny_roma_v1_outdoor


class RoMaMatcher(BaseMatcher):
    def __init__(
        self,
        variant: str = "outdoor",
        weight_path: Optional[str] = None,
        dinov2_weight_path: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        # repo_root = os.path.join(os.getcwd(), "third_party", "RoMa")
        # if repo_root not in sys.path:
        #     sys.path.append(repo_root)
        # if importlib.util.find_spec("romatch") is None:
        #     raise RuntimeError("romatch package not found in third_party/RoMa")
        

        weights = None
        dinov2_weights = None
        if weight_path:
            weights = torch.load(weight_path, map_location=self.device)
        if dinov2_weight_path:
            dinov2_weights = torch.load(dinov2_weight_path, map_location=self.device)

        if variant == "outdoor":
            self.model = roma_outdoor(
                device=self.device,
                weights=weights,
                dinov2_weights=dinov2_weights,
                use_custom_corr=False,
            )
        elif variant == "indoor":
            self.model = roma_indoor(
                device=self.device,
                weights=weights,
                dinov2_weights=dinov2_weights,
                use_custom_corr=False,
            )
        elif variant == "tiny_outdoor":
            self.model = tiny_roma_v1_outdoor(device=self.device, weights=weights)
        else:
            raise ValueError(f"Unsupported RoMa variant: {variant}")

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        img_a = Image.fromarray(img_a)
        img_b = Image.fromarray(img_b)
        warp, certainty = self.model.match(img_a, img_b, device=self.device)
        matches, certainty = self.model.sample(warp, certainty)
        if matches is None or len(matches) == 0:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        kpts_a, kpts_b = self.model.to_pixel_coordinates(matches, h_a, w_a, h_b, w_b)
        pts0 = kpts_a.detach().cpu().numpy()
        pts1 = kpts_b.detach().cpu().numpy()
        return MatchResult(pts0, pts1, match_time=time.perf_counter() - start)
