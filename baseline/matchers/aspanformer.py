import importlib.util
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

from .base import BaseMatcher, MatchResult


def _load_aspanformer_cfg(config_path: str):
    spec = importlib.util.spec_from_file_location("aspan_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load ASpanFormer config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "cfg"):
        raise RuntimeError("ASpanFormer config file must define cfg")
    return module.cfg


class ASpanFormerMatcher(BaseMatcher):
    def __init__(
        self,
        config_path: str,
        weight_path: str,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        repo_root = os.path.join(os.getcwd(), "third_party", "ml-aspanformer")
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        if importlib.util.find_spec("src.ASpanFormer.aspanformer") is None:
            raise RuntimeError("ASpanFormer source not found in third_party/ml-aspanformer")
        from src.ASpanFormer.aspanformer import ASpanFormer
        from src.utils.misc import lower_config

        cfg = _load_aspanformer_cfg(config_path)
        cfg = lower_config(cfg)
        if "aspan" not in cfg:
            raise RuntimeError("ASpanFormer config missing ASPAN section")
        self.model = ASpanFormer(config=cfg["aspan"]).to(self.device).eval()
        state = torch.load(weight_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state, strict=False)

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        image0 = self._to_gray_tensor(img_a, self.device)
        image1 = self._to_gray_tensor(img_b, self.device)
        data = {"image0": image0, "image1": image1}
        self.model(data, online_resize=True)
        mkpts0 = data.get("mkpts0_f", None)
        mkpts1 = data.get("mkpts1_f", None)
        if mkpts0 is None or mkpts1 is None or mkpts0.numel() == 0:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        pts0 = mkpts0.detach().cpu().numpy()
        pts1 = mkpts1.detach().cpu().numpy()
        return MatchResult(pts0, pts1, match_time=time.perf_counter() - start)
