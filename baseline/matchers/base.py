from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class MatchResult:
    pts0: np.ndarray
    pts1: np.ndarray
    scores: Optional[np.ndarray] = None
    match_time: Optional[float] = None


class BaseMatcher:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        raise NotImplementedError

    @staticmethod
    def _to_gray_tensor(img: np.ndarray, device: str) -> torch.Tensor:
        if img.ndim == 3 and img.shape[2] == 3:
            gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)
        else:
            gray = img.astype(np.float32)
        gray = torch.from_numpy(gray).to(device) / 255.0
        return gray.unsqueeze(0).unsqueeze(0)
