import time
from typing import Optional

import cv2
import numpy as np

from .base import BaseMatcher, MatchResult


class SIFTMatcher(BaseMatcher):
    def __init__(
        self,
        ratio_thresh: float = 0.75,
        fm_ransac_thresh: float = 3.0,
        fm_confidence: float = 0.99,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.ratio_thresh = ratio_thresh
        self.fm_ransac_thresh = fm_ransac_thresh
        self.fm_confidence = fm_confidence

    def match(self, img_a: np.ndarray, img_b: np.ndarray) -> MatchResult:
        start = time.perf_counter()
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
        gray_a = cv2.equalizeHist(gray_a)
        gray_b = cv2.equalizeHist(gray_b)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_a, None)
        kp2, des2 = sift.detectAndCompute(gray_b, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_thresh * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        pts_a = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_b = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        M, mask = cv2.findFundamentalMat(
            pts_a, pts_b, cv2.FM_RANSAC, self.fm_ransac_thresh, self.fm_confidence
        )
        if M is None:
            return MatchResult(np.empty((0, 2)), np.empty((0, 2)), match_time=time.perf_counter() - start)
        mask = mask.ravel().astype(bool)
        pts_a = pts_a[mask]
        pts_b = pts_b[mask]
        return MatchResult(pts_a, pts_b, match_time=time.perf_counter() - start)
