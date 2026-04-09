"""
Frame-difference motion detector
---------------------------------
Computes a per-pixel motion mask from consecutive frames and exposes a
method to score a bounding box by the fraction of "moving" pixels inside it.

Usage in the pipeline:
  1. Call `update(frame)` every frame — returns the motion mask (uint8, 0/255).
  2. Call `box_motion_score(bbox, frame_shape)` to get a [0,1] motion ratio
     for any normalized bbox.  Scores below `min_motion_ratio` mean the
     object is stationary and should be excluded from counting.

Design choices:
  - Three-frame running difference (abs(f[t] - f[t-2])) instead of two-frame
    to be more robust to single-frame noise and slight camera shake.
  - Gaussian blur before diff to suppress JPEG/sensor noise.
  - Otsu threshold on the diff image — adapts automatically to scene brightness.
  - Morphological close to fill small holes inside moving objects.
  - All ops on a downscaled gray image (1/2 resolution) for speed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


class MotionDetector:
    def __init__(
        self,
        scale: float = 0.5,          # process at this fraction of original size
        blur_ksize: int = 5,          # Gaussian blur kernel before diff
        morph_ksize: int = 7,         # morphological close kernel
        min_motion_ratio: float = 0.04,  # bbox must have ≥4% moving pixels
    ) -> None:
        self.scale = scale
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.morph_ksize = morph_ksize
        self.min_motion_ratio = min_motion_ratio
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
        )
        self._prev2: Optional[np.ndarray] = None   # frame t-2
        self._prev1: Optional[np.ndarray] = None   # frame t-1
        self._mask: Optional[np.ndarray] = None    # last computed mask (small res)
        self._mask_full: Optional[np.ndarray] = None  # upscaled to original size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Feed a new BGR frame.  Returns the full-resolution motion mask (uint8 0/255).
        Call this once per frame before querying box_motion_score().
        """
        h, w = frame.shape[:2]
        sh, sw = max(int(h * self.scale), 1), max(int(w * self.scale), 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(small, (self.blur_ksize, self.blur_ksize), 0)

        if self._prev2 is None:
            self._prev2 = blurred
            self._prev1 = blurred
            self._mask_full = np.zeros((h, w), dtype=np.uint8)
            return self._mask_full

        # Three-frame diff: combine t vs t-1 and t vs t-2
        diff1 = cv2.absdiff(blurred, self._prev1)
        diff2 = cv2.absdiff(blurred, self._prev2)
        diff = cv2.max(diff1, diff2)

        # Otsu threshold — adapts to scene
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological close to fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # Upscale back to original resolution (nearest — fast, binary mask)
        self._mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        self._prev2 = self._prev1
        self._prev1 = blurred
        return self._mask_full

    def box_motion_score(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
    ) -> float:
        """
        Return the fraction of pixels inside `bbox` that are marked as moving.
        `bbox` is normalized (x1,y1,x2,y2) in [0,1].
        `frame_shape` is (height, width).
        Returns 0.0 if no mask is available yet.
        """
        if self._mask_full is None:
            return 1.0   # no data yet — don't suppress
        h, w = frame_shape
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(bbox[2] * w))
        y2 = min(h, int(bbox[3] * h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = self._mask_full[y1:y2, x1:x2]
        return float(roi.mean()) / 255.0

    def is_moving(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int],
    ) -> bool:
        return self.box_motion_score(bbox, frame_shape) >= self.min_motion_ratio
