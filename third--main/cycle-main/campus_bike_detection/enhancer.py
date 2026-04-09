"""
Lightweight Dual-Branch Feature Enhancement with CBAM-style Attention
----------------------------------------------------------------------
Architecture:
  Branch-1 (Main):   CLAHE contrast normalization (LAB L-channel only)
  Branch-2 (Edge):   Laplacian sharpening
  Fusion:            cv2.addWeighted (single pass)
  CBAM Attention:
    - Channel Attention: per-channel gain via global avg/max → linear gate
    - Spatial Attention: GaussianBlur saliency map → linear blend (no float sigmoid)

All ops use OpenCV C++ paths — no full-frame float32 numpy loops.
Target overhead: < 2 ms per 640x640 frame on CPU.
"""
from __future__ import annotations

import cv2
import numpy as np


class DualBranchEnhancer:
    """
    Parameters
    ----------
    edge_weight : float
        Blend weight for the edge branch (0 = off, 0.25 default).
    clahe_clip : float
        CLAHE clip limit for the main branch.
    spatial_ksize : int
        Kernel size for spatial attention blur (must be odd).
    spatial_strength : float
        How strongly spatial attention modulates brightness (0~0.3 recommended).
    """

    def __init__(
        self,
        edge_weight: float = 0.25,
        clahe_clip: float = 2.0,
        spatial_ksize: int = 7,
        spatial_strength: float = 0.15,
    ) -> None:
        self.edge_weight = float(edge_weight)
        self.main_weight = 1.0 - self.edge_weight
        self.spatial_ksize = spatial_ksize if spatial_ksize % 2 == 1 else spatial_ksize + 1
        self.spatial_strength = float(spatial_strength)
        self._clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        # --- Branch 1: CLAHE on L channel ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        branch_main = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        # --- Branch 2: Unsharp / Laplacian sharpening ---
        # Use addWeighted trick: sharpened = 2*orig - blurred  (fast, no float)
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        branch_edge = cv2.addWeighted(frame, 2.0, blurred, -1.0, 0)

        # --- Fuse branches (single OpenCV call) ---
        fused = cv2.addWeighted(branch_main, self.main_weight, branch_edge, self.edge_weight, 0)

        # --- Channel Attention (per-channel gain, pure numpy on 3 scalars) ---
        fused = self._channel_attention(fused)

        # --- Spatial Attention (linear blend, no per-pixel sigmoid) ---
        if self.spatial_strength > 0:
            fused = self._spatial_attention(fused)

        return fused

    # ------------------------------------------------------------------
    # Attention modules — kept to O(C) or single-blur complexity
    # ------------------------------------------------------------------

    def _channel_attention(self, frame: np.ndarray) -> np.ndarray:
        """
        Per-channel gain from global mean stats only (3 scalar ops).
        Uses cv2.split/merge to stay in uint8 domain.
        """
        # cv2.mean is a single C++ call per channel
        means = np.array([cv2.mean(frame[:, :, c])[0] for c in range(3)], dtype=np.float32)
        global_mean = means.mean() + 1e-6
        # Gate in [0.88, 1.12] — subtle, avoids color cast
        gate = np.clip(means / global_mean, 0.88, 1.12).astype(np.float64)

        b, g, r = cv2.split(frame)
        b = cv2.convertScaleAbs(b, alpha=gate[0])
        g = cv2.convertScaleAbs(g, alpha=gate[1])
        r = cv2.convertScaleAbs(r, alpha=gate[2])
        return cv2.merge([b, g, r])

    def _spatial_attention(self, frame: np.ndarray) -> np.ndarray:
        """
        Spatial attention computed at 1/4 resolution then upsampled — ~16x cheaper.
        Uses cv2.addWeighted for the final blend, no full-res float multiply.
        """
        h, w = frame.shape[:2]
        sh, sw = max(h // 4, 1), max(w // 4, 1)

        # Compute saliency at small scale
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray_avg = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_max = cv2.max(cv2.max(small[:, :, 0], small[:, :, 1]), small[:, :, 2])
        combined = cv2.addWeighted(gray_avg, 0.5, gray_max, 0.5, 0)
        smoothed = cv2.GaussianBlur(combined, (self.spatial_ksize, self.spatial_ksize), 0)

        # Upsample mask back to full resolution
        mask = cv2.resize(smoothed, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert mask to a per-pixel alpha in [0, strength]
        # Use LUT to avoid float division: map [0,255] → [0, strength*255] as uint8
        lut = np.clip(
            np.arange(256, dtype=np.float32) * self.spatial_strength,
            0, 255
        ).astype(np.uint8)
        alpha_u8 = cv2.LUT(mask, lut)   # H x W, uint8, range [0, strength*255]

        # Blend: bright = frame + alpha_u8 (saturating add per channel)
        alpha3 = cv2.merge([alpha_u8, alpha_u8, alpha_u8])
        return cv2.add(frame, alpha3)
