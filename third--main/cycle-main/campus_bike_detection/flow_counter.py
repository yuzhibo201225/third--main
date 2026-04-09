from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Set, Tuple

from campus_bike_detection.models import CountLine, Track


class FlowCounter:
    """Direction-aware counting with multi-frame confirmation and spatial dedup.

    Key anti-flicker measures
    -------------------------
    1. **Multi-frame confirmation**: a crossing is only counted after the target
       has stayed on the new side for `confirm_frames` consecutive frames.
       Single-frame bbox jitter that flips the sign and flips back is ignored.

    2. **Spatial dedup on the line projection**: when a new track ID crosses,
       we project its position onto the count line and check whether another ID
       was already counted within `line_dedup_radius` along that projection.
       This catches ID re-births at the same physical location.

    3. **Debounce**: same ID cannot be counted twice within `debounce_frames`.
    """

    def __init__(
        self,
        line: CountLine,
        direction: str = "both",
        min_cross: float = 0.003,
        debounce_frames: int = 10,
        confirm_frames: int = 3,
        line_dedup_radius: float = 0.10,
    ) -> None:
        self.line = line
        self.direction = direction
        self.min_cross = min_cross
        self.debounce_frames = debounce_frames
        self.confirm_frames = confirm_frames
        self.line_dedup_radius = line_dedup_radius

        # track_id → current side value
        self.last_side: Dict[int, float] = {}
        # track_id → deque of recent side signs (+1 / -1) for confirmation window
        self.side_history: Dict[int, deque] = {}
        # track_id → frame index of last count event
        self.last_count_frame: Dict[int, int] = {}
        # IDs that have already been counted (one-shot per ID)
        self.counted_ids: Set[int] = set()
        # Projection coordinates along the line where counts occurred
        self.counted_projections: List[float] = []
        # Spatiotemporal crossing records: list of (frame_idx, proj, sign)
        # Used to suppress new IDs that appear near a recent crossing
        self._crossing_records: List[Tuple[int, float, int]] = []
        # How many frames a crossing "occupies" the zone (suppress re-count)
        # Use a generous window: occlusion by a tree trunk can last 60-120 frames
        self._crossing_cooldown: int = max(debounce_frames * 4, 90)

        self.total = 0
        self.forward = 0
        self.backward = 0

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _point_side(self, p: Tuple[float, float]) -> float:
        x1, y1 = self.line.start
        x2, y2 = self.line.end
        px, py = p
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _project_onto_line(self, p: Tuple[float, float]) -> float:
        """Scalar projection of p onto the line direction (normalized 0-1)."""
        x1, y1 = self.line.start
        x2, y2 = self.line.end
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-12:
            return 0.0
        return ((p[0] - x1) * dx + (p[1] - y1) * dy) / length_sq

    def _is_allowed_direction(self, prev: float, cur: float) -> bool:
        if self.direction == "both":
            return True
        if self.direction == "forward":
            return prev < 0 < cur
        if self.direction == "backward":
            return prev > 0 > cur
        return False

    def _near_counted_projection(self, proj: float) -> bool:
        return any(abs(proj - p) < self.line_dedup_radius for p in self.counted_projections)

    def _in_crossing_cooldown(self, proj: float, cur_sign: int, frame_idx: int) -> bool:
        """Return True if a recent crossing already covers this position+direction."""
        for rec_frame, rec_proj, rec_sign in self._crossing_records:
            if frame_idx - rec_frame > self._crossing_cooldown:
                continue
            # Use 3x radius to cover tree-trunk-width occlusion offsets
            if abs(proj - rec_proj) < self.line_dedup_radius * 3.0:
                if rec_sign == cur_sign:
                    return True
        return False

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, tracks: List[Track], frame_idx: int) -> int:
        for track in tracks:
            tid = track.track_id
            cx = (track.bbox[0] + track.bbox[2]) * 0.5
            cy = (track.bbox[1] + track.bbox[3]) * 0.5
            cur_side = self._point_side((cx, cy))
            cur_sign = 1 if cur_side >= 0 else -1

            # Maintain a rolling window of side signs
            if tid not in self.side_history:
                self.side_history[tid] = deque(maxlen=self.confirm_frames + 2)
            self.side_history[tid].append(cur_sign)

            prev_side = self.last_side.get(tid)
            self.last_side[tid] = cur_side

            if prev_side is None:
                continue

            # Already counted this ID
            if tid in self.counted_ids:
                continue

            # No sign change on the raw side value
            if prev_side * cur_side >= 0:
                continue

            if not self._is_allowed_direction(prev_side, cur_side):
                continue

            # Jitter guard: both sides must be meaningfully far from the line
            if min(abs(prev_side), abs(cur_side)) < self.min_cross:
                continue

            # Debounce guard
            last_count = self.last_count_frame.get(tid, -(10 ** 9))
            if frame_idx - last_count <= self.debounce_frames:
                continue

            # Multi-frame confirmation: the last `confirm_frames` signs must all
            # agree with the new side (cur_sign), meaning the target has settled
            # on the new side and is not just flickering.
            history = self.side_history[tid]
            recent = list(history)[-(self.confirm_frames):]
            if len(recent) < self.confirm_frames or any(s != cur_sign for s in recent):
                continue

            # Spatial dedup along the line projection
            proj = self._project_onto_line((cx, cy))
            if self._near_counted_projection(proj):
                continue

            # Spatiotemporal cooldown: suppress new IDs near a recent crossing
            if self._in_crossing_cooldown(proj, cur_sign, frame_idx):
                continue

            # All guards passed — record the count
            self.last_count_frame[tid] = frame_idx
            self.counted_ids.add(tid)
            self.counted_projections.append(proj)
            self._crossing_records.append((frame_idx, proj, cur_sign))
            # Prune old records to keep memory bounded
            if len(self._crossing_records) > 200:
                cutoff = frame_idx - self._crossing_cooldown
                self._crossing_records = [
                    r for r in self._crossing_records if r[0] >= cutoff
                ]
            self.total += 1
            if prev_side < 0 < cur_side:
                self.forward += 1
            elif prev_side > 0 > cur_side:
                self.backward += 1

        return self.total

    def snapshot_counts(self) -> Dict[str, int]:
        return {
            self.line.line_id: self.total,
            f"{self.line.line_id}_forward": self.forward,
            f"{self.line.line_id}_backward": self.backward,
        }
