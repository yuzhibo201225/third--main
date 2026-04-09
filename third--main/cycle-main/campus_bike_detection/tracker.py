from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from campus_bike_detection.models import Detection, Track

_F = np.eye(8, dtype=np.float32)
_F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0
_H = np.zeros((4, 8), dtype=np.float32)
_H[0, 0] = _H[1, 1] = _H[2, 2] = _H[3, 3] = 1.0
_Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]).astype(np.float32)
_R = np.diag([5e-4, 5e-4, 5e-4, 5e-4]).astype(np.float32)


def _bbox_to_z(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dtype=np.float32)

def _z_to_bbox(z):
    cx, cy, w, h = z[:4]
    w, h = max(w, 1e-4), max(h, 1e-4)
    return (cx-w/2, cy-h/2, cx+w/2, cy+h/2)

def _center(b):
    return (b[0]+b[2])*0.5, (b[1]+b[3])*0.5

def _iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0., ix2-ix1)*max(0., iy2-iy1)
    if inter <= 0: return 0.
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def _hungarian_match(cost):
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    if n <= 8 and m <= 8:
        # Exact minimum-cost assignment via DP (bitmask) to avoid
        # greedy mismatches that cause ID switches under occlusion.
        # We always assign the smaller dimension onto the larger one.
        if n <= m:
            rows, cols = n, m
            transposed = False
            work = cost
        else:
            rows, cols = m, n
            transposed = True
            work = cost.T

        size = 1 << cols
        inf = float("inf")
        dp = [inf] * size
        parent = [(-1, -1)] * size  # (prev_mask, chosen_col)
        dp[0] = 0.0

        for r in range(rows):
            new_dp = [inf] * size
            new_parent = [(-1, -1)] * size
            for mask in range(size):
                if dp[mask] == inf:
                    continue
                for c in range(cols):
                    if mask & (1 << c):
                        continue
                    nmask = mask | (1 << c)
                    cand = dp[mask] + float(work[r, c])
                    if cand < new_dp[nmask]:
                        new_dp[nmask] = cand
                        new_parent[nmask] = (mask, c)
            dp, parent = new_dp, new_parent

        best_mask = min(
            (mask for mask in range(size) if mask.bit_count() == rows),
            key=lambda mask: dp[mask],
        )

        chosen_cols = [0] * rows
        cur = best_mask
        for r in range(rows - 1, -1, -1):
            prev, c = parent[cur]
            chosen_cols[r] = c
            cur = prev

        if not transposed:
            return [(r, c) for r, c in enumerate(chosen_cols)]
        return [(c, r) for r, c in enumerate(chosen_cols)]
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


class _KalmanBox:
    def __init__(self, bbox):
        self.x = np.zeros(8, dtype=np.float32)
        self.x[:4] = _bbox_to_z(bbox)
        self.P = np.eye(8, dtype=np.float32) * 1e-2

    def predict(self):
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        return _z_to_bbox(self.x)

    def update(self, bbox):
        z = _bbox_to_z(bbox)
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(8, dtype=np.float32) - K @ _H) @ self.P
        return _z_to_bbox(self.x)


@dataclass
class _State:
    kf: _KalmanBox
    bbox: Tuple[float, float, float, float]
    misses: int
    hits: int = 0          # consecutive confirmed hits
    confirmed: bool = False
    traj: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class _DeadRecord:
    tid: int
    bbox: Tuple[float, float, float, float]
    traj: List[Tuple[float, float]]
    died_at_frame: int
    was_confirmed: bool = False  # confirmed or near-confirmed before death
    gmc_dx: float = 0.0
    gmc_dy: float = 0.0


class BikeTracker:
    """
    Kalman IoU tracker with:
    - Two-pass matching (IoU + center-distance fallback)
    - Tentative/confirmed states: new tracks must be seen for `confirm_hits`
      consecutive frames before being treated as real  filters out noise detections
    - Ghost frames during occlusion
    - Re-ID with Global Motion Compensation
    """

    def __init__(
        self,
        iou_thresh: float = 0.25,
        max_misses: int = 60,
        max_center_step: float = 0.35,
        max_area_ratio: float = 6.0,   # relaxed: partial occlusion changes bbox size a lot
        confirm_hits: int = 3,         # frames a new track must be seen before confirmed
        tentative_miss_tolerance: int = 2,  # keep tentative IDs alive for brief occlusion
        reid_frames: int = 90,
        reid_center_thresh: float = 0.25,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.max_center_step = max_center_step
        self.max_area_ratio = max_area_ratio
        self.confirm_hits = confirm_hits
        self.tentative_miss_tolerance = max(1, tentative_miss_tolerance)
        self.reid_frames = reid_frames
        self.reid_center_thresh = reid_center_thresh
        self.next_id = 1
        self.states: Dict[int, _State] = {}
        self.seen_ids: Set[int] = set()
        self.confirmed_ids: Set[int] = set()  # Track IDs that were confirmed
        self._frame_idx = 0
        self._dead: List[_DeadRecord] = []

    def update(self, detections: List[Detection]) -> List[Track]:
        self._frame_idx += 1
        cutoff = self._frame_idx - self.reid_frames
        self._dead = [d for d in self._dead if d.died_at_frame >= cutoff]

        prev_centers = {tid: _center(s.bbox) for tid, s in self.states.items()}

        for state in self.states.values():
            state.bbox = state.kf.predict()

        tids = list(self.states.keys())
        dets = list(detections)
        assigned: Dict[int, Detection] = {}
        matched_det_oids: Set[int] = set()
        matched_tids: Set[int] = set()

        # Pass 1  IoU Hungarian
        if tids and dets:
            n, m = len(tids), len(dets)
            cost = np.ones((n, m), dtype=np.float32)
            for i, tid in enumerate(tids):
                for j, det in enumerate(dets):
                    if not self._plausible(self.states[tid].bbox, det.bbox):
                        continue
                    cost[i, j] = 1.0 - _iou(self.states[tid].bbox, det.bbox)
            for i, j in _hungarian_match(cost):
                if cost[i, j] >= 1.0 - self.iou_thresh:
                    continue
                self._match(tids[i], dets[j], assigned, matched_det_oids, matched_tids)

        # Pass 2  center-distance fallback
        for tid in tids:
            if tid in matched_tids:
                continue
            pcx, pcy = _center(self.states[tid].bbox)
            best_j, best_dist = -1, self.max_center_step * 0.6
            for j, det in enumerate(dets):
                if id(det) in matched_det_oids:
                    continue
                ccx, ccy = _center(det.bbox)
                dist = ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5
                if dist < best_dist and self._similar_size(self.states[tid].bbox, det.bbox):
                    best_dist, best_j = dist, j
            if best_j >= 0:
                self._match(tid, dets[best_j], assigned, matched_det_oids, matched_tids)

        # GMC estimation
        gmc_dx, gmc_dy = self._estimate_gmc(prev_centers, matched_tids)
        for rec in self._dead:
            rec.gmc_dx += gmc_dx
            rec.gmc_dy += gmc_dy

        # Unmatched tracks -> ghost or expire
        for tid in tids:
            if tid in matched_tids:
                continue
            state = self.states[tid]
            state.misses += 1
            # Keep tentative tracks for a short window to handle
            # one/two-frame detector drop under partial occlusion.
            if not state.confirmed and state.misses > self.tentative_miss_tolerance:
                self._dead.append(_DeadRecord(
                    tid=tid, bbox=state.bbox,
                    traj=list(state.traj), died_at_frame=self._frame_idx,
                    was_confirmed=state.hits >= max(2, self.confirm_hits - 1),
                ))
                del self.states[tid]
            elif state.misses > self.max_misses:
                self._dead.append(_DeadRecord(
                    tid=tid, bbox=state.bbox,
                    traj=list(state.traj), died_at_frame=self._frame_idx,
                    was_confirmed=state.confirmed,
                ))
                del self.states[tid]
            else:
                ghost = Detection(bbox=state.bbox, confidence=0.0, class_id=-1)
                cx, cy = _center(state.bbox)
                state.traj.append((cx, cy))
                if len(state.traj) > 60:
                    state.traj = state.traj[-60:]
                assigned[tid] = ghost

        # Unmatched detections -> Re-ID or spawn
        for det in dets:
            if id(det) in matched_det_oids:
                continue
            reused = self._try_reid(det)
            if reused is not None:
                assigned[reused] = det
            else:
                self._spawn(det)

        # Only output confirmed tracks (or tentative tracks that have a live detection)
        result: List[Track] = []
        for tid, det in assigned.items():
            if tid not in self.states:
                continue
            state = self.states[tid]
            # Show tentative tracks only if they have a real detection this frame
            if not state.confirmed and det.confidence == 0.0:
                continue
            result.append(Track(
                track_id=tid,
                bbox=state.bbox,
                confidence=det.confidence,
                confirmed=state.confirmed,
                trajectory=state.traj,
            ))
        return result

    def _estimate_gmc(self, prev_centers, matched_tids):
        dxs, dys = [], []
        for tid in matched_tids:
            if tid not in prev_centers or tid not in self.states:
                continue
            px, py = prev_centers[tid]
            cx, cy = _center(self.states[tid].bbox)
            dxs.append(cx - px)
            dys.append(cy - py)
        if not dxs:
            return 0.0, 0.0
        return float(np.median(dxs)), float(np.median(dys))

    def _match(self, tid, det, assigned, matched_det_oids, matched_tids):
        state = self.states[tid]
        state.bbox = state.kf.update(det.bbox)
        state.misses = 0
        state.hits += 1
        if state.hits >= self.confirm_hits:
            state.confirmed = True
            self.confirmed_ids.add(tid)  # Track confirmed IDs
        cx, cy = _center(state.bbox)
        state.traj.append((cx, cy))
        if len(state.traj) > 60:
            state.traj = state.traj[-60:]
        assigned[tid] = det
        matched_det_oids.add(id(det))
        matched_tids.add(tid)

    def _try_reid(self, det: Detection) -> Optional[int]:
        ccx, ccy = _center(det.bbox)
        best: Optional[_DeadRecord] = None
        best_dist = self.reid_center_thresh
        for rec in self._dead:
            # Only Re-ID tracks that were confirmed before they died
            if not rec.was_confirmed:
                continue
                
            # Apply GMC to predict where the dead track should be now
            pcx = _center(rec.bbox)[0] + rec.gmc_dx
            pcy = _center(rec.bbox)[1] + rec.gmc_dy
            dist = ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5
            
            # Check size similarity
            if not self._similar_size(rec.bbox, det.bbox):
                continue
            
            # Check if the new detection is in a plausible direction
            # based on the trajectory of the dead track
            if len(rec.traj) >= 2:
                # Get movement direction from last 2 points in trajectory
                tx1, ty1 = rec.traj[-2]
                tx2, ty2 = rec.traj[-1]
                traj_dx, traj_dy = tx2 - tx1, ty2 - ty1
                traj_len = (traj_dx**2 + traj_dy**2)**0.5
                
                if traj_len > 0.01:  # meaningful movement
                    # Vector from last position to new detection
                    det_dx = ccx - (tx2 + rec.gmc_dx)
                    det_dy = ccy - (ty2 + rec.gmc_dy)
                    det_len = (det_dx**2 + det_dy**2)**0.5
                    
                    if det_len > 0.01:
                        # Cosine similarity: should be moving in similar direction
                        cos_sim = (traj_dx * det_dx + traj_dy * det_dy) / (traj_len * det_len)
                        # Reject if moving in opposite direction (cos < -0.3)
                        if cos_sim < -0.3:
                            continue
            
            if dist < best_dist:
                best_dist, best = dist, rec
                
        if best is None:
            return None
        self._dead = [d for d in self._dead if d.tid != best.tid]
        cx, cy = _center(det.bbox)
        traj = best.traj + [(cx, cy)]
        if len(traj) > 60:
            traj = traj[-60:]
        # Resurrect as confirmed (it was confirmed before it died)
        state = _State(kf=_KalmanBox(det.bbox), bbox=det.bbox, misses=0,
                       hits=self.confirm_hits, confirmed=True, traj=traj)
        self.states[best.tid] = state
        self.seen_ids.add(best.tid)
        self.confirmed_ids.add(best.tid)  # Mark as confirmed
        return best.tid

    def _spawn(self, det: Detection) -> int:
        cx, cy = _center(det.bbox)
        # Drop if too close to an existing live track
        for state in self.states.values():
            pcx, pcy = _center(state.bbox)
            dist = ((pcx-cx)**2 + (pcy-cy)**2)**0.5
            if dist < self.max_center_step * 0.4 and self._similar_size(state.bbox, det.bbox):
                return -1
        
        # Also check against recently dead tracks to avoid immediate re-spawn
        for rec in self._dead:
            if self._frame_idx - rec.died_at_frame > 30:  # Only check recent deaths
                continue
            pcx = _center(rec.bbox)[0] + rec.gmc_dx
            pcy = _center(rec.bbox)[1] + rec.gmc_dy
            dist = ((pcx-cx)**2 + (pcy-cy)**2)**0.5
            # Use a more generous threshold for spawn suppression
            if dist < self.reid_center_thresh * 1.5 and self._similar_size(rec.bbox, det.bbox):
                return -1
                
        tid = self.next_id
        self.next_id += 1
        self.states[tid] = _State(kf=_KalmanBox(det.bbox), bbox=det.bbox,
                                  misses=0, hits=1, confirmed=False, traj=[(cx, cy)])
        self.seen_ids.add(tid)
        return tid

    def _plausible(self, prev, cur) -> bool:
        pcx, pcy = _center(prev)
        ccx, ccy = _center(cur)
        if ((pcx-ccx)**2 + (pcy-ccy)**2)**0.5 > self.max_center_step:
            return False
        return self._similar_size(prev, cur)

    def _similar_size(self, a, b) -> bool:
        pa = max((a[2]-a[0])*(a[3]-a[1]), 1e-9)
        ca = max((b[2]-b[0])*(b[3]-b[1]), 1e-9)
        return max(pa/ca, ca/pa) <= self.max_area_ratio

    def total_unique(self) -> int:
        # Only count confirmed tracks to avoid noise IDs inflating the count
        return len(self.confirmed_ids)
