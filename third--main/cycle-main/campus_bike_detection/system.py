from __future__ import annotations

import time
from typing import List

import cv2

from campus_bike_detection.detector import BikeDetector
from campus_bike_detection.flow_counter import FlowCounter
from campus_bike_detection.models import SessionReport, SystemConfig
from campus_bike_detection.motion import MotionDetector
from campus_bike_detection.tracker import BikeTracker


class BikeDetectionSystem:
    def __init__(self, cfg: SystemConfig) -> None:
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source: {cfg.source}")

        self.detector = BikeDetector(
            model_path=cfg.model_path,
            backend=cfg.backend,
            device=cfg.device,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            enhance=cfg.enhance,
            edge_weight=cfg.enhance_edge_weight,
            clahe_clip=cfg.enhance_clahe_clip,
        )
        self.tracker = BikeTracker(
            iou_thresh=0.25,
            max_misses=150,             # much longer survival (5 sec @ 30fps)
            confirm_hits=3,             
            tentative_miss_tolerance=3, # avoid tentative ID reset on short occlusion
            reid_frames=180,            # even longer Re-ID window (6 sec)
            reid_center_thresh=0.30,    # more relaxed Re-ID threshold
            max_area_ratio=5.0,         # allow more size variation
        )
        self.counter = FlowCounter(
            cfg.line,
            direction=cfg.count_direction,
            min_cross=cfg.count_min_cross,
            debounce_frames=cfg.count_debounce_frames,
            confirm_frames=cfg.count_confirm_frames,
            line_dedup_radius=cfg.count_line_dedup_radius,
        )
        self.motion = MotionDetector(
            min_motion_ratio=cfg.motion_min_ratio,
        )

    def __enter__(self) -> BikeDetectionSystem:
        self.detector.warmup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self) -> SessionReport:
        frame_count = 0
        peak_count = 0
        fps_values: List[float] = []

        while True:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                break

            t0 = time.perf_counter()
            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)

            # Frame-difference motion filter:
            # only pass moving tracks to the flow counter
            motion_mask = self.motion.update(frame)
            h, w = frame.shape[:2]
            moving_tracks = [
                t for t in tracks
                if t.confirmed
                if t.confidence == 0.0  # ghost frames always pass (already tracked)
                or self.motion.is_moving(t.bbox, (h, w))
            ]

            current_count = len([t for t in tracks if t.confidence > 0])
            peak_count = max(peak_count, current_count)
            total_flow = self.counter.update(moving_tracks, frame_count)
            fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
            fps_values.append(fps)

            if self.cfg.show:
                self._draw(frame, tracks, current_count, total_flow, fps)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

        avg_fps = sum(fps_values) / max(len(fps_values), 1)
        return SessionReport(
            total_frames=frame_count,
            avg_fps=avg_fps,
            peak_count=peak_count,
            total_count=self.tracker.total_unique(),
            line_counts=self.counter.snapshot_counts(),
        )

    def _draw(self, frame, tracks, current_count: int, flow_count: int, fps: float) -> None:
        h, w = frame.shape[:2]
        for tr in tracks:
            if tr.confidence == 0.0:
                continue
            is_moving = self.motion.is_moving(tr.bbox, (h, w))
            color = (0, 255, 0) if is_moving else (160, 160, 160)  # green / gray
            x1, y1, x2, y2 = tr.bbox
            p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
            cv2.rectangle(frame, p1, p2, color, 2)
            cv2.putText(
                frame,
                f"ID:{tr.track_id} Conf:{tr.confidence:.2f}",
                (p1[0], max(15, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

            if self.cfg.draw_trails and tr.trajectory:
                pts = [(int(px * w), int(py * h)) for px, py in tr.trajectory[-20:]]
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i - 1], pts[i], (255, 180, 0), 1)
                side = self.counter.last_side.get(tr.track_id, 0.0)
                side_label = "A" if side >= 0 else "B"
                cv2.putText(
                    frame,
                    f"S:{side_label}",
                    (p1[0], min(h - 8, p2[1] + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 180, 0),
                    1,
                )

        ls = (int(self.cfg.line.start[0] * w), int(self.cfg.line.start[1] * h))
        le = (int(self.cfg.line.end[0] * w), int(self.cfg.line.end[1] * h))
        cv2.line(frame, ls, le, (0, 0, 255), 2)

        cv2.putText(frame, f"FPS:{fps:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Count:{current_count}", (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Flow:{flow_count}", (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(
            frame,
            f"Fwd:{self.counter.forward} Bwd:{self.counter.backward}",
            (10, 99),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (100, 230, 255),
            2,
        )
        cv2.imshow("Campus Bike Detection", frame)
