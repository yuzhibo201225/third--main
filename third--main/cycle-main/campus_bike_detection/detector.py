from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from campus_bike_detection.models import Detection

TARGET_CLASS_IDS = {1}
_MIN_AREA = 0.008


def _nms(detections: List[Detection], iou_thresh: float = 0.35) -> List[Detection]:
    detections = [d for d in detections if (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) >= _MIN_AREA]
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[Detection] = []
    suppressed = [False] * len(dets)
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(d)
        ax1, ay1, ax2, ay2 = d.bbox
        for j in range(i + 1, len(dets)):
            if suppressed[j]:
                continue
            bx1, by1, bx2, by2 = dets[j].bbox
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0., ix2-ix1) * max(0., iy2-iy1)
            if inter <= 0:
                continue
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            if inter / max(union, 1e-9) > iou_thresh:
                suppressed[j] = True
    keep = [d for d in keep if (d.bbox[2]-d.bbox[0]) / max(d.bbox[3]-d.bbox[1], 1e-4) >= 1.3]
    return keep


class BikeDetector:
    def __init__(self, model_path: str, backend: str = "auto", device: str = "cuda",
                 conf: float = 0.25, iou: float = 0.35, imgsz: int = 480,
                 enhance: bool = False, edge_weight: float = 0.25, clahe_clip: float = 2.0) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.backend = self._resolve_backend(backend)
        self.model = self._load_model()
        if enhance:
            from campus_bike_detection.enhancer import DualBranchEnhancer
            self._enhancer: Optional[DualBranchEnhancer] = DualBranchEnhancer(edge_weight=edge_weight, clahe_clip=clahe_clip)
        else:
            self._enhancer = None

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        ext = self.model_path.suffix.lower()
        if ext == ".pt":   return "pt"
        if ext == ".onnx": return "onnx"
        if ext in {".engine", ".trt"}: return "trt"
        raise ValueError(f"Unsupported model extension: {ext}")

    def _load_model(self):
        if self.backend in {"pt", "trt"}:
            from ultralytics import YOLO
            return YOLO(str(self.model_path))
        if self.backend == "onnx":
            import onnxruntime as ort
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"])
            return ort.InferenceSession(str(self.model_path), providers=providers)
        raise ValueError(f"Unsupported backend: {self.backend}")

    def warmup(self) -> None:
        _ = self.detect(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self._enhancer is not None:
            frame = self._enhancer(frame)
        if self.backend in {"pt", "trt"}:
            return self._detect_yolo(frame)
        return self._detect_onnx(frame)

    def _detect_yolo(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, classes=sorted(TARGET_CLASS_IDS), conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False, device=self.device, agnostic_nms=True, max_det=50)
        return self._from_ultralytics(results)

    def _from_ultralytics(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls not in TARGET_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0].tolist()]
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=float(box.conf[0].item()), class_id=cls))
        return _nms(detections, iou_thresh=0.35)

    def _detect_onnx(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.imgsz, self.imgsz))
        inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        out = self.model.run(None, {self.model.get_inputs()[0].name: inp})[0]
        rows = out[0] if out.ndim == 3 else out
        if rows.shape[0] == 6:
            rows = rows.T
        detections: List[Detection] = []
        for row in rows:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, conf, cls = [float(v) for v in row[:6]]
            if int(cls) not in TARGET_CLASS_IDS or conf < self.conf:
                continue
            detections.append(Detection(bbox=(max(0., x1/w), max(0., y1/h), min(1., x2/w), min(1., y2/h)), confidence=conf, class_id=int(cls)))
        return _nms(detections, iou_thresh=0.35)
