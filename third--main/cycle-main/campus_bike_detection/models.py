from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 normalized
    confidence: float
    class_id: int


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    trajectory: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class CountLine:
    line_id: str
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class SessionReport:
    total_frames: int
    avg_fps: float
    peak_count: int
    total_count: int
    line_counts: Dict[str, int]


@dataclass
class SystemConfig:
    source: Union[str, int]
    model_path: str
    backend: str = "auto"  # auto|pt|onnx|trt
    device: str = "cuda"  # cuda|cpu
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    show: bool = True
    line: CountLine = field(default_factory=lambda: CountLine("main", (0.05, 0.5), (0.95, 0.5)))
    count_direction: str = "both"  # both|forward|backward
    count_min_cross: float = 0.003
    count_debounce_frames: int = 10
    count_confirm_frames: int = 3
    count_line_dedup_radius: float = 0.10
    draw_trails: bool = True
    # Dual-branch feature enhancer
    enhance: bool = False
    enhance_edge_weight: float = 0.25
    enhance_clahe_clip: float = 2.0
    # Frame-difference motion filter
    motion_min_ratio: float = 0.04   # bbox moving-pixel fraction threshold