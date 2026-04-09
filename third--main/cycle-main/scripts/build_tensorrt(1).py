from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser("Build TensorRT engine from YOLO model")
    parser.add_argument("--model", required=True, help="input .pt or .onnx model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--device", default="0", help="GPU id, e.g. 0")
    args = parser.parse_args()

    yolo = YOLO(args.model)
    yolo.export(
        format="engine",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        device=args.device,
        simplify=True,
    )


if __name__ == "__main__":
    main()
