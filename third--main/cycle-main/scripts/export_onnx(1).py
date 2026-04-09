from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser("Export YOLOv8 to ONNX")
    parser.add_argument("--model", required=True, help="input .pt model")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="fp16 export")
    args = parser.parse_args()

    yolo = YOLO(args.model)
    yolo.export(format="onnx", imgsz=args.imgsz, half=args.half, simplify=True, opset=12)


if __name__ == "__main__":
    main()
