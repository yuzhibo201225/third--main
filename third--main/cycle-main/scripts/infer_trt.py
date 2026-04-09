from __future__ import annotations

import argparse
import time

import cv2

from campus_bike_detection.detector import BikeDetector


def main() -> None:
    parser = argparse.ArgumentParser("TensorRT benchmark")
    parser.add_argument("--model", required=True, help=".engine path")
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    detector = BikeDetector(args.model, backend="trt", device="cuda")
    detector.warmup()

    total = 0
    t0 = time.perf_counter()
    while total < 300:
        ok, frame = cap.read()
        if not ok:
            break
        _ = detector.detect(frame)
        total += 1
    dt = time.perf_counter() - t0
    print(f"frames={total}, fps={total/max(dt,1e-6):.2f}")


if __name__ == "__main__":
    main()
