from __future__ import annotations

import argparse

from campus_bike_detection.detector import BikeDetector


def main() -> None:
    parser = argparse.ArgumentParser("ONNX benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", default="0")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    import cv2
    import time

    cap = cv2.VideoCapture(src)
    detector = BikeDetector(args.model, backend="onnx", device=args.device)
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
