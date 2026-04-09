# Edge Deployment Guide (Jetson / RK3588 / Raspberry Pi)

## 1) Jetson (recommended TensorRT)
1. Install JetPack (includes CUDA + TensorRT).
2. Install Python deps:
   ```bash
   pip install -r requirements.txt
   ```
3. Export TensorRT engine:
   ```bash
   python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --half
   ```
4. Run:
   ```bash
   python -m campus_bike_detection.main --source 0 --model yolov8n.engine --backend trt --device cuda
   ```

## 2) RK3588 (recommended ONNX)
1. Use ONNX Runtime (CPU) or RKNN toolkit conversion if needed.
2. Export ONNX:
   ```bash
   python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt --imgsz 640
   ```
3. Run:
   ```bash
   python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu
   ```

## 3) Raspberry Pi
1. Install light OS + OpenCV + onnxruntime.
2. Use `imgsz=320` for speed:
   ```bash
   python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu --imgsz 320
   ```

## Performance Tuning (50~100 FPS target)
- Use TensorRT FP16 (Jetson / NVIDIA GPU).
- Use 640 -> 416/320 dynamic scaling.
- Disable display for benchmark (`--no-show`).
- Use smaller models (`yolov8n` / `yolov8n-int8` exported engine).
- Keep only bicycle class (already fixed to class id 1).
