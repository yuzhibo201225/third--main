# 校园自行车实时检测系统 — 项目说明

## 一、项目概述

本项目是一套面向校园场景的**自行车实时检测与流量统计系统**，核心链路为：

```
视频帧 → 目标检测 → 多目标追踪 → 跨线计数 → 可视化展示
```

支持三种推理后端（PyTorch / ONNX / TensorRT），可在 PC、Jetson、RK3588、树莓派等平台部署。

---

## 二、目录结构

```
campus_bike_detection/   # 核心包
  models.py              # 数据结构定义
  detector.py            # 多后端目标检测
  tracker.py             # IoU 多目标追踪
  flow_counter.py        # 跨线方向计数
  system.py              # 主循环与可视化
  main.py                # CLI 入口
  yolov8n.pt             # 预训练权重

scripts/                 # 工具脚本
  export_onnx.py         # 导出 ONNX 模型
  build_tensorrt.py      # 编译 TensorRT 引擎
  infer_onnx.py          # ONNX 推理基准测试
  infer_trt.py           # TRT 推理基准测试

data/                    # 测试视频
docs/                    # 文档
```

---

## 三、模块详解

### 1. `models.py` — 数据结构

全项目共用的纯数据类（`dataclass`），无业务逻辑，作为各模块之间的"契约"。

| 类 | 字段 | 说明 |
|---|---|---|
| `Detection` | bbox, confidence, class_id | 单帧单个检测框，坐标归一化到 [0,1] |
| `Track` | track_id, bbox, confidence, trajectory | 带 ID 的追踪目标，trajectory 存储历史中心点 |
| `CountLine` | line_id, start, end | 计数线，坐标同样归一化 |
| `SystemConfig` | source, model_path, backend, device, … | 系统全局配置，由 CLI 参数填充 |
| `SessionReport` | total_frames, avg_fps, peak_count, total_count, line_counts | 会话结束后的统计报告 |

所有坐标均使用**归一化坐标**（相对于帧宽高的比例），使得检测、追踪、计数逻辑与分辨率无关。

---

### 2. `detector.py` — 目标检测

**职责**：接收原始帧，返回当前帧中所有自行车的 `Detection` 列表。

**核心设计**：

- 固定只检测 COCO 类别 `bicycle`（class id = 1），过滤其他类别，降低误报。
- 支持三种后端，通过文件扩展名或 `--backend` 参数自动选择：

| 后端 | 文件格式 | 推理库 | 适用场景 |
|---|---|---|---|
| `pt` | `.pt` | ultralytics YOLO | 开发调试、GPU 服务器 |
| `onnx` | `.onnx` | onnxruntime | CPU 边缘设备、跨平台 |
| `trt` | `.engine` | ultralytics + TensorRT | Jetson / NVIDIA GPU 高性能部署 |

**ONNX 推理流程**：
1. 将帧 resize 到 `imgsz × imgsz`，BGR → RGB，归一化到 [0,1]。
2. 转为 `(1, 3, H, W)` 的 float32 张量送入 ONNX Runtime。
3. 解析输出行（兼容 `(N,6)` 和 `(6,N)` 两种布局），过滤置信度和类别，将像素坐标归一化后返回。

**warmup**：系统启动时用全零帧跑一次推理，预热模型（对 TensorRT 尤为重要，避免首帧延迟）。

---

### 3. `tracker.py` — 多目标追踪

**职责**：将每帧的 `Detection` 列表关联成跨帧连续的 `Track`，为每个目标分配稳定的 `track_id`。

**算法：带运动/尺度门控的 IoU 追踪器**

纯 Python 实现，无需额外依赖，逻辑如下：

```
每帧:
  对每个已有 track，在当前帧 detections 中找 IoU 最大的匹配
    → 匹配成功：更新 bbox、重置 miss 计数、追加轨迹点
    → 匹配失败：miss 计数 +1，超过 max_misses(20帧) 则删除
  未被匹配的 detection → 创建新 track，分配递增 ID
```

**两个额外门控**（`_is_plausible_match`）防止跨目标误匹配：

- **运动门控**：两帧间中心点位移 > `max_center_step`（默认 0.18，即帧宽/高的 18%）则拒绝匹配，过滤瞬移。
- **尺度门控**：前后帧 bbox 面积比 > `max_area_ratio`（默认 2.8）则拒绝，过滤突变。

**轨迹**：每个 track 保留最近 40 个中心点，用于可视化拖尾和计数方向判断。

**`total_unique()`**：返回整个会话中出现过的不重复 ID 数量，即"总共见过多少辆自行车"。

---

### 4. `flow_counter.py` — 跨线方向计数

**职责**：判断每个 track 是否穿越了计数线，并区分正向/反向，统计流量。

**原理：点在直线哪侧（叉积符号法）**

对于计数线 `(x1,y1)→(x2,y2)`，目标中心点 `(px,py)` 的"侧值"定义为：

```
side = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
```

- `side > 0`：点在线的左侧（A 侧）
- `side < 0`：点在线的右侧（B 侧）

当某帧的 `side` 符号与上一帧相反，说明目标穿越了计数线。

**三重防抖机制**，避免抖动误计：

1. **方向过滤**：`--count-direction` 可限定只统计正向（B→A）或反向（A→B）。
2. **最小穿越距离**（`min_cross`，默认 0.003）：要求前后两帧的 `|side|` 都大于阈值，过滤在线附近微小抖动。
3. **帧间防抖**（`debounce_frames`，默认 5 帧）：同一 ID 两次计数之间至少间隔 5 帧。
4. **单次计数**：每个 `track_id` 只计数一次（`counted_ids` 集合），防止同一辆车被重复统计。

---

### 5. `system.py` — 主循环与可视化

**职责**：串联所有模块，驱动逐帧处理，并将结果渲染到画面上。

**主循环（`run()`）**：

```
while 有帧:
    读帧
    detector.detect(frame)      → detections
    tracker.update(detections)  → tracks
    counter.update(tracks)      → flow_count
    计算 FPS
    可视化绘制（可选）
    按 'q' 退出
```

**可视化内容（`_draw()`）**：

- 每个 track 绘制绿色检测框 + `ID:x Conf:x.xx` 标签。
- 开启 `--draw-trails` 时，绘制橙色轨迹拖尾（最近 20 个点）及当前所在侧（A/B）。
- 红色计数线横跨画面。
- 左上角实时显示：FPS、当前帧内目标数（Count）、累计流量（Flow）、正向/反向计数（Fwd/Bwd）。

**资源管理**：实现了 `__enter__`/`__exit__`，以 `with` 语句使用，确保摄像头和窗口在任何情况下都能正确释放。

---

### 6. `main.py` — CLI 入口

解析命令行参数，构建 `SystemConfig`，启动 `BikeDetectionSystem`，会话结束后打印统计报告。

主要参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | `0` | 摄像头 ID 或视频文件路径 |
| `--model` | `yolov8n.pt` | 模型文件路径 |
| `--backend` | `auto` | 推理后端：auto/pt/onnx/trt |
| `--device` | `cuda` | 推理设备：cuda/cpu |
| `--conf` | `0.25` | 检测置信度阈值 |
| `--iou` | `0.5` | NMS IoU 阈值 |
| `--imgsz` | `640` | 推理输入尺寸 |
| `--line` | `0.05,0.5,0.95,0.5` | 计数线归一化坐标（水平中线） |
| `--count-direction` | `both` | 计数方向：both/forward/backward |
| `--no-trails` | — | 关闭轨迹拖尾和侧边调试信息 |
| `--no-show` | — | 不显示画面（适合无头服务器） |

---

## 四、工具脚本

### `scripts/export_onnx.py`
将 `.pt` 权重导出为 ONNX 格式，使用 ultralytics 的 `export` 接口，支持 FP16 和 opset 12，并自动做图优化（`simplify=True`）。

```bash
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt --imgsz 640
```

### `scripts/build_tensorrt.py`
将 `.pt` 或 `.onnx` 编译为 TensorRT `.engine`，支持 FP16 和 INT8 量化，适合 Jetson 等 NVIDIA 边缘设备。

```bash
python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --half
```

### `scripts/infer_onnx.py` / `scripts/infer_trt.py`
独立的推理基准脚本，跑 300 帧后输出平均 FPS，用于在目标设备上快速验证推理性能，不依赖追踪和计数逻辑。

---

## 五、数据流总览

```
VideoCapture
     │ BGR frame (H×W×3)
     ▼
BikeDetector.detect()
     │ List[Detection]  (归一化 bbox, conf, class_id=1)
     ▼
BikeTracker.update()
     │ List[Track]  (track_id, bbox, trajectory)
     ▼
FlowCounter.update()
     │ int  (累计穿线数)
     ▼
BikeDetectionSystem._draw()  →  cv2.imshow()
```

---

## 六、快速上手

```bash
# 安装依赖
pip install -r requirements.txt

# 使用摄像头实时检测
python -m campus_bike_detection.main --source 0 --backend pt --device cuda

# 使用视频文件
python -m campus_bike_detection.main --source data/IMG_1258.MP4 --backend pt --device cpu

# 导出 ONNX 并在 CPU 上运行
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt
python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu

# 无头模式（服务器/边缘设备）
python -m campus_bike_detection.main --source 0 --no-show --no-trails
```

---

## 七、边缘部署建议

| 平台 | 推荐后端 | 推荐参数 |
|---|---|---|
| NVIDIA Jetson | TensorRT FP16 | `--backend trt --device cuda` |
| RK3588 | ONNX CPU | `--backend onnx --device cpu` |
| 树莓派 | ONNX CPU | `--backend onnx --device cpu --imgsz 320 --no-show` |
| PC (NVIDIA GPU) | PyTorch 或 TRT | `--backend pt --device cuda` |

详见 `docs/EDGE_DEPLOYMENT.md`。

---

## 八、依赖

```
numpy>=1.24
opencv-python>=4.8
ultralytics>=8.2      # YOLOv8 推理（pt/trt 后端）
onnxruntime>=1.17     # ONNX 推理后端
```
