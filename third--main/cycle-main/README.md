# 校园自行车实时检测与流量统计系统

基于 YOLOv8n 的校园场景自行车实时检测、多目标追踪与流量计数系统。支持 PyTorch / ONNX / TensorRT 三种推理后端，CPU 下可达 70-90 FPS。

```
视频帧 → 目标检测 → 多目标追踪 → 跨线计数 → 可视化展示
```

---

## 目录

- [特性](#特性)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [命令行参数](#命令行参数)
- [模型导出](#模型导出)
- [边缘部署](#边缘部署)
- [模块说明](#模块说明)
- [性能指标](#性能指标)
- [常见问题](#常见问题)

---

## 特性

- **实时性能**：CPU 环境 70-90 FPS，TensorRT 后端 200+ FPS
- **多后端推理**：支持 `.pt`（PyTorch）、`.onnx`（ONNX Runtime）、`.engine`（TensorRT）
- **精准检测**：YOLOv8 内部 NMS + 自定义贪婪 NMS + 宽高比过滤，有效区分自行车与电动车，无需额外分类器
- **鲁棒追踪**：卡尔曼滤波 + 两阶段匹配，支持 5 秒遮挡保持 ID（幽灵模式）+ Re-ID
- **防重复计数**：五重保护机制（多帧确认、空间去重、时空冷却、Debounce、运动过滤）
- **可选增强**：轻量化双分支 CBAM 注意力增强，低光/雨雾场景提升检测率，开销 < 5ms/帧
- **跨平台部署**：支持 PC、Jetson、RK3588、树莓派等平台

---

## 项目结构

```
cycle-main/
├── campus_bike_detection/
│   ├── main.py            # CLI 入口
│   ├── system.py          # 主循环与可视化
│   ├── detector.py        # 多后端目标检测
│   ├── tracker.py         # 卡尔曼滤波多目标追踪
│   ├── flow_counter.py    # 跨线方向计数
│   ├── motion.py          # 帧差法运动检测
│   ├── enhancer.py        # 双分支 CBAM 特征增强
│   ├── models.py          # 数据结构定义
│   └── yolov8n.pt         # 预训练权重
├── scripts/
│   ├── export_onnx.py     # 导出 ONNX 模型
│   ├── build_tensorrt.py  # 编译 TensorRT 引擎
│   ├── infer_onnx.py      # ONNX 推理基准测试
│   └── infer_trt.py       # TensorRT 推理基准测试
├── data/                  # 测试视频
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   └── EDGE_DEPLOYMENT.md
├── requirements.txt
└── check_py38_compat.py
```

---

## 环境要求

- Python 3.8+
- （可选）CUDA 11.x+ 及对应 cuDNN，用于 GPU 推理
- （可选）TensorRT 8.x+，用于 `.engine` 后端

安装依赖：

```bash
pip install -r requirements.txt
```

核心依赖：

| 库 | 版本 | 用途 |
|---|---|---|
| ultralytics | >=8.2 | YOLOv8 推理（pt/trt 后端） |
| opencv-python | >=4.8 | 视频读取与可视化 |
| numpy | >=1.24 | 数值计算 |
| onnxruntime | >=1.17 | ONNX 推理后端 |

---

## 快速开始

### 使用摄像头实时检测

```bash
python -m campus_bike_detection.main --source 0 --backend pt --device cuda
```

### 使用视频文件

```bash
python -m campus_bike_detection.main --source data/666.mp4 --backend pt --device cpu
```

### 无头模式（服务器 / 边缘设备）

```bash
python -m campus_bike_detection.main --source data/666.mp4 --no-show --no-trails
```

### 启用图像增强（低光 / 雨雾场景）

```bash
python -m campus_bike_detection.main --source data/666.mp4 --enhance --enhance-clahe-clip 3.0
```

### 自定义计数线与方向

```bash
# 计数线坐标为归一化值 (x1,y1,x2,y2)，默认为水平中线
python -m campus_bike_detection.main \
    --source data/666.mp4 \
    --line 0.1,0.3,0.9,0.7 \
    --count-direction forward
```

---

## 命令行参数

### 检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | `0` | 摄像头 ID 或视频文件路径 |
| `--model` | `yolov8n.pt` | 模型文件路径 |
| `--backend` | `auto` | 推理后端：`auto` / `pt` / `onnx` / `trt` |
| `--device` | `cuda` | 推理设备：`cuda` / `cpu` |
| `--conf` | `0.35` | 检测置信度阈值 |
| `--iou` | `0.35` | NMS IoU 阈值 |
| `--imgsz` | `480` | 推理输入尺寸（像素） |

### 计数参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--line` | `0.05,0.5,0.95,0.5` | 计数线归一化坐标 `x1,y1,x2,y2` |
| `--count-direction` | `both` | 计数方向：`both` / `forward` / `backward` |
| `--count-min-cross` | `0.003` | 最小穿越距离，防止线附近抖动误计 |
| `--count-debounce-frames` | `10` | 同一 ID 两次计数最小间隔帧数 |
| `--count-confirm-frames` | `3` | 穿越后需连续确认的帧数 |
| `--count-line-dedup-radius` | `0.10` | 空间去重半径（归一化） |

### 增强参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--enhance` | 关闭 | 启用双分支 CBAM 增强器 |
| `--enhance-edge-weight` | `0.25` | 边缘分支融合权重 |
| `--enhance-clahe-clip` | `2.0` | CLAHE 对比度裁剪限制 |

### 运动过滤参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--motion-min-ratio` | `0.04` | 最小运动像素比例，低于此值视为静止 |

### 显示参数

| 参数 | 说明 |
|---|---|
| `--no-trails` | 关闭轨迹拖尾和侧边调试信息 |
| `--no-show` | 不显示画面，适合无头服务器 |

---

## 模型导出

### 导出 ONNX

```bash
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt --imgsz 640
```

导出后使用：

```bash
python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu
```

### 编译 TensorRT（需要 NVIDIA GPU）

```bash
# FP16 精度
python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --half

# INT8 量化（更快，精度略降）
python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --int8
```

导出后使用：

```bash
python -m campus_bike_detection.main --source 0 --model yolov8n.engine --backend trt --device cuda
```

### 推理基准测试

```bash
# 测试 ONNX 推理性能（跑 300 帧输出平均 FPS）
python scripts/infer_onnx.py --model yolov8n.onnx

# 测试 TensorRT 推理性能
python scripts/infer_trt.py --model yolov8n.engine
```

---

## 边缘部署

| 平台 | 推荐后端 | 推荐参数 |
|---|---|---|
| NVIDIA Jetson | TensorRT FP16 | `--backend trt --device cuda --imgsz 480` |
| RK3588 | ONNX CPU | `--backend onnx --device cpu` |
| 树莓派 | ONNX CPU | `--backend onnx --device cpu --imgsz 320 --no-show` |
| PC（NVIDIA GPU） | PyTorch 或 TRT | `--backend pt --device cuda` |
| PC（仅 CPU） | ONNX | `--backend onnx --device cpu` |

详细部署步骤见 [`docs/EDGE_DEPLOYMENT.md`](docs/EDGE_DEPLOYMENT.md)。

---

## 模块说明

### detector.py — 目标检测

- 固定检测 COCO `bicycle`（class_id=1），过滤摩托车等其他类别
- 后处理：YOLOv8 内部 NMS → 自定义贪婪 NMS → 宽高比过滤（`_nms` 函数）
- **宽高比过滤**：`w/h >= 1.3` 保留横向自行车，排除竖直电动车，无需额外分类器
- 最小面积过滤（`_MIN_AREA=0.008`）去除远处噪声

### tracker.py — 多目标追踪

- 卡尔曼滤波状态向量：`[cx, cy, w, h, vx, vy, vw, vh]`
- 两阶段匹配：IoU 匈牙利匹配 → 中心距离回退匹配
- 三种状态：`Tentative`（需 3 帧确认）→ `Confirmed` → `Ghost`（幽灵，最多 150 帧）
- **Re-ID**：120 帧窗口内可复活旧 ID，含轨迹一致性验证（点积检查运动方向）
- **全局运动补偿（GMC）**：中位数估算相机平移，提升 Re-ID 鲁棒性

### flow_counter.py — 流量计数

- 叉积符号法判断目标在计数线哪侧，符号翻转即为穿越
- 五重防重复机制：
  1. 多帧确认（连续 3 帧在新侧）
  2. 最小穿越距离（`min_cross=0.003`）
  3. Debounce（同 ID 间隔 10 帧）
  4. 空间去重（投影位置半径 0.10 内不重复计数）
  5. **时空冷却**（90 帧窗口内同区域同方向抑制新 ID，应对遮挡导致的 ID 切换）

### motion.py — 运动检测

- 三帧差分法，0.5 倍分辨率降采样处理（4 倍加速）
- Otsu 自适应阈值 + 形态学闭运算
- 运动评分 < `min_motion_ratio` 的目标不参与计数，过滤停车场景

### enhancer.py — 特征增强（可选）

- 双分支融合：CLAHE 对比度增强（主分支）+ Unsharp Masking 边缘增强（边缘分支）
- CBAM 注意力：通道注意力（增益裁剪）+ 空间注意力（1/4 分辨率计算，LUT 代替 sigmoid）
- 优化后开销 < 5ms/帧（原版 33ms，加速 6.6x）

---

## 性能指标

### 计算性能

| 配置 | 硬件 | FPS |
|---|---|---|
| 基础（imgsz=480） | CPU | 70-90 |
| 启用增强（imgsz=480） | CPU | 65-80 |
| 高分辨率（imgsz=640） | CPU | 50-60 |
| TensorRT FP16（imgsz=640） | RTX 3060 | 200+ |

### 计数精度

| 场景 | 重复计数率 | 漏计数率 |
|---|---|---|
| 无遮挡 | <1% | <2% |
| 轻度遮挡（<2s） | <3% | <5% |
| 重度遮挡（>3s） | <8% | <10% |

---

## 常见问题

**Q：如何提高检测精度？**
提高推理分辨率 `--imgsz 640`，启用增强器 `--enhance`，或适当降低置信度阈值 `--conf 0.30`。

**Q：如何减少重复计数？**
增加确认帧数 `--count-confirm-frames 5`，或增大空间去重半径 `--count-line-dedup-radius 0.15`。

**Q：如何提高帧率？**
降低推理分辨率 `--imgsz 416`，禁用可视化 `--no-show`，或使用 TensorRT 后端 `--backend trt`。

**Q：夜间 / 低光场景效果差？**
启用增强器并提高 CLAHE 强度：`--enhance --enhance-clahe-clip 3.0`，同时适当降低置信度阈值 `--conf 0.25`。

**Q：如何适配不同摄像头角度？**
调整计数线位置 `--line x1,y1,x2,y2`，俯视角可适当增大 `--count-min-cross 0.005`，侧视角可减小至 `0.002`。

---

## 依赖与许可

依赖库遵循各自开源协议（Ultralytics AGPL-3.0、OpenCV Apache-2.0 等）。

本项目采用 [MIT License](../LICENSE)。

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- SORT / DeepSORT 多目标追踪算法
