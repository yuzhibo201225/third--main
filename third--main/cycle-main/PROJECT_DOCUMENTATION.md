# 校园自行车实时检测与计数系统

## 项目概述

基于 YOLOv8n 的校园场景自行车实时检测、多目标追踪与流量统计系统。核心流水线：

```
视频帧 → [可选增强] → 目标检测 → 多目标追踪 → 运动过滤 → 跨线计数 → 可视化
```

支持 PyTorch / ONNX / TensorRT 三种推理后端，CPU 下可达 70-90 FPS。

### 核心特性

- 实时性能：CPU 70-90 FPS，TensorRT 200+ FPS
- 三层 NMS + 宽高比过滤（`w/h >= 1.3`），无需额外分类器区分自行车与电动车
- 卡尔曼滤波追踪，幽灵模式最长保持 150 帧（约 5 秒），Re-ID 窗口 180 帧
- 五重防重复计数：多帧确认 + 空间去重 + 时空冷却 + Debounce + 运动过滤
- 可选轻量 CBAM 增强器，开销 < 2ms/帧

### 技术栈

- 深度学习：Ultralytics YOLOv8、ONNX Runtime、TensorRT
- 计算机视觉：OpenCV
- 数值计算：NumPy、SciPy（大规模匈牙利匹配时）
- Python 3.8+

---

## 系统架构

```
BikeDetectionSystem (system.py)
  ├── BikeDetector   (detector.py)   多后端推理 + 三层 NMS
  ├── BikeTracker    (tracker.py)    卡尔曼滤波 + Re-ID
  ├── MotionDetector (motion.py)     帧差法运动过滤
  ├── FlowCounter    (flow_counter.py) 跨线方向计数
  └── DualBranchEnhancer (enhancer.py) [可选] CBAM 增强
```

---

## 目录结构

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
│   ├── export_onnx.py
│   ├── build_tensorrt.py
│   ├── infer_onnx.py
│   └── infer_trt.py
├── data/                  # 测试视频
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   └── EDGE_DEPLOYMENT.md
├── requirements.txt
└── check_py38_compat.py
```

---

## 模块详解

### 1. models.py — 数据结构

全项目共用的纯数据类，无业务逻辑，作为各模块之间的契约。

| 类 | 关键字段 | 说明 |
|---|---|---|
| `Detection` | `bbox, confidence, class_id` | 单帧单个检测框，坐标归一化到 [0,1] |
| `Track` | `track_id, bbox, confidence, trajectory` | 带 ID 的追踪目标，`trajectory` 存储历史中心点（最多 60 个） |
| `CountLine` | `line_id, start, end` | 计数线，坐标归一化 |
| `SystemConfig` | 全部 CLI 参数 | 系统全局配置，由 `main.py` 填充 |
| `SessionReport` | `total_frames, avg_fps, peak_count, total_count, line_counts` | 会话结束后的统计报告 |

所有坐标均使用归一化坐标（相对于帧宽高的比例），使检测、追踪、计数逻辑与分辨率无关。

`SystemConfig` 默认值（与代码一致）：

```python
conf=0.25, iou=0.5, imgsz=640, backend="auto", device="cuda"
count_direction="both", count_min_cross=0.003
count_debounce_frames=10, count_confirm_frames=3
count_line_dedup_radius=0.10
enhance=False, enhance_edge_weight=0.25, enhance_clahe_clip=2.0
motion_min_ratio=0.04
```

---

### 2. detector.py — 目标检测

**职责**：接收原始帧（可选先经增强器处理），返回当前帧中所有自行车的 `Detection` 列表。

#### 后端选择

通过文件扩展名或 `--backend` 参数自动选择：

| 后端 | 文件格式 | 推理库 | 适用场景 |
|---|---|---|---|
| `pt` | `.pt` | ultralytics YOLO | 开发调试、GPU 服务器 |
| `onnx` | `.onnx` | onnxruntime | CPU 边缘设备、跨平台 |
| `trt` | `.engine` / `.trt` | ultralytics + TensorRT | Jetson / NVIDIA GPU 高性能部署 |

#### 后处理过滤（`_nms` 函数）

pt/trt 后端先经过 YOLOv8 内部 NMS（`agnostic_nms=True, iou=0.35, max_det=50`），之后统一进入 `_nms()`：

```
步骤 1：最小面积过滤（_MIN_AREA=0.008，归一化面积），去除远处噪声小框
步骤 2：贪婪 NMS（按置信度排序，抑制 IoU > 0.35 的重叠框）
步骤 3：宽高比过滤（w/h >= 1.3），保留横向自行车，排除竖直电动车
```

onnx 后端在解析输出时先做置信度过滤，再进入同一个 `_nms()`。

宽高比过滤是区分自行车与电动车的核心手段，无需额外分类器，零额外计算。

#### ONNX 推理流程

1. resize 到 `imgsz × imgsz`，BGR → RGB，归一化到 [0,1]
2. 转为 `(1, 3, H, W)` float32 张量送入 ONNX Runtime
3. 兼容 `(N,6)` 和 `(6,N)` 两种输出布局，过滤置信度和类别，像素坐标归一化后返回

#### Warmup

系统启动时用全零帧跑一次推理，预热模型（对 TensorRT 尤为重要，避免首帧延迟）。

---

### 3. tracker.py — 多目标追踪

**职责**：将每帧的 `Detection` 列表关联成跨帧连续的 `Track`，为每个目标分配稳定的 `track_id`。

#### 卡尔曼滤波器（`_KalmanBox`）

- 状态向量：`[cx, cy, w, h, vx, vy, vw, vh]`（位置 + 速度）
- 预测：匀速运动模型（`_F` 矩阵）
- 更新：标准卡尔曼增益融合检测结果

#### 两阶段匹配

**Pass 1 — IoU 匈牙利匹配**

- 构建代价矩阵：`cost[i,j] = 1 - IoU(track_i, det_j)`
- 合理性门控（`_plausible`）：中心距离 > `max_center_step=0.35` 或面积比 > `max_area_ratio=5.0` 则跳过
- n ≤ 8 时用贪婪算法，否则调用 `scipy.optimize.linear_sum_assignment`

**Pass 2 — 中心距离回退匹配**

- 对 Pass 1 未匹配的轨迹，寻找最近的未匹配检测
- 阈值：`max_center_step * 0.6`，同时要求尺寸相似

#### 状态机

```
Tentative（新建）
  → 连续 confirm_hits=3 帧有检测 → Confirmed（确认）
  → 任意一帧未匹配 → 立即删除（不进入幽灵）

Confirmed（确认）
  → 未匹配 → Ghost（幽灵，最多 max_misses=150 帧）
  → 超过 max_misses → 死亡，进入 _dead 池

Ghost（幽灵）
  → 卡尔曼预测维持轨迹，confidence=0.0
  → 重新匹配到检测 → 恢复 Confirmed
```

Tentative 轨迹不参与计数，有效过滤瞬时噪声。

#### Re-ID 机制

**触发**：新检测无法匹配任何现有轨迹时。

**匹配流程**：
1. 遍历死亡池（`reid_frames=180` 帧内死亡的已确认轨迹）
2. 计算 GMC 校正后的中心距离，阈值 `reid_center_thresh=0.30`
3. 验证尺寸相似性（`max_area_ratio=5.0`）
4. 轨迹一致性检查（余弦相似度 < -0.3 则拒绝，防止反向目标误匹配）
5. 复活轨迹，继承历史轨迹和 ID，直接标记为 Confirmed

只有 `was_confirmed=True` 的死亡记录才参与 Re-ID，避免噪声 ID 被复活。

#### 全局运动补偿（GMC）

```python
# 计算所有匹配轨迹的位移中位数，估算相机平移
gmc_dx = median([cx_new - cx_prev for each matched track])
gmc_dy = median([cy_new - cy_prev for each matched track])
# 累积到死亡记录，Re-ID 时校正历史位置
corrected_x = rec.bbox_cx + rec.gmc_dx
```

#### Spawn 去重

新检测在以下情况下被抑制，不创建新 ID：
- 与现有轨迹中心距离 < `max_center_step * 0.4` 且尺寸相似
- 与 30 帧内死亡的轨迹（GMC 校正后）中心距离 < `reid_center_thresh * 1.5` 且尺寸相似

#### `total_unique()`

返回 `confirmed_ids` 集合大小，即整个会话中被确认过的不重复 ID 数量，排除噪声 ID。

---

### 4. flow_counter.py — 跨线方向计数

**职责**：判断每个 track 是否穿越了计数线，区分正向/反向，统计流量。

#### 几何原理

叉积符号法判断点在线哪侧：

```python
side = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
# side > 0: A 侧；side < 0: B 侧
```

相邻帧 `side` 符号翻转即为穿越。

线上投影（用于空间去重）：

```python
proj = dot(p - start, line_dir) / line_length²  # 归一化到 [0,1]
```

#### 五重防重复机制

**1. 多帧确认（`confirm_frames=3`）**
穿越后，目标必须连续 3 帧都在新侧，才记录为有效穿越。过滤单帧 bbox 抖动。

**2. 最小穿越距离（`min_cross=0.003`）**
穿越前后的 `|side|` 都必须 ≥ 0.003，防止在线附近徘徊的目标反复触发。

**3. Debounce（`debounce_frames=10`）**
同一 ID 两次计数之间至少间隔 10 帧。

**4. 空间去重（`line_dedup_radius=0.10`）**
记录每次计数的线上投影坐标，新 ID 穿越时检查投影位置是否与已计数位置重叠（半径 0.10），防止 ID 切换导致的同位置重复计数。

**5. 时空冷却（`_crossing_cooldown = max(debounce_frames*4, 90)` 帧）**
记录每次穿越事件 `(frame_idx, proj, sign)`，在冷却窗口内，同一区域（3 倍半径）+ 同方向的新 ID 被抑制。专门应对树干/柱子遮挡导致的 ID 切换：

```
Frame 100: ID=1 穿越计数线 → 计数 +1，记录冷却事件
Frame 110: ID=1 被遮挡死亡
Frame 120: 重新出现，分配 ID=2
Frame 125: ID=2 到达计数线 → 检测到冷却记录 → 抑制，不重复计数
```

每个 ID 只计数一次（`counted_ids` 集合）。

---

### 5. motion.py — 运动检测

**职责**：基于帧差法生成运动掩码，过滤静止目标，避免停放自行车被计数。

#### 算法流程

```
BGR 帧
  → 灰度化
  → 降采样到 0.5 倍分辨率（4 倍加速）
  → 高斯模糊（kernel=5，抑制噪声）
  → 三帧差分：diff = max(|f[t]-f[t-1]|, |f[t]-f[t-2]|)
  → Otsu 自适应阈值二值化
  → 形态学闭运算（椭圆核 7×7，填充小孔）
  → 最近邻上采样回原始分辨率
```

三帧差分比两帧差分对单帧噪声和轻微相机抖动更鲁棒。

#### 运动评分

```python
def box_motion_score(bbox, frame_shape) -> float:
    roi = mask[y1:y2, x1:x2]
    return roi.mean() / 255.0  # [0, 1]

def is_moving(bbox, frame_shape) -> bool:
    return box_motion_score(bbox, frame_shape) >= min_motion_ratio  # 默认 0.04
```

在 `system.py` 中，幽灵帧（`confidence=0.0`）自动豁免运动过滤，始终传入计数器（已被追踪，不应因静止而丢失 ID）。

---

### 6. enhancer.py — 特征增强（可选）

**职责**：在检测前对帧进行图像增强，提升低光/雨雾/低对比度场景的检测率。

#### 双分支融合

```
Branch 1（主分支）：CLAHE 对比度增强
  → 转 LAB 色彩空间，仅处理 L 通道
  → CLAHE(clipLimit=2.0, tileGridSize=8×8)
  → 转回 BGR

Branch 2（边缘分支）：Laplacian 锐化
  → sharpened = 2*orig - GaussianBlur(orig, 3×3)
  → 纯整数运算，无浮点

融合：fused = addWeighted(main, 0.75, edge, 0.25, 0)
```

#### CBAM 注意力

**通道注意力**：
```python
means = [cv2.mean(frame[:,:,c]) for c in range(3)]  # 3 个标量
gate = clip(means / global_mean, 0.88, 1.12)         # 增益范围 ±12%
# 用 cv2.convertScaleAbs 在 uint8 域应用增益
```

**空间注意力**（1/4 分辨率计算，约 16 倍加速）：
```python
small = resize(frame, 1/4)
saliency = 0.5*gray_avg + 0.5*gray_max
smoothed = GaussianBlur(saliency, kernel=7)
mask = resize(smoothed, full_res)
# LUT 代替 float sigmoid：lut[i] = clip(i * strength, 0, 255)
alpha = LUT(mask, lut)
output = cv2.add(frame, merge([alpha, alpha, alpha]))  # 饱和加法
```

全程使用 OpenCV C++ 路径，无 Python 循环，目标开销 < 2ms/帧（640×640）。

---

### 7. system.py — 系统集成

**主循环（`run()`）**：

```python
while 有帧:
    frame = cap.read()
    detections = detector.detect(frame)          # 检测
    tracks = tracker.update(detections)          # 追踪
    motion_mask = motion.update(frame)           # 运动掩码
    moving_tracks = [t for t in tracks
        if t.confidence == 0.0 or motion.is_moving(t.bbox, shape)]
    total_flow = counter.update(moving_tracks, frame_idx)
    fps = 1.0 / elapsed
    if show: _draw(frame, tracks, count, flow, fps)
    frame_idx += 1
```

**可视化（`_draw()`）**：
- 运动目标：绿色框 + `ID:x Conf:x.xx`
- 静止目标：灰色框
- 开启 `--draw-trails`：橙色轨迹拖尾（最近 20 点）+ 侧边标签（A/B）
- 红色计数线
- 左上角：FPS / Count / Flow / Fwd / Bwd

**资源管理**：实现 `__enter__`/`__exit__`，以 `with` 语句使用，确保摄像头和窗口在任何情况下都能正确释放。

**追踪器实例化参数**（`system.py` 中硬编码）：

```python
BikeTracker(
    iou_thresh=0.25,
    max_misses=150,          # 5 秒 @ 30fps
    confirm_hits=3,
    reid_frames=180,         # 6 秒 Re-ID 窗口
    reid_center_thresh=0.30,
    max_area_ratio=5.0,
)
```

---

### 8. main.py — CLI 入口

解析命令行参数，构建 `SystemConfig`，启动 `BikeDetectionSystem`，会话结束后打印统计报告。

#### 完整参数列表

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--source` | `0` | 摄像头 ID 或视频文件路径 |
| `--model` | `yolov8n.pt` | 模型文件路径 |
| `--backend` | `auto` | `auto` / `pt` / `onnx` / `trt` |
| `--device` | `cuda` | `cuda` / `cpu` |
| `--imgsz` | `480` | 推理输入尺寸 |
| `--conf` | `0.35` | 检测置信度阈值 |
| `--iou` | `0.35` | NMS IoU 阈值 |
| `--line` | `0.05,0.5,0.95,0.5` | 计数线归一化坐标 `x1,y1,x2,y2` |
| `--count-direction` | `both` | `both` / `forward` / `backward` |
| `--count-min-cross` | `0.003` | 最小穿越距离 |
| `--count-debounce-frames` | `10` | 同 ID 两次计数最小间隔帧数 |
| `--count-confirm-frames` | `3` | 穿越后需连续确认的帧数 |
| `--count-line-dedup-radius` | `0.10` | 空间去重半径 |
| `--enhance` | 关闭 | 启用双分支 CBAM 增强器 |
| `--enhance-edge-weight` | `0.25` | 边缘分支融合权重 |
| `--enhance-clahe-clip` | `2.0` | CLAHE 对比度裁剪限制 |
| `--motion-min-ratio` | `0.04` | 最小运动像素比例（0=关闭） |
| `--no-trails` | — | 关闭轨迹拖尾和侧边调试信息 |
| `--no-show` | — | 不显示画面（无头模式） |

会话结束后输出：

```
=== Session Report ===
Frames      : 402
Avg FPS     : 86.21
Peak Count  : 1
Total Bikes : 2
Line Counts : {'main': 2, 'main_forward': 1, 'main_backward': 1}
```

---

## 数据流总览

```
VideoCapture
     │ BGR frame (H×W×3)
     ▼
[DualBranchEnhancer]  ← 可选，--enhance
     │ enhanced frame
     ▼
BikeDetector.detect()
     │ List[Detection]  (归一化 bbox, conf, class_id=1)
     ▼
BikeTracker.update()
     │ List[Track]  (track_id, bbox, trajectory, confidence)
     ▼
MotionDetector.update() + is_moving()
     │ List[Track]  (仅运动目标，幽灵帧豁免)
     ▼
FlowCounter.update()
     │ int  (累计穿线数)
     ▼
BikeDetectionSystem._draw()  →  cv2.imshow()
```

---

## 快速上手

```bash
pip install -r requirements.txt

# 摄像头实时检测
python -m campus_bike_detection.main --source 0 --backend pt --device cuda

# 视频文件
python -m campus_bike_detection.main --source data/666.mp4 --backend pt --device cpu

# ONNX CPU 模式
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt
python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu

# 无头模式（服务器/边缘设备）
python -m campus_bike_detection.main --source 0 --no-show --no-trails

# 低光/雨雾场景
python -m campus_bike_detection.main --source data/666.mp4 --enhance --enhance-clahe-clip 3.0
```

---

## 模型导出

```bash
# 导出 ONNX
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt --imgsz 640

# 编译 TensorRT FP16
python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --half

# 推理基准测试（300 帧，输出平均 FPS）
python scripts/infer_onnx.py --model yolov8n.onnx
python scripts/infer_trt.py  --model yolov8n.engine
```

---

## 边缘部署

| 平台 | 推荐后端 | 推荐参数 |
|---|---|---|
| NVIDIA Jetson | TensorRT FP16 | `--backend trt --device cuda` |
| RK3588 | ONNX CPU | `--backend onnx --device cpu` |
| 树莓派 | ONNX CPU | `--backend onnx --device cpu --imgsz 320 --no-show` |
| PC（NVIDIA GPU） | PyTorch 或 TRT | `--backend pt --device cuda` |

详见 `docs/EDGE_DEPLOYMENT.md`。

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

## 技术难点与解决方案

### 遮挡导致的 ID 切换

**问题**：自行车被树干/行人遮挡，重新出现时分配新 ID，导致重复计数。

**解决**：
1. 幽灵模式（`max_misses=150`，约 5 秒）远超典型遮挡时长
2. Re-ID 窗口 180 帧，GMC 补偿相机抖动
3. 轨迹一致性验证（余弦相似度）防止反向目标误匹配
4. 时空冷却机制（90 帧）即使 ID 切换也不重复计数

### 低置信度噪声检测

**问题**：远处模糊物体、部分遮挡产生低置信度检测，被追踪器当作真实目标。

**解决**：
1. 置信度阈值 0.35（`main.py` 默认）
2. Tentative 状态需连续 3 帧确认
3. 最小面积过滤（`_MIN_AREA=0.008`）
4. `total_unique()` 只统计 `confirmed_ids`

### 电动车误检为自行车

**问题**：YOLOv8n 在 COCO 上，电动自行车容易被分类为 `bicycle`（class_id=1）。

**解决**：宽高比过滤 `w/h >= 1.3`，自行车横向，电动车竖直，零额外计算。

### 实时性能

**问题**：初版帧率不足，增强器开销 33ms/帧。

**解决**：
- 推理分辨率 640 → 480（27% 加速）
- n ≤ 8 时贪婪匹配代替匈牙利算法
- 增强器空间注意力降至 1/4 分辨率（16 倍加速）+ LUT 代替 sigmoid
- 运动检测降至 0.5 倍分辨率（4 倍加速）

---

## 参数调优指南

### 高密度场景

```bash
python -m campus_bike_detection.main --source video.mp4 \
    --conf 0.30 --imgsz 640 --count-confirm-frames 5
```

### 遮挡严重场景

在 `system.py` 中调整追踪器参数：`max_misses=180, reid_frames=240`

### 低光/雨雾

```bash
python -m campus_bike_detection.main --source video.mp4 \
    --enhance --enhance-clahe-clip 3.0 --enhance-edge-weight 0.3 --conf 0.25
```

### 性能优先

```bash
python -m campus_bike_detection.main --source video.mp4 \
    --imgsz 416 --no-show --motion-min-ratio 0.0
```

---

## 依赖

```
numpy>=1.24
opencv-python>=4.8
ultralytics>=8.2
onnxruntime>=1.17
```

---

## 许可证

MIT License。详见 LICENSE 文件。

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
- SORT / DeepSORT 多目标追踪算法
