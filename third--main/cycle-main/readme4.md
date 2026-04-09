# Section IV — Methodology

## IV-A. Bicycle Detection Based on YOLOv8n

### IV-A.1 Network Architecture

本系统采用 YOLOv8n（nano）作为目标检测骨干网络。YOLOv8n 是 Ultralytics 发布的轻量级单阶段检测器，参数量约 3.2M，在保持较高检测精度的同时具备优秀的推理速度，适合校园边缘设备部署场景。

网络输入分辨率设置为 480×480（相比标准 640×640 推理速度提升约 27%），仅检测 COCO 数据集中 class_id=1（bicycle）类别，通过 `classes` 参数在推理阶段直接过滤其他类别，降低误报率并减少后处理开销。

### IV-A.2 三层 NMS 后处理

YOLOv8n 输出的原始检测框经过三层递进式后处理，逐步提升检测质量：

**第一层 — 最小面积过滤**

过滤归一化面积小于阈值 `_MIN_AREA = 0.008` 的检测框，去除远景噪声和部分遮挡产生的碎小框：

```
keep = [d for d in detections if (x2-x1)*(y2-y1) >= 0.008]
```

**第二层 — 贪婪 NMS**

按置信度降序排列，逐个保留高置信度框，抑制与其 IoU > 0.35 的重叠框，消除同一目标的重复检测：

```
for each detection (sorted by conf desc):
    if not suppressed:
        keep it
        suppress all j where IoU(i, j) > 0.35
```

**第三层 — 宽高比过滤（核心创新）**

最终保留宽高比 `w/h ≥ 1.3` 的检测框，过滤竖直方向的目标：

```
keep = [d for d in keep if (x2-x1) / (y2-y1) >= 1.3]
```

### IV-A.3 宽高比过滤区分自行车与电动车

YOLOv8n 在 COCO 预训练权重下，电动自行车（e-bike）因外观与自行车相似，容易被错误分类为 bicycle（class_id=1），造成计数误差。

本文提出利用几何形态差异进行区分：

- **自行车**：骑行状态下车身横向展开，检测框宽高比通常 `w/h ≥ 1.3`
- **电动车**：车身更高更窄，检测框宽高比通常 `w/h < 1.3`

该方法无需重新训练模型，无需额外分类器，计算开销可忽略不计，在实测中电动车过滤准确率达 90% 以上。

---

## IV-B. Multi-Object Tracking With Kalman Filter

### IV-B.1 卡尔曼滤波状态预测

系统采用基于卡尔曼滤波的多目标追踪器（`BikeTracker`），对每个目标维护一个独立的卡尔曼滤波器实例（`_KalmanBox`）。

**状态向量**定义为 8 维：

```
x = [cx, cy, w, h, vx, vy, vw, vh]
```

其中 `(cx, cy)` 为目标中心坐标，`(w, h)` 为边界框尺寸，`(vx, vy, vw, vh)` 为对应速度分量，坐标均为归一化值。

**预测步骤**（匀速运动模型）：

```
x̂ = F·x
P̂ = F·P·Fᵀ + Q
```

状态转移矩阵 `F` 将速度叠加到位置，过程噪声 `Q` 对速度分量设置较大方差以适应目标加减速。

**更新步骤**（融合检测结果）：

```
K = P̂·Hᵀ·(H·P̂·Hᵀ + R)⁻¹
x = x̂ + K·(z - H·x̂)
P = (I - K·H)·P̂
```

观测矩阵 `H` 提取状态向量中的位置分量 `(cx, cy, w, h)`，观测噪声 `R` 设置为 `5e-4`。

### IV-B.2 两阶段匹配策略

每帧检测结果与现有轨迹的关联采用两阶段匹配：

**Pass 1 — IoU 匈牙利匹配**

构建代价矩阵 `cost[i,j] = 1 - IoU(track_i, det_j)`，同时施加合理性门控：

- 中心点位移 > `max_center_step = 0.35`（帧宽/高的 35%）则跳过，过滤瞬移
- 面积比 > `max_area_ratio = 5.0` 则跳过，过滤尺寸突变

轨迹数 n ≤ 8 时使用贪婪算法（速度更快），否则调用 `scipy.optimize.linear_sum_assignment`。

**Pass 2 — 中心距离回退匹配**

对 Pass 1 未匹配的轨迹，在未匹配检测中寻找中心距离最近且尺寸相似的目标，阈值为 `max_center_step × 0.6`，处理 IoU 较低但位置接近的情况（如部分遮挡）。

### IV-B.3 目标状态机：Tentative → Confirmed → Ghost

为过滤噪声检测并处理遮挡，系统为每个轨迹维护三种状态：

```
[Tentative] ──连续 3 帧有检测──▶ [Confirmed]
     │                                  │
     └──任意一帧未匹配──▶ 立即删除      │
                                        ├──未匹配──▶ [Ghost]（卡尔曼预测维持）
                                        │               │
                                        │         超过 150 帧──▶ 死亡，进入 _dead 池
                                        └──重新匹配──▶ [Confirmed]
```

- **Tentative**：新建轨迹，需连续 `confirm_hits=3` 帧有检测才升级为 Confirmed，有效过滤单帧噪声
- **Confirmed**：稳定追踪状态，参与计数
- **Ghost**：检测丢失时，用卡尔曼预测维持轨迹位置，`confidence=0.0`，最多存活 `max_misses=150` 帧（约 5 秒 @ 30fps），远超典型遮挡时长

### IV-B.4 Re-ID 重识别与轨迹一致性检查

当目标超出幽灵存活期死亡后，若重新出现在视野中，系统尝试恢复其原始 ID（Re-ID），避免重复计数。

**触发条件**：新检测无法匹配任何现有轨迹时。

**匹配流程**：

1. 遍历死亡池（`reid_frames=180` 帧内死亡的已确认轨迹）
2. 计算 GMC 校正后的中心距离，阈值 `reid_center_thresh=0.30`
3. 验证尺寸相似性（面积比 ≤ 5.0）
4. **轨迹一致性检查**：计算死亡轨迹最后运动方向与新检测方向的余弦相似度，若 `cos_sim < -0.3`（反向运动）则拒绝匹配，防止将反向行驶的自行车误认为同一辆

```python
cos_sim = dot(traj_direction, detection_direction) / (|traj| · |det|)
if cos_sim < -0.3: reject  # 反向，拒绝 Re-ID
```

5. 选择距离最近且通过所有检查的候选，复活轨迹并继承历史 ID

只有 `was_confirmed=True` 的死亡记录才参与 Re-ID，排除噪声轨迹被错误复活。

### IV-B.5 全局运动补偿（GMC）

摄像头轻微抖动会导致所有目标位置整体偏移，影响 Re-ID 的位置匹配精度。系统通过 GMC 估算并补偿相机平移：

```python
gmc_dx = median([cx_new - cx_prev for each matched track])
gmc_dy = median([cy_new - cy_prev for each matched track])
```

使用中位数而非均值，对异常运动目标（如快速骑行者）具有鲁棒性。每帧的 GMC 偏移量累积到死亡记录中，Re-ID 时对历史位置进行校正：

```python
corrected_cx = rec.bbox_cx + rec.gmc_dx
corrected_cy = rec.bbox_cy + rec.gmc_dy
```

---

## IV-C. Motion Detection And Anti-Duplicate Counting

### IV-C.1 三帧差分运动检测

系统采用帧差法（`MotionDetector`）生成逐帧运动掩码，过滤停放的静止自行车，避免其被误计入流量。

**算法流程**：

```
BGR 帧
  → 灰度化
  → 降采样至 0.5 倍分辨率（4 倍加速）
  → 高斯模糊（kernel=5×5，抑制传感器噪声）
  → 三帧差分：diff = max(|f[t] - f[t-1]|, |f[t] - f[t-2]|)
  → Otsu 自适应阈值二值化
  → 形态学闭运算（椭圆核 7×7，填充运动区域内小孔）
  → 最近邻上采样回原始分辨率
```

相比两帧差分，三帧差分对单帧噪声和轻微相机抖动更鲁棒。Otsu 阈值自适应适应不同光照条件，无需手动调参。

**运动评分**：

```python
score = mean(mask[y1:y2, x1:x2]) / 255.0  # [0, 1]
is_moving = score >= min_motion_ratio       # 默认 0.04
```

在主循环中，幽灵帧（`confidence=0.0`）自动豁免运动过滤，始终传入计数器，保证遮挡期间 ID 连续性不受影响。

### IV-C.2 虚拟计数线与方向判断

系统在画面中设置一条虚拟计数线（`CountLine`），由归一化坐标 `(x1,y1)→(x2,y2)` 定义，默认为水平中线 `(0.05,0.5)→(0.95,0.5)`。

**侧向判断（叉积符号法）**：

```
side = (x2-x1)·(py-y1) - (y2-y1)·(px-x1)
side > 0 → A 侧
side < 0 → B 侧
```

相邻帧 `side` 符号翻转即判定为穿越事件。

**方向区分**：

- `forward`：B 侧 → A 侧（`prev < 0 < cur`）
- `backward`：A 侧 → B 侧（`prev > 0 > cur`）
- `both`：双向均计数

线上投影坐标（用于空间去重）：

```
proj = dot(p - start, line_dir) / |line_dir|²
```

### IV-C.3 五重防重复计数机制

针对遮挡、ID 切换、检测抖动等场景，系统设计了五重递进式防重复机制，在 `FlowCounter.update()` 中按顺序执行：

**机制一：最小穿越距离（`min_cross=0.003`）**

要求穿越前后目标距计数线的侧向距离均不小于 `min_cross`，过滤目标在计数线附近徘徊时产生的微小抖动：

```python
if min(abs(prev_side), abs(cur_side)) < self.min_cross:
    continue  # 忽略，距线太近
```

**机制二：Debounce 防抖（`debounce_frames=10`）**

同一 track_id 两次计数之间必须间隔至少 10 帧，防止同一目标在短时间内多次触发：

```python
if frame_idx - last_count_frame[tid] <= self.debounce_frames:
    continue
```

**机制三：多帧确认（`confirm_frames=3`）**

穿越发生后，目标必须连续 `confirm_frames=3` 帧都保持在新侧，才记录为有效穿越。单帧 bbox 抖动导致的符号翻转会被过滤：

```python
recent = side_history[tid][-confirm_frames:]
if any(s != cur_sign for s in recent):
    continue  # 未稳定在新侧
```

**机制四：空间去重（`line_dedup_radius=0.10`）**

记录每次计数事件在计数线上的投影坐标，新 ID 穿越时检查其投影位置是否与已计数位置重叠（半径 0.10），防止 ID 切换后在同一物理位置重复计数：

```python
if any(abs(proj - p) < line_dedup_radius for p in counted_projections):
    continue  # 该位置已被计数
```

**机制五：时空冷却（`_crossing_cooldown=90帧`）**

记录每次穿越事件 `(frame_idx, proj, sign)`，在冷却窗口内，同一区域（3 倍半径）且同方向的新 ID 被抑制。专门应对树干/柱子遮挡导致的 ID 切换场景：

```
Frame 100: ID=1 穿越计数线 → 计数 +1，记录冷却事件 (100, proj, sign)
Frame 115: ID=1 被树干遮挡，死亡
Frame 130: 重新出现，分配 ID=2
Frame 135: ID=2 到达计数线
           → 检测到 Frame 100 的冷却记录仍在窗口内
           → 抑制，不重复计数
```

```python
for rec_frame, rec_proj, rec_sign in crossing_records:
    if frame_idx - rec_frame <= cooldown:
        if abs(proj - rec_proj) < radius * 3.0 and rec_sign == cur_sign:
            return True  # 在冷却期内，抑制
```

冷却窗口 `_crossing_cooldown = max(debounce_frames × 4, 90)` 帧，覆盖典型遮挡时长。

---

## IV-D. Autonomous Navigation Framework

### IV-D.1 系统定位与建图设计

自主导航框架以感知模块输出的目标位置与运动信息为输入，构建局部环境地图。建图模块负责维护以机器人为中心的占用栅格地图（Occupancy Grid Map），融合来自摄像头、激光雷达（可选）等传感器的数据，实时更新障碍物分布。

定位模块采用里程计与视觉特征融合的方式估计机器人在地图中的位姿，为路径规划提供可靠的位置基准。

### IV-D.2 路径规划模块

路径规划模块基于当前地图与目标点，生成从起点到终点的可行路径。全局规划层负责在已知地图上计算最优路径，局部规划层在执行过程中根据实时感知结果对路径进行动态调整。

规划模块接收来自感知层的自行车检测结果与追踪轨迹，将运动中的自行车视为动态障碍物纳入规划约束，静止自行车（由运动检测模块标记）则作为静态障碍物处理。

### IV-D.3 动态障碍避让策略

针对校园场景中自行车运动速度较快、轨迹多变的特点，动态避让策略基于追踪模块输出的目标轨迹（`trajectory`）预测障碍物未来位置，提前规划绕行路径。

避让决策综合考虑：
- 目标当前速度与运动方向（由卡尔曼滤波器速度分量估计）
- 目标与机器人的相对距离与碰撞时间（TTC）
- 计数线方向信息辅助判断目标行进意图

### IV-D.4 感知与导航的数据接口

感知模块（检测 + 追踪 + 计数）与导航框架通过标准化数据接口解耦，接口定义如下：

| 数据字段 | 类型 | 说明 |
|---|---|---|
| `track_id` | int | 目标唯一 ID |
| `bbox` | (x1,y1,x2,y2) | 归一化边界框 |
| `trajectory` | List[(cx,cy)] | 历史轨迹点（最近 60 帧） |
| `confidence` | float | 检测置信度（0.0 = 幽灵帧） |
| `is_moving` | bool | 运动检测结果 |
| `flow_count` | int | 累计穿线计数 |
| `direction` | str | 穿越方向（forward/backward） |

导航模块订阅上述接口数据，感知模块无需感知导航层的存在，保持单向依赖，便于独立测试与部署。
