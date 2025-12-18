# YOLO-OBB 卫星影像裁剪融合与坐标转换工具

基于 YOLO 旋转框检测结果，对高分卫星影像（GF1/GF2等）进行多光谱与全色影像的裁剪、配准、融合处理，并支持将分块图坐标转换回原始大图坐标。

## 功能概述

1. **影像裁剪融合** (`yolo_clip_fuse.py`)
   - 根据 YOLO-OBB 检测框裁剪 MSS（多光谱）和 PAN（全色）影像
   - RPC 坐标转换实现 MSS→PAN 精确配准
   - 支持多种融合算法：Gram-Schmidt、Brovey、IHS
   - 自动保存裁剪位置元数据

2. **坐标转换** (`convert_coords_to_global.py`)
   - 将分块图的局部坐标转换为原始大图的全局坐标
   - 支持 YOLO txt 标签和 JSON 分割结果的转换

## 目录结构

```
.
├── pic/                          # 输入：原始卫星影像
│   └── {scene_name}/
│       ├── *-MSS*.tiff          # 多光谱影像
│       ├── *-PAN*.tiff          # 全色影像
│       ├── *.rpb                # RPC参数文件（可选）
│       └── *.xml                # 元数据文件
│
├── labels/                       # 输入：YOLO-OBB 标签
│   └── {scene_name}-MSS*.txt    # 格式: class x1 y1 x2 y2 x3 y3 x4 y4 [score]
│
├── yolo_outputs/                 # 输出：裁剪融合结果
│   └── {scene_name}/
│       ├── det_000_fused.png    # 融合后的裁剪图
│       ├── det_000_ms_crop.png  # MSS 裁剪图
│       ├── det_000_pan_crop.png # PAN 裁剪图
│       └── det_000_metadata.json # 位置元数据 ⭐
│
├── results_json/                 # 输入：分割结果（相对于分块图）
│   └── {scene_name}/
│       └── det_000_fused.json   # 分割多边形（归一化坐标）
│
├── global_outputs/               # 输出：全局坐标结果
│   ├── crop_metadata.json       # 所有分块的元数据汇总
│   ├── global_labels/           # YOLO格式全局标签
│   │   └── {scene_name}/
│   │       └── det_000_global.txt
│   └── global_results_json/     # 转换后的分割JSON
│       └── {scene_name}/
│           └── det_000_global.json
│
├── yolo_clip_fuse.py            # 主程序：裁剪融合
├── convert_coords_to_global.py  # 坐标转换脚本
└── image_utils.py               # 图像处理工具函数
```

## 数据格式

### YOLO-OBB 标签格式 (输入)
```
class x1 y1 x2 y2 x3 y3 x4 y4 [score]
```
- 4个角点的归一化坐标（相对于 MSS 影像，范围 0-1）
- 示例：`1 0.260344 0.437152 0.293657 0.380854 0.268787 0.364420 0.235474 0.420719 0.909819`

### 分割结果 JSON 格式 (输入)
```json
[
  {
    "Key_area_id": "0",
    "Key_area_type": "genset",
    "polygon": [[0.435, 0.575], [0.433, 0.576], ...]
  }
]
```
- `polygon`: 归一化坐标，相对于分块图（范围 0-1）

### 位置元数据 JSON 格式 (输出)
```json
{
  "det_idx": 0,
  "global_offset_x": 12345,
  "global_offset_y": 6789,
  "crop_width": 2450,
  "crop_height": 2966,
  "pan_size": [29200, 27068],
  "mss_size": [7300, 6767],
  "class_id": "1",
  "score": 0.909819
}
```

### 全局坐标 JSON 格式 (输出)
```json
[
  {
    "Key_area_id": "0",
    "Key_area_type": "genset",
    "polygon": [[0.0234, 0.0456], ...],
    "polygon_pixel": [[683.5, 1234.2], ...],
    "crop_metadata": {
      "global_offset_x": 12345,
      "global_offset_y": 6789,
      "crop_width": 2450,
      "crop_height": 2966
    }
  }
]
```

## 使用方法

### 1. 裁剪融合

```bash
python yolo_clip_fuse.py \
    --input-dir pic \
    --label-dir labels \
    --output-dir yolo_outputs \
    --method gram_schmidt \
    --scale 1.3
```

**参数说明：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-dir` | `pic` | 原始影像根目录 |
| `--label-dir` | `labels` | YOLO 标签目录 |
| `--output-dir` | `yolo_outputs` | 输出目录 |
| `--scale` | `1.3` | 检测框放大倍数 |
| `--method` | `gram_schmidt` | 融合算法: `brovey`, `ihs`, `gram_schmidt`, `mean` |
| `--scene-expand` | `3.0` | 场景窗口放大倍数 |
| `--preview-method` | `clahe` | 预览增强方式 |

### 2. 坐标转换

```bash
python convert_coords_to_global.py \
    --input-dir pic \
    --label-dir labels \
    --output-dir yolo_outputs \
    --results-json-dir results_json \
    --global-output-dir global_outputs
```

## 坐标转换原理

### 坐标系统

```
原始大图 (PAN)                    分块图 (Crop)
┌────────────────────┐           ┌──────────┐
│                    │           │          │
│   ┌──────────┐     │           │  (lx,ly) │
│   │  Crop    │     │    ──►    │    ●     │
│   │   ●      │     │           │          │
│   └──────────┘     │           └──────────┘
│  (gx,gy)           │
└────────────────────┘
```

### 转换公式

**局部 → 全局（像素坐标）：**
```python
global_x = global_offset_x + local_x
global_y = global_offset_y + local_y
```

**归一化坐标转换：**
```python
# 分块图归一化 → 分块图像素
local_px = norm_x * crop_width
local_py = norm_y * crop_height

# 分块图像素 → 原图像素
global_px = global_offset_x + local_px
global_py = global_offset_y + local_py

# 原图像素 → 原图归一化
global_norm_x = global_px / pan_width
global_norm_y = global_py / pan_height
```

## 处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    yolo_clip_fuse.py                            │
├─────────────────────────────────────────────────────────────────┤
│  1. 读取 YOLO-OBB 标签（归一化到 MSS）                           │
│  2. MSS 归一化坐标 → MSS 像素坐标                                │
│  3. MSS 像素坐标 → PAN 像素坐标（RPC/仿射变换）                   │
│  4. 计算对齐窗口（自适应大小）                                    │
│  5. 生成控制点网格                                               │
│  6. Warp MSS 到 PAN 坐标系                                       │
│  7. 特征配准微调（可选）                                         │
│  8. 计算有效重叠区域                                             │
│  9. 裁剪并融合                                                   │
│ 10. 保存结果 + 元数据 ⭐                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                convert_coords_to_global.py                      │
├─────────────────────────────────────────────────────────────────┤
│  1. 加载元数据（优先使用精确元数据，否则估算）                    │
│  2. 读取 results_json 分割结果                                   │
│  3. 转换坐标：分块归一化 → 原图归一化/像素                        │
│  4. 保存全局 YOLO 标签                                           │
│  5. 保存全局 JSON 结果                                           │
└─────────────────────────────────────────────────────────────────┘
```

## 依赖

```
numpy
pillow
rasterio
opencv-python
```

## 注意事项

1. **精度问题**：如果没有 `det_xxx_metadata.json`，坐标转换脚本会通过重新计算来估算偏移，精度可能略低。建议重新运行 `yolo_clip_fuse.py` 生成精确元数据。

2. **RPC 文件**：如果影像目录中存在 `.rpb` 文件，将使用 RPC 进行精确配准；否则使用仿射变换。

3. **内存管理**：程序逐个处理检测框，避免内存溢出。

4. **支持格式**：
   - 输入影像：GeoTIFF (`.tif`, `.tiff`)
   - 输出裁剪：PNG
   - 坐标文件：TXT (YOLO格式), JSON

## 示例

```bash
# 完整流程
python yolo_clip_fuse.py
python convert_coords_to_global.py

# 查看转换后的全局坐标
cat global_outputs/global_results_json/GF2_xxx/det_000_global.json
```
