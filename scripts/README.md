# 一键运行脚本

基于 `rs_processor` 框架的一键运行脚本集合。

## 脚本列表

| 脚本 | 功能 | 说明 |
|------|------|------|
| `run_full_pipeline.py` | 完整流水线（推荐） | 整图融合 → 裁剪检测框 |
| `run_fuse.py` | 整图融合 | PAN/MSS 配准 + 锐化 |
| `run_crop.py` | YOLO 检测框裁剪 | 从融合图裁剪 |
| `run_convert.py` | 坐标转换 | 分块 -> 全局 |
| `run_convert_global.py` | 坐标转换到全局 | 含经纬度输出 |
| `run_pipeline.py` | 分步流水线 | 调用其他脚本 |

## 使用方法

### 1. 修改配置

每个脚本顶部都有 `用户配置区`，修改路径后即可运行：

```python
# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"              # 影像根目录
LABEL_DIR = r"F:\labels_export"         # YOLO 标签目录
OUTPUT_DIR = r"F:\output"               # 输出目录
```

### 2. 运行脚本

```bash
# 方式1：直接运行（使用脚本内配置）
python scripts/run_full_pipeline.py

# 方式2：命令行参数
python scripts/run_full_pipeline.py --input-dir F:\data --label-dir F:\labels --output-dir F:\output
```

## 脚本详解

### run_full_pipeline.py（推荐）

完整流水线，新流程：

**Step 1: 整图融合**
1. 读取 PAN/MSS 影像
2. RPC 粗配准 + 特征点精配准
3. Gram-Schmidt 全色锐化
4. 保存融合后的 TIFF 和 PNG 预览

**Step 2: 裁剪检测框**
1. 读取 YOLO OBB 标签
2. 从融合图直接裁剪（不需要扩大范围）
3. 保存裁剪图片和元数据 JSON

```bash
python scripts/run_full_pipeline.py \
    --input-dir F:\code\pic \
    --label-dir F:\labels_export \
    --output-dir F:\output \
    --scale 1.3 \
    --no-tiff  # 可选：不保存融合 TIFF
```

输出结构：
```
output/
├── fused/                    # 融合结果
│   └── scene_name/
│       ├── fused.tif         # 融合后的 TIFF
│       ├── fused_preview.png # PNG 预览
│       └── fusion_metadata.json
│
└── crops/                    # 裁剪结果
    └── scene_name/
        ├── det_000.png       # 裁剪图
        ├── det_000_metadata.json
        └── crops_summary.json
```

### run_convert_global.py

坐标转换到全局：

1. 读取裁剪元数据（`det_xxx_metadata.json`）
2. 将分块图的检测坐标转换为原始大图坐标
3. 转换 `results_json` 中的分割坐标
4. 输出全局 YOLO 标签和 JSON（含经纬度）

```bash
python scripts/run_convert_global.py \
    --input-dir F:\code\pic \
    --crop-dir F:\output\crops \
    --results-json-dir F:\results_json \
    --output-dir F:\output\global_coords
```

### run_fuse.py

单独的整图融合（不裁剪）：

```bash
# 批量模式
python scripts/run_fuse.py --batch --input-dir F:\data --output-dir F:\output

# 单文件模式
python scripts/run_fuse.py --pan pan.tif --mss mss.tif --output fused.tif
```

### run_crop.py

从融合后的影像裁剪检测框：

```bash
python scripts/run_crop.py --batch \
    --label-dir F:\labels \
    --fused-dir F:\fused \
    --output-dir F:\crops
```

### run_convert.py

坐标转换（使用 CLI 模块）：

```bash
python scripts/run_convert.py --batch \
    --crops-root F:\crops \
    --results-root F:\results_json \
    --output-root F:\global
```

## 依赖

```bash
pip install rs-processor[dev]

# 如需 Arosics 精配准
pip install rs-processor[arosics]
```
