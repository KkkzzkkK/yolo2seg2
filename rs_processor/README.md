# rs_processor - 遥感影像处理框架

模块化的遥感影像处理框架，提供 RPC 坐标转换、全色锐化、影像配准、裁剪导出等功能。

## 安装

```bash
pip install -e .
```

或安装依赖：

```bash
pip install numpy opencv-python rasterio pillow hypothesis pytest
```

### 可选：安装 Arosics（推荐）

Arosics 是专门为遥感影像配准设计的库，支持亚像素精度自动配准：

```bash
pip install arosics
# 或
pip install -e ".[arosics]"
```

## 模块结构

```
rs_processor/
├── core/           # 基础模块
│   ├── rpc_utils.py      # RPC 坐标转换
│   ├── pan_sharpen.py    # Gram-Schmidt 全色锐化
│   ├── normalize.py      # CLAHE 归一化
│   └── tiff_io.py        # TIFF 读写
├── processing/     # 处理模块
│   ├── registration.py   # 影像配准
│   ├── label_processor.py # YOLO 标签处理
│   ├── coord_converter.py # 坐标转换
│   ├── tile_processor.py  # 分块处理
│   ├── image_merger.py    # 分块合成
│   └── crop_exporter.py   # 裁剪导出
└── cli/            # 命令行入口
    ├── fuse_cli.py       # 整图融合
    ├── crop_cli.py       # 裁剪导出
    └── convert_cli.py    # 坐标转换
```

## CLI 使用

### 整图融合

```bash
python -m rs_processor.cli.fuse_cli --pan PAN.tif --mss MSS.tif --output fused.png
```

参数：
- `--pan`: PAN 影像路径
- `--mss`: MSS 影像路径
- `--output`: 输出路径
- `--format`: 输出格式 (png/tiff)
- `--tile-size`: 分块大小 (默认 8000)
- `--overlap`: 重叠区域 (默认 200)
- `--no-feature-refine`: 禁用特征点精配准

### 裁剪导出

```bash
python -m rs_processor.cli.crop_cli --image fused.png --labels labels.txt --output crops/
```

参数：
- `--image`: 融合后图像路径
- `--labels`: YOLO 标签路径
- `--output`: 输出目录
- `--fusion-metadata`: 融合元数据路径
- `--crop-multiple`: 裁剪尺寸倍数 (默认 64)
- `--scale`: 扩展比例 (默认 1.3)

### 坐标转换

```bash
python -m rs_processor.cli.convert_cli --metadata crops/ --output global/
```

参数：
- `--metadata`: 裁剪元数据目录
- `--json`: 分割结果 JSON 目录
- `--output`: 输出目录
- `--fusion-metadata`: 融合元数据路径

## API 使用

### RPC 坐标转换

```python
from rs_processor.core import parse_rpb_file, ground_to_image, image_to_ground

# 解析 RPB 文件
rpc = parse_rpb_file('image.rpb')

# 地理坐标 -> 像素坐标
col, row = ground_to_image(lon=116.3, lat=39.5, height=50.0, rpc=rpc)

# 像素坐标 -> 地理坐标
lon, lat = image_to_ground(col=1000, row=1000, rpc=rpc)
```

### 全色锐化

```python
from rs_processor.core import gram_schmidt_sharpen

# pan: (H, W), mss_bands: List[(H, W)]
sharpened = gram_schmidt_sharpen(pan, mss_bands)
```

### 影像配准

```python
from rs_processor.processing import RegistrationProcessor

# 默认使用 auto 模式：优先 arosics，不可用时回退到 ORB
processor = RegistrationProcessor(enable_feature_refine=True)

# 指定使用 arosics（需安装）
processor = RegistrationProcessor(refine_method='arosics')

# 指定使用 ORB 特征点匹配
processor = RegistrationProcessor(refine_method='orb')

result = processor.register(pan_data, pan_rpc, mss_data, mss_rpc, window)

# result.aligned_mss: 配准后的 MSS
# result.offset_info: 配准偏移信息
```

配准流程：
1. **RPC 粗配准** - 使用 RPC 参数计算整体平移，移动到大致位置
2. **精配准** - 使用 Arosics 或 ORB 特征点匹配计算变换矩阵

### 裁剪导出

```python
from rs_processor.processing import CropExporter, LabelProcessor

# 读取标签
detections = LabelProcessor.read_yolo_obb('labels.txt')

# 批量裁剪
exporter = CropExporter(crop_multiple=64)
results = exporter.batch_crop('image.png', detections, 'output/')
```

## 测试

```bash
python -m pytest rs_processor/ -v
```

当前测试覆盖 69 个属性测试，验证：
- RPC 坐标转换 Round-Trip
- Gram-Schmidt 波段数量不变量
- CLAHE 输出类型和范围
- 配准结果结构完整性
- 分块覆盖完整性
- 裁剪窗口包含检测框
- 批量处理错误隔离

## 输出格式

### 融合元数据 JSON

```json
{
  "source": {
    "pan_path": "/path/to/pan.tif",
    "mss_path": "/path/to/mss.tif"
  },
  "output": {
    "path": "/path/to/fused.png",
    "width": 29200,
    "height": 27068,
    "coordinate_system": "pan"
  },
  "processing": {
    "tile_size": 8000,
    "overlap": 200
  }
}
```

### 裁剪元数据 JSON

```json
{
  "det_idx": 0,
  "global_offset_x": 12345,
  "global_offset_y": 6789,
  "crop_width": 2048,
  "crop_height": 2048,
  "class_id": "1",
  "score": 0.95
}
```
