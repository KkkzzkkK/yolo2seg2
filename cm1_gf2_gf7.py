# -*- coding: utf-8 -*-
"""
简单的遥感影像标注脚本
功能：
- 遍历数据文件夹中的所有TIFF影像
- 应用Shapefile标注
- 可选智能裁剪或分块处理
- 不进行全色锐化融合
"""

import os
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from PIL import Image
import warnings
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
import cv2

# 导入工具模块
from image_utils import (
    parse_rpb_file, 
    get_image_geo_bounds,
    convert_polygon_with_rpc,
    percentile_normalize,
    process_rgb_to_8bit,
    process_rgb_to_8bit_with_global_stats,
    calculate_smart_crop_window,
    get_polygon_pixel_bounds,
)

# 导入通用业务逻辑模块
from processing_common import (
    match_annotations_to_image,
    filter_annotations_in_tile,
    generate_tiles,
    generate_tiles_preserve_annotations,
    collect_global_statistics,
)

# 设置 OpenCV 最大图像像素限制
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '5368709120'

# ============================================================================
# 用户配置区
# ============================================================================


# Shapefile 文件路径
SHP_FILE_PATH = r"E:\251007\目标4\目标4.shp"

# 数据文件夹路径
DATA_FOLDER_PATH = r"E:\250827\浙江大学1化工厂"

# 输出文件夹路径
OUTPUT_FOLDER_PATH = r"E:\251007\generate3"


# 要处理的TIFF文件后缀（可以指定多个）
TIFF_SUFFIXES = ['-NAD', '-MSS1', '-BWDMUX','-MSS2','-MUX','-MSS']  # 修改这里以匹配你需要的文件

# 处理模式选择
# 'full': 处理整张影像（不裁剪、不分块）
# 'crop': 根据每个标注智能裁剪影像
# 'tile': 分块模式处理整张影像
PROCESSING_MODE = 'full'

# 智能裁剪配置（仅在 PROCESSING_MODE='crop' 时有效）
CROP_SIZE = 5000  # 裁剪窗口目标尺寸（像素）
MIN_COVERAGE_RATIO = 0.05  # 最小覆盖率阈值（多边形面积占裁剪窗口的最小比例）

# 分块配置（仅在 PROCESSING_MODE='tile' 时有效）
TILE_SIZE = 20000  # 分块大小（像素）
TILE_OVERLAP = 200  # 分块重叠区域（像素）
PRESERVE_ANNOTATIONS = False  # 是否启用标注保护分块
TILE_SIZE_RANGE = (7000, 12000)  # 标注保护模式的分块尺寸范围
ANNOTATION_MARGIN = 500  # 标注到分块边界的最小距离（像素）

# 分块输出策略（仅在 PROCESSING_MODE='tile' 时有效）
# 'all': 输出所有分块到同一文件夹
# 'annotated_only': 只输出包含标注的分块
# 'unannotated_only': 只输出不包含标注的分块
# 'separate': 有标注和无标注分别输出到不同子文件夹
TILE_OUTPUT_STRATEGY = 'all'

# RGB 归一化方法
# 'clahe': CLAHE 独立处理每个波段
# 'percentile_rgb': RGB 联合百分位拉伸
# 'clahe_global': 使用全局范围+CLAHE（分块模式推荐）
RGB_NORMALIZATION_METHOD = 'clahe'

# 是否启用全局统计归一化（分块模式推荐启用）
USE_GLOBAL_STATS = False

# 全局统计采样率（0-1之间，1.0表示使用全部数据）
GLOBAL_STATS_SAMPLE_RATIO = 1

# Shapefile 类别列候选名称
CLASS_LABEL_COLUMN_CANDIDATES = ['类型', '类别', '标注类别', '类别名称']

# 是否保存裁剪信息的 JSON 文件（仅裁剪模式）
SAVE_CROP_INFO = True

# ============================================================================
# 影像扫描
# ============================================================================

def scan_tiff_images(data_folder, tiff_suffixes):
    """
    扫描数据文件夹，查找符合后缀的TIFF影像
    
    参数:
        data_folder: 数据文件夹路径
        tiff_suffixes: TIFF文件后缀列表
    
    返回:
        image_items: 影像信息列表
    """
    image_items = []
    
    print("=" * 80)
    print("扫描数据文件夹...")
    print("=" * 80)
    
    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"\n扫描: {subdir}")
        
        # 收集所有 TIFF 文件
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                file_base = os.path.splitext(filename)[0]
                
                # 检查是否匹配指定后缀
                if any(file_base.endswith(suffix) for suffix in tiff_suffixes):
                    tif_path = os.path.join(subdir_path, filename)
                    rpb_path = os.path.join(subdir_path, f"{file_base}.rpb")
                    
                    # 检查 RPB 文件是否存在
                    if os.path.exists(rpb_path):
                        image_items.append({
                            'tif_path': tif_path,
                            'rpb_path': rpb_path,
                            'base_name': file_base,
                            'folder': subdir
                        })
                        print(f"  找到: {file_base}")
                    else:
                        print(f"  跳过: {file_base} (缺少RPB文件)")
    
    print(f"\n扫描完成! 共 {len(image_items)} 个影像")
    print("=" * 80)
    
    return image_items


# ============================================================================
# 影像处理
# ============================================================================

def process_single_image(tif_src, window, global_stats=None):
    """
    处理单个影像窗口（不融合）
    
    参数:
        tif_src: rasterio数据源
        window: 读取窗口
        global_stats: 全局统计信息字典（可选）
    
    返回:
        image_8bit: RGB图像 (H, W, 3) uint8
    """
    num_bands = tif_src.count
    
    if num_bands == 1:
        # 单通道全色
        pan_tile = tif_src.read(1, window=window)
        pan_norm = percentile_normalize(pan_tile)
        image_8bit = (np.stack([pan_norm]*3, axis=-1) * 255).astype(np.uint8)
    
    elif num_bands >= 3:
        # 多通道 - 读取前3个波段作为RGB（假设顺序是B, G, R或R, G, B）
        bands = [tif_src.read(i+1, window=window) for i in range(3)]
        
        # 检查是否使用全局统计
        if global_stats and RGB_NORMALIZATION_METHOD in ['clahe_global', 'percentile_global', 'percentile_rgb_global']:
            # 使用全局统计方法
            image_8bit = process_rgb_to_8bit_with_global_stats(
                bands[2], bands[1], bands[0],  # 假设是 B, G, R 顺序，转为 R, G, B
                global_stats=global_stats,
                method=RGB_NORMALIZATION_METHOD
            )
        else:
            # 使用局部统计方法
            image_8bit = process_rgb_to_8bit(
                bands[2], bands[1], bands[0],  # 假设是 B, G, R 顺序，转为 R, G, B
                method=RGB_NORMALIZATION_METHOD
            )
    
    else:
        return None
    
    return image_8bit


# ============================================================================
# 主处理流程 - 统一处理函数
# ============================================================================

def process_all(shp_path, data_folder, output_folder, mode='full'):
    """
    统一的影像处理函数
    
    参数:
        mode: 处理模式
            - 'full': 整张影像（不裁剪、不分块）
            - 'crop': 智能裁剪（根据标注）
            - 'tile': 分块处理
    """
    
    # 打印模式信息
    print("\n" + "=" * 80)
    if mode == 'full':
        print("整图模式 - 处理完整影像")
    elif mode == 'crop':
        print("智能裁剪模式 - 根据标注裁剪")
        print(f"裁剪尺寸: {CROP_SIZE}x{CROP_SIZE}")
        print(f"最小覆盖率: {MIN_COVERAGE_RATIO * 100:.1f}%")
    elif mode == 'tile':
        print("分块模式 - 大图分块处理")
        if PRESERVE_ANNOTATIONS:
            print(f"分块策略: 标注保护（尺寸范围: {TILE_SIZE_RANGE[0]}-{TILE_SIZE_RANGE[1]}）")
            print(f"标注边距: {ANNOTATION_MARGIN} 像素")
        else:
            print(f"分块策略: 标准（固定尺寸: {TILE_SIZE}x{TILE_SIZE}）")
        print(f"重叠区域: {TILE_OVERLAP} 像素")
        
        # 显示输出策略
        strategy_desc = {
            'all': '输出所有分块',
            'annotated_only': '仅输出有标注的分块',
            'unannotated_only': '仅输出无标注的分块',
            'separate': '分别输出（有/无标注）'
        }
        print(f"输出策略: {strategy_desc.get(TILE_OUTPUT_STRATEGY, TILE_OUTPUT_STRATEGY)}")
    print(f"RGB归一化: {RGB_NORMALIZATION_METHOD}")
    print("=" * 80)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 为分块模式的分离策略创建子文件夹
    if mode == 'tile' and TILE_OUTPUT_STRATEGY == 'separate':
        os.makedirs(os.path.join(output_folder, 'with_annotations'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'without_annotations'), exist_ok=True)
    
    # 1. 扫描影像
    image_items = scan_tiff_images(data_folder, TIFF_SUFFIXES)
    if not image_items:
        print("错误: 未找到影像文件")
        return
    
    # 2. 读取 Shapefile
    print(f"\n读取 Shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)
    print(f"标注数量: {len(gdf)}, CRS: {gdf.crs}")
    
    # 识别类别列
    class_col = None
    for cand in CLASS_LABEL_COLUMN_CANDIDATES:
        if cand in gdf.columns:
            class_col = cand
            print(f"类别列: {class_col}")
            break
    
    transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)
    
    # 3. 处理每个影像
    total_outputs = 0
    outputs_with_annotations = 0
    outputs_without_annotations = 0
    
    print(f"\n" + "=" * 80)
    print(f"开始处理 {len(image_items)} 个影像")
    print("=" * 80)
    
    for idx, item in enumerate(image_items):
        print(f"\n[{idx+1}/{len(image_items)}] {item['base_name']}")
        
        try:
            # 解析 RPC
            rpc_params = parse_rpb_file(item['rpb_path'])
            geo_bounds = get_image_geo_bounds(rpc_params)
            
            # 获取影像尺寸
            with rasterio.open(item['tif_path']) as src:
                img_w, img_h = src.width, src.height
            
            print(f"  尺寸: {img_w}x{img_h}")
            print(f"  地理范围: 经度[{geo_bounds[0]:.4f}, {geo_bounds[2]:.4f}], "
                  f"纬度[{geo_bounds[1]:.4f}, {geo_bounds[3]:.4f}]")
            
            # 匹配标注
            all_annotations = []
            for row_idx, row in gdf.iterrows():
                if not isinstance(row.geometry, Polygon):
                    continue
                
                center = row.geometry.centroid
                lon, lat = transformer.transform(center.x, center.y)
                
                if (geo_bounds[0] <= lon <= geo_bounds[2] and 
                    geo_bounds[1] <= lat <= geo_bounds[3]):
                    
                    pixel_coords = convert_polygon_with_rpc(
                        row.geometry, rpc_params, transformer,
                        rpc_params.get('heightOffset', 0)
                    )
                    
                    label = row.get(class_col, 'unknown') if class_col else 'unknown'
                    all_annotations.append({
                        'label': str(label),
                        'pixel_coords': pixel_coords,
                        'row_idx': row_idx
                    })
            
            print(f"  匹配标注: {len(all_annotations)} 个")
            
            # 整图模式：即使没有标注也处理
            # 裁剪/分块模式：没有标注则跳过
            if mode != 'full' and len(all_annotations) == 0:
                print(f"  跳过（无匹配标注）")
                continue
            
            # 收集全局统计信息（如果启用）
            global_stats = {}
            if USE_GLOBAL_STATS and RGB_NORMALIZATION_METHOD in ['clahe_global', 'percentile_global', 'percentile_rgb_global']:
                print(f"  收集全局统计信息...")
                tif_src = rasterio.open(item['tif_path'])
                num_bands = tif_src.count
                
                if num_bands >= 3:
                    step = max(1, int(1.0 / np.sqrt(GLOBAL_STATS_SAMPLE_RATIO)))
                    
                    blue_samples = tif_src.read(1)[::step, ::step].flatten()
                    green_samples = tif_src.read(2)[::step, ::step].flatten()
                    red_samples = tif_src.read(3)[::step, ::step].flatten()
                    
                    global_stats['clahe_global'] = {
                        'r_min': float(np.min(red_samples)),
                        'r_max': float(np.max(red_samples)),
                        'g_min': float(np.min(green_samples)),
                        'g_max': float(np.max(green_samples)),
                        'b_min': float(np.min(blue_samples)),
                        'b_max': float(np.max(blue_samples)),
                    }
                    
                    r_low, r_high = np.percentile(red_samples, (2, 98))
                    g_low, g_high = np.percentile(green_samples, (2, 98))
                    b_low, b_high = np.percentile(blue_samples, (2, 98))
                    
                    global_stats['percentile_global'] = {
                        'r_low': float(r_low), 'r_high': float(r_high),
                        'g_low': float(g_low), 'g_high': float(g_high),
                        'b_low': float(b_low), 'b_high': float(b_high),
                    }
                    
                    all_samples = np.concatenate([red_samples, green_samples, blue_samples])
                    p_low, p_high = np.percentile(all_samples, (2, 98))
                    global_stats['percentile_rgb_global'] = {
                        'p_low': float(p_low), 'p_high': float(p_high),
                    }
                    
                    print(f"  全局统计完成!")
                
                tif_src.close()
            
            # 根据模式生成窗口列表
            windows = []
            
            if mode == 'full':
                # 整图模式：处理完整影像
                windows.append({
                    'window': Window(0, 0, img_w, img_h),
                    'name_suffix': '',
                    'annotations': all_annotations
                })
                print(f"  模式: 处理整张影像")
            
            elif mode == 'crop':
                # 裁剪模式：为每个标注生成裁剪窗口
                for poly_idx, ann in enumerate(all_annotations):
                    poly_bounds = get_polygon_pixel_bounds(ann['pixel_coords'])
                    center_x = (poly_bounds[0] + poly_bounds[2]) / 2
                    center_y = (poly_bounds[1] + poly_bounds[3]) / 2
                    
                    window_tuple, adjusted = calculate_smart_crop_window(
                        (center_x, center_y), CROP_SIZE, img_w, img_h,
                        polygon_bounds=poly_bounds,
                        min_coverage_ratio=MIN_COVERAGE_RATIO
                    )
                    
                    x, y, w, h = window_tuple
                    windows.append({
                        'window': Window(x, y, w, h),
                        'name_suffix': f"_crop{poly_idx:03d}_{ann['label']}",
                        'annotations': [ann],
                        'adjusted': adjusted
                    })
                print(f"  模式: 智能裁剪 - {len(windows)} 个窗口")
            
            elif mode == 'tile':
                # 分块模式：生成分块窗口（内置智能质量控制）
                if PRESERVE_ANNOTATIONS and len(all_annotations) > 0:
                    tiles, metadata = generate_tiles_preserve_annotations(
                        img_w, img_h, all_annotations,
                        tile_size_range=TILE_SIZE_RANGE,
                        overlap=TILE_OVERLAP,
                        min_annotation_margin=ANNOTATION_MARGIN
                    )
                    print(f"  分块策略: {metadata['strategy']}")
                    if metadata.get('annotations_split', 0) == 0:
                        print(f"  ✓ 所有标注完整保留")
                else:
                    tiles = generate_tiles(img_w, img_h, TILE_SIZE, TILE_OVERLAP)
                    print(f"  分块策略: 标准模式")
                
                for x, y, w, h, tile_idx in tiles:
                    windows.append({
                        'window': Window(x, y, w, h),
                        'name_suffix': f"_tile{tile_idx:03d}",
                        'annotations': all_annotations  # 将在后面过滤
                    })
                print(f"  分块数量: {len(windows)} 个")
            
            # 打开影像源
            tif_src = rasterio.open(item['tif_path'])
            
            # 处理每个窗口
            for win_idx, win_data in enumerate(windows):
                window = win_data['window']
                x, y = window.col_off, window.row_off
                w, h = window.width, window.height
                
                # 提前过滤并转换标注
                if mode == 'tile':
                    # 分块模式：过滤相交的标注
                    tile_anns = filter_annotations_in_tile(win_data['annotations'], x, y, w, h)
                else:
                    # 整图/裁剪模式：转换标注坐标
                    tile_anns = []
                    for ann in win_data['annotations']:
                        tile_coords = [[c[0] - x, c[1] - y] for c in ann['pixel_coords']]
                        tile_coords = [[np.clip(c[0], 0, w), np.clip(c[1], 0, h)] for c in tile_coords]
                        
                        # 去重
                        filtered = [tile_coords[0]]
                        for c in tile_coords[1:]:
                            if c[0] != filtered[-1][0] or c[1] != filtered[-1][1]:
                                filtered.append(c)
                        
                        if len(filtered) >= 3:
                            tile_anns.append({
                                'label': ann['label'],
                                'pixel_coords': filtered
                            })
                
                # 根据分块输出策略决定是否处理和保存（优化性能）
                has_annotations = len(tile_anns) > 0
                should_process = True
                save_folder = output_folder
                
                if mode == 'tile':
                    if TILE_OUTPUT_STRATEGY == 'annotated_only' and not has_annotations:
                        should_process = False
                    elif TILE_OUTPUT_STRATEGY == 'unannotated_only' and has_annotations:
                        should_process = False
                    elif TILE_OUTPUT_STRATEGY == 'separate':
                        if has_annotations:
                            save_folder = os.path.join(output_folder, 'with_annotations')
                        else:
                            save_folder = os.path.join(output_folder, 'without_annotations')
                
                # 跳过不需要处理的分块（节省计算资源）
                if not should_process:
                    continue
                
                # 读取并处理影像
                output_img = process_single_image(tif_src, window, global_stats=global_stats)
                
                if output_img is None:
                    print(f"    窗口 {win_idx+1} 处理失败，跳过")
                    continue
                
                # 保存 PNG
                output_name = f"{item['base_name']}{win_data['name_suffix']}"
                png_path = os.path.join(save_folder, f"{output_name}.png")
                cv2.imwrite(png_path, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
                
                # 保存 LabelMe JSON
                labelme_data = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": [
                        {
                            "label": ann['label'],
                            "points": ann['pixel_coords'],
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {}
                        }
                        for ann in tile_anns
                    ],
                    "imagePath": f"{output_name}.png",
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w
                }
                
                json_path = os.path.join(save_folder, f"{output_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                # 保存额外信息（裁剪模式）
                if mode == 'crop' and SAVE_CROP_INFO:
                    info = {
                        'image_name': item['base_name'],
                        'crop_window': {'x': x, 'y': y, 'width': w, 'height': h},
                        'adjusted': win_data.get('adjusted', False),
                    }
                    info_json_path = os.path.join(save_folder, f"{output_name}_info.json")
                    with open(info_json_path, 'w', encoding='utf-8') as f:
                        json.dump(info, f, indent=2, ensure_ascii=False)
                
                total_outputs += 1
                if has_annotations:
                    outputs_with_annotations += 1
                else:
                    outputs_without_annotations += 1
            
            # 关闭影像源
            tif_src.close()
            
            print(f"  完成! 生成 {len(windows)} 个输出")
        
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n" + "=" * 80)
    print(f"处理完成! 共生成 {total_outputs} 个影像文件")
    
    # 在分块模式下显示详细统计
    if mode == 'tile' and TILE_OUTPUT_STRATEGY in ['separate', 'annotated_only', 'unannotated_only']:
        print(f"  - 包含标注: {outputs_with_annotations} 个")
        print(f"  - 无标注: {outputs_without_annotations} 个")
        
        if TILE_OUTPUT_STRATEGY == 'separate':
            print(f"\n输出位置:")
            print(f"  - 有标注: {os.path.join(output_folder, 'with_annotations')}")
            print(f"  - 无标注: {os.path.join(output_folder, 'without_annotations')}")
        else:
            print(f"输出位置: {output_folder}")
    else:
        print(f"输出位置: {output_folder}")
    
    print("=" * 80)


# ============================================================================
# 脚本入口
# ============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    
    # 统一处理入口
    process_all(
        shp_path=SHP_FILE_PATH,
        data_folder=DATA_FOLDER_PATH,
        output_folder=OUTPUT_FOLDER_PATH,
        mode=PROCESSING_MODE
    )

