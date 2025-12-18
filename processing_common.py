# -*- coding: utf-8 -*-
"""
遥感影像处理通用业务逻辑模块
包含影像扫描、配对、标注处理等通用业务函数
"""

import os
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer
import rasterio
from rasterio.windows import Window

from image_utils import (
    parse_rpb_file,
    ground_to_image_rpc,
    image_to_ground_rpc,
    get_image_geo_bounds,
    convert_polygon_with_rpc,
    calculate_band_correlations,
    gram_schmidt_pan_sharpening,
    process_rgb_to_8bit,
    percentile_normalize,
)
import cv2
import numpy as np

# ============================================================================
# 影像扫描和配对
# ============================================================================

def scan_and_pair_images(data_folder, pan_mss_pairs, single_suffixes, enable_pan_sharpening=True):
    """
    扫描数据文件夹，配对全色和多光谱影像
    
    参数:
        data_folder: 数据文件夹路径
        pan_mss_pairs: 全色-多光谱配对列表，如 [('-PAN1', '-MSS1'), ('-BWDPAN', '-BWDMUX')]
        single_suffixes: 单独处理的后缀列表，如 ['-NAD']
        enable_pan_sharpening: 是否启用全色锐化配对
    
    返回:
        image_items: 影像配对信息列表
    """
    image_items = []
    
    print("=" * 80)
    print("扫描数据文件夹并配对影像...")
    print("=" * 80)
    
    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if not os.path.isdir(subdir_path):
            continue
        
        print(f"\n扫描: {subdir}")
        
        # 收集所有 TIFF 文件
        tiff_files = {}
        for filename in os.listdir(subdir_path):
            if filename.lower().endswith(('.tif', '.tiff')):
                file_base = os.path.splitext(filename)[0]
                tiff_files[file_base] = os.path.join(subdir_path, filename)
        
        # 配对全色和多光谱影像
        paired_bases = set()
        
        if enable_pan_sharpening:
            for pan_suffix, mss_suffix in pan_mss_pairs:
                pan_files = [base for base in tiff_files.keys() if base.endswith(pan_suffix)]
                
                for pan_base in pan_files:
                    prefix = pan_base[:-len(pan_suffix)]
                    mss_base = prefix + mss_suffix
                    
                    if mss_base in tiff_files:
                        pan_path = tiff_files[pan_base]
                        mss_path = tiff_files[mss_base]
                        pan_rpb = os.path.join(subdir_path, f"{pan_base}.rpb")
                        mss_rpb = os.path.join(subdir_path, f"{mss_base}.rpb")
                        
                        if os.path.exists(pan_rpb) and os.path.exists(mss_rpb):
                            image_items.append({
                                'type': 'paired',
                                'pan_path': pan_path,
                                'mss_path': mss_path,
                                'pan_rpb_path': pan_rpb,
                                'mss_rpb_path': mss_rpb,
                                'base_name': pan_base,
                                'folder': subdir
                            })
                            print(f"  [融合] {pan_base} + {mss_base}")
                            paired_bases.add(pan_base)
                            paired_bases.add(mss_base)
        
        # 单独影像
        for base_name, tif_path in tiff_files.items():
            if base_name in paired_bases:
                continue
            
            if any(base_name.endswith(suffix) for suffix in single_suffixes):
                rpb_path = os.path.join(subdir_path, f"{base_name}.rpb")
                if os.path.exists(rpb_path):
                    image_items.append({
                        'type': 'single',
                        'tif_path': tif_path,
                        'rpb_path': rpb_path,
                        'base_name': base_name,
                        'folder': subdir
                    })
                    print(f"  [单独] {base_name}")
    
    print(f"\n扫描完成! 共 {len(image_items)} 个影像")
    print("=" * 80)
    
    return image_items


# ============================================================================
# 标注匹配
# ============================================================================

def match_annotations_to_image(gdf, rpc_params, class_col_candidates=None):
    """
    将 Shapefile 中的标注匹配到影像坐标系统
    
    参数:
        gdf: GeoDataFrame (Shapefile 数据)
        rpc_params: RPC 参数字典
        class_col_candidates: 类别列候选名称列表
    
    返回:
        matched_annotations: 匹配的标注列表
        class_col: 使用的类别列名称
    """
    # 识别类别列
    class_col = None
    if class_col_candidates:
        for cand in class_col_candidates:
            if cand in gdf.columns:
                class_col = cand
                break
    
    # 获取影像地理范围
    geo_bounds = get_image_geo_bounds(rpc_params)
    
    # 创建坐标转换器
    transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)
    
    # 匹配标注
    matched_annotations = []
    for row_idx, row in gdf.iterrows():
        if not isinstance(row.geometry, Polygon):
            continue
        
        center = row.geometry.centroid
        lon, lat = transformer.transform(center.x, center.y)
        
        # 检查是否在影像范围内
        if (geo_bounds[0] <= lon <= geo_bounds[2] and 
            geo_bounds[1] <= lat <= geo_bounds[3]):
            
            # 转换为像素坐标
            pixel_coords = convert_polygon_with_rpc(
                row.geometry, rpc_params, transformer,
                rpc_params.get('heightOffset', 0)
            )
            
            # 获取标签
            label = row.get(class_col, 'unknown') if class_col else 'unknown'
            
            matched_annotations.append({
                'label': str(label),
                'pixel_coords': pixel_coords,
                'row_idx': row_idx
            })
    
    return matched_annotations, class_col


# ============================================================================
# 波段相关性分析
# ============================================================================

def analyze_band_correlations_from_item(item, sample_ratio=0.1):
    """
    从影像配对信息中分析波段相关性，计算最优 Gram-Schmidt 权重
    
    参数:
        item: 影像配对信息字典（必须是 'paired' 类型）
        sample_ratio: 采样率 (0-1)
    
    返回:
        weights: 归一化权重数组
        correlations: 相关系数数组
    """
    if item['type'] != 'paired':
        return None, None
    
    try:
        print(f"  分析波段相关性（采样率: {sample_ratio:.1%}）...")
        
        pan_src = rasterio.open(item['pan_path'])
        mss_src = rasterio.open(item['mss_path'])
        
        # 采样策略：读取影像中心区域的一个子块
        pan_h, pan_w = pan_src.height, pan_src.width
        mss_h, mss_w = mss_src.height, mss_src.width
        
        # 计算采样窗口（中心区域）
        sample_size = int(min(pan_w, pan_h) * np.sqrt(sample_ratio))
        sample_size = min(sample_size, 2000)  # 最大2000像素
        sample_size = max(sample_size, 500)   # 最小500像素
        
        pan_x = (pan_w - sample_size) // 2
        pan_y = (pan_h - sample_size) // 2
        
        # 读取全色波段采样
        pan_sample = pan_src.read(1, window=Window(pan_x, pan_y, sample_size, sample_size))
        
        # 计算对应的多光谱窗口
        scale_x = mss_w / pan_w
        scale_y = mss_h / pan_h
        mss_x = int(pan_x * scale_x)
        mss_y = int(pan_y * scale_y)
        mss_sample_w = int(sample_size * scale_x)
        mss_sample_h = int(sample_size * scale_y)
        
        # 读取多光谱波段采样
        mss_samples = []
        num_bands = min(mss_src.count, 4)
        for i in range(1, num_bands + 1):
            mss_band = mss_src.read(i, window=Window(mss_x, mss_y, mss_sample_w, mss_sample_h))
            mss_samples.append(mss_band)
        
        # 计算相关性
        correlations, weights = calculate_band_correlations(pan_sample, mss_samples, sample_ratio=1.0)
        
        pan_src.close()
        mss_src.close()
        
        # 显示结果
        band_names = ['Blue', 'Green', 'Red', 'NIR'][:len(correlations)]
        print(f"  波段相关系数:")
        for i, (name, corr) in enumerate(zip(band_names, correlations)):
            print(f"    {name:6s}: {corr:6.4f} -> 权重: {weights[i]:.4f}")
        
        return weights, correlations
    
    except Exception as e:
        print(f"  警告: 波段相关性分析失败: {e}")
        return None, None


# ============================================================================
# 全局统计信息收集
# ============================================================================

def collect_global_statistics(item, sample_ratio=0.3):
    """
    收集影像的全局统计信息（用于归一化）
    使用采样策略以提高速度
    
    参数:
        item: 影像信息字典
        sample_ratio: 采样率 (0-1)
    
    返回: 
        global_stats: 全局统计信息字典
    """
    print(f"  收集全局统计信息（采样率: {sample_ratio:.1%}）...")
    
    global_stats = {}
    
    try:
        if item['type'] == 'paired':
            # 全色融合影像：需要先融合后统计
            mss_src = rasterio.open(item['mss_path'])
            
            # 采样策略：每隔 n 个像素采样一次
            step = max(1, int(1.0 / np.sqrt(sample_ratio)))
            
            # 读取采样数据
            sampled_bands = []
            num_bands = min(mss_src.count, 4)
            
            for i in range(num_bands):
                band_data = mss_src.read(i+1)[::step, ::step]
                sampled_bands.append(band_data.flatten())
            
            # 假设波段顺序是 [B, G, R, NIR]，提取 RGB
            if len(sampled_bands) >= 3:
                blue_samples = sampled_bands[0]
                green_samples = sampled_bands[1]
                red_samples = sampled_bands[2]
                
                # 计算各种统计信息
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
                    'r_low': float(r_low),
                    'r_high': float(r_high),
                    'g_low': float(g_low),
                    'g_high': float(g_high),
                    'b_low': float(b_low),
                    'b_high': float(b_high),
                }
                
                # RGB 联合统计
                all_samples = np.concatenate([red_samples, green_samples, blue_samples])
                p_low, p_high = np.percentile(all_samples, (2, 98))
                global_stats['percentile_rgb_global'] = {
                    'p_low': float(p_low),
                    'p_high': float(p_high),
                }
            
            if len(sampled_bands) >= 4:
                nir_samples = sampled_bands[3]
                nir_min = float(np.min(nir_samples))
                nir_max = float(np.max(nir_samples))
                nir_low, nir_high = np.percentile(nir_samples, (2, 98))
                
                global_stats['nir_global'] = {
                    'nir_min': nir_min,
                    'nir_max': nir_max,
                    'nir_low': float(nir_low),
                    'nir_high': float(nir_high),
                }
            
            mss_src.close()
        
        else:
            # 单独影像
            tif_src = rasterio.open(item['tif_path'])
            num_bands = tif_src.count
            
            if num_bands >= 3:
                step = max(1, int(1.0 / np.sqrt(sample_ratio)))
                
                # 读取 RGB 波段采样
                red_samples = tif_src.read(3)[::step, ::step].flatten()
                green_samples = tif_src.read(2)[::step, ::step].flatten()
                blue_samples = tif_src.read(1)[::step, ::step].flatten()
                
                # 统计信息
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
                    'r_low': float(r_low),
                    'r_high': float(r_high),
                    'g_low': float(g_low),
                    'g_high': float(g_high),
                    'b_low': float(b_low),
                    'b_high': float(b_high),
                }
                
                all_samples = np.concatenate([red_samples, green_samples, blue_samples])
                p_low, p_high = np.percentile(all_samples, (2, 98))
                global_stats['percentile_rgb_global'] = {
                    'p_low': float(p_low),
                    'p_high': float(p_high),
                }
            
            if num_bands >= 4:
                nir_samples = tif_src.read(4)[::step, ::step].flatten()
                nir_min = float(np.min(nir_samples))
                nir_max = float(np.max(nir_samples))
                nir_low, nir_high = np.percentile(nir_samples, (2, 98))
                
                global_stats['nir_global'] = {
                    'nir_min': nir_min,
                    'nir_max': nir_max,
                    'nir_low': float(nir_low),
                    'nir_high': float(nir_high),
                }
            
            tif_src.close()
        
        print(f"  全局统计完成!")
        if 'percentile_rgb_global' in global_stats:
            print(f"    RGB联合百分位: [{global_stats['percentile_rgb_global']['p_low']:.1f}, "
                  f"{global_stats['percentile_rgb_global']['p_high']:.1f}]")
    
    except Exception as e:
        print(f"  警告: 收集全局统计失败: {e}")
        global_stats = {}
    
    return global_stats


# ============================================================================
# RPC几何变换（通用函数）
# ============================================================================

def rpc_warp_and_fuse(pan_tile, window, pan_rpc, mss_src, mss_rpc, 
                      pan_sharpening_method='gram_schmidt',
                      rgb_normalization_method='clahe',
                      gs_weights=None,
                      warp_interpolation='INTER_LANCZOS4',
                      warp_border_mode='BORDER_CONSTANT',
                      rpc_iterations=5):
    """
    通用RPC几何变换和全色锐化融合函数
    
    参数:
        pan_tile: 已读取的全色影像分块 (H, W)
        window: 分块窗口 (rasterio.Window)
        pan_rpc: 全色RPC参数
        mss_src: 多光谱影像源 (rasterio DatasetReader)
        mss_rpc: 多光谱RPC参数
        pan_sharpening_method: 全色锐化方法
        rgb_normalization_method: RGB归一化方法
        gs_weights: Gram-Schmidt权重
        warp_interpolation: 插值方法
        warp_border_mode: 边界填充模式
        rpc_iterations: RPC反投影迭代次数
    
    返回:
        image_8bit: 融合后的RGB图像 (H, W, 3) uint8
    """
    # 检查实际读取尺寸（边界裁剪检测）
    actual_height, actual_width = pan_tile.shape
    if actual_width != window.width or actual_height != window.height:
        window = Window(window.col_off, window.row_off, actual_width, actual_height)
    
    h_avg = pan_rpc.get('heightOffset', 0)
    
    # 计算PAN角点的地理坐标
    pan_corners_pix = [
        (window.col_off, window.row_off),
        (window.col_off + window.width, window.row_off),
        (window.col_off + window.width, window.row_off + window.height),
        (window.col_off, window.row_off + window.height),
    ]
    
    ground_corners_lonlat = [
        image_to_ground_rpc(px, py, pan_rpc, h_avg, iterations=rpc_iterations) 
        for px, py in pan_corners_pix
    ]
    
    # 转换为MSS像素坐标
    mss_corners_pix = np.float32([
        ground_to_image_rpc(lon, lat, h_avg, mss_rpc) for lon, lat in ground_corners_lonlat
    ])
    
    # 计算MSS读取窗口
    x_coords = mss_corners_pix[:, 0]
    y_coords = mss_corners_pix[:, 1]
    mss_x_min, mss_x_max = np.min(x_coords), np.max(x_coords)
    mss_y_min, mss_y_max = np.min(y_coords), np.max(y_coords)
    
    buffer = 20
    mss_col_off = int(np.floor(mss_x_min)) - buffer
    mss_row_off = int(np.floor(mss_y_min)) - buffer
    mss_width = int(np.ceil(mss_x_max - mss_x_min)) + 2 * buffer
    mss_height = int(np.ceil(mss_y_max - mss_y_min)) + 2 * buffer
    
    # 边界检查
    mss_col_off = max(0, min(mss_col_off, mss_src.width - 1))
    mss_row_off = max(0, min(mss_row_off, mss_src.height - 1))
    if mss_col_off + mss_width > mss_src.width:
        mss_width = mss_src.width - mss_col_off
    if mss_row_off + mss_height > mss_src.height:
        mss_height = mss_src.height - mss_row_off
    
    if mss_width <= 0 or mss_height <= 0:
        return None
    
    mss_read_window = Window(mss_col_off, mss_row_off, mss_width, mss_height)
    
    # 读取多光谱数据
    try:
        mss_sub_tiles = []
        num_bands = min(mss_src.count, 4)
        for i in range(1, num_bands + 1):
            tile = mss_src.read(i, window=mss_read_window)
            if tile.size == 0:
                return None
            mss_sub_tiles.append(tile)
    except Exception as e:
        return None
    
    # 计算透视变换矩阵
    dst_corners = np.float32([
        [0, 0], 
        [window.width, 0], 
        [window.width, window.height], 
        [0, window.height]
    ])
    
    src_corners = mss_corners_pix - np.float32([mss_col_off, mss_row_off])
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    
    # 执行几何变换
    interp_flag = getattr(cv2, warp_interpolation, cv2.INTER_CUBIC)
    border_mode = getattr(cv2, warp_border_mode, cv2.BORDER_CONSTANT)
    
    mss_warped_bands = []
    for band in mss_sub_tiles:
        if border_mode == cv2.BORDER_CONSTANT:
            band_mean = np.mean(band)
            warped_band = cv2.warpPerspective(
                band, M, (window.width, window.height), 
                flags=interp_flag,
                borderMode=border_mode,
                borderValue=band_mean
            )
        else:
            warped_band = cv2.warpPerspective(
                band, M, (window.width, window.height), 
                flags=interp_flag,
                borderMode=border_mode
            )
        mss_warped_bands.append(warped_band)
    
    # 全色锐化
    if pan_sharpening_method == 'brovey':
        sharpened = brovey_pan_sharpening(pan_tile, mss_warped_bands)
    elif pan_sharpening_method == 'ihs':
        sharpened = ihs_pan_sharpening(pan_tile, mss_warped_bands)
    elif pan_sharpening_method == 'gram_schmidt':
        sharpened = gram_schmidt_pan_sharpening(pan_tile, mss_warped_bands, weights=gs_weights)
    else:
        sharpened = simple_mean_pan_sharpening(pan_tile, mss_warped_bands)
    
    # RGB归一化
    image_8bit = process_rgb_to_8bit(
        sharpened[2], sharpened[1], sharpened[0],
        method=rgb_normalization_method
    )
    
    return image_8bit


# ============================================================================
# 标注过滤（用于分块）
# ============================================================================

def filter_annotations_in_tile(annotations, x_offset, y_offset, width, height):
    """
    过滤分块中的标注，并转换为相对坐标
    
    参数:
        annotations: 标注列表
        x_offset, y_offset: 分块偏移量
        width, height: 分块尺寸
    
    返回:
        tile_annotations: 过滤后的标注列表
    """
    tile_annotations = []
    
    for ann in annotations:
        pixel_coords = ann['pixel_coords']
        
        # 检查多边形是否与分块相交
        xs = [c[0] for c in pixel_coords]
        ys = [c[1] for c in pixel_coords]
        
        if not (max(xs) >= x_offset and min(xs) <= x_offset + width and
                max(ys) >= y_offset and min(ys) <= y_offset + height):
            continue
        
        # 转换为相对坐标
        tile_coords = [[c[0] - x_offset, c[1] - y_offset] for c in pixel_coords]
        
        # 裁剪到边界
        tile_coords = [[np.clip(c[0], 0, width), np.clip(c[1], 0, height)] 
                       for c in tile_coords]
        
        # 去重
        filtered = [tile_coords[0]]
        for c in tile_coords[1:]:
            if c[0] != filtered[-1][0] or c[1] != filtered[-1][1]:
                filtered.append(c)
        
        if len(filtered) >= 3:
            tile_annotations.append({
                'label': ann['label'],
                'pixel_coords': filtered
            })
    
    return tile_annotations


# ============================================================================
# 分块生成（用于大图分块处理）
# ============================================================================

def generate_tiles(img_width, img_height, tile_size, overlap):
    """
    生成分块窗口坐标（固定大小）
    
    参数:
        img_width, img_height: 影像尺寸
        tile_size: 分块大小
        overlap: 重叠区域大小
    
    返回:
        tiles: 分块列表 [(x, y, width, height, tile_idx), ...]
    """
    tiles = []
    tile_idx = 0
    
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            w = min(tile_size, img_width - x)
            h = min(tile_size, img_height - y)
            
            tiles.append((x, y, w, h, tile_idx))
            tile_idx += 1
            
            x += tile_size - overlap
            if x >= img_width:
                break
        
        y += tile_size - overlap
        if y >= img_height:
            break
    
    return tiles


def generate_tiles_annotation_first(img_width, img_height, annotations,
                                    tile_size_range=(8000, 12000),
                                    overlap=200,
                                    min_annotation_margin=1000,
                                    min_boundary_tile_size=2000):
    """
    基于标注优先的分块策略
    
    核心策略：
    1. **提取标注区域**：找到每个标注的边界框并扩展边距
    2. **扩大到目标尺寸**：将标注区域扩大到目标分块大小（可重叠）
    3. **生成有标注图块**：优先生成包含标注的图块
    4. **生成边界图块**：对剩余区域（无标注区域）进行分块
    5. **过滤小图块**：舍去太小的边界图块
    
    参数:
        img_width, img_height: 影像尺寸
        annotations: 标注列表，每个标注包含 'pixel_coords'
        tile_size_range: 分块大小范围 (min_size, max_size)
        overlap: 重叠区域大小
        min_annotation_margin: 标注到分块边界的最小距离
        min_boundary_tile_size: 边界图块的最小尺寸阈值（小于此值舍弃）
    
    返回:
        tiles: 分块列表 [(x, y, width, height, tile_idx), ...]
        metadata: 元数据字典，包含统计信息
    """
    if isinstance(tile_size_range, int):
        tile_size_range = (tile_size_range, tile_size_range)
    
    min_tile_size, max_tile_size = tile_size_range
    target_tile_size = max_tile_size  # 使用最大尺寸作为目标尺寸
    
    # 如果没有标注，使用标准分块
    if not annotations:
        return generate_tiles(img_width, img_height, max_tile_size, overlap), {
            'strategy': 'standard_no_annotations',
            'annotated_tiles': 0,
            'boundary_tiles': 0,
            'discarded_tiles': 0
        }
    
    print(f"  使用标注优先分块策略（标注数: {len(annotations)}，目标尺寸: {target_tile_size}）")
    
    # ========================================================================
    # 步骤1: 提取所有标注的边界框
    # ========================================================================
    annotation_bounds = []
    for idx, ann in enumerate(annotations):
        coords = ann['pixel_coords']
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        annotation_bounds.append({
            'id': idx,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2,
            'width': max_x - min_x,
            'height': max_y - min_y,
        })
    
    # ========================================================================
    # 步骤2: 为每个标注生成扩展后的图块（包含边距）
    # ========================================================================
    annotated_tiles = []
    occupied_regions = []  # 记录已占用的区域，用于后续过滤
    
    for ann_bound in annotation_bounds:
        # 计算扩展后的区域（标注 + 边距）
        ann_with_margin_min_x = ann_bound['min_x'] - min_annotation_margin
        ann_with_margin_max_x = ann_bound['max_x'] + min_annotation_margin
        ann_with_margin_min_y = ann_bound['min_y'] - min_annotation_margin
        ann_with_margin_max_y = ann_bound['max_y'] + min_annotation_margin
        
        ann_with_margin_width = ann_with_margin_max_x - ann_with_margin_min_x
        ann_with_margin_height = ann_with_margin_max_y - ann_with_margin_min_y
        
        # 计算中心点
        center_x = ann_bound['center_x']
        center_y = ann_bound['center_y']
        
        # 扩大到目标尺寸（以中心为基准）
        tile_w = max(ann_with_margin_width, target_tile_size)
        tile_h = max(ann_with_margin_height, target_tile_size)
        
        # 限制最大尺寸（避免过大）
        tile_w = min(tile_w, max_tile_size * 1.5)
        tile_h = min(tile_h, max_tile_size * 1.5)
        
        # 计算图块位置（以标注中心为基准）
        tile_x = int(center_x - tile_w / 2)
        tile_y = int(center_y - tile_h / 2)
        tile_w = int(tile_w)
        tile_h = int(tile_h)
        
        # 边界检查和调整
        if tile_x < 0:
            tile_x = 0
        if tile_y < 0:
            tile_y = 0
        if tile_x + tile_w > img_width:
            tile_w = img_width - tile_x
        if tile_y + tile_h > img_height:
            tile_h = img_height - tile_y
        
        # 再次检查尺寸
        if tile_w < min_tile_size or tile_h < min_tile_size:
            # 如果太小，使用最小尺寸
            tile_w = min(min_tile_size, img_width)
            tile_h = min(min_tile_size, img_height)
            tile_x = max(0, min(int(center_x - tile_w / 2), img_width - tile_w))
            tile_y = max(0, min(int(center_y - tile_h / 2), img_height - tile_h))
        
        annotated_tiles.append({
            'x': tile_x,
            'y': tile_y,
            'w': tile_w,
            'h': tile_h,
            'annotation_ids': [ann_bound['id']],
            'type': 'annotated'
        })
        
        # 记录占用区域（扩展重叠区域）
        occupied_regions.append({
            'min_x': tile_x - overlap // 2,
            'max_x': tile_x + tile_w + overlap // 2,
            'min_y': tile_y - overlap // 2,
            'max_y': tile_y + tile_h + overlap // 2,
        })
    
    print(f"    生成有标注图块: {len(annotated_tiles)} 个")
    
    # ========================================================================
    # 步骤3: 合并重叠的标注图块（可选优化）
    # ========================================================================
    # 简化版：检测高度重叠的图块并合并
    merged_annotated_tiles = []
    merged_flags = [False] * len(annotated_tiles)
    
    for i in range(len(annotated_tiles)):
        if merged_flags[i]:
            continue
        
        current_tile = annotated_tiles[i].copy()
        merged_flags[i] = True
        
        # 检查是否有其他图块与当前图块高度重叠
        for j in range(i + 1, len(annotated_tiles)):
            if merged_flags[j]:
                continue
            
            other_tile = annotated_tiles[j]
            
            # 计算重叠区域
            overlap_min_x = max(current_tile['x'], other_tile['x'])
            overlap_max_x = min(current_tile['x'] + current_tile['w'], other_tile['x'] + other_tile['w'])
            overlap_min_y = max(current_tile['y'], other_tile['y'])
            overlap_max_y = min(current_tile['y'] + current_tile['h'], other_tile['y'] + other_tile['h'])
            
            if overlap_min_x < overlap_max_x and overlap_min_y < overlap_max_y:
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                current_area = current_tile['w'] * current_tile['h']
                other_area = other_tile['w'] * other_tile['h']
                
                # 如果重叠超过50%，考虑合并
                if overlap_area > 0.5 * min(current_area, other_area):
                    # 合并两个图块
                    merged_min_x = min(current_tile['x'], other_tile['x'])
                    merged_min_y = min(current_tile['y'], other_tile['y'])
                    merged_max_x = max(current_tile['x'] + current_tile['w'], other_tile['x'] + other_tile['w'])
                    merged_max_y = max(current_tile['y'] + current_tile['h'], other_tile['y'] + other_tile['h'])
                    
                    merged_w = merged_max_x - merged_min_x
                    merged_h = merged_max_y - merged_min_y
                    
                    # 检查合并后的尺寸是否合理
                    if merged_w <= max_tile_size * 2 and merged_h <= max_tile_size * 2:
                        current_tile = {
                            'x': merged_min_x,
                            'y': merged_min_y,
                            'w': merged_w,
                            'h': merged_h,
                            'annotation_ids': current_tile['annotation_ids'] + other_tile['annotation_ids'],
                            'type': 'annotated'
                        }
                        merged_flags[j] = True
        
        merged_annotated_tiles.append(current_tile)
    
    if len(merged_annotated_tiles) < len(annotated_tiles):
        print(f"    合并重叠图块: {len(annotated_tiles)} -> {len(merged_annotated_tiles)}")
    
    annotated_tiles = merged_annotated_tiles
    
    # ========================================================================
    # 步骤4: 生成边界图块（覆盖没有标注的区域）
    # ========================================================================
    boundary_tiles = []
    
    def is_in_occupied_region(x, y, w, h):
        """检查图块是否与已占用区域重叠"""
        tile_min_x, tile_max_x = x, x + w
        tile_min_y, tile_max_y = y, y + h
        
        for region in occupied_regions:
            # 检查是否有重叠
            if not (tile_max_x <= region['min_x'] or tile_min_x >= region['max_x'] or
                    tile_max_y <= region['min_y'] or tile_min_y >= region['max_y']):
                # 计算重叠面积
                overlap_min_x = max(tile_min_x, region['min_x'])
                overlap_max_x = min(tile_max_x, region['max_x'])
                overlap_min_y = max(tile_min_y, region['min_y'])
                overlap_max_y = min(tile_max_y, region['max_y'])
                
                overlap_area = (overlap_max_x - overlap_min_x) * (overlap_max_y - overlap_min_y)
                tile_area = w * h
                
                # 如果重叠超过30%，认为已被占用
                if overlap_area > 0.3 * tile_area:
                    return True
        
        return False
    
    # 使用标准网格划分边界区域
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            w = min(target_tile_size, img_width - x)
            h = min(target_tile_size, img_height - y)
            
            # 检查是否太小
            if w < min_boundary_tile_size or h < min_boundary_tile_size:
                x += target_tile_size - overlap
                if x >= img_width:
                    break
                continue
            
            # 检查是否在已占用区域
            if not is_in_occupied_region(x, y, w, h):
                boundary_tiles.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'annotation_ids': [],
                    'type': 'boundary'
                })
            
            x += target_tile_size - overlap
            if x >= img_width:
                break
        
        y += target_tile_size - overlap
        if y >= img_height:
            break
    
    print(f"    生成边界图块: {len(boundary_tiles)} 个（最小尺寸阈值: {min_boundary_tile_size}）")
    
    # ========================================================================
    # 步骤5: 合并所有图块并分配索引
    # ========================================================================
    all_tiles = annotated_tiles + boundary_tiles
    tiles = []
    
    for idx, tile_info in enumerate(all_tiles):
        tiles.append((tile_info['x'], tile_info['y'], tile_info['w'], tile_info['h'], idx))
    
    # 统计信息
    metadata = {
        'strategy': 'annotation_first',
        'annotated_tiles': len(annotated_tiles),
        'boundary_tiles': len(boundary_tiles),
        'total_tiles': len(tiles),
        'annotations_split': 0,  # 使用此策略不会切割标注
        'min_boundary_tile_size': min_boundary_tile_size,
        'target_tile_size': target_tile_size,
    }
    
    print(f"  ✓ 分块完成: {len(tiles)} 个图块（有标注: {len(annotated_tiles)}，边界: {len(boundary_tiles)}）")
    
    return tiles, metadata


def generate_tiles_preserve_annotations(img_width, img_height, annotations, 
                                        tile_size_range=(8000, 12000), 
                                        overlap=200,
                                        min_annotation_margin=100):
    """
    生成完整分块网格，严格保护标注完整性，覆盖整个影像区域
    
    核心策略：
    1. **收集禁区**：标注+边距区域标记为禁止切割区
    2. **智能分割线**：在标注间的空白区域生成分割线
    3. **生成初始分块**：基于分割线生成网格
    4. **受限合并**：检测切割标注的分块，小范围合并(限制最大尺寸)
    5. **全覆盖保证**：确保输出覆盖整个影像(除了极小的边缘区域)
    
    参数:
        img_width, img_height: 影像尺寸
        annotations: 标注列表，每个标注包含 'pixel_coords'
        tile_size_range: 分块大小范围 (min_size, max_size)
        overlap: 重叠区域大小
        min_annotation_margin: 标注到分块边界的最小距离
    
    返回:
        tiles: 分块列表 [(x, y, width, height, tile_idx), ...]
        metadata: 元数据字典，包含统计信息
    """
    if isinstance(tile_size_range, int):
        tile_size_range = (tile_size_range, tile_size_range)
    
    min_tile_size, max_tile_size = tile_size_range
    min_edge_tile_size = 200  # 边缘最小分块尺寸阈值
    max_merge_size = max_tile_size * 1.8  # 合并后的最大尺寸(单边)
    
    # 如果没有标注，使用标准分块
    if not annotations:
        return generate_tiles(img_width, img_height, max_tile_size, overlap), {
            'strategy': 'standard',
            'annotations_split': 0,
            'coverage_rate': 1.0
        }
    
    # 解析所有标注的边界框（扩展边距形成禁区）
    forbidden_zones_x = []  # 垂直方向禁区 [(x_min, x_max), ...]
    forbidden_zones_y = []  # 水平方向禁区 [(y_min, y_max), ...]
    annotation_bounds = []
    
    for idx, ann in enumerate(annotations):
        coords = ann['pixel_coords']
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 添加边距形成禁区
        forbidden_x_min = min_x - min_annotation_margin
        forbidden_x_max = max_x + min_annotation_margin
        forbidden_y_min = min_y - min_annotation_margin
        forbidden_y_max = max_y + min_annotation_margin
        
        forbidden_zones_x.append((forbidden_x_min, forbidden_x_max))
        forbidden_zones_y.append((forbidden_y_min, forbidden_y_max))
        
        annotation_bounds.append({
            'id': idx,
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
        })
    
    # 检查是否有超大标注
    max_ann_width = max([b['width'] for b in annotation_bounds])
    max_ann_height = max([b['height'] for b in annotation_bounds])
    
    # 超大标注处理策略
    if (max_ann_width + 2 * min_annotation_margin > max_tile_size or 
        max_ann_height + 2 * min_annotation_margin > max_tile_size):
        print(f"  检测到超大标注 ({max_ann_width:.0f}x{max_ann_height:.0f})，使用整张图模式")
        
        # 对于非常大的影像（单边>20000），即使有超大标注也建议分块处理以避免RPC精度问题
        max_img_dimension = max(img_width, img_height)
        if max_img_dimension > 20000:
            print(f"  ⚠ 警告: 影像尺寸过大 ({img_width}x{img_height})")
            print(f"  建议: 考虑增加 TILE_SIZE_RANGE 上限或使用多个重叠分块")
            print(f"  当前策略: 仍使用整张图，但RPC对齐可能出现偏移伪影")
        
        return [(0, 0, img_width, img_height, 0)], {
            'strategy': 'full_image',
            'reason': 'large_annotation',
            'annotations_split': 0,
            'coverage_rate': 1.0,
            'max_dimension': max_img_dimension
        }
    
    print(f"  生成完整分块网格（保护 {len(annotation_bounds)} 个标注，边距 {min_annotation_margin}px）...")
    
    # 步骤1：智能生成垂直分割线（密集生成，在空白区域多插入）
    candidate_vertical_splits = []
    x = max_tile_size
    while x < img_width:
        candidate_vertical_splits.append(x)
        x += (max_tile_size - overlap)
    
    # 步骤2：过滤和扩展垂直分割线
    valid_vertical_splits = []
    
    for split_x in candidate_vertical_splits:
        # 检查此分割线是否穿过任何禁区
        in_forbidden_zone = False
        for x_min, x_max in forbidden_zones_x:
            if x_min < split_x < x_max:
                in_forbidden_zone = True
                break
        
        if not in_forbidden_zone:
            valid_vertical_splits.append(split_x)
    
    # 步骤2.5：在大间距处插入额外分割线（密集分割策略）
    i = 0
    while i < len(valid_vertical_splits):
        prev_x = valid_vertical_splits[i-1] if i > 0 else 0
        curr_x = valid_vertical_splits[i]
        gap = curr_x - prev_x
        
        # 如果间距过大，插入中间分割线
        if gap > max_tile_size * 1.5:
            # 计算需要插入的分割线数量
            num_splits = int(gap / max_tile_size)
            step = gap / (num_splits + 1)
            
            inserted = []
            for j in range(1, num_splits + 1):
                new_x = int(prev_x + step * j)
                
                # 确保不在禁区内
                in_forbidden = False
                for x_min, x_max in forbidden_zones_x:
                    if x_min < new_x < x_max:
                        in_forbidden = True
                        break
                
                if not in_forbidden:
                    inserted.append(new_x)
            
            # 插入新分割线
            for new_x in inserted:
                valid_vertical_splits.insert(i, new_x)
                i += 1
        
        i += 1
    
    valid_vertical_splits.sort()
    
    # 步骤3：智能生成水平分割线（密集生成）
    candidate_horizontal_splits = []
    y = max_tile_size
    while y < img_height:
        candidate_horizontal_splits.append(y)
        y += (max_tile_size - overlap)
    
    # 步骤4：过滤和扩展水平分割线
    valid_horizontal_splits = []
    
    for split_y in candidate_horizontal_splits:
        in_forbidden_zone = False
        for y_min, y_max in forbidden_zones_y:
            if y_min < split_y < y_max:
                in_forbidden_zone = True
                break
        
        if not in_forbidden_zone:
            valid_horizontal_splits.append(split_y)
    
    # 步骤4.5：在大间距处插入额外分割线
    i = 0
    while i < len(valid_horizontal_splits):
        prev_y = valid_horizontal_splits[i-1] if i > 0 else 0
        curr_y = valid_horizontal_splits[i]
        gap = curr_y - prev_y
        
        if gap > max_tile_size * 1.5:
            num_splits = int(gap / max_tile_size)
            step = gap / (num_splits + 1)
            
            inserted = []
            for j in range(1, num_splits + 1):
                new_y = int(prev_y + step * j)
                
                in_forbidden = False
                for y_min, y_max in forbidden_zones_y:
                    if y_min < new_y < y_max:
                        in_forbidden = True
                        break
                
                if not in_forbidden:
                    inserted.append(new_y)
            
            for new_y in inserted:
                valid_horizontal_splits.insert(i, new_y)
                i += 1
        
        i += 1
    
    valid_horizontal_splits.sort()
    
    # 步骤5：根据有效分割线生成初始分块网格
    x_boundaries = [0] + valid_vertical_splits + [img_width]
    y_boundaries = [0] + valid_horizontal_splits + [img_height]
    
    print(f"  分割线: 垂直 {len(valid_vertical_splits)} 条，水平 {len(valid_horizontal_splits)} 条")
    print(f"  初始网格: {len(x_boundaries)-1} × {len(y_boundaries)-1} = {(len(x_boundaries)-1) * (len(y_boundaries)-1)} 个网格单元")
    
    # 生成初始网格（二维数组，记录每个网格单元的状态）
    grid = []
    for i in range(len(y_boundaries) - 1):
        row = []
        for j in range(len(x_boundaries) - 1):
            x = x_boundaries[j]
            y = y_boundaries[i]
            w = x_boundaries[j + 1] - x
            h = y_boundaries[i + 1] - y
            
            row.append({
                'x': x, 'y': y, 'w': w, 'h': h,
                'row': i, 'col': j,
                'merged': False,  # 是否已被合并
                'contains_annotations': [],  # 包含的标注ID列表
                'splits_annotations': []  # 切割的标注ID列表
            })
        grid.append(row)
    
    # 步骤6：检测每个网格单元与标注的关系
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            
            for ann in annotation_bounds:
                # 标注完全在网格内
                ann_fully_inside = (ann['min_x'] >= x and ann['max_x'] <= x + w and
                                   ann['min_y'] >= y and ann['max_y'] <= y + h)
                
                # 标注完全在网格外
                ann_fully_outside = (ann['max_x'] <= x or ann['min_x'] >= x + w or
                                    ann['max_y'] <= y or ann['min_y'] >= y + h)
                
                if ann_fully_inside:
                    cell['contains_annotations'].append(ann['id'])
                elif not ann_fully_outside:
                    # 部分相交 = 切割
                    cell['splits_annotations'].append(ann['id'])
    
    # 步骤7：优先处理切割标注的网格 - 必须合并以保护标注
    def try_merge_to_protect_annotation(i, j):
        """必须合并以保护标注，优先级最高，不受尺寸限制"""
        if grid[i][j]['merged']:
            return False
        
        if not grid[i][j]['splits_annotations']:
            return False
        
        current_w = grid[i][j]['w']
        current_h = grid[i][j]['h']
        
        # 尝试向右合并
        if j + 1 < len(grid[i]) and not grid[i][j+1]['merged']:
            next_w = grid[i][j+1]['w']
            merged_w = current_w + next_w
            
            merged_cell = {
                'x': grid[i][j]['x'],
                'y': grid[i][j]['y'],
                'w': merged_w,
                'h': current_h,
                'row': i, 'col': j,
                'merged': False,
                'contains_annotations': [],
                'splits_annotations': []
            }
            
            # 检查合并后是否还切割标注
            x, y, w, h = merged_cell['x'], merged_cell['y'], merged_cell['w'], merged_cell['h']
            still_splits = False
            
            for ann in annotation_bounds:
                ann_fully_inside = (ann['min_x'] >= x and ann['max_x'] <= x + w and
                                   ann['min_y'] >= y and ann['max_y'] <= y + h)
                ann_fully_outside = (ann['max_x'] <= x or ann['min_x'] >= x + w or
                                    ann['max_y'] <= y or ann['min_y'] >= y + h)
                
                if ann_fully_inside:
                    merged_cell['contains_annotations'].append(ann['id'])
                elif not ann_fully_outside:
                    merged_cell['splits_annotations'].append(ann['id'])
                    still_splits = True
            
            # 如果合并后不再切割，则必须执行合并（不管尺寸多大）
            if not still_splits:
                grid[i][j] = merged_cell
                grid[i][j+1]['merged'] = True
                if merged_w > max_merge_size:
                    print(f"    ⚠ 为保护标注，生成超大分块: {merged_w}x{current_h}")
                return True
        
        # 尝试向下合并
        if i + 1 < len(grid) and not grid[i+1][j]['merged']:
            next_h = grid[i+1][j]['h']
            merged_h = current_h + next_h
            
            merged_cell = {
                'x': grid[i][j]['x'],
                'y': grid[i][j]['y'],
                'w': current_w,
                'h': merged_h,
                'row': i, 'col': j,
                'merged': False,
                'contains_annotations': [],
                'splits_annotations': []
            }
            
            x, y, w, h = merged_cell['x'], merged_cell['y'], merged_cell['w'], merged_cell['h']
            still_splits = False
            
            for ann in annotation_bounds:
                ann_fully_inside = (ann['min_x'] >= x and ann['max_x'] <= x + w and
                                   ann['min_y'] >= y and ann['max_y'] <= y + h)
                ann_fully_outside = (ann['max_x'] <= x or ann['min_x'] >= x + w or
                                    ann['max_y'] <= y or ann['min_y'] >= y + h)
                
                if ann_fully_inside:
                    merged_cell['contains_annotations'].append(ann['id'])
                elif not ann_fully_outside:
                    merged_cell['splits_annotations'].append(ann['id'])
                    still_splits = True
            
            if not still_splits:
                grid[i][j] = merged_cell
                grid[i+1][j]['merged'] = True
                if merged_h > max_merge_size:
                    print(f"    ⚠ 为保护标注，生成超大分块: {current_w}x{merged_h}")
                return True
        
        return False
    
    # 多次迭代合并，优先保护标注
    print(f"  开始合并以保护标注...")
    max_merge_iterations = 20  # 增加迭代次数以确保所有标注都被保护
    for iteration in range(max_merge_iterations):
        merged_count = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if try_merge_to_protect_annotation(i, j):
                    merged_count += 1
        
        if merged_count == 0:
            break
        if merged_count > 0:
            print(f"    保护标注迭代 {iteration + 1}: 合并了 {merged_count} 个网格单元")
    
    # 步骤8：收集最终的有效分块
    tiles = []
    tile_idx = 0
    annotations_split_count = 0
    annotations_split_set = set()  # 记录被切割的标注ID
    discarded_count = 0
    total_area = 0
    covered_area = 0
    oversized_tiles = 0
    
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell['merged']:
                continue
            
            x, y, w, h = cell['x'], cell['y'], cell['w'], cell['h']
            total_area += w * h
            
            # 检查是否为边缘极小区域（可舍弃）
            is_edge = (x == 0 or y == 0 or x + w == img_width or y + h == img_height)
            is_tiny = (w < min_edge_tile_size or h < min_edge_tile_size)
            has_no_annotation = (len(cell['contains_annotations']) == 0 and 
                                len(cell['splits_annotations']) == 0)
            
            if is_edge and is_tiny and has_no_annotation:
                discarded_count += 1
                continue
            
            # 检查是否超大
            if w > max_merge_size or h > max_merge_size:
                oversized_tiles += 1
            
            # 最终验证：不应该切割标注
            if cell['splits_annotations']:
                for ann_id in cell['splits_annotations']:
                    annotations_split_set.add(ann_id)
                print(f"    ⚠⚠⚠ 网格 ({x}, {y}, {w}x{h}) 仍切割标注 {cell['splits_annotations']} - 这不应该发生!")
            
            tiles.append((x, y, w, h, tile_idx))
            covered_area += w * h
            tile_idx += 1
    
    annotations_split_count = len(annotations_split_set)
    
    # 计算覆盖率
    coverage_rate = covered_area / total_area if total_area > 0 else 0.0
    
    # 元数据
    metadata = {
        'strategy': 'annotation_protection_priority',
        'total_annotations': len(annotation_bounds),
        'annotations_split': annotations_split_count,
        'tile_count': len(tiles),
        'discarded_tiny_edge_tiles': discarded_count,
        'oversized_tiles': oversized_tiles,
        'max_merge_size': max_merge_size,
        'tile_size_range': tile_size_range,
        'coverage_rate': coverage_rate,
        'total_area': total_area,
        'covered_area': covered_area,
    }
    
    if annotations_split_count == 0:
        print(f"  ✓ 分块完成: {len(tiles)} 个有效分块，所有标注完整保留")
        print(f"    覆盖率: {coverage_rate:.1%}，舍弃边缘极小区域: {discarded_count} 个")
        if oversized_tiles > 0:
            print(f"    ⚠ 有 {oversized_tiles} 个分块超过推荐尺寸(>{max_merge_size:.0f}px)，但这是为了保护标注")
    else:
        print(f"  ⚠⚠⚠ 分块完成: {len(tiles)} 个分块，仍有 {annotations_split_count} 个标注被切割")
        print(f" 标注在图片边缘 无法保护")
    
    return tiles, metadata
