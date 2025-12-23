# -*- coding: utf-8 -*-
"""
遥感影像处理主脚本 - 支持分块和全色锐化融合
策略：先分块 -> 再融合 -> 保存所有分块

重要改进：
- 自动读取并使用所有多光谱波段（包括近红外 NIR 波段）
- 全色锐化算法会利用 NIR 波段来计算强度分量，提高融合质量
- 许多卫星的全色波段覆盖可见光+近红外范围，因此使用 NIR 可以获得更准确的色彩
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

# 导入通用工具模块
from image_utils import (
    parse_rpb_file, ground_to_image_rpc, get_image_geo_bounds,
    convert_polygon_with_rpc, image_to_ground_rpc,
    gram_schmidt_pan_sharpening,  
    percentile_normalize,
    calculate_band_correlations,
    ihs_pan_sharpening,
    brovey_pan_sharpening,
    simple_mean_pan_sharpening,
    process_multispectral_to_8bit,
)

# 导入通用业务逻辑模块
from processing_common import (
    generate_tiles_annotation_first,
    scan_and_pair_images,
    generate_tiles,
    generate_tiles_preserve_annotations,
    analyze_band_correlations_from_item,
    collect_global_statistics,
    filter_annotations_in_tile,
    match_annotations_to_image,
)

# 设置 OpenCV 最大图像像素限制
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '5368709120'

# ============================================================================
# 用户配置区
# ============================================================================

# Shapefile 文件路径
SHP_FILE_PATH = r"F:\shuchu\biaozhubaocun\biaozhu_ExportFeatures.shp"

# 数据文件夹路径
DATA_FOLDER_PATH = r"F:\shuru"

# 输出文件夹路径
OUTPUT_FOLDER_PATH = r"F:\shuchu"

# 分块大小（像素） - 简化为固定小分块，避免RPC非线性误差
TILE_SIZE = 10000  # 减小到4000，提高RPC精度

# 分块重叠区域（像素）
TILE_OVERLAP = 500  # 增加重叠，避免边界伪影

# 是否启用标注保护分块（优先保证标注完整性）
PRESERVE_ANNOTATIONS = True  #

# 分块策略选择
# 'annotation_first': 标注优先策略（先根据标注生成图块，再生成边界图块）
# 'grid_protection': 网格保护策略（生成网格并合并以保护标注）
TILING_STRATEGY = 'grid_protection'

# 标注保护模式的分块尺寸范围（启用 PRESERVE_ANNOTATIONS 时使用）
TILE_SIZE_RANGE = (4000, 7000)  # (最小尺寸, 最大尺寸)
# 注意：太大的分块可能导致RPC对齐精度下降，建议不超过8000

# 标注到分块边界的最小距离（像素）
ANNOTATION_MARGIN = 1000

# 边界图块最小尺寸阈值（仅用于 annotation_first 策略）
# 小于此尺寸的边界图块将被舍弃
MIN_BOUNDARY_TILE_SIZE = 1000

# 分块尺寸阈值（用于RPC控制点自动升级）
LARGE_TILE_THRESHOLD = 7000  # 像素
ULTRA_LARGE_TILE_THRESHOLD = 12000  # 像素

# RANSAC 参数配置（用于RPC几何校正的离群点检测）
USE_RANSAC = False

# 特征配准（用于修正RPC误差）
ENABLE_FEATURE_REFINEMENT = True  # 启用基于特征的精细配准
FEATURE_DETECTOR = 'ORB'  # 'ORB' 或 'SIFT' (SIFT更精确但需要opencv-contrib)
FEATURE_MAX_FEATURES = 1000  # 提取的特征点数量
FEATURE_MATCH_RATIO = 0.7  # 特征匹配比率阈值（越小越严格）
FEATURE_MIN_MATCHES = 10  # 最少匹配点数量
FEATURE_DEBUG = False  # 是否输出特征匹配的诊断信息
FEATURE_DEBUG_OUTPUT = r"C:\Users\Administrator\Desktop\scratch\feature_debug"  # 诊断输出文件夹

# 分块输出策略
# 'all': 输出所有分块到同一文件夹
# 'annotated_only': 只输出包含标注的分块
# 'unannotated_only': 只输出不包含标注的分块
# 'separate': 有标注和无标注分别输出到不同子文件夹
TILE_OUTPUT_STRATEGY = 'annotated_only'

# 是否启用全色锐化融合
ENABLE_PAN_SHARPENING = True

# 全色和多光谱影像的配对后缀
PAN_MSS_PAIRS = [
    ('-PAN1', '-MSS1'),  # GF2
    ('-PAN2', '-MSS2'),  # GF2
    ('-BWDPAN', '-BWDMUX'),  # GF7
    ('-PAN', '-MUX'),  # GF7
]


PAN_SUFFIXES = ['-PAN1', '-PAN2', '-PAN', '-BWDPAN']
MSS_SUFFIXES = ['-MSS1', '-MSS2', '-MUX', '-BWDMUX']
PAN_MSS_PAIRS = [(pan, mss) for pan in PAN_SUFFIXES for mss in MSS_SUFFIXES] 


# 单独处理的后缀
SINGLE_SUFFIXES = ['-NAD']


# 启用 RPC 校准（推荐）
ENABLE_RPC_ALIGNMENT = True

# RPC 对齐控制点数量（更多控制点 = 更高精度但更慢）
# 'corners_4': 仅使用4个角点（快速，可能有精度问题）
# 'grid_9': 使用3x3网格9个点（推荐，平衡精度和速度）
# 'grid_16': 使用4x4网格16个点（最精确，较慢）
# 'grid_36': 使用6x6网格36个点（最高精度）
RPC_CONTROL_POINTS = 'grid_36'  # 增加到36点，大幅提升精度

# RPC 反投影迭代次数（提高可以改善对齐，但更慢）
RPC_INVERSE_ITERATIONS = 20  # 增加到20次，提高精度

# 几何变换插值方法（影响对齐质量）
# 'INTER_CUBIC': 三次插值（快速，适合大多数情况）
# 'INTER_LANCZOS4': Lanczos插值（最高质量，推荐用于解决伪影问题）
# 'INTER_LINEAR': 线性插值（最快，质量较低）
WARP_INTERPOLATION = 'INTER_LANCZOS4'

# 边界填充方法（影响边缘伪影）
# 'BORDER_CONSTANT': 常数填充（推荐，避免反射伪影）
# 'BORDER_REFLECT': 反射填充（可能产生伪影）
# 'BORDER_REPLICATE': 复制边缘（折中选择）
WARP_BORDER_MODE = 'BORDER_CONSTANT'

# 自动估计 PAN 与 MSS 的全局像素偏移（基于相关性）
AUTO_ESTIMATE_OFFSET = False

# 全色锐化方法
# 'simple_mean': 简单均值法（快速，效果一般）
# 'brovey': Brovey 变换（色彩保真度较好）
# 'ihs': IHS 强度-色调-饱和度变换（色彩保真度高，细节增强明显，推荐）
# 'gram_schmidt': Gram-Schmidt 自适应方法（光谱保真度最佳，支持波段权重配置）
PAN_SHARPENING_METHOD = 'gram_schmidt'  # 推荐：ihs、gram_schmidt 或 wavelet

#

GRAM_SCHMIDT_WEIGHTS = [0.12, 0.28, 0.32, 0.28]

# 是否自动计算 Gram-Schmidt 权重（基于波段相关性分析）
# 如果启用，将覆盖 GRAM_SCHMIDT_WEIGHTS 设置
AUTO_CALCULATE_GS_WEIGHTS = False

# 自动计算权重时的采样率（0-1之间，用于加速计算）
GS_WEIGHT_SAMPLE_RATIO = 0.5 

# RGB 归一化方法
# 局部统计方法（基于每个分块，可能导致分块间不一致）：
#   'clahe': CLAHE独立处理每个波段（对比度强，但可能色彩失真）
# 全局统计方法（基于整个大图，保证分块间一致性）：
#   'clahe_global': 使用全局范围+CLAHE（推荐用于对比度增强）
RGB_NORMALIZATION_METHOD = 'clahe_global'

# 是否启用全局统计归一化（启用，保证分块间一致性）
USE_GLOBAL_STATS = True

# 全局统计采样率（0-1之间，1.0表示使用全部数据）
GLOBAL_STATS_SAMPLE_RATIO = 1.0  # 使用全部数据，不再采样

# 对齐质量诊断
ENABLE_ALIGNMENT_DIAGNOSTIC = False
DIAGNOSTIC_OUTPUT_FOLDER = r"C:\Users\Administrator\Desktop\scratch\alignment_diagnostic"

# Shapefile 类别列候选名称
CLASS_LABEL_COLUMN_CANDIDATES = ['类型', '类别', '标注类别', '类别名称']


# ============================================================================
# 分块处理（先分块再融合）与对齐
# ============================================================================

def refine_alignment_with_features(pan_tile, mss_warped_bands, output_size, tile_name=None):
    """
    使用特征匹配精细校正MSS和PAN之间的对齐误差
    
    参数:
        pan_tile: 全色影像分块 (H, W)
        mss_warped_bands: RPC粗对齐后的MSS波段列表 [(H, W), ...]
        output_size: 输出尺寸 (width, height)
        tile_name: 分块名称（用于调试输出）
    
    返回:
        (精细变换矩阵, 内点数, 总匹配数) or None（如果失败）
    """
    try:
        # 1. 准备用于匹配的灰度影像
        pan_gray = cv2.normalize(pan_tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 2. MSS转灰度（使用所有波段的强度均值）
        if len(mss_warped_bands) >= 3:
            mss_intensity = np.mean([mss_warped_bands[0], mss_warped_bands[1], mss_warped_bands[2]], axis=0)
        else:
            mss_intensity = mss_warped_bands[0]
        mss_gray = cv2.normalize(mss_intensity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # 3. 特征检测
        if FEATURE_DETECTOR == 'SIFT':
            try:
                detector = cv2.SIFT_create(nfeatures=FEATURE_MAX_FEATURES)
            except AttributeError:
                detector = cv2.ORB_create(nfeatures=FEATURE_MAX_FEATURES)
        else:  # ORB
            detector = cv2.ORB_create(nfeatures=FEATURE_MAX_FEATURES)
        
        kp1, des1 = detector.detectAndCompute(pan_gray, None)
        kp2, des2 = detector.detectAndCompute(mss_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        
        # 4. 特征匹配
        if FEATURE_DETECTOR == 'SIFT' and des1.dtype == np.float32:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des2, des1, k=2)
            # Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < FEATURE_MATCH_RATIO * n.distance:
                        good_matches.append(m)
        else:  # ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des2, des1)
            good_matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches) * FEATURE_MATCH_RATIO)]
        
        if len(good_matches) < FEATURE_MIN_MATCHES:
            return None
        
        # 5. 提取匹配点坐标
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 6. 计算精细变换矩阵
        M_refine, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        
        if M_refine is None:
            return None
        
        # 统计内点数量
        inliers = np.sum(mask) if mask is not None else 0
        
        # 7. 可选的调试输出
        if FEATURE_DEBUG and tile_name:
            os.makedirs(FEATURE_DEBUG_OUTPUT, exist_ok=True)
            # 绘制匹配结果
            match_img = cv2.drawMatches(
                mss_gray, kp2, pan_gray, kp1, 
                [m for i, m in enumerate(good_matches) if mask[i]], 
                None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            debug_path = os.path.join(FEATURE_DEBUG_OUTPUT, f"{tile_name}_matches.png")
            cv2.imwrite(debug_path, match_img)
        
        return M_refine, inliers, len(good_matches)
        
    except Exception as e:
        return None


def process_tile_with_rpc_warp(pan_src, mss_src, window, pan_rpc, mss_rpc, global_stats=None, gs_weights=None, tile_name=None):
    """
    处理单个分块：通过 RPC 反向和正向投影进行精确几何校正，然后融合.
    
    参数:
        global_stats: 全局统计信息字典（可选）
        gs_weights: Gram-Schmidt 权重（可选，仅用于 gram_schmidt 方法）
        tile_name: 分块名称（可选，用于调试输出）
    """
    # 1. 读取全色分块并确定真实影像范围
    pan_tile = pan_src.read(1, window=window)
    if pan_tile.size == 0:
        return None
    
    # 确定真实的有效窗口（处理边界裁剪）
    actual_height, actual_width = pan_tile.shape
    effective_window = Window(
        window.col_off,
        window.row_off,
        min(actual_width, pan_src.width - window.col_off),
        min(actual_height, pan_src.height - window.row_off)
    )
    
    # 2. 基于真实范围生成控制点
    h_avg = pan_rpc.get('heightOffset', 0)
    max_dimension = max(effective_window.width, effective_window.height)
    
    # 根据分块尺寸选择控制点策略
    if max_dimension > ULTRA_LARGE_TILE_THRESHOLD:
        actual_mode = 'grid_36'
    elif max_dimension > LARGE_TILE_THRESHOLD or RPC_CONTROL_POINTS == 'grid_16':
        actual_mode = 'grid_16'
    elif RPC_CONTROL_POINTS == 'grid_9':
        actual_mode = 'grid_9'
    else:
        actual_mode = 'corners_4'
    
    # 生成控制点（基于真实有效范围）
    if actual_mode == 'grid_36':
        control_points = [
            (effective_window.col_off + effective_window.width * j / 5.0,
             effective_window.row_off + effective_window.height * i / 5.0)
            for i in range(6) for j in range(6)
        ]
    elif actual_mode == 'grid_16':
        control_points = [
            (effective_window.col_off + effective_window.width * j / 3.0,
             effective_window.row_off + effective_window.height * i / 3.0)
            for i in range(4) for j in range(4)
        ]
    elif actual_mode == 'grid_9':
        control_points = [
            (effective_window.col_off + effective_window.width * j / 2.0,
             effective_window.row_off + effective_window.height * i / 2.0)
            for i in range(3) for j in range(3)
        ]
    else:  # corners_4
        control_points = [
            (effective_window.col_off, effective_window.row_off),
            (effective_window.col_off + effective_window.width, effective_window.row_off),
            (effective_window.col_off + effective_window.width, effective_window.row_off + effective_window.height),
            (effective_window.col_off, effective_window.row_off + effective_window.height),
        ]
    
    # 3. 将 PAN 控制点转换为地理坐标，再转换为 MSS 影像坐标
    ground_control_lonlat = [
        image_to_ground_rpc(px, py, pan_rpc, h_avg, iterations=RPC_INVERSE_ITERATIONS) 
        for px, py in control_points
    ]
    mss_control_pix = np.float32([
        ground_to_image_rpc(lon, lat, h_avg, mss_rpc) for lon, lat in ground_control_lonlat
    ])
    
    # 4. 计算 MSS 影像的读取窗口（包含控制点的最小外包矩形）
    buffer = 20
    mss_x_min, mss_x_max = np.min(mss_control_pix[:, 0]), np.max(mss_control_pix[:, 0])
    mss_y_min, mss_y_max = np.min(mss_control_pix[:, 1]), np.max(mss_control_pix[:, 1])
    
    mss_col_off = max(0, int(np.floor(mss_x_min)) - buffer)
    mss_row_off = max(0, int(np.floor(mss_y_min)) - buffer)
    mss_width = min(int(np.ceil(mss_x_max - mss_x_min)) + 2 * buffer, mss_src.width - mss_col_off)
    mss_height = min(int(np.ceil(mss_y_max - mss_y_min)) + 2 * buffer, mss_src.height - mss_row_off)
    
    if mss_width <= 0 or mss_height <= 0:
        return None
        
    mss_read_window = Window(mss_col_off, mss_row_off, mss_width, mss_height)

    # 5. 读取多光谱数据
    try:
        num_bands = min(mss_src.count, 4)
        mss_sub_tiles = [mss_src.read(i, window=mss_read_window) for i in range(1, num_bands + 1)]
        if any(tile.size == 0 for tile in mss_sub_tiles):
            return None
    except:
        return None

    # 6. 计算几何变换矩阵
    dst_control_points = np.float32([
        [px - effective_window.col_off, py - effective_window.row_off] for px, py in control_points
    ])
    src_control_points = mss_control_pix - np.float32([mss_col_off, mss_row_off])
    
    if len(control_points) == 4:
        M = cv2.getPerspectiveTransform(src_control_points, dst_control_points)
    else:
        M, _ = cv2.findHomography(src_control_points, dst_control_points, 0 if not USE_RANSAC else cv2.RANSAC)
        if M is None:
            return None
    
    # 7. 执行几何变换（RPC粗对齐）
    interp_flag = getattr(cv2, WARP_INTERPOLATION, cv2.INTER_CUBIC)
    border_mode = getattr(cv2, WARP_BORDER_MODE, cv2.BORDER_CONSTANT)
    output_size = (effective_window.width, effective_window.height)
    
    mss_warped_bands = []
    for band in mss_sub_tiles:
        border_value = np.mean(band) if border_mode == cv2.BORDER_CONSTANT else 0
        warped_band = cv2.warpPerspective(
            band, M, output_size, 
            flags=interp_flag,
            borderMode=border_mode,
            borderValue=border_value
        )
        mss_warped_bands.append(warped_band)

    # 7.5. 特征配准精细校正（修正RPC误差）
    if ENABLE_FEATURE_REFINEMENT:
        refinement_result = refine_alignment_with_features(pan_tile, mss_warped_bands, output_size, tile_name)
        if refinement_result is not None:
            M_refine, inliers, total_matches = refinement_result
            # 重新对MSS波段进行精细变换
            mss_refined_bands = []
            for band in mss_warped_bands:
                refined_band = cv2.warpPerspective(
                    band, M_refine, output_size,
                    flags=interp_flag,
                    borderMode=border_mode,
                    borderValue=np.mean(band) if border_mode == cv2.BORDER_CONSTANT else 0
                )
                mss_refined_bands.append(refined_band)
            mss_warped_bands = mss_refined_bands

    # 8. 全色锐化
    if PAN_SHARPENING_METHOD == 'brovey':
        sharpened = brovey_pan_sharpening(pan_tile, mss_warped_bands)
    elif PAN_SHARPENING_METHOD == 'ihs':
        sharpened = ihs_pan_sharpening(pan_tile, mss_warped_bands)
    elif PAN_SHARPENING_METHOD == 'gram_schmidt':
        weights = gs_weights if gs_weights is not None else GRAM_SCHMIDT_WEIGHTS
        sharpened = gram_schmidt_pan_sharpening(pan_tile, mss_warped_bands, weights=weights)
    else:
        sharpened = simple_mean_pan_sharpening(pan_tile, mss_warped_bands)

    # 9. 统⼀化输出 RGB+NIR （可选）并提升显示效果
    image_8bit = process_multispectral_to_8bit(
        sharpened,
        global_stats=global_stats,
        method=RGB_NORMALIZATION_METHOD
    )
    return image_8bit


def process_tile_single(tif_src, window, global_stats=None):
    """处理单个影像分块（非融合）"""
    num_bands = tif_src.count
    
    if num_bands == 1:
        pan_tile = tif_src.read(1, window=window)
        pan_norm = percentile_normalize(pan_tile)
        image_8bit = (np.stack([pan_norm]*3, axis=-1) * 255).astype(np.uint8)
    elif num_bands >= 3:
        band_count = min(num_bands, 4)
        bands = [tif_src.read(i+1, window=window) for i in range(band_count)]
        image_8bit = process_multispectral_to_8bit(
            bands,
            global_stats=global_stats,
            method=RGB_NORMALIZATION_METHOD
        )
    else:
        return None
    
    return image_8bit

# ============================================================================
# 主处理流程
# ============================================================================

def process_all(shp_path, data_folder, output_folder):
    """主处理函数"""
    
    print(f"\n{'='*80}")
    print(f"遥感影像处理 | 分块: {TILE_SIZE}px | 重叠: {TILE_OVERLAP}px | "
          f"锐化: {PAN_SHARPENING_METHOD if ENABLE_PAN_SHARPENING else '禁用'}")
    if ENABLE_FEATURE_REFINEMENT:
        print(f"特征配准: 启用 ({FEATURE_DETECTOR}, {FEATURE_MAX_FEATURES}特征点)")
    print(f"{'='*80}")
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 为分离策略创建子文件夹
    if TILE_OUTPUT_STRATEGY == 'separate':
        os.makedirs(os.path.join(output_folder, 'with_annotations'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'without_annotations'), exist_ok=True)
    
    # 1. 扫描影像
    image_items = scan_and_pair_images(
        data_folder, 
        PAN_MSS_PAIRS, 
        SINGLE_SUFFIXES, 
        ENABLE_PAN_SHARPENING
    )
    if not image_items:
        print("错误: 未找到影像文件")
        return
    
    # 2. 读取 Shapefile
    gdf = gpd.read_file(shp_path)
    print(f"标注: {len(gdf)} 个 | CRS: {gdf.crs}")
    
    # 识别类别列
    class_col = None
    for cand in CLASS_LABEL_COLUMN_CANDIDATES:
        if cand in gdf.columns:
            class_col = cand
            break
    
    transformer = Transformer.from_crs(gdf.crs, "EPSG:4326", always_xy=True)
    
    # 3. 处理每个影像
    total_tiles = 0
    tiles_with_annotations = 0
    tiles_without_annotations = 0
    
    print(f"\n处理 {len(image_items)} 个影像...")
    
    for idx, item in enumerate(image_items):
        print(f"\n[{idx+1}/{len(image_items)}] {item['base_name']}")
        
        try:
            # 解析 RPC
            if item['type'] == 'paired':
                pan_rpc_params = parse_rpb_file(item['pan_rpb_path'])
                mss_rpc_params = parse_rpb_file(item['mss_rpb_path'])
                rpc_params = pan_rpc_params
            else:
                rpc_params = parse_rpb_file(item['rpb_path'])

            geo_bounds = get_image_geo_bounds(rpc_params)
            
            # 获取影像尺寸
            if item['type'] == 'paired':
                with rasterio.open(item['pan_path']) as src:
                    img_w, img_h = src.width, src.height
            else:
                with rasterio.open(item['tif_path']) as src:
                    img_w, img_h = src.width, src.height
            
            # 匹配标注
            all_annotations = []
            for _, row in gdf.iterrows():
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
                        'pixel_coords': pixel_coords
                    })
            
            print(f"  尺寸: {img_w}x{img_h} | 标注: {len(all_annotations)} 个")
            
            # 自动计算 Gram-Schmidt 权重（如果启用）
            gs_weights_calculated = None
            if AUTO_CALCULATE_GS_WEIGHTS and PAN_SHARPENING_METHOD == 'gram_schmidt' and item['type'] == 'paired':
                weights, correlations = analyze_band_correlations_from_item(item, sample_ratio=GS_WEIGHT_SAMPLE_RATIO)
                if weights is not None:
                    gs_weights_calculated = weights.tolist()
            
            # 收集全局统计信息（如果启用）
            global_stats = {}
            if USE_GLOBAL_STATS and RGB_NORMALIZATION_METHOD in ['clahe_global', 'percentile_global', 'percentile_rgb_global']:
                global_stats = collect_global_statistics(item, sample_ratio=GLOBAL_STATS_SAMPLE_RATIO)
            
            # 生成分块
            if PRESERVE_ANNOTATIONS and len(all_annotations) > 0:
                if TILING_STRATEGY == 'annotation_first':
                    tiles, metadata = generate_tiles_annotation_first(
                        img_w, img_h, 
                        all_annotations,
                        tile_size_range=TILE_SIZE_RANGE,
                        overlap=TILE_OVERLAP,
                        min_annotation_margin=ANNOTATION_MARGIN
                    )
                else:
                    tiles, metadata = generate_tiles_preserve_annotations(
                        img_w, img_h, 
                        all_annotations,
                        tile_size_range=TILE_SIZE_RANGE,
                        overlap=TILE_OVERLAP,
                        min_annotation_margin=ANNOTATION_MARGIN
                    )
                print(f"  分块: {len(tiles)} 个 ({metadata['strategy']})")
            else:
                tiles = generate_tiles(img_w, img_h, TILE_SIZE, TILE_OVERLAP)
                print(f"  分块: {len(tiles)} 个")
            
            # 打开影像源
            if item['type'] == 'paired':
                pan_src = rasterio.open(item['pan_path'])
                mss_src = rasterio.open(item['mss_path'])
            else:
                tif_src = rasterio.open(item['tif_path'])
            
            # 处理每个分块
            for x, y, w, h, tile_idx in tiles:
                # 提前过滤标注并判断是否需要处理（优化性能）
                tile_anns = filter_annotations_in_tile(all_annotations, x, y, w, h)
                has_annotations = len(tile_anns) > 0
                
                # 根据输出策略决定是否处理和保存
                should_process = True
                save_folder = output_folder
                
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
                
                window = Window(x, y, w, h)
                tile_name = f"{item['base_name']}_tile{tile_idx:03d}"
                
                # 读取并处理分块
                if item['type'] == 'paired':
                    tile_img = process_tile_with_rpc_warp(
                        pan_src, mss_src, window, pan_rpc_params, mss_rpc_params,
                        global_stats=global_stats,
                        gs_weights=gs_weights_calculated,
                        tile_name=tile_name
                    )
                else:
                    tile_img = process_tile_single(tif_src, window, global_stats=global_stats)
                
                if tile_img is None:
                    continue
                
                # 保存 PNG
                png_path = os.path.join(save_folder, f"{tile_name}.png")
                if tile_img.ndim == 3 and tile_img.shape[2] == 4:
                    save_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2BGRA)
                else:
                    save_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(png_path, save_img)
                
                # 保存 JSON（即使没有标注也保存）
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
                    "imagePath": f"{tile_name}.png",
                    "imageData": None,
                    "imageHeight": h,
                    "imageWidth": w
                }
                
                json_path = os.path.join(save_folder, f"{tile_name}.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)
                
                total_tiles += 1
                if has_annotations:
                    tiles_with_annotations += 1
                else:
                    tiles_without_annotations += 1
            
            # 关闭影像源
            if item['type'] == 'paired':
                pan_src.close()
                mss_src.close()
            else:
                tif_src.close()
            
            print(f"  ✓ 完成")
        
        except Exception as e:
            print(f"  ✗ 错误: {e}")
    
    print(f"\n{'='*80}")
    print(f"处理完成 | 共生成 {total_tiles} 个分块 | 有标注: {tiles_with_annotations} | 无标注: {tiles_without_annotations}")
    print(f"输出位置: {output_folder}")
    print(f"{'='*80}")

# ============================================================================
# 脚本入口
# ============================================================================

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
    
    process_all(
        shp_path=SHP_FILE_PATH,
        data_folder=DATA_FOLDER_PATH,
        output_folder=OUTPUT_FOLDER_PATH
    )
