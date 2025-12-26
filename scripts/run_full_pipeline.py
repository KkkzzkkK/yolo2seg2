# -*- coding: utf-8 -*-
"""
一键运行脚本：完整流水线（新流程）

流程：
1. 整图配准 + 锐化 + 保存融合结果
2. 读取 YOLO 标签 + 从融合图裁剪 + 保存元数据

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_full_pipeline.py
"""

import os
import sys
import json
import math
from typing import List, Tuple, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from PIL import Image
import rasterio
from rasterio.windows import Window

from rs_processor.core.rpc_utils import RPCParams, parse_rpb_file, ground_to_image, image_to_ground
from rs_processor.core.pan_sharpen import gram_schmidt_sharpen, calculate_band_correlations
from rs_processor.core.normalize import bands_to_rgb_uint8
from rs_processor.processing.registration import RegistrationProcessor
from rs_processor.processing.label_processor import LabelProcessor, Detection

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic\pt"              # 影像根目录（每个场景一个子文件夹）
LABEL_DIR = r"F:\1218\labels_export"         # YOLO 标签目录
OUTPUT_DIR = r"F:\1218\output"                   # 输出根目录


# 融合参数
TILE_SIZE = 4096                        # 分块大小（处理大图时分块）
OVERLAP = 256                           # 分块重叠
SHARPEN_METHOD = "gram_schmidt"         # 锐化方法
REFINE_METHOD = "auto"                  # 精配准方法: auto, orb, arosics
SAVE_FUSED_TIFF = True                  # 是否保存融合后的 TIFF
SAVE_FUSED_PNG = True                   # 是否保存融合后的 PNG 预览

# 裁剪参数
BOX_SCALE = 1.3                         # 检测框放大倍数
CROP_MULTIPLE = 64                      # 裁剪尺寸对齐倍数
MIN_CROP_SIZE = 256                     # 最小裁剪尺寸
MAX_DETECTIONS = 500                    # 最大检测数量
PREVIEW_METHOD = "clahe"                # 预览增强方法


def find_scene_files(scene_dir: str) -> Dict:
    """在场景目录中查找 PAN/MSS 文件"""
    tiffs = [
        f for f in os.listdir(scene_dir)
        if f.lower().endswith(('.tif', '.tiff')) and 'thumb' not in f.lower()
    ]
    
    if len(tiffs) < 2:
        raise FileNotFoundError(f"{scene_dir} 中未找到足够的 TIFF 文件")
    
    infos = []
    for tif in tiffs:
        path = os.path.join(scene_dir, tif)
        try:
            with rasterio.open(path) as ds:
                res_x, res_y = ds.res if ds.res else (1.0, 1.0)
                rpb_path = os.path.splitext(path)[0] + '.rpb'
                infos.append({
                    'path': path,
                    'bands': ds.count,
                    'res': max(abs(res_x), abs(res_y)),
                    'width': ds.width,
                    'height': ds.height,
                    'has_rpb': os.path.exists(rpb_path),
                    'rpb_path': rpb_path if os.path.exists(rpb_path) else None,
                })
        except Exception as e:
            print(f"[warn] 跳过 {tif}: {e}")
    
    # 选择 PAN（单波段，分辨率最高）
    pan_candidates = [i for i in infos if i['bands'] == 1]
    pan_candidates.sort(key=lambda x: (0 if x['has_rpb'] else 1, x['res']))
    
    # 选择 MSS（多波段）
    mss_candidates = [i for i in infos if i['bands'] >= 3]
    mss_candidates.sort(key=lambda x: (0 if x['has_rpb'] else 1, x['res']))
    
    if not pan_candidates or not mss_candidates:
        raise FileNotFoundError(f"未能在 {scene_dir} 找到 PAN/MSS 配对")
    
    return {
        'pan': pan_candidates[0],
        'mss': mss_candidates[0],
    }


def round_up_to_multiple(val: int, multiple: int) -> int:
    """向上取整到指定倍数"""
    return ((val + multiple - 1) // multiple) * multiple


def compute_square_crop(
    cx: float, cy: float,
    det_w: float, det_h: float,
    scale: float,
    img_w: int, img_h: int,
    multiple: int = 64,
    min_size: int = 256
) -> Tuple[int, int, int, int]:
    """计算正方形裁剪窗口
    
    Args:
        cx, cy: 检测框中心
        det_w, det_h: 检测框宽高
        scale: 放大倍数
        img_w, img_h: 图像尺寸
        multiple: 对齐倍数
        min_size: 最小尺寸
    
    Returns:
        (xmin, ymin, width, height)
    """
    # 计算放大后的尺寸
    w = det_w * scale
    h = det_h * scale
    
    # 取最大边长，向上取整到 multiple 的倍数
    side = max(w, h, min_size)
    side = round_up_to_multiple(int(side), multiple)
    side = min(side, img_w, img_h)
    side = (side // multiple) * multiple
    if side < min_size:
        side = min_size
    
    half = side / 2
    
    # 以中心为基准计算边界
    xmin = int(cx - half)
    ymin = int(cy - half)
    xmax = int(cx + half)
    ymax = int(cy + half)
    
    # 移动窗口确保在有效范围内
    if xmin < 0:
        xmax -= xmin
        xmin = 0
    if xmax > img_w:
        xmin -= (xmax - img_w)
        xmax = img_w
    if xmin < 0:
        xmin = 0
    
    if ymin < 0:
        ymax -= ymin
        ymin = 0
    if ymax > img_h:
        ymin -= (ymax - img_h)
        ymax = img_h
    if ymin < 0:
        ymin = 0
    
    return xmin, ymin, xmax - xmin, ymax - ymin


# ============================================================================
# Step 1: 整图配准 + 锐化 + 保存（分块处理大图）
# ============================================================================
def fuse_scene(
    scene_name: str,
    scene_dir: str,
    output_dir: str,
) -> Optional[Dict]:
    """融合单个场景的整图（分块处理以节省内存）
    
    配准流程：
    1. RPC 粗配准 - 使用原始尺寸计算全局偏移
    2. 特征点精配准 - 在采样数据上进行 ORB 匹配
    
    Returns:
        融合结果信息，包含输出路径和元数据
    """
    print(f"\n{'='*60}")
    print(f"Step 1: 整图融合 - {scene_name}")
    print(f"{'='*60}")
    
    # 查找文件
    files = find_scene_files(scene_dir)
    pan_info = files['pan']
    mss_info = files['mss']
    
    print(f"PAN: {os.path.basename(pan_info['path'])} ({pan_info['width']}x{pan_info['height']})")
    print(f"MSS: {os.path.basename(mss_info['path'])} ({mss_info['width']}x{mss_info['height']})")
    
    # 解析 RPC
    pan_rpc = parse_rpb_file(pan_info['rpb_path']) if pan_info['rpb_path'] else None
    mss_rpc = parse_rpb_file(mss_info['rpb_path']) if mss_info['rpb_path'] else None
    
    if pan_rpc:
        print(f"PAN RPC: 已加载")
    if mss_rpc:
        print(f"MSS RPC: 已加载")
    
    fused_dir = os.path.join(output_dir, 'fused', scene_name)
    os.makedirs(fused_dir, exist_ok=True)
    
    with rasterio.open(pan_info['path']) as pan_ds, rasterio.open(mss_info['path']) as mss_ds:
        pan_w, pan_h = pan_ds.width, pan_ds.height
        mss_w, mss_h = mss_ds.width, mss_ds.height
        num_bands = min(mss_ds.count, 4)
        
        # 计算 MSS 到 PAN 的缩放比例
        scale_x = pan_w / mss_w
        scale_y = pan_h / mss_h
        print(f"缩放比例: x={scale_x:.4f}, y={scale_y:.4f}")
        
        # ================================================================
        # 采样并进行两阶段配准
        # ================================================================
        print("[配准] 采样并进行两阶段配准...")
        
        # 采样参数 - 采样到约 2000x2000 大小
        sample_size = 2000
        sample_step_pan = max(1, max(pan_w, pan_h) // sample_size)
        sample_step_mss = max(1, max(mss_w, mss_h) // sample_size)
        
        # 读取 PAN 采样数据
        pan_sample = pan_ds.read(1)[::sample_step_pan, ::sample_step_pan]
        sample_h, sample_w = pan_sample.shape
        
        # 读取 MSS 采样数据并重采样到 PAN 采样尺寸
        mss_sample_list = []
        for b in range(num_bands):
            mss_band = mss_ds.read(b + 1)[::sample_step_mss, ::sample_step_mss]
            # 重采样到与 PAN 采样相同尺寸
            mss_resized = cv2.resize(mss_band, (sample_w, sample_h), interpolation=cv2.INTER_CUBIC)
            mss_sample_list.append(mss_resized)
        mss_sample = np.stack(mss_sample_list, axis=0)
        
        # 使用 RegistrationProcessor 进行配准
        rpc_offset_orig = (0.0, 0.0)
        feature_offset_orig = (0.0, 0.0)
        feature_match_count = 0
        feature_refine_success = False
        
        if pan_rpc and mss_rpc:
            registrator = RegistrationProcessor(
                enable_feature_refine=True,
                feature_max=2000,
                feature_min_match=10,
            )
            
            # 在采样数据上做配准，但使用原始尺寸计算 RPC 偏移
            offset_info = registrator.register_sampled(
                pan_sample=pan_sample,
                mss_sample=mss_sample,
                pan_rpc=pan_rpc,
                mss_rpc=mss_rpc,
                pan_full_size=(pan_w, pan_h),
                mss_full_size=(mss_w, mss_h),
                sample_step=sample_step_pan,
            )
            
            rpc_offset_orig = offset_info.rpc_offset
            feature_offset_orig = offset_info.feature_offset
            feature_match_count = offset_info.feature_match_count
            feature_refine_success = offset_info.feature_refine_success
            
            print(f"[粗配准] RPC 偏移: dx={rpc_offset_orig[0]:.2f}, dy={rpc_offset_orig[1]:.2f} (PAN 像素)")
            print(f"[精配准] 特征点偏移: dx={feature_offset_orig[0]:.2f}, dy={feature_offset_orig[1]:.2f}, 匹配点: {feature_match_count}")
        else:
            print("[配准] 无 RPC，使用简单缩放")
        
        # 总偏移（在 PAN 像素空间）
        total_offset_pan = (
            rpc_offset_orig[0] + feature_offset_orig[0],
            rpc_offset_orig[1] + feature_offset_orig[1]
        )
        
        print(f"[总偏移] dx={total_offset_pan[0]:.2f}, dy={total_offset_pan[1]:.2f} (PAN 像素)")
        
        # ================================================================
        # 计算全局权重
        # ================================================================
        print("计算全局锐化权重...")
        _, global_weights = calculate_band_correlations(pan_sample, [mss_sample[b] for b in range(num_bands)])
        print(f"权重: {global_weights}")
        
        # ================================================================
        # 分块处理
        # ================================================================
        tile_size = TILE_SIZE
        overlap = OVERLAP
        
        n_cols = math.ceil(pan_w / (tile_size - overlap))
        n_rows = math.ceil(pan_h / (tile_size - overlap))
        total_tiles = n_cols * n_rows
        
        print(f"分块处理: {n_cols}x{n_rows} = {total_tiles} 块 (每块 {tile_size}x{tile_size})")
        
        # 创建输出 TIFF
        tiff_path = os.path.join(fused_dir, 'fused.tif') if SAVE_FUSED_TIFF else None
        
        if tiff_path:
            profile = pan_ds.profile.copy()
            profile.update(
                count=num_bands,
                dtype='float32',
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            fused_ds = rasterio.open(tiff_path, 'w', **profile)
        else:
            fused_ds = None
        
        # 分块处理
        processed = 0
        for row in range(n_rows):
            for col in range(n_cols):
                # 计算 PAN 分块范围
                pan_x0 = col * (tile_size - overlap)
                pan_y0 = row * (tile_size - overlap)
                pan_x1 = min(pan_x0 + tile_size, pan_w)
                pan_y1 = min(pan_y0 + tile_size, pan_h)
                
                tile_w = pan_x1 - pan_x0
                tile_h = pan_y1 - pan_y0
                
                if tile_w <= 0 or tile_h <= 0:
                    continue
                
                # 读取 PAN 分块
                pan_window = Window(pan_x0, pan_y0, tile_w, tile_h)
                pan_tile = pan_ds.read(1, window=pan_window)
                
                # 计算对应的 MSS 范围（应用总偏移）
                # 偏移是 MSS 相对于 PAN 的位移，所以 MSS 窗口要减去偏移
                mss_x0 = int((pan_x0 - total_offset_pan[0]) / scale_x)
                mss_y0 = int((pan_y0 - total_offset_pan[1]) / scale_y)
                mss_x1 = int((pan_x1 - total_offset_pan[0]) / scale_x) + 1
                mss_y1 = int((pan_y1 - total_offset_pan[1]) / scale_y) + 1
                
                # 裁剪到有效范围
                mss_x0 = max(0, mss_x0)
                mss_y0 = max(0, mss_y0)
                mss_x1 = min(mss_w, mss_x1)
                mss_y1 = min(mss_h, mss_y1)
                
                if mss_x1 <= mss_x0 or mss_y1 <= mss_y0:
                    # MSS 窗口无效，用零填充
                    mss_aligned = np.zeros((num_bands, tile_h, tile_w), dtype=np.float32)
                else:
                    # 读取 MSS 分块
                    mss_window = Window(mss_x0, mss_y0, mss_x1 - mss_x0, mss_y1 - mss_y0)
                    mss_tile = np.stack([
                        mss_ds.read(b + 1, window=mss_window)
                        for b in range(num_bands)
                    ], axis=0)
                    
                    # 重采样 MSS 到 PAN 分辨率
                    mss_aligned = np.stack([
                        cv2.resize(mss_tile[b], (tile_w, tile_h), interpolation=cv2.INTER_CUBIC)
                        for b in range(num_bands)
                    ], axis=0)
                
                # 全色锐化（使用全局权重）
                mss_bands = [mss_aligned[b] for b in range(num_bands)]
                fused_bands = gram_schmidt_sharpen(pan_tile, mss_bands, global_weights)
                
                # 写入输出
                if fused_ds:
                    for b, band in enumerate(fused_bands):
                        fused_ds.write(band.astype(np.float32), b + 1, window=pan_window)
                
                processed += 1
                if processed % 10 == 0 or processed == total_tiles:
                    print(f"  进度: {processed}/{total_tiles} ({100*processed/total_tiles:.1f}%)")
        
        if fused_ds:
            fused_ds.close()
            print(f"保存 TIFF: {tiff_path}")
        
        # 生成 PNG 预览（降采样）
        if SAVE_FUSED_PNG:
            png_path = os.path.join(fused_dir, 'fused_preview.png')
            print(f"生成 PNG 预览...")
            
            # 降采样读取
            preview_max_size = 4096
            preview_scale = min(1.0, preview_max_size / max(pan_w, pan_h))
            preview_w = int(pan_w * preview_scale)
            preview_h = int(pan_h * preview_scale)
            
            if tiff_path and os.path.exists(tiff_path):
                with rasterio.open(tiff_path) as src:
                    preview_data = src.read(
                        out_shape=(num_bands, preview_h, preview_w),
                        resampling=rasterio.enums.Resampling.bilinear
                    )
            else:
                preview_data = np.stack([
                    cv2.resize(pan_ds.read(1), (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                    for _ in range(num_bands)
                ], axis=0)
            
            rgb = bands_to_rgb_uint8(preview_data)
            Image.fromarray(rgb).save(png_path)
            print(f"保存 PNG 预览: {png_path}")
        
        # 保存融合元数据（包含配准信息）
        metadata = {
            'scene': scene_name,
            'pan_path': pan_info['path'],
            'mss_path': mss_info['path'],
            'pan_size': [pan_w, pan_h],
            'mss_size': [mss_w, mss_h],
            'fused_size': [pan_w, pan_h],
            'fused_tiff': tiff_path,
            'num_bands': num_bands,
            'has_pan_rpc': pan_rpc is not None,
            'has_mss_rpc': mss_rpc is not None,
            'sharpen_method': SHARPEN_METHOD,
            'refine_method': REFINE_METHOD,
            'tile_size': tile_size,
            'overlap': overlap,
            'global_weights': [float(w) for w in global_weights],
            'scale_x': float(scale_x),
            'scale_y': float(scale_y),
            # 配准偏移信息
            'registration': {
                'rpc_offset_pan_px': [float(x) for x in rpc_offset_orig],
                'feature_offset_pan_px': [float(x) for x in feature_offset_orig],
                'feature_match_count': int(feature_match_count),
                'feature_refine_success': bool(feature_refine_success),
                'total_offset_pan_px': [float(x) for x in total_offset_pan],
            },
        }
        
        if pan_rpc:
            metadata['pan_rpc_path'] = pan_info['rpb_path']
        if mss_rpc:
            metadata['mss_rpc_path'] = mss_info['rpb_path']
        
        meta_path = os.path.join(fused_dir, 'fusion_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"融合完成！输出: {fused_dir}")
        
        return {
            'fused_tiff': tiff_path,
            'fused_dir': fused_dir,
            'metadata': metadata,
            'pan_rpc': pan_rpc,
        }


# ============================================================================
# Step 2: 读取标签 + 裁剪 + 保存元数据
# ============================================================================
def crop_detections(
    scene_name: str,
    label_path: str,
    fused_tiff: str,
    fused_dir: str,
    fusion_metadata: Dict,
    pan_rpc: Optional[RPCParams],
    output_dir: str,
) -> List[Dict]:
    """从融合图裁剪检测框
    
    Args:
        scene_name: 场景名称
        label_path: 标签文件路径
        fused_tiff: 融合后的 TIFF 文件路径
        fused_dir: 融合输出目录
        fusion_metadata: 融合元数据
        pan_rpc: PAN 的 RPC 参数
        output_dir: 输出目录
    
    Returns:
        裁剪结果列表
    """
    print(f"\n{'='*60}")
    print(f"Step 2: 裁剪检测框 - {scene_name}")
    print(f"{'='*60}")
    
    # 读取标签
    detections = LabelProcessor.read_yolo_obb(label_path)
    if not detections:
        print(f"[skip] 没有检测框")
        return []
    
    print(f"检测框数量: {len(detections)}")
    
    # 打开融合后的 TIFF
    if not fused_tiff or not os.path.exists(fused_tiff):
        print(f"[error] 融合 TIFF 不存在: {fused_tiff}")
        return []
    
    crops_dir = os.path.join(output_dir, 'crops', scene_name)
    os.makedirs(crops_dir, exist_ok=True)
    
    results = []
    
    with rasterio.open(fused_tiff) as fused_ds:
        img_w, img_h = fused_ds.width, fused_ds.height
        num_bands = fused_ds.count
        h_avg = pan_rpc.height_offset if pan_rpc else 0
        
        for det_idx, det in enumerate(detections[:MAX_DETECTIONS]):
            try:
                # 归一化坐标 -> 像素坐标（相对于融合图，即 PAN 尺寸）
                points_px = [
                    (det.points[i] * img_w, det.points[i+1] * img_h)
                    for i in range(0, 8, 2)
                ]
                
                # 计算检测框边界和中心
                xs = [p[0] for p in points_px]
                ys = [p[1] for p in points_px]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                det_w = max(xs) - min(xs)
                det_h = max(ys) - min(ys)
                
                # 计算裁剪窗口
                crop_x, crop_y, crop_w, crop_h = compute_square_crop(
                    cx, cy, det_w, det_h, BOX_SCALE,
                    img_w, img_h, CROP_MULTIPLE, MIN_CROP_SIZE
                )
                
                if crop_w <= 0 or crop_h <= 0:
                    print(f"[{det_idx}] 裁剪区域无效，跳过")
                    continue
                
                # 从 TIFF 读取裁剪区域
                crop_window = Window(crop_x, crop_y, crop_w, crop_h)
                crop_data = fused_ds.read(window=crop_window)
                
                # 转换为 RGB 并保存
                rgb = bands_to_rgb_uint8(crop_data)
                
                crop_path = os.path.join(crops_dir, f"det_{det_idx:03d}.png")
                Image.fromarray(rgb).save(crop_path)
                
                # 计算检测框在裁剪图中的相对位置
                rel_points = [
                    [float((x - crop_x) / crop_w), float((y - crop_y) / crop_h)]
                    for x, y in points_px
                ]
                
                # 计算经纬度（如果有 RPC）
                geo_points = []
                if pan_rpc:
                    for x, y in points_px:
                        lon, lat = image_to_ground(x, y, pan_rpc, h_avg)
                        geo_points.append([float(lon), float(lat)])
                
                # 保存元数据
                crop_metadata = {
                    'det_idx': int(det_idx),
                    'class_id': int(det.class_id),
                    'score': float(det.score) if det.score is not None else None,
                    'global_offset_x': int(crop_x),
                    'global_offset_y': int(crop_y),
                    'crop_width': int(crop_w),
                    'crop_height': int(crop_h),
                    'fused_size': [int(img_w), int(img_h)],
                    'original_poly_norm': [
                        [float(det.points[i]), float(det.points[i+1])]
                        for i in range(0, 8, 2)
                    ],
                    'poly_in_crop_norm': rel_points,
                    'geo_polygon': geo_points if geo_points else None,
                    'fusion_metadata_path': os.path.join(fused_dir, 'fusion_metadata.json'),
                }
                
                meta_path = os.path.join(crops_dir, f"det_{det_idx:03d}_metadata.json")
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(crop_metadata, f, ensure_ascii=False, indent=2)
                
                print(f"[{det_idx}] 保存: {crop_path} ({crop_w}x{crop_h})")
                results.append({'det_idx': det_idx, 'success': True, 'path': crop_path})
                
            except Exception as e:
                print(f"[{det_idx}] 错误: {e}")
                results.append({'det_idx': det_idx, 'success': False, 'error': str(e)})
    
    # 保存汇总元数据
    summary_path = os.path.join(crops_dir, 'crops_summary.json')
    summary = {
        'scene': scene_name,
        'total_detections': len(detections),
        'processed': len(results),
        'success': sum(1 for r in results if r.get('success', False)),
        'fusion_metadata_path': os.path.join(fused_dir, 'fusion_metadata.json'),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"裁剪完成！成功: {summary['success']}/{summary['processed']}")
    
    return results


# ============================================================================
# 主流程
# ============================================================================
def process_scene(
    scene_name: str,
    label_path: str,
    scene_dir: str,
    output_dir: str,
) -> Dict:
    """处理单个场景的完整流程"""
    
    # Step 1: 整图融合
    fuse_result = fuse_scene(scene_name, scene_dir, output_dir)
    
    if not fuse_result:
        return {'scene': scene_name, 'success': False, 'error': '融合失败'}
    
    # Step 2: 裁剪检测框
    crop_results = crop_detections(
        scene_name=scene_name,
        label_path=label_path,
        fused_tiff=fuse_result['fused_tiff'],
        fused_dir=fuse_result['fused_dir'],
        fusion_metadata=fuse_result['metadata'],
        pan_rpc=fuse_result['pan_rpc'],
        output_dir=output_dir,
    )
    
    return {
        'scene': scene_name,
        'success': True,
        'fused_dir': fuse_result['fused_dir'],
        'crops_count': len(crop_results),
        'crops_success': sum(1 for r in crop_results if r.get('success', False)),
    }


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找所有标签文件
    label_files = [
        os.path.join(LABEL_DIR, f)
        for f in os.listdir(LABEL_DIR)
        if f.lower().endswith('.txt')
    ]
    
    if not label_files:
        print(f"未找到标签文件: {LABEL_DIR}")
        return
    
    print(f"找到 {len(label_files)} 个标签文件")
    
    all_results = []
    
    for label_path in label_files:
        label_stem = os.path.splitext(os.path.basename(label_path))[0]
        if '-' in label_stem:
            scene_name = label_stem.rsplit('-', 1)[0]
        else:
            scene_name = label_stem
        
        scene_dir = os.path.join(INPUT_DIR, scene_name)
        if not os.path.isdir(scene_dir):
            # 尝试直接使用 INPUT_DIR
            if any(f.lower().endswith(('.tif', '.tiff')) for f in os.listdir(INPUT_DIR)):
                scene_dir = INPUT_DIR
            else:
                print(f"[warn] 未找到场景目录: {scene_dir}")
                continue
        
        try:
            result = process_scene(scene_name, label_path, scene_dir, OUTPUT_DIR)
            all_results.append(result)
        except Exception as e:
            print(f"[error] {scene_name}: {e}")
            all_results.append({'scene': scene_name, 'success': False, 'error': str(e)})
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"全部完成！")
    print(f"{'='*60}")
    
    success_count = sum(1 for r in all_results if r.get('success', False))
    print(f"场景: {success_count}/{len(all_results)} 成功")
    
    total_crops = sum(r.get('crops_success', 0) for r in all_results)
    print(f"裁剪: {total_crops} 个")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"  - 融合结果: {OUTPUT_DIR}/fused/")
    print(f"  - 裁剪结果: {OUTPUT_DIR}/crops/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="完整流水线：整图融合 + 裁剪")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="影像根目录")
    parser.add_argument("--label-dir", default=LABEL_DIR, help="标签目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE, help="分块大小")
    parser.add_argument("--scale", type=float, default=BOX_SCALE, help="检测框放大倍数")
    parser.add_argument("--no-tiff", action="store_true", help="不保存融合 TIFF")
    parser.add_argument("--no-png", action="store_true", help="不保存融合 PNG 预览")
    
    args = parser.parse_args()
    
    INPUT_DIR = args.input_dir
    LABEL_DIR = args.label_dir
    OUTPUT_DIR = args.output_dir
    TILE_SIZE = args.tile_size
    BOX_SCALE = args.scale
    SAVE_FUSED_TIFF = not args.no_tiff
    SAVE_FUSED_PNG = not args.no_png
    
    main()
