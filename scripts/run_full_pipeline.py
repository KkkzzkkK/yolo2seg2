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
from rs_processor.core.normalize import clahe_normalize, bands_to_rgb_uint8
from rs_processor.core.tiff_io import read_tiff, write_tiff, write_png
from rs_processor.processing.registration import RegistrationProcessor
from rs_processor.processing.label_processor import LabelProcessor, Detection
from rs_processor.processing.tile_processor import TileProcessor, TileInfo

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"              # 影像根目录（每个场景一个子文件夹）
LABEL_DIR = r"F:\labels_export"         # YOLO 标签目录
OUTPUT_DIR = r"F:\output"               # 输出根目录

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
# Step 1: 整图配准 + 锐化 + 保存
# ============================================================================
def fuse_scene(
    scene_name: str,
    scene_dir: str,
    output_dir: str,
) -> Optional[Dict]:
    """融合单个场景的整图
    
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
    
    # 初始化配准处理器
    registrator = RegistrationProcessor(
        enable_feature_refine=True,
        refine_method=REFINE_METHOD,
    )
    
    fused_dir = os.path.join(output_dir, 'fused', scene_name)
    os.makedirs(fused_dir, exist_ok=True)
    
    with rasterio.open(pan_info['path']) as pan_ds, rasterio.open(mss_info['path']) as mss_ds:
        pan_w, pan_h = pan_ds.width, pan_ds.height
        mss_w, mss_h = mss_ds.width, mss_ds.height
        num_bands = min(mss_ds.count, 4)
        
        print(f"开始融合... (分块大小: {TILE_SIZE}, 重叠: {OVERLAP})")
        
        # 读取整个 PAN 和 MSS
        pan_data = pan_ds.read(1)
        mss_data = np.stack([mss_ds.read(b + 1) for b in range(num_bands)], axis=0)
        
        # 配准 MSS 到 PAN
        if pan_rpc and mss_rpc:
            print("执行 RPC 配准...")
            reg_result = registrator.register(
                pan_data=pan_data,
                pan_rpc=pan_rpc,
                mss_data=mss_data,
                mss_rpc=mss_rpc,
                pan_window=(0, 0, pan_w, pan_h),
            )
            mss_aligned = reg_result.aligned_mss
            print(f"配准完成: 特征点匹配 {reg_result.offset_info.feature_match_count}")
        else:
            print("无 RPC，使用简单重采样...")
            mss_aligned = np.stack([
                cv2.resize(mss_data[b], (pan_w, pan_h), interpolation=cv2.INTER_CUBIC)
                for b in range(num_bands)
            ], axis=0)
        
        # 全色锐化
        print("执行 Gram-Schmidt 锐化...")
        mss_bands = [mss_aligned[b] for b in range(mss_aligned.shape[0])]
        _, weights = calculate_band_correlations(pan_data, mss_bands)
        fused_bands = gram_schmidt_sharpen(pan_data, mss_bands, weights)
        
        # 堆叠为数组
        fused_data = np.stack(fused_bands, axis=0)
        
        # 保存融合结果
        if SAVE_FUSED_TIFF:
            tiff_path = os.path.join(fused_dir, 'fused.tif')
            print(f"保存 TIFF: {tiff_path}")
            
            # 使用 PAN 的 profile
            profile = pan_ds.profile.copy()
            profile.update(
                count=fused_data.shape[0],
                dtype=fused_data.dtype,
            )
            
            with rasterio.open(tiff_path, 'w', **profile) as dst:
                for b in range(fused_data.shape[0]):
                    dst.write(fused_data[b], b + 1)
        
        if SAVE_FUSED_PNG:
            png_path = os.path.join(fused_dir, 'fused_preview.png')
            print(f"保存 PNG 预览: {png_path}")
            rgb = bands_to_rgb_uint8(fused_data, method=PREVIEW_METHOD)
            Image.fromarray(rgb).save(png_path)
        
        # 保存融合元数据
        metadata = {
            'scene': scene_name,
            'pan_path': pan_info['path'],
            'mss_path': mss_info['path'],
            'pan_size': [pan_w, pan_h],
            'mss_size': [mss_w, mss_h],
            'fused_size': [pan_w, pan_h],
            'num_bands': fused_data.shape[0],
            'has_pan_rpc': pan_rpc is not None,
            'has_mss_rpc': mss_rpc is not None,
            'sharpen_method': SHARPEN_METHOD,
            'refine_method': REFINE_METHOD,
        }
        
        if pan_rpc:
            metadata['pan_rpc_path'] = pan_info['rpb_path']
        
        meta_path = os.path.join(fused_dir, 'fusion_metadata.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"融合完成！输出: {fused_dir}")
        
        return {
            'fused_data': fused_data,
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
    fused_data: np.ndarray,
    fused_dir: str,
    fusion_metadata: Dict,
    pan_rpc: Optional[RPCParams],
    output_dir: str,
) -> List[Dict]:
    """从融合图裁剪检测框
    
    Args:
        scene_name: 场景名称
        label_path: 标签文件路径
        fused_data: 融合后的图像数据 (bands, H, W)
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
    
    _, img_h, img_w = fused_data.shape
    h_avg = pan_rpc.height_offset if pan_rpc else 0
    
    crops_dir = os.path.join(output_dir, 'crops', scene_name)
    os.makedirs(crops_dir, exist_ok=True)
    
    results = []
    
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
            
            # 裁剪
            crop_data = fused_data[:, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            # 转换为 RGB 并保存
            rgb = bands_to_rgb_uint8(crop_data, method=PREVIEW_METHOD)
            
            crop_path = os.path.join(crops_dir, f"det_{det_idx:03d}.png")
            Image.fromarray(rgb).save(crop_path)
            
            # 计算检测框在裁剪图中的相对位置
            rel_points = [
                ((x - crop_x) / crop_w, (y - crop_y) / crop_h)
                for x, y in points_px
            ]
            
            # 计算经纬度（如果有 RPC）
            geo_points = []
            if pan_rpc:
                for x, y in points_px:
                    lon, lat = image_to_ground(x, y, pan_rpc, h_avg)
                    geo_points.append([lon, lat])
            
            # 保存元数据
            crop_metadata = {
                'det_idx': det_idx,
                'class_id': det.class_id,
                'score': det.score,
                'global_offset_x': crop_x,
                'global_offset_y': crop_y,
                'crop_width': crop_w,
                'crop_height': crop_h,
                'fused_size': [img_w, img_h],
                'original_poly_norm': [
                    [det.points[i], det.points[i+1]]
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
        fused_data=fuse_result['fused_data'],
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
