# -*- coding: utf-8 -*-
"""
一键运行脚本：坐标转换到全局（类似原 convert_coords_to_global.py）

功能：
1. 读取裁剪元数据（det_xxx_metadata.json）
2. 将分块图的检测坐标转换为原始大图坐标
3. 转换 results_json 中的分割坐标
4. 输出全局 YOLO 标签和 JSON（含经纬度）

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_convert_global.py
"""

import os
import sys
import json
from typing import List, Dict, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_processor.core.rpc_utils import parse_rpb_file, image_to_ground
from rs_processor.processing.coord_converter import CoordConverter, CropMetadata, FusionMetadata

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"                    # 影像根目录
CROP_DIR = r"F:\output\yolo_crops"            # 裁剪输出目录（含元数据）
RESULTS_JSON_DIR = r"F:\results_json"         # 检测结果 JSON 目录
OUTPUT_DIR = r"F:\output\global_coords"       # 全局坐标输出目录


def find_scene_rpb(scene_name: str, input_dir: str) -> Optional[str]:
    """查找场景的 PAN RPB 文件"""
    scene_dir = os.path.join(input_dir, scene_name)
    if not os.path.isdir(scene_dir):
        return None
    
    for f in os.listdir(scene_dir):
        if f.lower().endswith('.rpb'):
            # 优先选择 PAN 的 RPB
            stem = os.path.splitext(f)[0].upper()
            if 'PAN' in stem or 'BWD' in stem:
                return os.path.join(scene_dir, f)
    
    # 如果没有明确的 PAN RPB，返回第一个找到的
    for f in os.listdir(scene_dir):
        if f.lower().endswith('.rpb'):
            return os.path.join(scene_dir, f)
    
    return None


def load_crop_metadata(crop_dir: str) -> List[Dict]:
    """加载裁剪元数据"""
    metadata_list = []
    
    for f in os.listdir(crop_dir):
        if f.endswith('_metadata.json') and f.startswith('det_'):
            path = os.path.join(crop_dir, f)
            try:
                with open(path, 'r', encoding='utf-8') as fp:
                    meta = json.load(fp)
                    metadata_list.append(meta)
            except Exception as e:
                print(f"[warn] 加载元数据失败 {f}: {e}")
    
    # 按 det_idx 排序
    metadata_list.sort(key=lambda x: x.get('det_idx', 0))
    return metadata_list


def convert_json_to_global(
    json_path: str,
    metadata: Dict,
    output_path: str,
    pan_rpc=None,
):
    """将 results_json 中的坐标转换为全局坐标"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    crop_w = metadata['crop_width']
    crop_h = metadata['crop_height']
    global_x = metadata['global_offset_x']
    global_y = metadata['global_offset_y']
    pan_w, pan_h = metadata['pan_size']
    h_avg = pan_rpc.height_offset if pan_rpc else 0
    
    def convert_points(points, is_normalized=False):
        """转换点坐标"""
        new_points_norm = []
        new_points_pixel = []
        geo_points = []
        
        for point in points:
            if is_normalized:
                local_x = point[0] * crop_w
                local_y = point[1] * crop_h
            else:
                local_x = point[0]
                local_y = point[1]
            
            # 局部 -> 全局像素
            global_px = global_x + local_x
            global_py = global_y + local_y
            
            # 全局像素 -> 归一化
            norm_x = global_px / pan_w
            norm_y = global_py / pan_h
            
            new_points_norm.append([norm_x, norm_y])
            new_points_pixel.append([global_px, global_py])
            
            # 全局像素 -> 经纬度
            if pan_rpc:
                lon, lat = image_to_ground(global_px, global_py, pan_rpc, h_avg)
                geo_points.append([lon, lat])
            else:
                geo_points.append([0.0, 0.0])
        
        return new_points_norm, new_points_pixel, geo_points
    
    # 检测 JSON 格式
    if isinstance(data, dict) and 'shapes' in data:
        # labelme 格式
        converted_data = data.copy()
        converted_shapes = []
        
        for shape in data['shapes']:
            new_shape = shape.copy()
            if 'points' in shape:
                _, new_points_pixel, geo_points = convert_points(shape['points'], is_normalized=False)
                new_shape['points'] = new_points_pixel
                new_shape['geo_points'] = geo_points
                new_shape['global_offset'] = {
                    'x': global_x,
                    'y': global_y,
                    'crop_width': crop_w,
                    'crop_height': crop_h,
                }
            converted_shapes.append(new_shape)
        
        converted_data['shapes'] = converted_shapes
        converted_data['global_metadata'] = {
            'pan_size': [pan_w, pan_h],
            'crop_offset': [global_x, global_y],
            'crop_size': [crop_w, crop_h],
        }
    
    elif isinstance(data, list):
        # 简单列表格式
        converted_data = []
        for item in data:
            new_item = item.copy() if isinstance(item, dict) else item
            if isinstance(item, dict) and 'polygon' in item:
                new_polygon, _, geo_polygon = convert_points(item['polygon'], is_normalized=True)
                new_item['polygon'] = new_polygon
                new_item['geo_polygon'] = geo_polygon
            converted_data.append(new_item)
    else:
        converted_data = data
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


def save_global_yolo_label(metadata: Dict, output_dir: str) -> str:
    """保存全局 YOLO 标签"""
    det_idx = metadata['det_idx']
    pan_w, pan_h = metadata['pan_size']
    global_x = metadata['global_offset_x']
    global_y = metadata['global_offset_y']
    crop_w = metadata['crop_width']
    crop_h = metadata['crop_height']
    
    # 裁剪区域的四个角
    corners = [
        (global_x, global_y),
        (global_x + crop_w, global_y),
        (global_x + crop_w, global_y + crop_h),
        (global_x, global_y + crop_h),
    ]
    
    # 归一化
    corners_norm = [(x / pan_w, y / pan_h) for x, y in corners]
    
    output_path = os.path.join(output_dir, f"det_{det_idx:03d}_global.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cls_id = metadata.get('class_id', '0')
    score = metadata.get('score')
    score_str = f" {score}" if score else ""
    
    coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in corners_norm])
    line = f"{cls_id} {coords_str}{score_str}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(line)
    
    return output_path


def process_scene(
    scene_name: str,
    crop_dir: str,
    results_json_dir: Optional[str],
    output_dir: str,
    pan_rpc=None,
) -> List[Dict]:
    """处理单个场景"""
    print(f"\n{'='*60}")
    print(f"处理场景: {scene_name}")
    print(f"{'='*60}")
    
    # 加载元数据
    metadata_list = load_crop_metadata(crop_dir)
    if not metadata_list:
        print(f"[warn] 未找到元数据")
        return []
    
    print(f"找到 {len(metadata_list)} 个裁剪元数据")
    
    results = []
    
    # 保存全局 YOLO 标签
    labels_dir = os.path.join(output_dir, 'global_labels')
    for meta in metadata_list:
        try:
            label_path = save_global_yolo_label(meta, labels_dir)
            print(f"[{meta['det_idx']}] 保存标签: {os.path.basename(label_path)}")
            results.append({'det_idx': meta['det_idx'], 'success': True, 'label': label_path})
        except Exception as e:
            print(f"[{meta['det_idx']}] 保存标签失败: {e}")
            results.append({'det_idx': meta['det_idx'], 'success': False, 'error': str(e)})
    
    # 转换 results_json
    if results_json_dir and os.path.isdir(results_json_dir):
        json_output_dir = os.path.join(output_dir, 'global_results_json')
        meta_by_idx = {m['det_idx']: m for m in metadata_list}
        
        for fn in os.listdir(results_json_dir):
            if not fn.lower().endswith('.json'):
                continue
            
            # 解析 det_idx
            det_idx = -1
            parts = os.path.splitext(fn)[0].split('_')
            if len(parts) >= 2:
                try:
                    det_idx = int(parts[1])
                except ValueError:
                    pass
            
            if det_idx < 0 or det_idx not in meta_by_idx:
                continue
            
            meta = meta_by_idx[det_idx]
            json_path = os.path.join(results_json_dir, fn)
            output_path = os.path.join(json_output_dir, fn)
            
            try:
                convert_json_to_global(json_path, meta, output_path, pan_rpc)
                print(f"[{det_idx}] 转换 JSON: {fn}")
            except Exception as e:
                print(f"[{det_idx}] 转换 JSON 失败: {e}")
    
    return results


def main():
    """主函数"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 查找所有场景目录
    scene_dirs = [
        d for d in os.listdir(CROP_DIR)
        if os.path.isdir(os.path.join(CROP_DIR, d))
    ]
    
    if not scene_dirs:
        print(f"未找到场景目录: {CROP_DIR}")
        return
    
    all_metadata = {}
    total_success = 0
    total_failed = 0
    
    for scene_name in scene_dirs:
        scene_crop_dir = os.path.join(CROP_DIR, scene_name)
        scene_json_dir = os.path.join(RESULTS_JSON_DIR, scene_name) if RESULTS_JSON_DIR else None
        scene_output_dir = os.path.join(OUTPUT_DIR, scene_name)
        
        # 查找 RPB
        rpb_path = find_scene_rpb(scene_name, INPUT_DIR)
        pan_rpc = None
        if rpb_path:
            pan_rpc = parse_rpb_file(rpb_path)
            print(f"加载 RPB: {os.path.basename(rpb_path)}")
        else:
            print(f"[warn] 未找到 RPB，经纬度将填充为 0")
        
        try:
            results = process_scene(
                scene_name,
                scene_crop_dir,
                scene_json_dir if scene_json_dir and os.path.isdir(scene_json_dir) else None,
                scene_output_dir,
                pan_rpc,
            )
            
            success = sum(1 for r in results if r.get('success', False))
            total_success += success
            total_failed += len(results) - success
            
            # 保存元数据
            metadata_list = load_crop_metadata(scene_crop_dir)
            all_metadata[scene_name] = metadata_list
            
        except Exception as e:
            print(f"[error] {scene_name}: {e}")
            total_failed += 1
    
    # 保存汇总元数据
    metadata_path = os.path.join(OUTPUT_DIR, 'crop_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"全部完成！")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"元数据文件: {metadata_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="坐标转换到全局一键运行脚本")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="影像根目录")
    parser.add_argument("--crop-dir", default=CROP_DIR, help="裁剪输出目录")
    parser.add_argument("--results-json-dir", default=RESULTS_JSON_DIR, help="检测结果 JSON 目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录")
    
    args = parser.parse_args()
    
    INPUT_DIR = args.input_dir
    CROP_DIR = args.crop_dir
    RESULTS_JSON_DIR = args.results_json_dir
    OUTPUT_DIR = args.output_dir
    
    main()
