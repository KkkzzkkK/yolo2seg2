# -*- coding: utf-8 -*-
"""
将分块图的坐标转换为原始大图的坐标。

功能：
1. 重新计算每个分块在原始PAN图中的位置（偏移）
2. 保存每个分块的位置元数据（YOLO txt格式 + JSON）
3. 转换 results_json 中的分割坐标到原图坐标

输出：
- global_labels/: 每个分块对应的全局坐标txt文件（YOLO格式）
- global_results_json/: 转换后的JSON文件（坐标相对于原图）
- crop_metadata.json: 所有分块的位置元数据
"""

import argparse
import json
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import rasterio
from PIL import Image

from image_utils import (
    parse_rpb_file,
    ground_to_image_rpc,
    image_to_ground_rpc,
)


# ============================================================================
# 配置
# ============================================================================
DEFAULT_INPUT_DIR = r"F:\code\pic"              # 存放影像的根目录（每个影像一个子文件夹）
DEFAULT_LABEL_DIR = r"F:\1218\labels_export"           # YOLO 标签文件夹（每个影像一个 txt）
DEFAULT_OUTPUT_DIR = r"I:\251218\yolo_pic"
DEFAULT_RESULTS_JSON_DIR = r"I:\251218\results_json_global"
DEFAULT_GLOBAL_OUTPUT_DIR = r"I:\251218\results_json_global_converted"
DEFAULT_BOX_SCALE = 1.3
DEFAULT_SCENE_EXPAND = 3.0


def _list_some_dirs(root: str, limit: int = 10) -> List[str]:
    try:
        names = [
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ]
    except Exception:
        return []
    names.sort()
    return names[:limit]


def find_scene_dir(pic_root: str, scene_name: str) -> str:
    """在 pic_root 下定位场景目录。

    兼容以下数据组织方式：
    1) pic_root/scene_name/xxx.tif
    2) pic_root 本身就是场景目录（直接包含 tif）
    3) pic_root 下存在相近命名的目录（大小写不敏感、前缀/包含匹配）
    4) 两级目录：pic_root/*/scene_name
    """
    # 1) 直拼路径
    direct = os.path.join(pic_root, scene_name)
    if os.path.isdir(direct):
        return direct

    # 2) pic_root 本身就是场景目录
    try:
        if any(
            f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower()
            for f in os.listdir(pic_root)
            if os.path.isfile(os.path.join(pic_root, f))
        ):
            return pic_root
    except Exception:
        pass

    scene_lower = scene_name.lower()
    candidates: List[str] = []

    # 3) 在同级子目录中做大小写不敏感匹配
    try:
        for d in os.listdir(pic_root):
            full = os.path.join(pic_root, d)
            if not os.path.isdir(full):
                continue
            dl = d.lower()
            if dl == scene_lower:
                return full
            if dl.startswith(scene_lower) or scene_lower in dl:
                candidates.append(full)
    except Exception:
        candidates = []

    if len(candidates) == 1:
        return candidates[0]

    # 4) 两级目录查找：pic_root/*/scene_name
    try:
        for d in os.listdir(pic_root):
            full = os.path.join(pic_root, d)
            if not os.path.isdir(full):
                continue
            try:
                for c in os.listdir(full):
                    sub = os.path.join(full, c)
                    if os.path.isdir(sub) and c.lower() == scene_lower:
                        return sub
            except Exception:
                continue
    except Exception:
        pass

    sample_dirs = _list_some_dirs(pic_root, limit=10)
    msg = (
        f"Scene folder not found under pic_root.\n"
        f"  scene_name: {scene_name}\n"
        f"  pic_root:  {pic_root}\n"
        f"  tried:     {direct}\n"
    )
    if candidates:
        show = "\n".join([f"    - {p}" for p in candidates[:10]])
        msg += f"  candidates (name match):\n{show}\n"
    if sample_dirs:
        msg += "  sample subfolders under pic_root:\n" + "\n".join([f"    - {d}" for d in sample_dirs]) + "\n"
    msg += "  hint: 请检查 --input-dir 是否指向影像根目录（包含场景子文件夹），或直接指向该场景目录。"
    raise FileNotFoundError(msg)


def find_scene_json_dir(results_json_root: str, scene_name: str) -> str:
    """在 results_json_root 下定位场景对应的 JSON 目录。

    兼容：
    1) results_json_root/scene_name/*.json
    2) 大小写不敏感的同级子目录匹配
    3) 两级目录：results_json_root/*/scene_name
    """
    direct = os.path.join(results_json_root, scene_name)
    if os.path.isdir(direct):
        return direct

    scene_lower = scene_name.lower()

    # 同级目录大小写不敏感匹配
    try:
        for d in os.listdir(results_json_root):
            full = os.path.join(results_json_root, d)
            if os.path.isdir(full) and d.lower() == scene_lower:
                return full
    except Exception:
        pass

    # 两级目录查找
    try:
        for d in os.listdir(results_json_root):
            full = os.path.join(results_json_root, d)
            if not os.path.isdir(full):
                continue
            try:
                for c in os.listdir(full):
                    sub = os.path.join(full, c)
                    if os.path.isdir(sub) and c.lower() == scene_lower:
                        return sub
            except Exception:
                continue
    except Exception:
        pass

    return ""


def try_parse_det_idx_from_name(filename: str) -> int:
    """从文件名解析 det_idx。

    支持：seg_000_global.json / det_000_global.json
    返回：det_idx 或 -1
    """
    name = os.path.basename(filename)
    if not name.lower().endswith(".json"):
        return -1
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    # 期望：seg / det + idx + global
    if len(parts) >= 3 and parts[0].lower() in ("seg", "det") and parts[2].lower() == "global":
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1


def parse_args():
    parser = argparse.ArgumentParser(description="Convert crop coordinates to global coordinates")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-json-dir", default=DEFAULT_RESULTS_JSON_DIR)
    parser.add_argument("--global-output-dir", default=DEFAULT_GLOBAL_OUTPUT_DIR)
    parser.add_argument("--scale", type=float, default=DEFAULT_BOX_SCALE)
    parser.add_argument("--scene-expand", type=float, default=DEFAULT_SCENE_EXPAND)
    return parser.parse_args()


def read_labels(label_path: str) -> List[dict]:
    """读取YOLO OBB标签文件"""
    detections = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in (9, 10):
                continue
            cls_id = parts[0]
            try:
                coords = list(map(float, parts[1:9]))
            except ValueError:
                continue
            poly_norm = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            score = float(parts[9]) if len(parts) == 10 else None
            detections.append({"cls": cls_id, "poly_norm": poly_norm, "score": score})
    return detections


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def expand_polygon(poly_px: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    """以多边形中心为基准放大"""
    cx = sum(p[0] for p in poly_px) / len(poly_px)
    cy = sum(p[1] for p in poly_px) / len(poly_px)
    return [((x - cx) * scale + cx, (y - cy) * scale + cy) for x, y in poly_px]


def guess_suffix(stem: str) -> str:
    if "-" in stem:
        return stem.rsplit("-", 1)[1].upper()
    return ""


def looks_like_pan(suffix: str) -> bool:
    s = suffix.upper()
    return any(tag in s for tag in ["PAN", "BWDPAN"])


def resolve_scene_files(label_file: str, pic_root: str):
    """查找MSS和PAN文件"""
    label_stem = os.path.splitext(os.path.basename(label_file))[0]
    if "-" in label_stem:
        base_prefix, suffix_hint = label_stem.rsplit("-", 1)
    else:
        base_prefix, suffix_hint = label_stem, ""

    scene_dir = find_scene_dir(pic_root, base_prefix)

    tiffs = [
        f for f in os.listdir(scene_dir)
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower()
    ]
    if len(tiffs) < 2:
        raise FileNotFoundError(f"{scene_dir} 中未找到足够的 .tiff/.tif 文件（需要至少 PAN+MSS 两个）")

    infos = []
    for tif in tiffs:
        path = os.path.join(scene_dir, tif)
        try:
            with rasterio.open(path) as ds:
                res_x, res_y = ds.res if ds.res else (1.0, 1.0)
                infos.append({
                    "path": path,
                    "bands": ds.count,
                    "res": max(abs(res_x), abs(res_y)),
                    "suffix": guess_suffix(os.path.splitext(tif)[0]),
                    "has_rpb": os.path.exists(os.path.splitext(path)[0] + ".rpb"),
                })
        except Exception as exc:
            print(f"[warn] 跳过 {tif}: {exc}")

    suffix_upper = suffix_hint.upper()
    want_pan_suffix = suffix_upper if looks_like_pan(suffix_upper) else ""
    want_mss_suffix = "" if want_pan_suffix else suffix_upper

    def pick_best(candidates, want_suffix: str, prefer_pan: bool):
        filtered = [c for c in candidates if c["bands"] == 1] if prefer_pan else [c for c in candidates if c["bands"] >= 3]
        if not filtered:
            return None
        filtered.sort(key=lambda c: (
            0 if c["has_rpb"] else 1,
            0 if want_suffix and c["suffix"] == want_suffix else 1,
            c["res"],
            -c["bands"],
            os.path.basename(c["path"]),
        ))
        return filtered[0]

    pan_info = pick_best(infos, want_pan_suffix, prefer_pan=True)
    remaining = [c for c in infos if pan_info and c["path"] != pan_info["path"]]
    mss_info = pick_best(remaining, want_mss_suffix, prefer_pan=False)

    if not pan_info or not mss_info:
        raise FileNotFoundError(f"未能在 {scene_dir} 找到 PAN/MSS 配对")

    def rpb_of(info):
        return os.path.splitext(info["path"])[0] + ".rpb" if info["has_rpb"] else None

    return {
        "scene": base_prefix,
        "pan": pan_info["path"],
        "mss": mss_info["path"],
        "pan_rpb": rpb_of(pan_info),
        "mss_rpb": rpb_of(mss_info),
        "suffix": suffix_upper,
    }


def transform_points_rpc(points: List[Tuple[float, float]],
                         src_rpc: dict, dst_rpc: dict,
                         h_avg: float, iterations: int = 20) -> np.ndarray:
    """使用RPC将点从源影像坐标转换到目标影像坐标"""
    result = []
    for px, py in points:
        lon, lat = image_to_ground_rpc(px, py, src_rpc, h_avg, iterations=iterations)
        dst_x, dst_y = ground_to_image_rpc(lon, lat, h_avg, dst_rpc)
        result.append((dst_x, dst_y))
    return np.float32(result)


def compute_align_window(pan_poly: List[Tuple[float, float]],
                         pan_W: int, pan_H: int,
                         scene_expand: float,
                         min_window_size: int = 1000) -> Tuple[int, int, int, int]:
    """计算对齐窗口（与原代码相同的逻辑）"""
    pan_xs = [p[0] for p in pan_poly]
    pan_ys = [p[1] for p in pan_poly]
    pan_cx = (min(pan_xs) + max(pan_xs)) / 2
    pan_cy = (min(pan_ys) + max(pan_ys)) / 2

    det_w = max(pan_xs) - min(pan_xs)
    det_h = max(pan_ys) - min(pan_ys)
    det_size = max(det_w, det_h)

    if det_size < 200:
        effective_expand = max(scene_expand, 10.0)
    elif det_size < 500:
        effective_expand = max(scene_expand, 5.0)
    elif det_size < 1000:
        effective_expand = max(scene_expand, 3.0)
    else:
        effective_expand = scene_expand

    pan_half_w = det_w / 2 * effective_expand
    pan_half_h = det_h / 2 * effective_expand
    pan_half_w = max(pan_half_w, min_window_size / 2)
    pan_half_h = max(pan_half_h, min_window_size / 2)

    col_off = int(math.floor(pan_cx - pan_half_w))
    row_off = int(math.floor(pan_cy - pan_half_h))
    col_end = int(math.ceil(pan_cx + pan_half_w))
    row_end = int(math.ceil(pan_cy + pan_half_h))

    return col_off, row_off, col_end - col_off, row_end - row_off


def generate_control_points(col_off: int, row_off: int,
                            width: int, height: int,
                            grid_n: int = None) -> List[Tuple[float, float]]:
    """生成控制点网格"""
    if grid_n is None:
        max_dim = max(width, height)
        if max_dim > 5000:
            grid_n = 5
        elif max_dim > 2000:
            grid_n = 4
        else:
            grid_n = 3

    control_points = []
    for i in range(grid_n):
        for j in range(grid_n):
            cpx = col_off + width * j / max(grid_n - 1, 1)
            cpy = row_off + height * i / max(grid_n - 1, 1)
            control_points.append((cpx, cpy))
    return control_points


def load_existing_metadata(scene_crop_dir: str, det_idx: int) -> Dict:
    """
    尝试加载 yolo_clip_fuse.py 保存的精确元数据。
    如果存在 det_xxx_metadata.json，直接使用它（最精确）。
    """
    metadata_path = os.path.join(scene_crop_dir, f"det_{det_idx:03d}_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def compute_crop_metadata(label_path: str, pic_root: str, crop_output_dir: str, args) -> List[Dict]:
    """
    计算每个检测框裁剪后在原图中的位置。

    优先使用 yolo_clip_fuse.py 保存的精确元数据（det_xxx_metadata.json）。
    如果不存在，则通过重新计算来估算（精度较低）。

    返回元数据列表，每个元素包含:
    - det_idx: 检测框索引
    - global_offset_x: 分块在原图的X起点
    - global_offset_y: 分块在原图的Y起点
    - crop_width: 分块宽度
    - crop_height: 分块高度
    - pan_size: 原始PAN图尺寸 [W, H]
    - mss_size: 原始MSS图尺寸 [W, H]
    - original_poly_norm: 原始归一化多边形（相对于MSS）
    """
    info = resolve_scene_files(label_path, pic_root)
    scene_name = info["scene"]

    detections = read_labels(label_path)
    if not detections:
        return []

    scene_crop_dir = os.path.join(crop_output_dir, scene_name)
    metadata_list = []

    # 首先尝试加载精确元数据
    for det_idx, det in enumerate(detections):
        existing_meta = load_existing_metadata(scene_crop_dir, det_idx)
        if existing_meta:
            existing_meta["scene"] = scene_name
            existing_meta["source"] = "exact"  # 标记为精确元数据
            metadata_list.append(existing_meta)
            print(f"[{det_idx}] 使用精确元数据: offset=({existing_meta['global_offset_x']}, {existing_meta['global_offset_y']})")
            continue

    # 如果有精确元数据，直接返回
    if metadata_list:
        return metadata_list

    # 否则，通过重新计算来估算
    print(f"[info] 未找到精确元数据，尝试重新计算...")

    mss_rpc = parse_rpb_file(info["mss_rpb"]) if info.get("mss_rpb") else None
    pan_rpc = parse_rpb_file(info["pan_rpb"]) if info.get("pan_rpb") else None

    with rasterio.open(info["mss"]) as ms_ds, rasterio.open(info["pan"]) as pan_ds:
        ms_w, ms_h = ms_ds.width, ms_ds.height
        pan_W, pan_H = pan_ds.width, pan_ds.height
        h_avg = pan_rpc.get("heightOffset", 0) if pan_rpc else 0

        for det_idx, det in enumerate(detections):
            crop_path = os.path.join(scene_crop_dir, f"det_{det_idx:03d}_fused.png")
            if not os.path.exists(crop_path):
                print(f"[warn] 未找到裁剪图: {crop_path}")
                continue

            try:
                with Image.open(crop_path) as img:
                    crop_w, crop_h = img.size

                # 重新计算裁剪逻辑（与原代码相同）
                poly_ms_px = [(x * ms_w, y * ms_h) for x, y in det["poly_norm"]]
                poly_ms_px_expanded = expand_polygon(poly_ms_px, args.scale)

                # MSS -> PAN 坐标转换
                if pan_rpc and mss_rpc:
                    pan_poly = list(transform_points_rpc(
                        poly_ms_px_expanded, mss_rpc, pan_rpc, h_avg, iterations=20
                    ))
                    pan_poly = [(float(p[0]), float(p[1])) for p in pan_poly]
                else:
                    ms_transform = ms_ds.transform
                    pan_transform = pan_ds.transform
                    pan_poly = []
                    for x_ms, y_ms in poly_ms_px_expanded:
                        gx, gy = ms_transform * (x_ms, y_ms)
                        px, py = ~pan_transform * (gx, gy)
                        pan_poly.append((px, py))

                # 计算对齐窗口
                pan_col_off, pan_row_off, pan_width, pan_height = compute_align_window(
                    pan_poly, pan_W, pan_H, args.scene_expand, min_window_size=1000
                )

                # 限制窗口大小
                pan_width = min(pan_width, pan_W)
                pan_height = min(pan_height, pan_H)

                # 移动窗口到图像内
                if pan_col_off < 0:
                    pan_col_off = 0
                elif pan_col_off + pan_width > pan_W:
                    pan_col_off = pan_W - pan_width

                if pan_row_off < 0:
                    pan_row_off = 0
                elif pan_row_off + pan_height > pan_H:
                    pan_row_off = pan_H - pan_height

                # 计算检测框在窗口内的相对位置
                rel_poly = [(px - pan_col_off, py - pan_row_off) for px, py in pan_poly]
                xs = [p[0] for p in rel_poly]
                ys = [p[1] for p in rel_poly]

                det_xmin = int(clamp(min(xs), 0, pan_width - 1))
                det_xmax = int(clamp(max(xs), 0, pan_width - 1))
                det_ymin = int(clamp(min(ys), 0, pan_height - 1))
                det_ymax = int(clamp(max(ys), 0, pan_height - 1))

                # 估算最终裁剪的偏移
                # 根据实际裁剪尺寸反推偏移，假设裁剪是以检测框为中心的
                det_center_x = (det_xmin + det_xmax) / 2
                det_center_y = (det_ymin + det_ymax) / 2

                xmin = int(det_center_x - crop_w / 2)
                ymin = int(det_center_y - crop_h / 2)

                # 确保在窗口范围内
                xmin = max(0, min(xmin, pan_width - crop_w))
                ymin = max(0, min(ymin, pan_height - crop_h))

                # 全局偏移
                global_x = pan_col_off + xmin
                global_y = pan_row_off + ymin

                metadata = {
                    "det_idx": det_idx,
                    "scene": scene_name,
                    "global_offset_x": global_x,
                    "global_offset_y": global_y,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                    "pan_size": [pan_W, pan_H],
                    "mss_size": [ms_w, ms_h],
                    "original_poly_norm": det["poly_norm"],
                    "class_id": det["cls"],
                    "score": det["score"],
                    "source": "estimated",  # 标记为估算值
                    # 用于调试
                    "pan_window": {
                        "col_off": pan_col_off,
                        "row_off": pan_row_off,
                        "width": pan_width,
                        "height": pan_height
                    },
                    "local_crop_offset": {
                        "xmin": xmin,
                        "ymin": ymin
                    }
                }
                metadata_list.append(metadata)
                print(f"[{det_idx}] 估算偏移: ({global_x}, {global_y}), 裁剪尺寸: ({crop_w}, {crop_h})")

            except Exception as exc:
                print(f"[{det_idx}] error: {exc}")

    return metadata_list


def convert_json_to_global(json_path: str, metadata: Dict, output_path: str, pan_rpc: dict = None):
    """
    将results_json中的坐标转换为相对于原图的坐标。

    支持两种输入JSON格式:
    1. labelme格式: {"version": ..., "shapes": [{"points": [...], ...}]}
    2. 简单列表格式: [{"polygon": [...], ...}]

    输出JSON格式: 保持原格式，坐标转换为原图像素坐标和经纬度坐标
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    crop_w = metadata["crop_width"]
    crop_h = metadata["crop_height"]
    global_x = metadata["global_offset_x"]
    global_y = metadata["global_offset_y"]
    pan_W, pan_H = metadata["pan_size"]
    h_avg = pan_rpc.get("heightOffset", 0) if pan_rpc else 0

    def convert_points(points, is_normalized=False):
        """转换点坐标到全局坐标和经纬度"""
        new_points = []
        new_points_pixel = []
        geo_points = []
        for point in points:
            if is_normalized:
                # 归一化坐标 -> 分块像素坐标
                local_x = point[0] * crop_w
                local_y = point[1] * crop_h
            else:
                # 已经是像素坐标
                local_x = point[0]
                local_y = point[1]
            # 分块像素坐标 -> 原图像素坐标
            global_px = global_x + local_x
            global_py = global_y + local_y
            # 原图像素坐标 -> 原图归一化坐标
            norm_x = global_px / pan_W
            norm_y = global_py / pan_H
            new_points.append([norm_x, norm_y])
            new_points_pixel.append([global_px, global_py])

            # 原图像素坐标 -> 经纬度坐标
            if pan_rpc:
                lon, lat = image_to_ground_rpc(global_px, global_py, pan_rpc, h_avg, iterations=20)
                geo_points.append([lon, lat])
            else:
                geo_points.append([0.0, 0.0])  # 如果没有RPC，填充0

        return new_points, new_points_pixel, geo_points

    # 检测JSON格式
    if isinstance(data, dict) and "shapes" in data:
        # labelme格式
        converted_data = data.copy()
        converted_shapes = []
        for shape in data["shapes"]:
            new_shape = shape.copy()
            if "points" in shape:
                # labelme的points是像素坐标，不是归一化的
                _, new_points_pixel, geo_points = convert_points(shape["points"], is_normalized=False)
                new_shape["points"] = new_points_pixel
                new_shape["geo_points"] = geo_points  # 添加经纬度坐标
                new_shape["global_offset"] = {
                    "x": global_x,
                    "y": global_y,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                }
            converted_shapes.append(new_shape)
        converted_data["shapes"] = converted_shapes
        # 添加全局元数据
        converted_data["global_metadata"] = {
            "pan_size": [pan_W, pan_H],
            "crop_offset": [global_x, global_y],
            "crop_size": [crop_w, crop_h],
        }
    elif isinstance(data, list):
        # 简单列表格式
        converted_data = []
        for item in data:
            new_item = item.copy() if isinstance(item, dict) else item
            if isinstance(item, dict) and "polygon" in item:
                new_polygon, _, geo_polygon = convert_points(item["polygon"], is_normalized=True)
                new_item["polygon"] = new_polygon  # 归一化到原图
                new_item["geo_polygon"] = geo_polygon  # 经纬度坐标
            converted_data.append(new_item)
    else:
        # 未知格式，直接返回原数据
        converted_data = data

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)


def save_global_yolo_label(metadata: Dict, output_dir: str):
    """
    保存YOLO格式的全局标签（相对于原图）。

    格式: class x1 y1 x2 y2 x3 y3 x4 y4 [score]
    """
    scene = metadata["scene"]
    det_idx = metadata["det_idx"]
    pan_W, pan_H = metadata["pan_size"]
    global_x = metadata["global_offset_x"]
    global_y = metadata["global_offset_y"]
    crop_w = metadata["crop_width"]
    crop_h = metadata["crop_height"]

    # 原始多边形是相对于MSS的归一化坐标
    # 这里我们需要保存检测框在PAN图中的全局位置
    # 使用裁剪区域的四个角点作为边界框

    # 裁剪区域的四个角（像素坐标）
    x1, y1 = global_x, global_y
    x2, y2 = global_x + crop_w, global_y
    x3, y3 = global_x + crop_w, global_y + crop_h
    x4, y4 = global_x, global_y + crop_h

    # 归一化到原图
    corners_norm = [
        (x1 / pan_W, y1 / pan_H),
        (x2 / pan_W, y2 / pan_H),
        (x3 / pan_W, y3 / pan_H),
        (x4 / pan_W, y4 / pan_H),
    ]

    output_path = os.path.join(output_dir, scene, f"det_{det_idx:03d}_global.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cls_id = metadata["class_id"]
    score = metadata.get("score", "")
    score_str = f" {score}" if score else ""

    coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in corners_norm])
    line = f"{cls_id} {coords_str}{score_str}\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(line)

    return output_path


def main():
    args = parse_args()

    label_dir = args.label_dir if os.path.isabs(args.label_dir) else os.path.join(os.getcwd(), args.label_dir)
    pic_root = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(os.getcwd(), args.input_dir)
    crop_output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    results_json_dir = args.results_json_dir if os.path.isabs(args.results_json_dir) else os.path.join(os.getcwd(), args.results_json_dir)
    global_output_dir = args.global_output_dir if os.path.isabs(args.global_output_dir) else os.path.join(os.getcwd(), args.global_output_dir)

    os.makedirs(global_output_dir, exist_ok=True)

    # 查找所有标签文件
    label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(".txt")]

    all_metadata = {}

    for label_path in label_files:
        label_stem = os.path.splitext(os.path.basename(label_path))[0]
        if "-" in label_stem:
            scene_name = label_stem.rsplit("-", 1)[0]
        else:
            scene_name = label_stem

        print(f"\n{'='*60}")
        print(f"处理场景: {scene_name}")
        print(f"{'='*60}")

        try:
            # 解析场景文件，获取 PAN RPB 路径
            info = resolve_scene_files(label_path, pic_root)
            pan_rpc = None
            if info.get("pan_rpb") and os.path.exists(info["pan_rpb"]):
                pan_rpc = parse_rpb_file(info["pan_rpb"])
                print(f"  加载 RPB 文件: {os.path.basename(info['pan_rpb'])}")
            else:
                print(f"  [warn] 未找到 RPB 文件，经纬度将填充为 0")

            metadata_list = compute_crop_metadata(label_path, pic_root, crop_output_dir, args)
            all_metadata[scene_name] = metadata_list

            meta_by_det_idx = {m.get("det_idx"): m for m in metadata_list if isinstance(m, dict) and "det_idx" in m}

            # 保存全局YOLO标签
            global_labels_dir = os.path.join(global_output_dir, "global_labels")
            for meta in metadata_list:
                save_global_yolo_label(meta, global_labels_dir)

            # 转换 results_json
            scene_json_dir = find_scene_json_dir(results_json_dir, scene_name)
            global_json_dir = os.path.join(global_output_dir, "global_results_json", scene_name)

            if scene_json_dir and os.path.isdir(scene_json_dir):
                converted_cnt = 0
                skipped_cnt = 0
                for root, _, files in os.walk(scene_json_dir):
                    for fn in files:
                        if not fn.lower().endswith(".json"):
                            continue
                        det_idx = try_parse_det_idx_from_name(fn)
                        if det_idx < 0:
                            continue
                        meta = meta_by_det_idx.get(det_idx)
                        if not meta:
                            skipped_cnt += 1
                            continue
                        json_path = os.path.join(root, fn)
                        output_json_path = os.path.join(global_json_dir, fn)
                        convert_json_to_global(json_path, meta, output_json_path, pan_rpc)
                        converted_cnt += 1
                        print(f"  转换JSON: {fn}")

                if converted_cnt == 0:
                    print(f"  [warn] 未在 {scene_json_dir} 中找到可转换的 seg/det JSON")
                elif skipped_cnt > 0:
                    print(f"  [warn] 有 {skipped_cnt} 个 JSON 未找到对应 det_idx 元数据，已跳过")
            else:
                print(f"  [info] 未找到该场景的 results_json 目录，跳过 JSON 转换")
        except Exception as exc:
            print(f"[error] {scene_name}: {exc}")

    # 保存所有元数据
    metadata_path = os.path.join(global_output_dir, "crop_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n完成！输出目录: {global_output_dir}")
    print(f"元数据文件: {metadata_path}")


if __name__ == "__main__":
    main()
