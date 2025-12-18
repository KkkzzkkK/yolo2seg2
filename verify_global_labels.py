# -*- coding: utf-8 -*-
"""
验证 global_labels JSON 坐标是否正确的脚本。

功能：
1. 将小图贴回原图的对应位置，验证位置是否正确
2. 在原图上绘制 JSON 中的多边形标注，可视化验证

输出：
- verification_outputs/paste_back/: 小图贴回原图的结果
- verification_outputs/visualize_json/: JSON多边形可视化结果
"""

import argparse
import json
import os
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import rasterio
from PIL import Image

# 允许读取大尺寸图像
Image.MAX_IMAGE_PIXELS = None
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)


# ============================================================================
# 配置
# ============================================================================
DEFAULT_PIC_DIR = "pic"
DEFAULT_YOLO_OUTPUTS_DIR = "yolo_outputs"
DEFAULT_GLOBAL_OUTPUTS_DIR = "global_outputs"
DEFAULT_VERIFICATION_DIR = "verification_outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Verify global labels")
    parser.add_argument("--pic-dir", default=DEFAULT_PIC_DIR, help="原始影像目录")
    parser.add_argument("--yolo-outputs-dir", default=DEFAULT_YOLO_OUTPUTS_DIR, help="裁剪图输出目录")
    parser.add_argument("--global-outputs-dir", default=DEFAULT_GLOBAL_OUTPUTS_DIR, help="全局坐标输出目录")
    parser.add_argument("--verification-dir", default=DEFAULT_VERIFICATION_DIR, help="验证结果输出目录")
    parser.add_argument("--scene", type=str, help="指定场景名称（可选，不指定则处理所有场景）")
    parser.add_argument("--det-idx", type=int, help="指定检测框索引（可选，不指定则处理所有）")
    parser.add_argument("--mode", choices=["paste", "visualize", "both"], default="both",
                        help="验证模式：paste=贴回原图, visualize=可视化JSON, both=两者都做")
    parser.add_argument("--scale", type=float, default=0.25,
                        help="输出图像缩放比例（原图太大，建议0.1-0.5）")
    parser.add_argument("--context-expand", type=float, default=2.0,
                        help="显示上下文区域的扩展倍数（相对于小图尺寸）")
    parser.add_argument("--use-mss", action="store_true",
                        help="贴回MSS图像而不是PAN图像")
    return parser.parse_args()


def find_pan_tiff(scene_dir: str) -> Optional[str]:
    """在场景目录中查找PAN TIFF文件"""
    if not os.path.isdir(scene_dir):
        return None

    for f in os.listdir(scene_dir):
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower():
            stem = os.path.splitext(f)[0]
            # 判断是否是PAN图像
            suffix = stem.rsplit("-", 1)[-1].upper() if "-" in stem else ""
            if any(tag in suffix for tag in ["PAN", "BWDPAN"]):
                return os.path.join(scene_dir, f)

    # 如果没找到明确的PAN，尝试找单波段的
    for f in os.listdir(scene_dir):
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower():
            path = os.path.join(scene_dir, f)
            try:
                with rasterio.open(path) as ds:
                    if ds.count == 1:
                        return path
            except:
                pass
    return None


def find_mss_tiff(scene_dir: str) -> Optional[str]:
    """在场景目录中查找MSS TIFF文件"""
    if not os.path.isdir(scene_dir):
        return None

    for f in os.listdir(scene_dir):
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower():
            stem = os.path.splitext(f)[0]
            suffix = stem.rsplit("-", 1)[-1].upper() if "-" in stem else ""
            # 判断是否是MSS图像
            if any(tag in suffix for tag in ["MSS", "MUX"]):
                return os.path.join(scene_dir, f)

    # 如果没找到明确的MSS，尝试找多波段的
    for f in os.listdir(scene_dir):
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower():
            path = os.path.join(scene_dir, f)
            try:
                with rasterio.open(path) as ds:
                    if ds.count >= 3:
                        return path
            except:
                pass
    return None


def read_pan_region(pan_path: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    """读取PAN图像的指定区域"""
    with rasterio.open(pan_path) as ds:
        # 确保窗口在图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, ds.width - x)
        h = min(h, ds.height - y)

        if w <= 0 or h <= 0:
            return None

        window = rasterio.windows.Window(x, y, w, h)
        data = ds.read(1, window=window)

        # 归一化到0-255
        if data.dtype != np.uint8:
            p2, p98 = np.percentile(data[data > 0], [2, 98]) if np.any(data > 0) else (0, 1)
            if p98 > p2:
                data = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
            else:
                data = np.zeros_like(data, dtype=np.uint8)

        return data


def read_mss_region(mss_path: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    """读取MSS图像的指定区域，返回RGB图像"""
    with rasterio.open(mss_path) as ds:
        # 确保窗口在图像范围内
        x = max(0, x)
        y = max(0, y)
        w = min(w, ds.width - x)
        h = min(h, ds.height - y)

        if w <= 0 or h <= 0:
            return None

        window = rasterio.windows.Window(x, y, w, h)

        # 读取多波段数据
        if ds.count >= 4:
            # 假设波段顺序: B, G, R, NIR (或类似)
            # 使用 R, G, B 波段 (索引 3, 2, 1)
            r = ds.read(3, window=window)
            g = ds.read(2, window=window)
            b = ds.read(1, window=window)
        elif ds.count >= 3:
            r = ds.read(1, window=window)
            g = ds.read(2, window=window)
            b = ds.read(3, window=window)
        else:
            # 单波段
            data = ds.read(1, window=window)
            r = g = b = data

        # 归一化到0-255
        result = np.zeros((h, w, 3), dtype=np.uint8)
        for i, band in enumerate([b, g, r]):  # OpenCV uses BGR
            if band.dtype != np.uint8:
                valid = band[band > 0]
                if len(valid) > 0:
                    p2, p98 = np.percentile(valid, [2, 98])
                    if p98 > p2:
                        band = np.clip((band - p2) / (p98 - p2) * 255, 0, 255)
                    else:
                        band = np.zeros_like(band)
            result[:, :, i] = band.astype(np.uint8)

        return result


def get_pan_size(pan_path: str) -> Tuple[int, int]:
    """获取PAN图像尺寸"""
    with rasterio.open(pan_path) as ds:
        return ds.width, ds.height


def paste_crop_back(
    pan_path: str,
    crop_img: np.ndarray,
    metadata: Dict,
    context_expand: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    将裁剪图贴回原图的对应位置。

    返回：
    - result_img: 包含原图背景和贴回的小图的区域图像
    - region_info: 区域信息
    """
    global_x = metadata["global_offset_x"]
    global_y = metadata["global_offset_y"]
    crop_w = metadata["crop_width"]
    crop_h = metadata["crop_height"]

    # 计算显示区域（包含上下文）
    expand_w = int(crop_w * context_expand)
    expand_h = int(crop_h * context_expand)

    region_x = max(0, global_x - expand_w // 2)
    region_y = max(0, global_y - expand_h // 2)
    region_w = crop_w + expand_w
    region_h = crop_h + expand_h

    # 读取原图区域
    pan_region = read_pan_region(pan_path, region_x, region_y, region_w, region_h)
    if pan_region is None:
        return None, None

    # 转换为3通道用于绘制
    result = cv2.cvtColor(pan_region, cv2.COLOR_GRAY2BGR)

    # 计算小图在显示区域中的位置
    paste_x = global_x - region_x
    paste_y = global_y - region_y

    # 确保小图与目标区域尺寸匹配
    crop_rgb = crop_img
    if len(crop_rgb.shape) == 2:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2BGR)
    elif crop_rgb.shape[2] == 4:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGBA2BGR)

    # 调整小图大小以匹配元数据中的尺寸
    if crop_rgb.shape[0] != crop_h or crop_rgb.shape[1] != crop_w:
        crop_rgb = cv2.resize(crop_rgb, (crop_w, crop_h))

    # 计算实际可贴入的区域
    paste_x_end = min(paste_x + crop_w, result.shape[1])
    paste_y_end = min(paste_y + crop_h, result.shape[0])
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    crop_x_start = 0 if paste_x >= 0 else -paste_x
    crop_y_start = 0 if paste_y >= 0 else -paste_y
    crop_x_end = crop_x_start + (paste_x_end - paste_x)
    crop_y_end = crop_y_start + (paste_y_end - paste_y)

    # 创建左右对比图
    comparison = np.zeros((result.shape[0], result.shape[1] * 2 + 10, 3), dtype=np.uint8)
    comparison[:, :result.shape[1]] = result  # 左边是原图
    comparison[:, result.shape[1] + 10:] = result.copy()  # 右边是贴回后的图

    # 在右边贴入小图
    right_img = comparison[:, result.shape[1] + 10:]
    if paste_y_end > paste_y and paste_x_end > paste_x:
        right_img[paste_y:paste_y_end, paste_x:paste_x_end] = \
            crop_rgb[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # 在右边绘制小图边界
    cv2.rectangle(right_img, (paste_x, paste_y),
                  (paste_x_end - 1, paste_y_end - 1), (0, 255, 0), 2)

    # 添加标注
    cv2.putText(comparison, "Original PAN", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(comparison, "Crop Pasted Back", (result.shape[1] + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    region_info = {
        "region_x": region_x,
        "region_y": region_y,
        "region_w": region_w,
        "region_h": region_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
    }

    return comparison, region_info


def visualize_json_polygons(
    pan_path: str,
    json_data: List[Dict],
    metadata: Dict,
    context_expand: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    在原图上可视化JSON中的多边形标注。
    使用归一化坐标 polygon 字段来验证！

    返回：
    - result_img: 带有多边形标注的图像
    - region_info: 区域信息
    """
    # 获取原图尺寸
    pan_W, pan_H = metadata["pan_size"]

    # 从归一化坐标计算像素坐标的中心区域
    all_pixel_coords = []
    for item in json_data:
        if "polygon" in item:
            for pt in item["polygon"]:
                px = pt[0] * pan_W
                py = pt[1] * pan_H
                all_pixel_coords.append((px, py))

    if not all_pixel_coords:
        return None, None

    # 计算显示区域（基于多边形的边界）
    xs = [p[0] for p in all_pixel_coords]
    ys = [p[1] for p in all_pixel_coords]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    # 扩展显示区域
    expand_w = max(int(bbox_w * context_expand), 500)
    expand_h = max(int(bbox_h * context_expand), 500)

    region_x = int(max(0, center_x - expand_w / 2))
    region_y = int(max(0, center_y - expand_h / 2))
    region_w = int(expand_w)
    region_h = int(expand_h)

    # 读取原图区域
    pan_region = read_pan_region(pan_path, region_x, region_y, region_w, region_h)
    if pan_region is None:
        return None, None

    # 转换为3通道
    result = cv2.cvtColor(pan_region, cv2.COLOR_GRAY2BGR)

    # 颜色映射
    colors = {
        "storage_tank": (0, 255, 0),      # 绿色
        "genset": (0, 0, 255),             # 红色
        "containment_vessel": (255, 0, 0), # 蓝色
        "cooling_tower": (255, 255, 0),    # 青色
        "default": (0, 165, 255),          # 橙色
    }

    # 绘制多边形 - 使用归一化坐标！
    for item in json_data:
        if "polygon" not in item:
            continue

        # 从归一化坐标计算像素坐标，再转到显示区域坐标
        polygon_norm = item["polygon"]
        pts = np.array([[p[0] * pan_W - region_x, p[1] * pan_H - region_y]
                        for p in polygon_norm], dtype=np.int32)

        # 获取颜色
        area_type = item.get("Key_area_type", "default")
        color = colors.get(area_type, colors["default"])

        # 绘制多边形
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)

        # 绘制标签
        if len(pts) > 0:
            label = f"{area_type}_{item.get('Key_area_id', '?')}"
            text_pos = (int(pts[:, 0].min()), int(pts[:, 1].min()) - 5)
            cv2.putText(result, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 添加标题说明
    cv2.putText(result, "Using NORMALIZED coords (polygon)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    region_info = {
        "region_x": region_x,
        "region_y": region_y,
        "region_w": region_w,
        "region_h": region_h,
        "num_polygons": len([i for i in json_data if "polygon" in i]),
    }

    return result, region_info


def load_crop_metadata(global_outputs_dir: str) -> Dict:
    """加载裁剪元数据"""
    metadata_path = os.path.join(global_outputs_dir, "crop_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def paste_crop_back_mss(
    mss_path: str,
    crop_img: np.ndarray,
    metadata: Dict,
    context_expand: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    将裁剪图贴回MSS图像的对应位置。
    """
    global_x = metadata["global_offset_x"]
    global_y = metadata["global_offset_y"]
    crop_w = metadata["crop_width"]
    crop_h = metadata["crop_height"]

    # 计算显示区域
    expand_w = int(crop_w * context_expand)
    expand_h = int(crop_h * context_expand)

    region_x = max(0, global_x - expand_w // 2)
    region_y = max(0, global_y - expand_h // 2)
    region_w = crop_w + expand_w
    region_h = crop_h + expand_h

    # 读取MSS区域
    mss_region = read_mss_region(mss_path, region_x, region_y, region_w, region_h)
    if mss_region is None:
        return None, None

    result = mss_region.copy()

    # 计算小图在显示区域中的位置
    paste_x = global_x - region_x
    paste_y = global_y - region_y

    # 确保小图与目标区域尺寸匹配
    crop_rgb = crop_img
    if len(crop_rgb.shape) == 2:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2BGR)
    elif crop_rgb.shape[2] == 4:
        crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_RGBA2BGR)

    # 缩放小图到MSS尺寸
    if crop_w > 0 and crop_h > 0:
        crop_rgb = cv2.resize(crop_rgb, (crop_w, crop_h))

    # 计算实际可贴入的区域
    paste_x_end = min(paste_x + crop_w, result.shape[1])
    paste_y_end = min(paste_y + crop_h, result.shape[0])
    paste_x = max(0, paste_x)
    paste_y = max(0, paste_y)

    crop_x_start = 0
    crop_y_start = 0
    crop_x_end = paste_x_end - paste_x
    crop_y_end = paste_y_end - paste_y

    # 创建左右对比图
    comparison = np.zeros((result.shape[0], result.shape[1] * 2 + 10, 3), dtype=np.uint8)
    comparison[:, :result.shape[1]] = result  # 左边是原图
    comparison[:, result.shape[1] + 10:] = result.copy()  # 右边是贴回后的图

    # 在右边贴入小图
    right_img = comparison[:, result.shape[1] + 10:]
    if paste_y_end > paste_y and paste_x_end > paste_x:
        right_img[paste_y:paste_y_end, paste_x:paste_x_end] = \
            crop_rgb[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # 在右边绘制小图边界
    cv2.rectangle(right_img, (paste_x, paste_y),
                  (paste_x_end - 1, paste_y_end - 1), (0, 255, 0), 2)

    # 添加标注
    cv2.putText(comparison, "Original MSS", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(comparison, "Crop Pasted Back", (result.shape[1] + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    region_info = {
        "region_x": region_x,
        "region_y": region_y,
        "region_w": region_w,
        "region_h": region_h,
        "paste_x": paste_x,
        "paste_y": paste_y,
    }

    return comparison, region_info


def visualize_json_polygons_mss(
    mss_path: str,
    json_data: List[Dict],
    display_meta: Dict,
    original_meta: Dict,
    context_expand: float = 2.0
) -> Tuple[np.ndarray, Dict]:
    """
    在MSS图像上可视化JSON中的多边形标注。
    使用归一化坐标 polygon 字段来验证！
    """
    # 获取原图尺寸（PAN和MSS）
    pan_W, pan_H = original_meta.get("pan_size", [1, 1])
    mss_W, mss_H = original_meta.get("mss_size", [1, 1])

    # 从归一化坐标计算MSS像素坐标
    all_pixel_coords = []
    for item in json_data:
        if "polygon" in item:
            for pt in item["polygon"]:
                # 归一化坐标是相对于PAN的，需要转换到MSS
                px = pt[0] * mss_W
                py = pt[1] * mss_H
                all_pixel_coords.append((px, py))

    if not all_pixel_coords:
        return None, None

    # 计算显示区域
    xs = [p[0] for p in all_pixel_coords]
    ys = [p[1] for p in all_pixel_coords]
    center_x = (min(xs) + max(xs)) / 2
    center_y = (min(ys) + max(ys)) / 2
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    expand_w = max(int(bbox_w * context_expand), 200)
    expand_h = max(int(bbox_h * context_expand), 200)

    region_x = int(max(0, center_x - expand_w / 2))
    region_y = int(max(0, center_y - expand_h / 2))
    region_w = int(expand_w)
    region_h = int(expand_h)

    # 读取MSS区域
    mss_region = read_mss_region(mss_path, region_x, region_y, region_w, region_h)
    if mss_region is None:
        return None, None

    result = mss_region.copy()

    # 颜色映射
    colors = {
        "storage_tank": (0, 255, 0),
        "genset": (0, 0, 255),
        "containment_vessel": (255, 0, 0),
        "cooling_tower": (255, 255, 0),
        "default": (0, 165, 255),
    }

    # 绘制多边形 - 使用归一化坐标！
    for item in json_data:
        if "polygon" not in item:
            continue

        # 从归一化坐标计算MSS像素坐标
        polygon_norm = item["polygon"]
        pts = np.array([[p[0] * mss_W - region_x, p[1] * mss_H - region_y]
                        for p in polygon_norm], dtype=np.int32)

        area_type = item.get("Key_area_type", "default")
        color = colors.get(area_type, colors["default"])

        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)

        if len(pts) > 0:
            label = f"{area_type}_{item.get('Key_area_id', '?')}"
            text_pos = (int(pts[:, 0].min()), int(pts[:, 1].min()) - 5)
            cv2.putText(result, label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 添加标题
    cv2.putText(result, "Using NORMALIZED coords (polygon) on MSS", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    region_info = {
        "region_x": region_x,
        "region_y": region_y,
        "region_w": region_w,
        "region_h": region_h,
        "num_polygons": len([i for i in json_data if "polygon" in i]),
    }

    return result, region_info


def load_global_json(global_outputs_dir: str, scene: str, det_idx: int) -> Optional[List[Dict]]:
    """加载全局坐标JSON"""
    json_path = os.path.join(global_outputs_dir, "global_results_json", scene,
                             f"det_{det_idx:03d}_global.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    args = parse_args()

    # 转换为绝对路径
    pic_dir = os.path.abspath(args.pic_dir)
    yolo_outputs_dir = os.path.abspath(args.yolo_outputs_dir)
    global_outputs_dir = os.path.abspath(args.global_outputs_dir)
    verification_dir = os.path.abspath(args.verification_dir)

    # 创建输出目录
    suffix = "_mss" if args.use_mss else ""
    paste_dir = os.path.join(verification_dir, f"paste_back{suffix}")
    visualize_dir = os.path.join(verification_dir, f"visualize_json{suffix}")
    os.makedirs(paste_dir, exist_ok=True)
    os.makedirs(visualize_dir, exist_ok=True)

    # 加载元数据
    all_metadata = load_crop_metadata(global_outputs_dir)
    if not all_metadata:
        print("[error] 未找到 crop_metadata.json")
        return

    # 确定要处理的场景
    scenes_to_process = [args.scene] if args.scene else list(all_metadata.keys())

    for scene in scenes_to_process:
        if scene not in all_metadata:
            print(f"[warn] 场景 {scene} 不在元数据中")
            continue

        print(f"\n{'='*60}")
        print(f"处理场景: {scene}")
        print(f"{'='*60}")

        scene_pic_dir = os.path.join(pic_dir, scene)

        if args.use_mss:
            # MSS模式
            img_path = find_mss_tiff(scene_pic_dir)
            if not img_path:
                print(f"[warn] 未找到MSS图像: {scene_pic_dir}")
                continue
            print(f"MSS图像: {img_path}")
            img_w, img_h = get_pan_size(img_path)
            print(f"MSS尺寸: {img_w} x {img_h}")
        else:
            # PAN模式
            img_path = find_pan_tiff(scene_pic_dir)
            if not img_path:
                print(f"[warn] 未找到PAN图像: {scene_pic_dir}")
                continue
            print(f"PAN图像: {img_path}")
            img_w, img_h = get_pan_size(img_path)
            print(f"PAN尺寸: {img_w} x {img_h}")

        # 处理该场景的所有检测框
        scene_metadata = all_metadata[scene]
        for meta in scene_metadata:
            det_idx = meta["det_idx"]

            # 如果指定了det_idx，只处理指定的
            if args.det_idx is not None and det_idx != args.det_idx:
                continue

            # 计算坐标（MSS模式需要转换）
            if args.use_mss:
                # 获取PAN和MSS尺寸比例
                pan_w, pan_h = meta.get("pan_size", [img_w * 4, img_h * 4])
                mss_w, mss_h = meta.get("mss_size", [img_w, img_h])
                scale_x = mss_w / pan_w
                scale_y = mss_h / pan_h

                # 转换坐标
                display_meta = meta.copy()
                display_meta["global_offset_x"] = int(meta["global_offset_x"] * scale_x)
                display_meta["global_offset_y"] = int(meta["global_offset_y"] * scale_y)
                display_meta["crop_width"] = int(meta["crop_width"] * scale_x)
                display_meta["crop_height"] = int(meta["crop_height"] * scale_y)
                print(f"\n  [{det_idx}] MSS offset=({display_meta['global_offset_x']}, {display_meta['global_offset_y']}), "
                      f"size=({display_meta['crop_width']}, {display_meta['crop_height']})")
            else:
                display_meta = meta
                print(f"\n  [{det_idx}] offset=({meta['global_offset_x']}, {meta['global_offset_y']}), "
                      f"size=({meta['crop_width']}, {meta['crop_height']})")

            # 读取裁剪图
            crop_path = os.path.join(yolo_outputs_dir, scene, f"det_{det_idx:03d}_fused.png")
            if not os.path.exists(crop_path):
                print(f"      [warn] 裁剪图不存在: {crop_path}")
                continue

            crop_img = cv2.imread(crop_path)
            if crop_img is None:
                print(f"      [warn] 无法读取裁剪图: {crop_path}")
                continue

            # 模式1: 贴回原图
            if args.mode in ["paste", "both"]:
                if args.use_mss:
                    comparison, region_info = paste_crop_back_mss(
                        img_path, crop_img, display_meta, args.context_expand
                    )
                else:
                    comparison, region_info = paste_crop_back(
                        img_path, crop_img, meta, args.context_expand
                    )
                if comparison is not None:
                    # 缩放
                    if args.scale != 1.0:
                        new_size = (int(comparison.shape[1] * args.scale),
                                    int(comparison.shape[0] * args.scale))
                        comparison = cv2.resize(comparison, new_size)

                    output_path = os.path.join(paste_dir, scene, f"det_{det_idx:03d}_comparison.png")
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, comparison)
                    print(f"      [paste] 保存: {output_path}")

            # 模式2: 可视化JSON
            if args.mode in ["visualize", "both"]:
                json_data = load_global_json(global_outputs_dir, scene, det_idx)
                if json_data:
                    if args.use_mss:
                        result, region_info = visualize_json_polygons_mss(
                            img_path, json_data, display_meta, meta, args.context_expand
                        )
                    else:
                        result, region_info = visualize_json_polygons(
                            img_path, json_data, meta, args.context_expand
                        )
                    if result is not None:
                        # 缩放
                        if args.scale != 1.0:
                            new_size = (int(result.shape[1] * args.scale),
                                        int(result.shape[0] * args.scale))
                            result = cv2.resize(result, new_size)

                        output_path = os.path.join(visualize_dir, scene,
                                                   f"det_{det_idx:03d}_polygons.png")
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        cv2.imwrite(output_path, result)
                        print(f"      [visualize] 保存: {output_path} "
                              f"(多边形数: {region_info['num_polygons']})")
                else:
                    print(f"      [warn] 未找到JSON文件")

    print(f"\n完成！结果保存在: {verification_dir}")


if __name__ == "__main__":
    main()
