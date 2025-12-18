# -*- coding: utf-8 -*-
"""
Crop, register, and pan-sharpen patches around YOLO-OBB detections.

Inputs (per run):
- 一个包含 1 组多光谱 TIFF 和全色 TIFF 的文件夹（可带 .rpb，若有则使用 RPC 精配准；.xml/.jpg 仅作预览不参与处理）
- 对应多光谱的 YOLO 多边形标签（class x1 y1 x2 y2 x3 y3 x4 y4 [score]，归一化到多光谱尺寸）

Per detection 输出:
- MSS 裁剪 PNG
- PAN 裁剪 PNG
- MSS→PAN 精配准 PNG（RPC 仿射+可选特征细调）
- 融合 PNG（IHS/Brovey/Gram-Schmidt，默认 GS，可自动权重）
"""

import argparse
import math
import os
from typing import List, Tuple

import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

from image_utils import (
    parse_rpb_file,
    ground_to_image_rpc,
    image_to_ground_rpc,
    percentile_normalize,
    process_multispectral_to_8bit,
    brovey_pan_sharpening,
    ihs_pan_sharpening,
    gram_schmidt_pan_sharpening,
    simple_mean_pan_sharpening,
    calculate_band_correlations,
)
import cv2

# ============================================================================
# 用户配置区（如果不想用命令行参数，直接修改下面的默认值）
# ============================================================================
DEFAULT_INPUT_DIR = "pic"              # 存放影像的根目录（每个影像一个子文件夹）
DEFAULT_LABEL_DIR = "labels"           # YOLO 标签文件夹（每个影像一个 txt）
DEFAULT_OUTPUT_DIR = "yolo_output"






# ##########################################
DEFAULT_MS_PATH = "multispectral.tif"  # 仅用于单文件模式
DEFAULT_PAN_PATH = "pan.tif"           # 仅用于单文件模式
DEFAULT_LABELS = "labels.txt"          # 仅用于单文件模式
DEFAULT_BOX_SCALE = 1.3                # 放大检测框，形成更大矩形
DEFAULT_MAX_DETECTIONS = 500           # 保护性上限
DEFAULT_SHARPEN_METHOD = "gram_schmidt"  # brovey/ihs/gram_schmidt/mean
DEFAULT_SCENE_EXPAND = 3.0             # 计算场景大窗时，对所有检测外包框的放大倍数
DEFAULT_CROP_MULTIPLE = 256            # 裁剪尺寸的倍数（256, 512 等正方形）
ENABLE_FEATURE_REFINE = True           # ORB/SIFT 特征微调
FEATURE_MAX = 1200
FEATURE_MIN_MATCH = 18
FEATURE_RATIO = 0.75
SHARPEN_SAMPLE_RATIO = 0.2             # 计算GS权重的采样率
PREVIEW_METHOD = "clahe"               # CLAHE 独立通道增强到 PNG
SAVE_RAW_16BIT = False                 # 如需保存未增强的16bit裁剪，设为 True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop, register, and fuse PAN/MSS patches from YOLO-OBB labels.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Root folder of imagery; each scene in its own subfolder.")
    parser.add_argument("--label-dir", default=DEFAULT_LABEL_DIR, help="Folder containing YOLO txt labels (one per scene).")
    parser.add_argument("--ms-path", default=DEFAULT_MS_PATH, help="Multispectral TIFF path (relative or absolute, single-scene fallback).")
    parser.add_argument("--pan-path", default=DEFAULT_PAN_PATH, help="Panchromatic TIFF path (relative or absolute, single-scene fallback).")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="YOLO rotated label file (class cx cy w h angle), normalized to the multispectral image; single-scene fallback.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Destination folder for crops and fused PNGs.")
    parser.add_argument("--scale", type=float, default=DEFAULT_BOX_SCALE, help="Scale factor to enlarge each detected box before cropping.")
    parser.add_argument("--max-detections", type=int, default=DEFAULT_MAX_DETECTIONS, help="Optional cap to avoid runaway processing.")
    parser.add_argument("--method", default=DEFAULT_SHARPEN_METHOD, choices=["brovey", "ihs", "gram_schmidt", "mean"], help="Pan-sharpen method.")
    parser.add_argument("--no-refine", action="store_true", help="Disable feature-based refinement.")
    parser.add_argument("--preview-method", default=PREVIEW_METHOD, choices=["clahe", "percentile_rgb_global", "percentile_global"], help="8-bit PNG增强方式，默认独立通道CLAHE。")
    parser.add_argument("--save-raw", action="store_true", help="同时保存未增强的16bit裁剪（PNG）。")
    parser.add_argument("--scene-expand", type=float, default=DEFAULT_SCENE_EXPAND, help="场景大窗放大倍数（相对于所有检测外包框）。")
    parser.add_argument("--crop-multiple", type=int, default=DEFAULT_CROP_MULTIPLE, help="裁剪尺寸的倍数（如 256, 512），输出正方形。")
    return parser.parse_args()


def read_labels(label_path: str) -> List[dict]:
    detections = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) not in (9, 10):
                print(f"[warn] 忽略无效标签行: {line}")
                continue
            cls_id = parts[0]
            try:
                coords = list(map(float, parts[1:9]))
            except ValueError:
                print(f"[warn] 解析标签失败: {line}")
                continue
            poly_norm = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            score = float(parts[9]) if len(parts) == 10 else None
            detections.append({"cls": cls_id, "poly_norm": poly_norm, "score": score})
    return detections


def clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, val))


def round_up_to_multiple(val: int, multiple: int = 256) -> int:
    """将值向上取整到指定倍数"""
    return ((val + multiple - 1) // multiple) * multiple


def compute_square_crop(xmin: int, xmax: int, ymin: int, ymax: int,
                         max_w: int, max_h: int,
                         multiple: int = 256,
                         min_size: int = 256) -> Tuple[int, int, int, int]:
    """
    将矩形区域扩展为正方形，大小是 multiple 的倍数。

    参数:
        xmin, xmax, ymin, ymax: 原始区域边界
        max_w, max_h: 可用区域的最大宽高
        multiple: 目标尺寸的倍数（默认 256）
        min_size: 最小尺寸

    返回: (new_xmin, new_xmax, new_ymin, new_ymax)
    """
    # 计算当前区域的宽高和中心
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # 取最大边长，向上取整到 multiple 的倍数
    side = max(w, h, min_size)
    side = round_up_to_multiple(side, multiple)

    # 确保不超过可用区域
    side = min(side, max_w, max_h)
    # 再次确保是 multiple 的倍数（向下取）
    side = (side // multiple) * multiple
    if side < min_size:
        side = min_size

    half = side / 2

    # 以中心为基准计算新边界
    new_xmin = int(cx - half)
    new_xmax = int(cx + half)
    new_ymin = int(cy - half)
    new_ymax = int(cy + half)

    # 移动窗口确保在有效范围内
    if new_xmin < 0:
        new_xmax -= new_xmin
        new_xmin = 0
    if new_xmax > max_w:
        new_xmin -= (new_xmax - max_w)
        new_xmax = max_w
    if new_xmin < 0:
        new_xmin = 0

    if new_ymin < 0:
        new_ymax -= new_ymin
        new_ymin = 0
    if new_ymax > max_h:
        new_ymin -= (new_ymax - max_h)
        new_ymax = max_h
    if new_ymin < 0:
        new_ymin = 0

    return new_xmin, new_xmax, new_ymin, new_ymax


def single_band_to_uint8(band: np.ndarray) -> np.ndarray:
    norm = percentile_normalize(band)
    return (np.clip(norm, 0, 1) * 255).astype(np.uint8)


def bands_to_uint8(bands: List[np.ndarray], method: str) -> np.ndarray:
    """
    Convert bands to uint8 RGB preview.

    注意：process_multispectral_to_8bit 期望输入顺序为 [B, G, R, NIR]
    大多数卫星多光谱数据（如 GF2/GF7）的波段顺序就是 [B, G, R, NIR]，
    所以这里直接传入即可。
    """
    return process_multispectral_to_8bit(bands, method=method)


def save_raw_16bit(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = array
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    Image.fromarray(arr.astype(np.uint16)).save(path, bits=16)


def guess_suffix(stem: str) -> str:
    if "-" in stem:
        return stem.rsplit("-", 1)[1].upper()
    return ""


def looks_like_pan(suffix: str) -> bool:
    s = suffix.upper()
    return any(tag in s for tag in ["PAN", "BWDPAN"])


def resolve_scene_files(label_file: str, pic_root: str):
    """Find MSS, PAN, and RPCs for a label file (only .tiff + optional .rpb are considered)."""
    label_stem = os.path.splitext(os.path.basename(label_file))[0]
    if "-" in label_stem:
        base_prefix, suffix_hint = label_stem.rsplit("-", 1)
    else:
        base_prefix, suffix_hint = label_stem, ""

    scene_dir = os.path.join(pic_root, base_prefix)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Scene folder not found: {scene_dir}")

    tiffs = [
        f
        for f in os.listdir(scene_dir)
        if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower()
    ]
    if len(tiffs) < 2:
        raise FileNotFoundError(f"{scene_dir} 中未找到足够的 .tiff 文件")

    infos = []
    for tif in tiffs:
        path = os.path.join(scene_dir, tif)
        try:
            with rasterio.open(path) as ds:
                res_x, res_y = ds.res if ds.res else (1.0, 1.0)
                infos.append(
                    {
                        "path": path,
                        "bands": ds.count,
                        "res": max(abs(res_x), abs(res_y)),
                        "suffix": guess_suffix(os.path.splitext(tif)[0]),
                        "has_rpb": os.path.exists(os.path.splitext(path)[0] + ".rpb"),
                    }
                )
        except Exception as exc:
            print(f"[warn] 跳过 {tif}: {exc}")

    if len(infos) < 2:
        raise FileNotFoundError(f"{scene_dir} 中可用的 .tiff 少于 2 个")

    suffix_upper = suffix_hint.upper()
    want_pan_suffix = suffix_upper if looks_like_pan(suffix_upper) else ""
    want_mss_suffix = "" if want_pan_suffix else suffix_upper

    def pick_best(candidates, want_suffix: str, prefer_pan: bool):
        filtered = [c for c in candidates if c["bands"] == 1] if prefer_pan else [c for c in candidates if c["bands"] >= 3]
        if not filtered:
            return None
        filtered.sort(
            key=lambda c: (
                0 if c["has_rpb"] else 1,
                0 if want_suffix and c["suffix"] == want_suffix else 1,
                c["res"],
                -c["bands"],
                os.path.basename(c["path"]),
            )
        )
        return filtered[0]

    pan_info = pick_best(infos, want_pan_suffix, prefer_pan=True)
    remaining = [c for c in infos if pan_info and c["path"] != pan_info["path"]]
    mss_info = pick_best(remaining, want_mss_suffix, prefer_pan=False)

    if not pan_info or not mss_info:
        raise FileNotFoundError(f"未能在 {scene_dir} 找到 PAN/MSS 配对 (label={label_stem})")

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


def expand_polygon(poly_px: List[Tuple[float, float]], scale: float) -> List[Tuple[float, float]]:
    """以多边形中心为基准，按 scale 倍放大多边形"""
    cx = sum(p[0] for p in poly_px) / len(poly_px)
    cy = sum(p[1] for p in poly_px) / len(poly_px)
    return [((x - cx) * scale + cx, (y - cy) * scale + cy) for x, y in poly_px]


def compute_align_window(pan_poly: List[Tuple[float, float]],
                         pan_W: int, pan_H: int,
                         scene_expand: float,
                         min_window_size: int = 1000) -> Tuple[int, int, int, int]:
    """
    计算对齐窗口大小，自适应目标尺寸。

    策略：
    - 小目标：放大更多，确保对齐窗口足够大（至少 min_window_size）
    - 大目标：放大少一点，避免窗口过大

    返回: (col_off, row_off, width, height)
           col_off/row_off 可能为负数（表示窗口超出图像边界，需要后续移动）
    """
    pan_xs = [p[0] for p in pan_poly]
    pan_ys = [p[1] for p in pan_poly]
    pan_cx = (min(pan_xs) + max(pan_xs)) / 2
    pan_cy = (min(pan_ys) + max(pan_ys)) / 2

    # 检测框的实际尺寸
    det_w = max(pan_xs) - min(pan_xs)
    det_h = max(pan_ys) - min(pan_ys)
    det_size = max(det_w, det_h)

    # 自适应放大策略：小目标放大更多，大目标放大少一点
    if det_size < 200:
        # 很小的目标：窗口至少是目标的 10 倍
        effective_expand = max(scene_expand, 10.0)
    elif det_size < 500:
        # 小目标：窗口至少是目标的 5 倍
        effective_expand = max(scene_expand, 5.0)
    elif det_size < 1000:
        # 中等目标：窗口是目标的 3 倍
        effective_expand = max(scene_expand, 3.0)
    else:
        # 大目标：使用用户指定的放大倍数
        effective_expand = scene_expand

    # 计算窗口半宽半高
    pan_half_w = det_w / 2 * effective_expand
    pan_half_h = det_h / 2 * effective_expand

    # 确保窗口有最小尺寸
    pan_half_w = max(pan_half_w, min_window_size / 2)
    pan_half_h = max(pan_half_h, min_window_size / 2)

    # 以检测框中心为基准计算窗口
    col_off = int(math.floor(pan_cx - pan_half_w))
    row_off = int(math.floor(pan_cy - pan_half_h))
    col_end = int(math.ceil(pan_cx + pan_half_w))
    row_end = int(math.ceil(pan_cy + pan_half_h))

    win_w = col_end - col_off
    win_h = row_end - row_off

    return col_off, row_off, win_w, win_h


def estimate_translation(pan_uint8: np.ndarray,
                         mss_uint8: np.ndarray,
                         max_shift: int = 5) -> Tuple[float, float]:
    """
    使用相位相关估计小范围平移偏差。

    返回 dx, dy（应用到 MSS，使其对齐 PAN）。
    """
    if pan_uint8.ndim == 3:
        pan_gray = cv2.cvtColor(pan_uint8, cv2.COLOR_RGB2GRAY)
    else:
        pan_gray = pan_uint8
    if mss_uint8.ndim == 3:
        mss_gray = cv2.cvtColor(mss_uint8, cv2.COLOR_RGB2GRAY)
    else:
        mss_gray = mss_uint8

    # 为稳定性裁剪相同尺寸
    h = min(pan_gray.shape[0], mss_gray.shape[0])
    w = min(pan_gray.shape[1], mss_gray.shape[1])
    pan_crop = pan_gray[:h, :w].astype(np.float32)
    mss_crop = mss_gray[:h, :w].astype(np.float32)

    shift, response = cv2.phaseCorrelate(mss_crop, pan_crop)
    dx, dy = shift  # note: phaseCorrelate 返回 (x, y)
    if abs(dx) > max_shift or abs(dy) > max_shift or not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0
    return dx, dy


def generate_control_points(col_off: int, row_off: int,
                            width: int, height: int,
                            grid_n: int = None) -> List[Tuple[float, float]]:
    """
    在窗口内均匀生成控制点网格。

    参数:
        col_off, row_off: 窗口左上角坐标
        width, height: 窗口尺寸
        grid_n: 网格大小，None 则自动根据窗口大小选择

    返回: 控制点列表 [(x, y), ...]
    """
    if grid_n is None:
        max_dim = max(width, height)
        if max_dim > 5000:
            grid_n = 5
        elif max_dim > 2000:
            grid_n = 4
        else:
            grid_n = 3  # 最少 3x3=9 个控制点

    control_points = []
    for i in range(grid_n):
        for j in range(grid_n):
            cpx = col_off + width * j / max(grid_n - 1, 1)
            cpy = row_off + height * i / max(grid_n - 1, 1)
            control_points.append((cpx, cpy))

    return control_points


def transform_points_rpc(points: List[Tuple[float, float]],
                         src_rpc: dict, dst_rpc: dict,
                         h_avg: float, iterations: int = 20) -> np.ndarray:
    """
    使用 RPC 将点从源影像坐标转换到目标影像坐标。

    流程: src_pixel -> ground(lon,lat) -> dst_pixel
    """
    result = []
    for px, py in points:
        lon, lat = image_to_ground_rpc(px, py, src_rpc, h_avg, iterations=iterations)
        dst_x, dst_y = ground_to_image_rpc(lon, lat, h_avg, dst_rpc)
        result.append((dst_x, dst_y))
    return np.float32(result)


def warp_mss_to_pan(mss_tiles: List[np.ndarray],
                    src_control: np.ndarray,
                    dst_control: np.ndarray,
                    output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用控制点将 MSS 波段 warp 到 PAN 窗口大小。

    参数:
        mss_tiles: MSS 波段列表
        src_control: MSS 窗口内的控制点坐标
        dst_control: PAN 窗口内的控制点坐标
        output_size: 输出尺寸 (width, height)

    返回: (对齐后的 MSS 数组 (bands, height, width), 有效区域 mask (height, width))
    """
    # 计算变换矩阵
    if len(src_control) == 4:
        M = cv2.getPerspectiveTransform(src_control, dst_control)
    else:
        M, _ = cv2.findHomography(src_control, dst_control, 0)

    if M is None:
        raise ValueError("变换矩阵计算失败")

    # Warp 每个波段（使用 BORDER_CONSTANT，超出范围填 0）
    mss_aligned = []
    for band in mss_tiles:
        warped = cv2.warpPerspective(
            band.astype(np.float32),
            M,
            output_size,
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mss_aligned.append(warped)

    # 生成有效区域 mask：warp 一个全 1 的 mask，值 > 0.5 的区域为有效
    h, w = mss_tiles[0].shape
    ones_mask = np.ones((h, w), dtype=np.float32)
    valid_mask = cv2.warpPerspective(
        ones_mask,
        M,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid_mask = (valid_mask > 0.5).astype(np.uint8)

    return np.stack(mss_aligned, axis=0), valid_mask


def fuse_pan_mss(pan_crop: np.ndarray,
                 ms_crop: np.ndarray,
                 method: str = "gram_schmidt") -> List[np.ndarray]:
    """
    融合 PAN 和 MSS 裁剪。

    参数:
        pan_crop: PAN 裁剪 (H, W)
        ms_crop: MSS 裁剪 (bands, H, W)
        method: 融合方法

    返回: 融合后的波段列表
    """
    ms_band_list = [ms_crop[i] for i in range(min(ms_crop.shape[0], 4))]

    weights = None
    if method == "gram_schmidt" and len(ms_band_list) >= 3:
        _, weights = calculate_band_correlations(pan_crop, ms_band_list, sample_ratio=SHARPEN_SAMPLE_RATIO)

    if method == "brovey":
        return brovey_pan_sharpening(pan_crop, ms_band_list)
    elif method == "ihs":
        return ihs_pan_sharpening(pan_crop, ms_band_list)
    elif method == "gram_schmidt":
        return gram_schmidt_pan_sharpening(pan_crop, ms_band_list, weights=weights)
    else:
        return simple_mean_pan_sharpening(pan_crop, ms_band_list)


def save_image(array: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if array.ndim == 2:
        img = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[0] in (1, 3, 4):
        img = Image.fromarray(np.transpose(array[:3], (1, 2, 0)), mode="RGB")
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        img = Image.fromarray(array[:, :, :3], mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape for saving: {array.shape}")
    img.save(path)


def find_feature_refine(pan_uint8: np.ndarray, mss_uint8: np.ndarray):
    """ORB/SIFT-based homography refine; returns 3x3 matrix or None."""
    if pan_uint8.ndim == 2:
        pan_gray = pan_uint8
    else:
        pan_gray = cv2.cvtColor(pan_uint8, cv2.COLOR_RGB2GRAY)
    if mss_uint8.ndim == 3:
        mss_gray = cv2.cvtColor(mss_uint8, cv2.COLOR_RGB2GRAY)
    else:
        mss_gray = mss_uint8

    # 优先使用 SIFT（更稳定），失败则 ORB
    detector = None
    if hasattr(cv2, "SIFT_create"):
        try:
            detector = cv2.SIFT_create(nfeatures=FEATURE_MAX)
        except Exception:
            detector = None
    if detector is None:
        detector = cv2.ORB_create(nfeatures=FEATURE_MAX)

    kp1, des1 = detector.detectAndCompute(pan_gray, None)
    kp2, des2 = detector.detectAndCompute(mss_gray, None)
    if des1 is None or des2 is None or len(kp1) < FEATURE_MIN_MATCH or len(kp2) < FEATURE_MIN_MATCH:
        return None

    norm_flag = cv2.NORM_L2 if des1.dtype == np.float32 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm_flag, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < FEATURE_RATIO * n.distance:
            good.append(m)
    if len(good) < FEATURE_MIN_MATCH:
        return None

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    return H


def process_label_file(label_path: str, pic_root: str, out_root: str, args) -> None:
    info = resolve_scene_files(label_path, pic_root)
    scene_out = os.path.join(out_root, info["scene"])
    os.makedirs(scene_out, exist_ok=True)
    print(
        f"[files] MSS={os.path.basename(info['mss'])} ({'RPC' if info['mss_rpb'] else 'no RPC'}) | "
        f"PAN={os.path.basename(info['pan'])} ({'RPC' if info['pan_rpb'] else 'no RPC'})"
    )

    detections = read_labels(label_path)
    if not detections:
        print(f"[skip] {os.path.basename(label_path)} 没有检测框")
        return

    mss_rpc = parse_rpb_file(info["mss_rpb"]) if info.get("mss_rpb") else None
    pan_rpc = parse_rpb_file(info["pan_rpb"]) if info.get("pan_rpb") else None

    with rasterio.open(info["mss"]) as ms_ds, rasterio.open(info["pan"]) as pan_ds:
        if ms_ds.crs and pan_ds.crs and ms_ds.crs != pan_ds.crs and not (mss_rpc and pan_rpc):
            raise ValueError(f"CRS mismatch and no RPC: MS={ms_ds.crs}, PAN={pan_ds.crs}")
        print(f"[scene] {info['scene']} | MS {ms_ds.width}x{ms_ds.height} | PAN {pan_ds.width}x{pan_ds.height} | {len(detections)} dets")

        ms_w, ms_h = ms_ds.width, ms_ds.height
        pan_W, pan_H = pan_ds.width, pan_ds.height
        h_avg = pan_rpc.get("heightOffset", 0) if pan_rpc else 0
        num_bands = min(ms_ds.count, 4)

        # ====================================================================
        # 逐个检测框处理，避免内存溢出
        # ====================================================================
        for det_idx, det in enumerate(detections[: args.max_detections]):
            try:
                # 1) MSS 归一化坐标 -> MSS 像素坐标 -> PAN 像素坐标
                poly_ms_px = [(x * ms_w, y * ms_h) for x, y in det["poly_norm"]]
                poly_ms_px_expanded = expand_polygon(poly_ms_px, args.scale)

                # 坐标转换：MSS -> PAN
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

                # 2) 计算对齐窗口（自适应大小，以检测框为中心）
                pan_col_off, pan_row_off, pan_width, pan_height = compute_align_window(
                    pan_poly, pan_W, pan_H, args.scene_expand, min_window_size=1000
                )

                if pan_width <= 100 or pan_height <= 100:
                    print(f"[{det_idx}] PAN 窗口过小，跳过")
                    continue

                # 3) 移动窗口使其完全在图像内（保持窗口大小不变）
                # 确保窗口尺寸不超过图像尺寸
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

                # 4) 读取 PAN 窗口
                pan_window = Window(pan_col_off, pan_row_off, pan_width, pan_height)
                pan_patch = pan_ds.read(1, window=pan_window)
                actual_pan_h, actual_pan_w = pan_patch.shape

                # 5) 生成控制点并计算 MSS 窗口
                buffer = 60
                control_points_pan = generate_control_points(pan_col_off, pan_row_off, pan_width, pan_height)
                if pan_rpc and mss_rpc:
                    mss_control_pix = transform_points_rpc(control_points_pan, pan_rpc, mss_rpc, h_avg, iterations=20)
                else:
                    pan_transform = pan_ds.transform
                    ms_transform = ms_ds.transform
                    ground_control = [pan_transform * (cpx, cpy) for cpx, cpy in control_points_pan]
                    mss_control_pix = np.float32([~ms_transform * (gx, gy) for gx, gy in ground_control])

                mss_x_min, mss_x_max = np.min(mss_control_pix[:, 0]), np.max(mss_control_pix[:, 0])
                mss_y_min, mss_y_max = np.min(mss_control_pix[:, 1]), np.max(mss_control_pix[:, 1])

                # 计算理想的 MSS 窗口（不考虑边界）
                ideal_mss_col_off = int(np.floor(mss_x_min)) - buffer
                ideal_mss_row_off = int(np.floor(mss_y_min)) - buffer
                ideal_mss_col_end = int(np.ceil(mss_x_max)) + buffer
                ideal_mss_row_end = int(np.ceil(mss_y_max)) + buffer

                # 实际读取的 MSS 窗口（受图像边界限制）
                mss_col_off = max(0, ideal_mss_col_off)
                mss_row_off = max(0, ideal_mss_row_off)
                mss_col_end = min(ms_w, ideal_mss_col_end)
                mss_row_end = min(ms_h, ideal_mss_row_end)

                mss_width = mss_col_end - mss_col_off
                mss_height = mss_row_end - mss_row_off

                if mss_width <= 0 or mss_height <= 0:
                    print(f"[{det_idx}] MSS 窗口无效，跳过")
                    continue

                # 6) 读取 MSS 窗口
                mss_read_window = Window(mss_col_off, mss_row_off, mss_width, mss_height)
                mss_tiles = [ms_ds.read(b + 1, window=mss_read_window) for b in range(num_bands)]

                if any(tile.size == 0 for tile in mss_tiles):
                    print(f"[{det_idx}] MSS 数据为空，跳过")
                    continue

                # 7) Warp MSS 到 PAN 窗口
                # dst_control: 相对于 PAN 窗口的局部坐标
                dst_control = np.float32([
                    [cpx - pan_col_off, cpy - pan_row_off]
                    for cpx, cpy in control_points_pan
                ])
                # src_control: 相对于实际读取的 MSS 窗口的局部坐标
                # 注意：控制点可能落在 MSS 窗口外（负数或超出宽高），这是正常的
                src_control = mss_control_pix - np.float32([mss_col_off, mss_row_off])
                output_size = (actual_pan_w, actual_pan_h)

                mss_aligned, valid_mask = warp_mss_to_pan(mss_tiles, src_control, dst_control, output_size)

                # 8) 可选特征细配准
                if not args.no_refine and ENABLE_FEATURE_REFINE:
                    aligned_preview = bands_to_uint8(
                        [mss_aligned[i] for i in range(min(mss_aligned.shape[0], 4))],
                        args.preview_method
                    )
                    # 先用相位相关做小范围平移修正，再尝试特征细配准
                    dx, dy = estimate_translation(single_band_to_uint8(pan_patch), aligned_preview, max_shift=5)
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
                        shifted = [cv2.warpAffine(
                            band, M_shift, output_size,
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        ) for band in mss_aligned]
                        mss_aligned = np.stack(shifted, axis=0)
                        # 同步更新 mask
                        valid_mask = cv2.warpAffine(
                            valid_mask.astype(np.float32), M_shift, output_size,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        valid_mask = (valid_mask > 0.5).astype(np.uint8)
                        aligned_preview = bands_to_uint8(
                            [mss_aligned[i] for i in range(min(mss_aligned.shape[0], 4))],
                            args.preview_method
                        )

                    refine_H = find_feature_refine(single_band_to_uint8(pan_patch), aligned_preview)
                    if refine_H is not None:
                        refined = []
                        for band in mss_aligned:
                            refined.append(cv2.warpPerspective(
                                band, refine_H, output_size,
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0,
                            ))
                        mss_aligned = np.stack(refined, axis=0)
                        # 同步更新 mask
                        valid_mask = cv2.warpPerspective(
                            valid_mask.astype(np.float32), refine_H, output_size,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        valid_mask = (valid_mask > 0.5).astype(np.uint8)

                # 9) 计算有效重叠区域的边界框
                # 先找检测框区域
                rel_poly = [(px - pan_col_off, py - pan_row_off) for px, py in pan_poly]
                xs = [p[0] for p in rel_poly]
                ys = [p[1] for p in rel_poly]

                det_xmin = int(clamp(min(xs), 0, actual_pan_w - 1))
                det_xmax = int(clamp(max(xs), 0, actual_pan_w - 1))
                det_ymin = int(clamp(min(ys), 0, actual_pan_h - 1))
                det_ymax = int(clamp(max(ys), 0, actual_pan_h - 1))

                if det_xmax - det_xmin < 2 or det_ymax - det_ymin < 2:
                    print(f"[{det_idx}] 检测框区域过小，跳过")
                    continue

                # 在检测框区域内找有效 mask 的边界
                crop_mask = valid_mask[det_ymin:det_ymax, det_xmin:det_xmax]
                valid_rows = np.any(crop_mask > 0, axis=1)
                valid_cols = np.any(crop_mask > 0, axis=0)

                if not np.any(valid_rows) or not np.any(valid_cols):
                    print(f"[{det_idx}] 无有效重叠区域，跳过")
                    continue

                row_indices = np.where(valid_rows)[0]
                col_indices = np.where(valid_cols)[0]
                local_ymin, local_ymax = int(row_indices[0]), int(row_indices[-1] + 1)
                local_xmin, local_xmax = int(col_indices[0]), int(col_indices[-1] + 1)

                # 转换回全局坐标
                xmin = det_xmin + local_xmin
                xmax = det_xmin + local_xmax
                ymin = det_ymin + local_ymin
                ymax = det_ymin + local_ymax

                if xmax - xmin < 2 or ymax - ymin < 2:
                    print(f"[{det_idx}] 有效重叠区域过小，跳过")
                    continue

                # 调整为正方形，大小是 crop_multiple 的倍数
                crop_multiple = args.crop_multiple
                xmin, xmax, ymin, ymax = compute_square_crop(
                    xmin, xmax, ymin, ymax,
                    max_w=actual_pan_w, max_h=actual_pan_h,
                    multiple=crop_multiple, min_size=crop_multiple
                )

                # 再次检查尺寸有效性
                if xmax - xmin < crop_multiple or ymax - ymin < crop_multiple:
                    print(f"[{det_idx}] 正方形区域不足 {crop_multiple}，跳过")
                    continue

                pan_crop = pan_patch[ymin:ymax, xmin:xmax]
                ms_crop = mss_aligned[:, ymin:ymax, xmin:xmax]

                # 11) 保存裁剪结果
                # save_image(bands_to_uint8(list(ms_crop), args.preview_method),
                #            os.path.join(scene_out, f"det_{det_idx:03d}_ms_crop.png"))
                # save_image(single_band_to_uint8(pan_crop),
                #            os.path.join(scene_out, f"det_{det_idx:03d}_pan_crop.png"))

                # if args.save_raw or SAVE_RAW_16BIT:
                #     save_raw_16bit(ms_crop, os.path.join(scene_out, f"det_{det_idx:03d}_ms_crop_raw.png"))
                #     save_raw_16bit(pan_crop, os.path.join(scene_out, f"det_{det_idx:03d}_pan_crop_raw.png"))

                # 12) 融合
                fused_bands = fuse_pan_mss(pan_crop, ms_crop, args.method)
                fused_uint8 = bands_to_uint8(fused_bands, args.preview_method)
                save_image(fused_uint8, os.path.join(scene_out, f"det_{det_idx:03d}_fused.png"))

                if args.save_raw or SAVE_RAW_16BIT:
                    save_raw_16bit(np.stack(fused_bands, axis=0),
                                   os.path.join(scene_out, f"det_{det_idx:03d}_fused_raw.png"))

                # 13) 保存偏移元数据（用于坐标转换）
                crop_metadata = {
                    "det_idx": det_idx,
                    "global_offset_x": pan_col_off + xmin,
                    "global_offset_y": pan_row_off + ymin,
                    "crop_width": xmax - xmin,
                    "crop_height": ymax - ymin,
                    "pan_size": [pan_W, pan_H],
                    "mss_size": [ms_w, ms_h],
                    "class_id": det["cls"],
                    "score": det["score"],
                    "original_poly_norm": det["poly_norm"],
                }
                import json
                metadata_path = os.path.join(scene_out, f"det_{det_idx:03d}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as mf:
                    json.dump(crop_metadata, mf, ensure_ascii=False, indent=2)

                # 输出信息
                det_size = max(max(xs) - min(xs), max(ys) - min(ys))
                crop_size = xmax - xmin  # 正方形，宽高相同
                print(f"[{det_idx}] done | det_size: {det_size:.0f}px | align_window: {pan_width}x{pan_height} | crop: {crop_size}x{crop_size} | offset: ({pan_col_off + xmin}, {pan_row_off + ymin})")

            except Exception as exc:
                print(f"[{det_idx}] error: {exc}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    label_dir = args.label_dir if os.path.isabs(args.label_dir) else os.path.join(os.getcwd(), args.label_dir)
    pic_root = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(os.getcwd(), args.input_dir)
    label_files = []
    if os.path.isdir(label_dir):
        label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(".txt")]
    else:
        # fallback to single-scene paths
        label_files = [args.labels if os.path.isabs(args.labels) else os.path.join(args.input_dir, args.labels)]

    if not label_files:
        print("No label files found.")
        return

    for lf in label_files:
        try:
            process_label_file(lf, pic_root, args.output_dir, args)
        except Exception as exc:
            print(f"[error] {os.path.basename(lf)} -> {exc}")

    print(f"全部完成，输出路径: {args.output_dir}")


if __name__ == "__main__":
    main()
