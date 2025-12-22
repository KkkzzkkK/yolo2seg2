# -*- coding: utf-8 -*-
"""
Crop, register, and pan-sharpen patches around YOLO-OBB detections.

Inputs (per run):
- 一个包含 1 组多光谱 TIFF 和全色 TIFF 的文件夹（可带 .rpb，若有则使用 RPC 精配准；.xml/.jpg 仅作预览不参与处理）
- 对应多光谱的 YOLO 多边形标签（class x1 y1 x2 y2 x3 y3 x4 y4 [score]，归一化到多光谱尺寸）

Per detection 输出:
- 融合 PNG（Gram-Schmidt，自动权重）
"""

import json
import math
import os
from typing import List, Tuple

import cv2
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
    gram_schmidt_pan_sharpening,
    calculate_band_correlations,
)

# ============================================================================
# 用户配置区（直接修改下面的默认值即可运行）
# ============================================================================
INPUT_DIR = r"F:\code\pic\pt"              # 存放影像的根目录（每个影像一个子文件夹）
LABEL_DIR = r"F:\1218\labels_export"    # YOLO 标签文件夹（每个影像一个 txt）
OUTPUT_DIR = r"F:\1218\yolo_pic2"

BOX_SCALE = 1.3                         # 放大检测框，形成更大矩形
MAX_DETECTIONS = 500                    # 保护性上限
SCENE_EXPAND = 3.0                      # 计算场景大窗时，对所有检测外包框的放大倍数
CROP_MULTIPLE = 64                      # 裁剪尺寸的倍数（256, 512 等正方形）

# 特征配准参数
ENABLE_FEATURE_REFINE = True            # ORB/SIFT 特征微调
FEATURE_MAX = 1500                      # 增加特征点数量
FEATURE_MIN_MATCH = 12                  # 降低最小匹配要求
FEATURE_RATIO = 0.78                    # 稍微放宽匹配比例

# 融合参数（固定使用 Gram-Schmidt）
SHARPEN_SAMPLE_RATIO = 0.2              # 计算GS权重的采样率
PREVIEW_METHOD = "clahe"                # CLAHE 独立通道增强到 PNG
SAVE_RAW_16BIT = False                  # 如需保存未增强的16bit裁剪，设为 True

# 控制点网格参数（增加控制点数量）
CONTROL_GRID_SIZE_SMALL = 5             # 小窗口 (<2000) 使用 5x5=25 点
CONTROL_GRID_SIZE_MEDIUM = 7            # 中窗口 (2000-5000) 使用 7x7=49 点
CONTROL_GRID_SIZE_LARGE = 9             # 大窗口 (>5000) 使用 9x9=81 点


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
                        min_size: int = 256,
                        allow_padding: bool = True) -> Tuple[int, int, int, int, Tuple[int, int, int, int]]:
    """
    将矩形区域扩展为正方形，大小是 multiple 的倍数。
    允许超出边界，返回 padding 信息。

    参数:
        xmin, xmax, ymin, ymax: 原始区域边界
        max_w, max_h: 可用区域的最大宽高
        multiple: 目标尺寸的倍数（默认 256）
        min_size: 最小尺寸
        allow_padding: 是否允许超出边界（通过填充处理）

    返回: (new_xmin, new_xmax, new_ymin, new_ymax, (pad_left, pad_right, pad_top, pad_bottom))
    """
    # 计算当前区域的宽高和中心
    w = xmax - xmin
    h = ymax - ymin
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    # 取最大边长，向上取整到 multiple 的倍数
    side = max(w, h, min_size)
    side = round_up_to_multiple(side, multiple)

    half = side / 2

    # 以中心为基准计算新边界（允许为负或超出）
    new_xmin = int(cx - half)
    new_xmax = int(cx + half)
    new_ymin = int(cy - half)
    new_ymax = int(cy + half)

    if allow_padding:
        # 计算需要填充的量
        pad_left = max(0, -new_xmin)
        pad_top = max(0, -new_ymin)
        pad_right = max(0, new_xmax - max_w)
        pad_bottom = max(0, new_ymax - max_h)

        # 调整边界到有效范围
        actual_xmin = max(0, new_xmin)
        actual_xmax = min(max_w, new_xmax)
        actual_ymin = max(0, new_ymin)
        actual_ymax = min(max_h, new_ymax)

        return actual_xmin, actual_xmax, actual_ymin, actual_ymax, (pad_left, pad_right, pad_top, pad_bottom)
    else:
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

        return new_xmin, new_xmax, new_ymin, new_ymax, (0, 0, 0, 0)


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
           col_off/row_off 可能为负数（表示窗口超出图像边界，需要后续填充处理）
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
        effective_expand = max(scene_expand, 10.0)
    elif det_size < 500:
        effective_expand = max(scene_expand, 5.0)
    elif det_size < 1000:
        effective_expand = max(scene_expand, 3.0)
    else:
        effective_expand = scene_expand

    # 计算窗口半宽半高
    pan_half_w = det_w / 2 * effective_expand
    pan_half_h = det_h / 2 * effective_expand

    # 确保窗口有最小尺寸
    pan_half_w = max(pan_half_w, min_window_size / 2)
    pan_half_h = max(pan_half_h, min_window_size / 2)

    # 以检测框中心为基准计算窗口（允许超出边界）
    col_off = int(math.floor(pan_cx - pan_half_w))
    row_off = int(math.floor(pan_cy - pan_half_h))
    col_end = int(math.ceil(pan_cx + pan_half_w))
    row_end = int(math.ceil(pan_cy + pan_half_h))

    win_w = col_end - col_off
    win_h = row_end - row_off

    return col_off, row_off, win_w, win_h


def estimate_translation(pan_uint8: np.ndarray,
                         mss_uint8: np.ndarray,
                         max_shift: int = 8) -> Tuple[float, float]:
    """
    使用相位相关估计小范围平移偏差。
    增大 max_shift 允许更大的滑动范围。

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
    dx, dy = shift
    if abs(dx) > max_shift or abs(dy) > max_shift or not np.isfinite(dx) or not np.isfinite(dy):
        return 0.0, 0.0
    return dx, dy


def generate_control_points(col_off: int, row_off: int,
                            width: int, height: int,
                            grid_n: int = None) -> List[Tuple[float, float]]:
    """
    在窗口内均匀生成控制点网格。
    使用更多控制点以获得更精确的配准。

    参数:
        col_off, row_off: 窗口左上角坐标
        width, height: 窗口尺寸
        grid_n: 网格大小，None 则自动根据窗口大小选择

    返回: 控制点列表 [(x, y), ...]
    """
    if grid_n is None:
        max_dim = max(width, height)
        if max_dim > 5000:
            grid_n = CONTROL_GRID_SIZE_LARGE    # 9x9=81 点
        elif max_dim > 2000:
            grid_n = CONTROL_GRID_SIZE_MEDIUM   # 7x7=49 点
        else:
            grid_n = CONTROL_GRID_SIZE_SMALL    # 5x5=25 点

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


def compute_border_value(bands: List[np.ndarray], percentile: float = 10) -> List[float]:
    """
    计算每个波段的边界填充值。
    使用较低的百分位数，避免使用异常高值或零值填充边界。
    
    参数:
        bands: 波段数据列表
        percentile: 使用的百分位数（默认10，避免极端值）
    
    返回: 每个波段的填充值列表
    """
    border_values = []
    for b in bands:
        # 获取有效像素（非零且非 NaN）
        valid_mask = (b > 0) & np.isfinite(b)
        if np.any(valid_mask):
            valid_pixels = b[valid_mask]
            # 使用较低的百分位数作为填充值，避免过亮的边界
            fill_val = float(np.percentile(valid_pixels, percentile))
        else:
            fill_val = 0.0
        border_values.append(fill_val)
    return border_values


def warp_mss_to_pan(mss_tiles: List[np.ndarray],
                    src_control: np.ndarray,
                    dst_control: np.ndarray,
                    output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用控制点将 MSS 波段 warp 到 PAN 窗口大小。
    使用 Lanczos4 插值，边界填充使用低百分位值避免异常颜色。

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

    # 检查变换矩阵是否合理（避免极端变形）
    # 检查矩阵的行列式，避免过度缩放或翻转
    det = np.linalg.det(M[:2, :2])
    if abs(det) < 0.01 or abs(det) > 100:
        print(f"[warn] 变换矩阵行列式异常: {det:.4f}，可能导致图像变形")

    # 计算每个波段的填充值（使用低百分位避免异常值）
    border_values = compute_border_value(mss_tiles, percentile=10)

    # Warp 每个波段（使用 Lanczos4 插值）
    mss_aligned = []
    for band, border_val in zip(mss_tiles, border_values):
        warped = cv2.warpPerspective(
            band.astype(np.float32),
            M,
            output_size,
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        mss_aligned.append(warped)

    # 生成有效区域 mask
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

    # 额外检查：确保无效区域使用合理的填充值
    mss_stack = np.stack(mss_aligned, axis=0)
    invalid_mask = (valid_mask == 0)
    if np.any(invalid_mask):
        for i in range(mss_stack.shape[0]):
            mss_stack[i][invalid_mask] = border_values[i]

    return mss_stack, valid_mask


def fuse_pan_mss(pan_crop: np.ndarray,
                 ms_crop: np.ndarray,
                 valid_mask: np.ndarray = None) -> List[np.ndarray]:
    """
    使用 Gram-Schmidt 方法融合 PAN 和 MSS 裁剪。
    对于无效区域（对齐边界外），保留原始 MSS 值避免彩虹色伪影。

    参数:
        pan_crop: PAN 裁剪 (H, W)
        ms_crop: MSS 裁剪 (bands, H, W)
        valid_mask: 有效区域掩码 (H, W)，可选。1=有效，0=无效

    返回: 融合后的波段列表
    """
    ms_band_list = [ms_crop[i] for i in range(min(ms_crop.shape[0], 4))]

    # 如果有有效掩码，只使用有效区域计算相关性权重
    weights = None
    if len(ms_band_list) >= 3:
        if valid_mask is not None and np.sum(valid_mask) > 100:
            # 使用有效区域计算权重
            pan_valid = pan_crop.copy()
            ms_valid = [b.copy() for b in ms_band_list]
            # 将无效区域设为 NaN 以排除
            invalid = (valid_mask == 0)
            pan_valid[invalid] = np.nan
            for b in ms_valid:
                b[invalid] = np.nan
            try:
                _, weights = calculate_band_correlations(pan_valid, ms_valid, sample_ratio=SHARPEN_SAMPLE_RATIO)
            except Exception:
                weights = None
        else:
            _, weights = calculate_band_correlations(pan_crop, ms_band_list, sample_ratio=SHARPEN_SAMPLE_RATIO)

    fused = gram_schmidt_pan_sharpening(pan_crop, ms_band_list, weights=weights)

    # 对于无效区域，用原始 MSS 值替换融合结果，避免边界伪影
    if valid_mask is not None:
        invalid = (valid_mask == 0)
        if np.any(invalid):
            for i, band in enumerate(fused):
                if i < len(ms_band_list):
                    band[invalid] = ms_band_list[i][invalid]

    return fused


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


def read_with_padding(ds, band_idx: int, window: Window,
                      img_w: int, img_h: int,
                      fill_value: float = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    读取数据，支持超出边界的窗口（填充0）。

    参数:
        ds: rasterio dataset
        band_idx: 波段索引（1-based）
        window: 读取窗口（可能超出边界）
        img_w, img_h: 图像尺寸
        fill_value: 填充值

    返回: (数据数组, (pad_left, pad_right, pad_top, pad_bottom))
    """
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    # 计算有效读取范围
    read_col_off = max(0, col_off)
    read_row_off = max(0, row_off)
    read_col_end = min(img_w, col_off + width)
    read_row_end = min(img_h, row_off + height)

    # 计算 padding
    pad_left = max(0, -col_off)
    pad_top = max(0, -row_off)
    pad_right = max(0, (col_off + width) - img_w)
    pad_bottom = max(0, (row_off + height) - img_h)

    # 读取有效范围
    read_width = read_col_end - read_col_off
    read_height = read_row_end - read_row_off

    if read_width <= 0 or read_height <= 0:
        # 完全超出边界，返回全填充
        result = np.full((height, width), fill_value, dtype=np.float32)
        return result, (pad_left, pad_right, pad_top, pad_bottom)

    read_window = Window(read_col_off, read_row_off, read_width, read_height)
    data = ds.read(band_idx, window=read_window).astype(np.float32)

    # 创建带 padding 的结果
    result = np.full((height, width), fill_value, dtype=np.float32)
    result[pad_top:pad_top + read_height, pad_left:pad_left + read_width] = data

    return result, (pad_left, pad_right, pad_top, pad_bottom)


def process_label_file(label_path: str, pic_root: str, out_root: str) -> None:
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
        # 逐个检测框处理
        # ====================================================================
        for det_idx, det in enumerate(detections[:MAX_DETECTIONS]):
            try:
                # 1) MSS 归一化坐标 -> MSS 像素坐标 -> PAN 像素坐标
                poly_ms_px = [(x * ms_w, y * ms_h) for x, y in det["poly_norm"]]
                poly_ms_px_expanded = expand_polygon(poly_ms_px, BOX_SCALE)

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

                # 2) 计算对齐窗口（自适应大小，以检测框为中心，允许超出边界）
                pan_col_off, pan_row_off, pan_width, pan_height = compute_align_window(
                    pan_poly, pan_W, pan_H, SCENE_EXPAND, min_window_size=1000
                )

                if pan_width <= 100 or pan_height <= 100:
                    print(f"[{det_idx}] PAN 窗口过小，跳过")
                    continue

                # 3) 读取 PAN 窗口（支持超出边界填充0）
                pan_window = Window(pan_col_off, pan_row_off, pan_width, pan_height)
                pan_patch, pan_padding = read_with_padding(
                    pan_ds, 1, pan_window, pan_W, pan_H, fill_value=0
                )
                actual_pan_h, actual_pan_w = pan_patch.shape

                # 4) 生成控制点并计算 MSS 窗口
                buffer = 80  # 增加 buffer
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

                # 计算 MSS 窗口（允许超出边界）
                mss_col_off = int(np.floor(mss_x_min)) - buffer
                mss_row_off = int(np.floor(mss_y_min)) - buffer
                mss_col_end = int(np.ceil(mss_x_max)) + buffer
                mss_row_end = int(np.ceil(mss_y_max)) + buffer

                mss_width = mss_col_end - mss_col_off
                mss_height = mss_row_end - mss_row_off

                if mss_width <= 0 or mss_height <= 0:
                    print(f"[{det_idx}] MSS 窗口无效，跳过")
                    continue

                # 5) 读取 MSS 窗口（支持超出边界填充均值）
                mss_tiles = []
                for b in range(num_bands):
                    mss_win = Window(mss_col_off, mss_row_off, mss_width, mss_height)
                    # 先读取一小块计算均值
                    sample_col = max(0, min(mss_col_off, ms_w - 100))
                    sample_row = max(0, min(mss_row_off, ms_h - 100))
                    sample_w = min(100, ms_w - sample_col)
                    sample_h = min(100, ms_h - sample_row)
                    if sample_w > 0 and sample_h > 0:
                        sample = ms_ds.read(b + 1, window=Window(sample_col, sample_row, sample_w, sample_h))
                        fill_val = float(np.mean(sample[sample > 0])) if np.any(sample > 0) else 0
                    else:
                        fill_val = 0
                    tile, _ = read_with_padding(ms_ds, b + 1, mss_win, ms_w, ms_h, fill_value=fill_val)
                    mss_tiles.append(tile)

                if any(tile.size == 0 for tile in mss_tiles):
                    print(f"[{det_idx}] MSS 数据为空，跳过")
                    continue

                # 6) Warp MSS 到 PAN 窗口（使用 Lanczos4，边界填充均值）
                dst_control = np.float32([
                    [cpx - pan_col_off, cpy - pan_row_off]
                    for cpx, cpy in control_points_pan
                ])
                src_control = mss_control_pix - np.float32([mss_col_off, mss_row_off])
                output_size = (actual_pan_w, actual_pan_h)

                mss_aligned, valid_mask = warp_mss_to_pan(mss_tiles, src_control, dst_control, output_size)

                # 7) 特征细配准（启用滑动移动）
                if ENABLE_FEATURE_REFINE:
                    aligned_preview = bands_to_uint8(
                        [mss_aligned[i] for i in range(min(mss_aligned.shape[0], 4))],
                        PREVIEW_METHOD
                    )
                    # 相位相关做小范围平移修正（增大移动范围）
                    dx, dy = estimate_translation(single_band_to_uint8(pan_patch), aligned_preview, max_shift=8)
                    if abs(dx) > 0.1 or abs(dy) > 0.1:
                        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
                        # 使用 Lanczos4 插值进行滑动
                        border_vals = compute_border_value([mss_aligned[i] for i in range(mss_aligned.shape[0])])
                        shifted = []
                        for i, band in enumerate(mss_aligned):
                            shifted.append(cv2.warpAffine(
                                band, M_shift, output_size,
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_vals[i],
                            ))
                        mss_aligned = np.stack(shifted, axis=0)
                        valid_mask = cv2.warpAffine(
                            valid_mask.astype(np.float32), M_shift, output_size,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        valid_mask = (valid_mask > 0.5).astype(np.uint8)
                        aligned_preview = bands_to_uint8(
                            [mss_aligned[i] for i in range(min(mss_aligned.shape[0], 4))],
                            PREVIEW_METHOD
                        )

                    # SIFT/ORB 特征细配准
                    refine_H = find_feature_refine(single_band_to_uint8(pan_patch), aligned_preview)
                    if refine_H is not None:
                        border_vals = compute_border_value([mss_aligned[i] for i in range(mss_aligned.shape[0])])
                        refined = []
                        for i, band in enumerate(mss_aligned):
                            refined.append(cv2.warpPerspective(
                                band, refine_H, output_size,
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=border_vals[i],
                            ))
                        mss_aligned = np.stack(refined, axis=0)
                        valid_mask = cv2.warpPerspective(
                            valid_mask.astype(np.float32), refine_H, output_size,
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        valid_mask = (valid_mask > 0.5).astype(np.uint8)

                # 8) 计算裁剪区域（基于检测框，允许填充0）
                rel_poly = [(px - pan_col_off, py - pan_row_off) for px, py in pan_poly]
                xs = [p[0] for p in rel_poly]
                ys = [p[1] for p in rel_poly]

                det_xmin = int(min(xs))
                det_xmax = int(max(xs))
                det_ymin = int(min(ys))
                det_ymax = int(max(ys))

                if det_xmax - det_xmin < 2 or det_ymax - det_ymin < 2:
                    print(f"[{det_idx}] 检测框区域过小，跳过")
                    continue

                # 调整为正方形，允许填充
                xmin, xmax, ymin, ymax, padding = compute_square_crop(
                    det_xmin, det_xmax, det_ymin, det_ymax,
                    max_w=actual_pan_w, max_h=actual_pan_h,
                    multiple=CROP_MULTIPLE, min_size=CROP_MULTIPLE,
                    allow_padding=True
                )
                pad_left, pad_right, pad_top, pad_bottom = padding

                # 裁剪数据（包括有效掩码）
                pan_crop = pan_patch[ymin:ymax, xmin:xmax]
                ms_crop = mss_aligned[:, ymin:ymax, xmin:xmax]
                mask_crop = valid_mask[ymin:ymax, xmin:xmax]

                # 如果需要填充（边缘情况）
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    target_h = ymax - ymin + pad_top + pad_bottom
                    target_w = xmax - xmin + pad_left + pad_right

                    # PAN 填充 0
                    pan_padded = np.zeros((target_h, target_w), dtype=pan_crop.dtype)
                    pan_padded[pad_top:pad_top + pan_crop.shape[0],
                               pad_left:pad_left + pan_crop.shape[1]] = pan_crop
                    pan_crop = pan_padded

                    # MSS 填充均值
                    ms_border_vals = compute_border_value([ms_crop[i] for i in range(ms_crop.shape[0])])
                    ms_padded = np.zeros((ms_crop.shape[0], target_h, target_w), dtype=ms_crop.dtype)
                    for i in range(ms_crop.shape[0]):
                        ms_padded[i] = ms_border_vals[i]
                        ms_padded[i, pad_top:pad_top + ms_crop.shape[1],
                                  pad_left:pad_left + ms_crop.shape[2]] = ms_crop[i]
                    ms_crop = ms_padded

                    # 有效掩码填充 0（标记填充区域为无效）
                    mask_padded = np.zeros((target_h, target_w), dtype=mask_crop.dtype)
                    mask_padded[pad_top:pad_top + mask_crop.shape[0],
                                pad_left:pad_left + mask_crop.shape[1]] = mask_crop
                    mask_crop = mask_padded

                # 9) 融合（固定 Gram-Schmidt，传递有效掩码）
                fused_bands = fuse_pan_mss(pan_crop, ms_crop, valid_mask=mask_crop)
                fused_uint8 = bands_to_uint8(fused_bands, PREVIEW_METHOD)
                save_image(fused_uint8, os.path.join(scene_out, f"det_{det_idx:03d}_fused.png"))

                if SAVE_RAW_16BIT:
                    save_raw_16bit(np.stack(fused_bands, axis=0),
                                   os.path.join(scene_out, f"det_{det_idx:03d}_fused_raw.png"))

                # 10) 保存偏移元数据
                crop_metadata = {
                    "det_idx": det_idx,
                    "global_offset_x": pan_col_off + xmin - pad_left,
                    "global_offset_y": pan_row_off + ymin - pad_top,
                    "crop_width": pan_crop.shape[1],
                    "crop_height": pan_crop.shape[0],
                    "padding": {"left": pad_left, "right": pad_right, "top": pad_top, "bottom": pad_bottom},
                    "pan_size": [pan_W, pan_H],
                    "mss_size": [ms_w, ms_h],
                    "class_id": det["cls"],
                    "score": det["score"],
                    "original_poly_norm": det["poly_norm"],
                }
                metadata_path = os.path.join(scene_out, f"det_{det_idx:03d}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as mf:
                    json.dump(crop_metadata, mf, ensure_ascii=False, indent=2)

                det_size = max(max(xs) - min(xs), max(ys) - min(ys))
                crop_size = pan_crop.shape[0]
                print(f"[{det_idx}] done | det_size: {det_size:.0f}px | align_window: {pan_width}x{pan_height} | crop: {crop_size}x{crop_size} | padding: {padding}")

            except Exception as exc:
                import traceback
                print(f"[{det_idx}] error: {exc}")
                traceback.print_exc()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_dir = LABEL_DIR if os.path.isabs(LABEL_DIR) else os.path.join(os.getcwd(), LABEL_DIR)
    pic_root = INPUT_DIR if os.path.isabs(INPUT_DIR) else os.path.join(os.getcwd(), INPUT_DIR)
    label_files = []
    if os.path.isdir(label_dir):
        label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith(".txt")]
    else:
        print(f"Label directory not found: {label_dir}")
        return

    if not label_files:
        print("No label files found.")
        return

    for lf in label_files:
        try:
            process_label_file(lf, pic_root, OUTPUT_DIR)
        except Exception as exc:
            print(f"[error] {os.path.basename(lf)} -> {exc}")

    print(f"全部完成，输出路径: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
