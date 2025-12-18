# -*- coding: utf-8 -*-
"""
全色锐化脚本 - 保留RPB信息
从文件夹中读取多光谱和全色影像，进行全色锐化，输出带RPB信息的影像
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from image_utils import (
    parse_rpb_file,
    gram_schmidt_pan_sharpening,
    brovey_pan_sharpening,
    ihs_pan_sharpening,
    simple_mean_pan_sharpening,
    calculate_band_correlations,
)


def rpb_rpc_to_gdal_rpc_tags(rpc_params):
    """将 parse_rpb_file 解析出的 RPB 参数转换为 GeoTIFF/GDAL 风格的 RPC tags。

    返回的 dict 可用于 rasterio: dst.update_tags(ns='RPC', **tags)
    """
    if not rpc_params:
        return {}

    def _get_float(key):
        value = rpc_params.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def _coeffs_to_str(key):
        coeffs = rpc_params.get(key)
        if coeffs is None:
            return None
        arr = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if arr.size < 20:
            arr = np.pad(arr, (0, 20 - arr.size), mode="constant", constant_values=0)
        elif arr.size > 20:
            arr = arr[:20]
        return " ".join(f"{v:.16g}" for v in arr.tolist())

    tags = {}

    mapping = {
        "LINE_OFF": "lineOffset",
        "SAMP_OFF": "sampOffset",
        "LAT_OFF": "latOffset",
        "LONG_OFF": "longOffset",
        "HEIGHT_OFF": "heightOffset",
        "LINE_SCALE": "lineScale",
        "SAMP_SCALE": "sampScale",
        "LAT_SCALE": "latScale",
        "LONG_SCALE": "longScale",
        "HEIGHT_SCALE": "heightScale",
    }

    for out_key, in_key in mapping.items():
        val = _get_float(in_key)
        if val is not None:
            tags[out_key] = f"{val:.16g}"

    coeff_map = {
        "LINE_NUM_COEFF": "lineNumCoef",
        "LINE_DEN_COEFF": "lineDenCoef",
        "SAMP_NUM_COEFF": "sampNumCoef",
        "SAMP_DEN_COEFF": "sampDenCoef",
    }
    for out_key, in_key in coeff_map.items():
        coeff_str = _coeffs_to_str(in_key)
        if coeff_str:
            tags[out_key] = coeff_str

    return tags


def _resampling_from_name(name: str) -> Resampling:
    name = (name or "").lower()
    if name == "nearest":
        return Resampling.nearest
    if name == "bilinear":
        return Resampling.bilinear
    return Resampling.cubic


def _choose_stats_shape(width: int, height: int, max_dim: int) -> tuple[int, int]:
    if not max_dim or max_dim <= 0:
        return height, width
    scale = min(1.0, float(max_dim) / float(max(width, height)))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    return out_h, out_w


def _safe_tiff_block_size(dim: int, preferred: int = 256) -> int:
    """GeoTIFF tiled blocks must be multiples of 16 (common GDAL constraint).

    Returns a value in [16, dim] and divisible by 16.
    """
    if dim <= 0:
        return 256
    if dim < 16:
        return dim
    bs = min(int(preferred), int(dim))
    bs = bs - (bs % 16)
    return max(16, bs)


def _compute_weights_streaming(
    pan_ds,
    mss_ds,
    num_bands: int,
    tile_size: int,
    resampling: Resampling,
    sample_ratio: float,
):
    """通过分块+抽样计算 PAN 与各 MSS band 的相关系数，从而得到 GS 权重。"""
    sample_ratio = float(np.clip(sample_ratio or 0.1, 0.0001, 1.0))
    step = max(1, int(1.0 / np.sqrt(sample_ratio)))

    scale_x = float(mss_ds.width) / float(pan_ds.width)
    scale_y = float(mss_ds.height) / float(pan_ds.height)

    n = 0
    sum_pan = 0.0
    sum_pan2 = 0.0

    sum_b = np.zeros(num_bands, dtype=np.float64)
    sum_b2 = np.zeros(num_bands, dtype=np.float64)
    sum_pan_b = np.zeros(num_bands, dtype=np.float64)

    for row_off in range(0, pan_ds.height, tile_size):
        win_h = min(tile_size, pan_ds.height - row_off)
        for col_off in range(0, pan_ds.width, tile_size):
            win_w = min(tile_size, pan_ds.width - col_off)
            pan_win = Window(col_off, row_off, win_w, win_h)
            pan_tile = pan_ds.read(1, window=pan_win).astype(np.float64)

            mss_win = Window(
                col_off * scale_x,
                row_off * scale_y,
                win_w * scale_x,
                win_h * scale_y,
            )

            pan_s = pan_tile[::step, ::step]
            if pan_s.size == 0:
                continue

            pan_f = pan_s.reshape(-1)
            n_tile = pan_f.size
            n += n_tile
            sum_pan += float(pan_f.sum())
            sum_pan2 += float((pan_f * pan_f).sum())

            for bi in range(num_bands):
                band_tile = mss_ds.read(
                    bi + 1,
                    window=mss_win,
                    out_shape=(win_h, win_w),
                    resampling=resampling,
                ).astype(np.float64)
                band_f = band_tile[::step, ::step].reshape(-1)
                m = min(band_f.size, pan_f.size)
                if m <= 0:
                    continue
                band_f = band_f[:m]
                pan_f_adj = pan_f[:m]

                sum_b[bi] += float(band_f.sum())
                sum_b2[bi] += float((band_f * band_f).sum())
                sum_pan_b[bi] += float((pan_f_adj * band_f).sum())

    if n <= 1:
        return np.ones(num_bands, dtype=np.float32) / float(num_bands)

    mean_pan = sum_pan / n
    var_pan = max(0.0, (sum_pan2 / n) - mean_pan * mean_pan)
    std_pan = np.sqrt(var_pan) if var_pan > 0 else 0.0

    corrs = []
    for bi in range(num_bands):
        mean_b = sum_b[bi] / n
        var_b = max(0.0, (sum_b2[bi] / n) - mean_b * mean_b)
        std_b = np.sqrt(var_b) if var_b > 0 else 0.0
        cov = (sum_pan_b[bi] / n) - mean_pan * mean_b
        corr = cov / (std_pan * std_b) if (std_pan > 0 and std_b > 0) else 0.0
        corrs.append(corr)

    abs_corrs = np.abs(np.asarray(corrs, dtype=np.float64))
    denom = float(abs_corrs.sum())
    if denom > 0:
        return (abs_corrs / denom).astype(np.float32)
    return np.ones(num_bands, dtype=np.float32) / float(num_bands)


def _compute_global_stats_streaming(
    pan_ds,
    mss_ds,
    num_bands: int,
    tile_size: int,
    resampling: Resampling,
    weights,
    sample_ratio: float,
):
    """分块+抽样计算 IHS/GS 所需全局统计（不下采样，仍低内存）。"""
    sample_ratio = float(np.clip(sample_ratio or 0.1, 0.0001, 1.0))
    step = max(1, int(1.0 / np.sqrt(sample_ratio)))
    weights = np.asarray(weights, dtype=np.float64)

    scale_x = float(mss_ds.width) / float(pan_ds.width)
    scale_y = float(mss_ds.height) / float(pan_ds.height)

    n = 0
    sum_pan = 0.0
    sum_pan2 = 0.0
    sum_int = 0.0
    sum_int2 = 0.0
    sum_sim = 0.0
    sum_sim2 = 0.0

    sum_band = np.zeros(num_bands, dtype=np.float64)
    sum_sim_band = np.zeros(num_bands, dtype=np.float64)

    for row_off in range(0, pan_ds.height, tile_size):
        win_h = min(tile_size, pan_ds.height - row_off)
        for col_off in range(0, pan_ds.width, tile_size):
            win_w = min(tile_size, pan_ds.width - col_off)
            pan_win = Window(col_off, row_off, win_w, win_h)
            pan_tile = pan_ds.read(1, window=pan_win).astype(np.float64)

            mss_win = Window(
                col_off * scale_x,
                row_off * scale_y,
                win_w * scale_x,
                win_h * scale_y,
            )

            mss_tiles = []
            for bi in range(num_bands):
                band_tile = mss_ds.read(
                    bi + 1,
                    window=mss_win,
                    out_shape=(win_h, win_w),
                    resampling=resampling,
                ).astype(np.float64)
                mss_tiles.append(band_tile)

            pan_s = pan_tile[::step, ::step]
            if pan_s.size == 0:
                continue

            mss_stack = np.stack([b[::step, ::step] for b in mss_tiles], axis=0)
            intensity_s = mss_stack.mean(axis=0)
            sim_s = np.tensordot(weights, mss_stack, axes=(0, 0))

            pan_f = pan_s.reshape(-1)
            int_f = intensity_s.reshape(-1)
            sim_f = sim_s.reshape(-1)

            m = min(pan_f.size, int_f.size, sim_f.size)
            if m <= 0:
                continue
            pan_f = pan_f[:m]
            int_f = int_f[:m]
            sim_f = sim_f[:m]

            n += m
            sum_pan += float(pan_f.sum())
            sum_pan2 += float((pan_f * pan_f).sum())
            sum_int += float(int_f.sum())
            sum_int2 += float((int_f * int_f).sum())
            sum_sim += float(sim_f.sum())
            sum_sim2 += float((sim_f * sim_f).sum())

            for bi in range(num_bands):
                band_f = mss_tiles[bi][::step, ::step].reshape(-1)
                band_f = band_f[:m]
                sum_band[bi] += float(band_f.sum())
                sum_sim_band[bi] += float((band_f * sim_f).sum())

    if n <= 1:
        ihs_stats = (0.0, 0.0, 0.0, 0.0)
        gs_stats = (0.0, 0.0, np.zeros(num_bands, dtype=np.float32))
        return ihs_stats, gs_stats

    pan_mean = sum_pan / n
    pan_var = max(0.0, (sum_pan2 / n) - pan_mean * pan_mean)
    pan_std = float(np.sqrt(pan_var))

    int_mean = sum_int / n
    int_var = max(0.0, (sum_int2 / n) - int_mean * int_mean)
    int_std = float(np.sqrt(int_var))

    sim_mean = sum_sim / n
    sim_var = max(0.0, (sum_sim2 / n) - sim_mean * sim_mean)
    sim_std = float(np.sqrt(sim_var))

    g_coeffs = np.zeros(num_bands, dtype=np.float32)
    if sim_var > 1e-12:
        for bi in range(num_bands):
            band_mean = sum_band[bi] / n
            cov = (sum_sim_band[bi] / n) - band_mean * sim_mean
            g_coeffs[bi] = float(cov / sim_var)

    ihs_stats = (float(pan_mean), float(pan_std), float(int_mean), float(int_std))
    gs_stats = (float(sim_mean), float(sim_std), g_coeffs)
    return ihs_stats, gs_stats


def _compute_global_stats_downsampled(
    pan_ds,
    mss_ds,
    num_bands: int,
    stats_max_dim: int,
    sample_ratio_for_weights: float,
):
    """用下采样数据估计全局统计，避免整图入内存。

    返回:
      - weights: GS 权重（归一化）
      - ihs_stats: (pan_mean, pan_std, intensity_mean, intensity_std)
      - gs_stats: (sim_pan_mean, sim_pan_std, g_coeffs)
    """
    out_h, out_w = _choose_stats_shape(pan_ds.width, pan_ds.height, stats_max_dim)

    pan_small = pan_ds.read(1, out_shape=(out_h, out_w), resampling=Resampling.average).astype(np.float64)
    mss_small_bands: list[np.ndarray] = []
    for b in range(1, num_bands + 1):
        band_small = mss_ds.read(b, out_shape=(out_h, out_w), resampling=Resampling.average).astype(np.float64)
        mss_small_bands.append(band_small)

    # 用抽样比例控制相关性计算开销
    sample_ratio_for_weights = float(np.clip(sample_ratio_for_weights or 0.1, 0.0001, 1.0))
    step = max(1, int(1.0 / np.sqrt(sample_ratio_for_weights)))
    pan_s = pan_small[::step, ::step].reshape(-1)

    corrs = []
    for band in mss_small_bands:
        band_s = band[::step, ::step].reshape(-1)
        corr = np.corrcoef(pan_s, band_s)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        corrs.append(corr)
    corrs = np.asarray(corrs, dtype=np.float64)
    abs_corrs = np.abs(corrs)
    denom = float(abs_corrs.sum())
    if denom > 0:
        weights = (abs_corrs / denom).astype(np.float32)
    else:
        weights = (np.ones(num_bands, dtype=np.float32) / float(num_bands))

    pan_mean = float(pan_small.mean())
    pan_std = float(pan_small.std())
    intensity_small = np.mean(np.stack(mss_small_bands, axis=0), axis=0)
    intensity_mean = float(intensity_small.mean())
    intensity_std = float(intensity_small.std())
    ihs_stats = (pan_mean, pan_std, intensity_mean, intensity_std)

    sim_pan_small = np.zeros_like(intensity_small, dtype=np.float64)
    for i in range(num_bands):
        sim_pan_small += mss_small_bands[i] * float(weights[i])
    sim_pan_mean = float(sim_pan_small.mean())
    sim_pan_std = float(sim_pan_small.std())

    sim_flat = sim_pan_small.reshape(-1)
    sim_var = float(sim_flat.var())
    if sim_var <= 1e-12:
        g_coeffs = np.zeros(num_bands, dtype=np.float32)
    else:
        centered_sim = sim_flat - sim_pan_mean
        g = []
        for band in mss_small_bands:
            band_flat = band.reshape(-1)
            band_mean = float(band_flat.mean())
            cov = float(np.mean((band_flat - band_mean) * centered_sim))
            g.append(cov / sim_var)
        g_coeffs = np.asarray(g, dtype=np.float32)

    gs_stats = (sim_pan_mean, sim_pan_std, g_coeffs)
    return weights, ihs_stats, gs_stats


def _ihs_pan_sharpening_with_stats(pan_band, mss_bands, pan_mean, pan_std, intensity_mean, intensity_std):
    """IHS：使用全局统计做 PAN 归一化，避免分块之间色调跳变。"""
    pan_float = pan_band.astype(np.float32)
    mss_float = [band.astype(np.float32) for band in mss_bands]

    intensity_orig = np.mean(mss_float, axis=0)
    intensity_orig = np.where(intensity_orig == 0, 1, intensity_orig)

    if pan_std > 0:
        pan_normalized = (pan_float - float(pan_mean)) / float(pan_std) * float(intensity_std) + float(intensity_mean)
    else:
        pan_normalized = pan_float

    ratio = pan_normalized / intensity_orig

    sharpened_bands = []
    for band in mss_float:
        sharpened = band * ratio
        sharpened = np.clip(sharpened, 0, 65535)
        sharpened_bands.append(sharpened)
    return sharpened_bands


def _gram_schmidt_with_global_coeffs(pan_band, mss_bands, weights, pan_mean, pan_std, sim_pan_mean, sim_pan_std, g_coeffs):
    """GS：用全局(下采样)估计的 g_coeffs + stats，按 tile 精确套用原公式。"""
    pan_float = pan_band.astype(np.float32)
    mss_float = [b.astype(np.float32) for b in mss_bands]

    sim_pan = np.zeros_like(mss_float[0], dtype=np.float32)
    for i in range(len(mss_float)):
        sim_pan += mss_float[i] * float(weights[i])

    if pan_std > 0:
        pan_adjusted = (pan_float - float(pan_mean)) * (float(sim_pan_std) / float(pan_std)) + float(sim_pan_mean)
    else:
        pan_adjusted = pan_float

    delta = pan_adjusted - sim_pan

    fused = []
    for i, band in enumerate(mss_float):
        out = band + float(g_coeffs[i]) * delta
        out = np.clip(out, 0, 65535)
        fused.append(out)
    return fused

# ============================================================================
# 用户配置区
# ============================================================================
DEFAULT_INPUT_DIR = r"F:\code\pic\pt"
DEFAULT_OUTPUT_DIR = r"F:\平台测试样本检测结果\pan_sharpened"
DEFAULT_SHARPEN_METHOD = "gram_schmidt"  # brovey/ihs/gram_schmidt/mean
SHARPEN_SAMPLE_RATIO = 0.8  # 计算GS权重的采样率

# 分块处理默认参数（用于降低内存占用）
DEFAULT_TILE_SIZE = 30  # 0 表示整图处理（占内存大）
DEFAULT_STATS_MAX_DIM = 0  # 0: 不下采样，改用分块扫描统计（更准但会多一遍IO）
DEFAULT_RESAMPLING = "cubic"  # MSS -> PAN 分辨率重采样方式

def parse_args():
    parser = argparse.ArgumentParser(description="Pan-sharpen images while preserving RPB metadata.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                        help="Root folder of imagery; each scene in its own subfolder.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Destination folder for pan-sharpened images.")
    parser.add_argument("--method", default=DEFAULT_SHARPEN_METHOD,
                        choices=["brovey", "ihs", "gram_schmidt", "mean"],
                        help="Pan-sharpen method.")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help="Process by tiles to reduce memory. 0 means full image in memory.")
    parser.add_argument("--stats-max-dim", type=int, default=DEFAULT_STATS_MAX_DIM,
                        help="Max dimension for downsampled global stats (IHS/GS). 0 = streaming stats by tiles (no downsample).")
    parser.add_argument("--resampling", default=DEFAULT_RESAMPLING,
                        choices=["nearest", "bilinear", "cubic"],
                        help="Resampling method when scaling MSS to PAN grid in tiled mode.")
    return parser.parse_args()


def guess_suffix(stem):
    """从文件名推断后缀标识"""
    if "-" in stem:
        return stem.rsplit("-", 1)[1].upper()
    return ""


def looks_like_pan(suffix):
    """判断是否为全色影像"""
    s = suffix.upper()
    return any(tag in s for tag in ["PAN", "BWDPAN"])


def find_scene_pairs(pic_root):
    """
    在文件夹中查找所有场景的多光谱和全色影像对
    返回: [{"scene": scene_name, "mss": mss_path, "pan": pan_path, "mss_rpb": rpb_path, "pan_rpb": rpb_path}, ...]
    """
    scene_pairs = []

    # 遍历所有子文件夹
    for scene_name in os.listdir(pic_root):
        scene_dir = os.path.join(pic_root, scene_name)
        if not os.path.isdir(scene_dir):
            continue

        # 查找所有tiff文件
        tiffs = [
            f for f in os.listdir(scene_dir)
            if f.lower().endswith((".tif", ".tiff")) and "thumb" not in f.lower()
        ]

        if len(tiffs) < 2:
            continue

        # 分析每个tiff文件
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

        if len(infos) < 2:
            continue

        # 选择PAN和MSS
        def pick_best(candidates, prefer_pan):
            filtered = [c for c in candidates if c["bands"] == 1] if prefer_pan else [c for c in candidates if c["bands"] >= 3]
            if not filtered:
                return None
            filtered.sort(key=lambda c: (
                0 if c["has_rpb"] else 1,
                c["res"],
                -c["bands"],
                os.path.basename(c["path"]),
            ))
            return filtered[0]

        pan_info = pick_best(infos, prefer_pan=True)
        remaining = [c for c in infos if pan_info and c["path"] != pan_info["path"]]
        mss_info = pick_best(remaining, prefer_pan=False)

        if not pan_info or not mss_info:
            print(f"[warn] {scene_name} 未找到 PAN/MSS 配对")
            continue

        def rpb_of(info):
            return os.path.splitext(info["path"])[0] + ".rpb" if info["has_rpb"] else None

        scene_pairs.append({
            "scene": scene_name,
            "pan": pan_info["path"],
            "mss": mss_info["path"],
            "pan_rpb": rpb_of(pan_info),
            "mss_rpb": rpb_of(mss_info),
        })

    return scene_pairs


def sharpen_scene(scene_info, output_dir, method, tile_size, stats_max_dim, resampling_name):
    """
    对单个场景进行全色锐化
    """
    scene_name = scene_info["scene"]
    mss_path = scene_info["mss"]
    pan_path = scene_info["pan"]
    mss_rpb = scene_info.get("mss_rpb")
    pan_rpb = scene_info.get("pan_rpb")

    print(f"\n[{scene_name}] 开始处理...")
    print(f"  MSS: {os.path.basename(mss_path)} ({'RPC' if mss_rpb else 'no RPC'})")
    print(f"  PAN: {os.path.basename(pan_path)} ({'RPC' if pan_rpb else 'no RPC'})")

    # 读取影像
    with rasterio.open(mss_path) as mss_ds, rasterio.open(pan_path) as pan_ds:
        print(f"  MSS: {mss_ds.width}x{mss_ds.height}, {mss_ds.count} bands")
        print(f"  PAN: {pan_ds.width}x{pan_ds.height}, {pan_ds.count} band(s)")

        num_bands = min(mss_ds.count, 4)

        # 读取RPB信息（如果有）
        mss_rpc = parse_rpb_file(mss_rpb) if mss_rpb else None
        pan_rpc = parse_rpb_file(pan_rpb) if pan_rpb else None

        # ------------------------
        # 分块模式：极大降低内存占用
        # ------------------------
        if tile_size and tile_size > 0:
            resampling = _resampling_from_name(resampling_name)
            print(f"  分块处理启用: tile={tile_size}, resampling={resampling_name}")

            weights = None
            ihs_stats = None
            gs_stats = None
            if method in {"ihs", "gram_schmidt"}:
                if stats_max_dim and stats_max_dim > 0:
                    print(f"  估计全局统计(下采样 max_dim={stats_max_dim})...")
                    weights, ihs_stats, gs_stats = _compute_global_stats_downsampled(
                        pan_ds,
                        mss_ds,
                        num_bands=num_bands,
                        stats_max_dim=stats_max_dim,
                        sample_ratio_for_weights=SHARPEN_SAMPLE_RATIO,
                    )
                else:
                    print(f"  估计全局统计(分块扫描, 无下采样)...")
                    weights = _compute_weights_streaming(
                        pan_ds,
                        mss_ds,
                        num_bands=num_bands,
                        tile_size=tile_size,
                        resampling=resampling,
                        sample_ratio=SHARPEN_SAMPLE_RATIO,
                    )
                    ihs_stats, gs_stats = _compute_global_stats_streaming(
                        pan_ds,
                        mss_ds,
                        num_bands=num_bands,
                        tile_size=tile_size,
                        resampling=resampling,
                        weights=weights,
                        sample_ratio=SHARPEN_SAMPLE_RATIO,
                    )
                if method == "gram_schmidt":
                    print(f"  权重: {weights}")

            output_path = os.path.join(output_dir, f"{scene_name}_fused.tif")
            os.makedirs(output_dir, exist_ok=True)
            print(f"  保存结果到: {output_path}")

            profile = pan_ds.profile.copy()
            blockx = _safe_tiff_block_size(pan_ds.width, preferred=min(256, int(tile_size)))
            blocky = _safe_tiff_block_size(pan_ds.height, preferred=min(256, int(tile_size)))
            use_tiled = (pan_ds.width >= 16 and pan_ds.height >= 16)
            profile.update(
                driver="GTiff",
                width=pan_ds.width,
                height=pan_ds.height,
                count=num_bands,
                dtype=rasterio.uint16,
                nodata=0,
                compress="lzw",
                tiled=use_tiled,
                blockxsize=blockx if use_tiled else None,
                blockysize=blocky if use_tiled else None,
                bigtiff="yes",
            )

            scale_x = float(mss_ds.width) / float(pan_ds.width)
            scale_y = float(mss_ds.height) / float(pan_ds.height)

            with rasterio.open(output_path, "w", **profile) as dst:
                if pan_ds.transform is not None:
                    print(f"  已保留地理变换信息")
                if pan_ds.crs is not None:
                    print(f"  已保留投影信息")

                rpc_tags = {}
                if pan_rpb and pan_rpc:
                    rpc_tags = rpb_rpc_to_gdal_rpc_tags(pan_rpc)
                    if rpc_tags:
                        dst.update_tags(ns="RPC", **rpc_tags)
                        print(f"  已写入 PAN RPC 信息 (来自 .rpb)")
                elif mss_rpb and mss_rpc:
                    rpc_tags = rpb_rpc_to_gdal_rpc_tags(mss_rpc)
                    if rpc_tags:
                        print(f"  [警告] 使用 MSS RPC（可能不精确）")
                        dst.update_tags(ns="RPC", **rpc_tags)

                for row_off in range(0, pan_ds.height, tile_size):
                    win_h = min(tile_size, pan_ds.height - row_off)
                    for col_off in range(0, pan_ds.width, tile_size):
                        win_w = min(tile_size, pan_ds.width - col_off)
                        pan_win = Window(col_off, row_off, win_w, win_h)

                        pan_tile = pan_ds.read(1, window=pan_win).astype(np.float32)

                        mss_win = Window(
                            col_off * scale_x,
                            row_off * scale_y,
                            win_w * scale_x,
                            win_h * scale_y,
                        )
                        mss_bands_tile = []
                        for b in range(1, num_bands + 1):
                            band_tile = mss_ds.read(
                                b,
                                window=mss_win,
                                out_shape=(win_h, win_w),
                                resampling=resampling,
                            ).astype(np.float32)
                            mss_bands_tile.append(band_tile)

                        if method == "brovey":
                            fused_bands = brovey_pan_sharpening(pan_tile, mss_bands_tile)
                        elif method == "mean":
                            fused_bands = simple_mean_pan_sharpening(pan_tile, mss_bands_tile)
                        elif method == "ihs":
                            pan_mean, pan_std, intensity_mean, intensity_std = ihs_stats
                            fused_bands = _ihs_pan_sharpening_with_stats(
                                pan_tile,
                                mss_bands_tile,
                                pan_mean,
                                pan_std,
                                intensity_mean,
                                intensity_std,
                            )
                        else:
                            pan_mean, pan_std, intensity_mean, intensity_std = ihs_stats
                            sim_pan_mean, sim_pan_std, g_coeffs = gs_stats
                            fused_bands = _gram_schmidt_with_global_coeffs(
                                pan_tile,
                                mss_bands_tile,
                                weights,
                                pan_mean,
                                pan_std,
                                sim_pan_mean,
                                sim_pan_std,
                                g_coeffs,
                            )

                        for i, band in enumerate(fused_bands, start=1):
                            dst.write(band.astype(np.uint16), i, window=pan_win)

            print(f"  [{scene_name}] 完成!")
            return

        # ------------------------
        # 整图模式：保留原逻辑（但很吃内存）
        # ------------------------
        print("  [提示] tile_size=0 将整图读入内存，超大图可能占用非常大")

        mss_bands = [mss_ds.read(b + 1) for b in range(num_bands)]
        pan_band = pan_ds.read(1)

        weights = None
        if method == "gram_schmidt" and len(mss_bands) >= 3:
            print(f"  计算波段相关性...")
            _, weights = calculate_band_correlations(pan_band, mss_bands, sample_ratio=SHARPEN_SAMPLE_RATIO)
            print(f"  权重: {weights}")

        print(f"  执行全色锐化 (方法: {method})...")
        if method == "brovey":
            fused_bands = brovey_pan_sharpening(pan_band, mss_bands)
        elif method == "ihs":
            fused_bands = ihs_pan_sharpening(pan_band, mss_bands)
        elif method == "gram_schmidt":
            fused_bands = gram_schmidt_pan_sharpening(pan_band, mss_bands, weights=weights)
        else:
            fused_bands = simple_mean_pan_sharpening(pan_band, mss_bands)

        # 保存结果（使用 rasterio 保留元数据）
        output_path = os.path.join(output_dir, f"{scene_name}_fused.tif")
        os.makedirs(output_dir, exist_ok=True)

        print(f"  保存结果到: {output_path}")

        # 写入 GeoTIFF（以 PAN 的空间参考为准）
        profile = pan_ds.profile.copy()
        blockx = _safe_tiff_block_size(pan_ds.width, preferred=256)
        blocky = _safe_tiff_block_size(pan_ds.height, preferred=256)
        use_tiled = (pan_ds.width >= 16 and pan_ds.height >= 16)
        profile.update(
            driver="GTiff",
            width=pan_ds.width,
            height=pan_ds.height,
            count=len(fused_bands),
            dtype=rasterio.uint16,
            nodata=0,
            compress="lzw",
            tiled=use_tiled,
            blockxsize=blockx if use_tiled else None,
            blockysize=blocky if use_tiled else None,
            bigtiff="yes",
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i, band in enumerate(fused_bands, start=1):
                dst.write(band.astype(np.uint16), i)

            if pan_ds.transform is not None:
                print(f"  已保留地理变换信息")

            if pan_ds.crs is not None:
                print(f"  已保留投影信息")

            # 保留/写入 RPC 信息（优先使用 PAN 的 RPB；否则尝试 MSS 的）
            rpc_tags = {}
            if pan_rpb and pan_rpc:
                rpc_tags = rpb_rpc_to_gdal_rpc_tags(pan_rpc)
                if rpc_tags:
                    dst.update_tags(ns="RPC", **rpc_tags)
                    print(f"  已写入 PAN RPC 信息 (来自 .rpb)")
            elif mss_rpb and mss_rpc:
                rpc_tags = rpb_rpc_to_gdal_rpc_tags(mss_rpc)
                if rpc_tags:
                    print(f"  [警告] 使用 MSS RPC（可能不精确）")
                    dst.update_tags(ns="RPC", **rpc_tags)

        print(f"  [{scene_name}] 完成!")


def main():
    args = parse_args()

    pic_root = args.input_dir if os.path.isabs(args.input_dir) else os.path.join(os.getcwd(), args.input_dir)

    if not os.path.exists(pic_root):
        print(f"错误: 输入目录不存在: {pic_root}")
        return

    # 查找所有场景
    print(f"扫描输入目录: {pic_root}")
    scene_pairs = find_scene_pairs(pic_root)

    if not scene_pairs:
        print("未找到任何 PAN/MSS 影像对")
        return

    print(f"找到 {len(scene_pairs)} 个场景")

    # 处理每个场景
    for scene_info in scene_pairs:
        try:
            sharpen_scene(
                scene_info,
                args.output_dir,
                args.method,
                tile_size=args.tile_size,
                stats_max_dim=args.stats_max_dim,
                resampling_name=args.resampling,
            )
        except Exception as exc:
            print(f"[error] {scene_info['scene']} -> {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n全部完成！输出路径: {args.output_dir}")


if __name__ == "__main__":
    main()
