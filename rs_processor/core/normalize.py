"""
图像归一化模块 - CLAHE 算法

提供图像增强和归一化功能。
仅保留 CLAHE 方法，对比度增强效果最好。
"""

from typing import List, Tuple
import numpy as np
import cv2


def clahe_normalize(
    band: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """使用 CLAHE 进行归一化和局部对比度增强
    
    Args:
        band: 输入波段 (H, W)，任意数值类型
        clip_limit: CLAHE 裁剪限制
        tile_size: CLAHE 瓦片网格大小
        
    Returns:
        归一化后的波段 [0, 1] 浮点数组
    """
    # 处理无效值
    band = np.nan_to_num(band, nan=0.0, posinf=65535.0, neginf=0.0)
    
    # 归一化到 0-255
    band_min = np.min(band)
    band_max = np.max(band)
    
    if band_max > band_min:
        band_norm = (band - band_min) / (band_max - band_min) * 255
    else:
        band_norm = np.zeros_like(band, dtype=np.float32)
    
    band_uint8 = band_norm.astype(np.uint8)
    
    # 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced = clahe.apply(band_uint8)
    
    # 转换回 [0, 1]
    normalized = enhanced.astype(np.float32) / 255.0
    
    return normalized


def _nir_overexposure_fusion(
    rgb_norm: np.ndarray,
    nir_norm: np.ndarray,
    threshold: int = 235,
    kernel_size: Tuple[int, int] = (7, 7)
) -> np.ndarray:
    """使用 NIR 波段修复 RGB 过曝区域
    
    Args:
        rgb_norm: RGB 归一化图像 [0,1] (H, W, 3)
        nir_norm: NIR 波段归一化图像 [0,1] (H, W)
        threshold: 过曝阈值 (0-255)
        kernel_size: 高斯模糊核大小
        
    Returns:
        融合后的 RGB uint8 图像 (H, W, 3)
    """
    # 转换到 HSV 颜色空间
    rgb_8bit = (rgb_norm * 255).astype(np.uint8)
    hsv_image = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2HSV)
    h, s, v_orig = cv2.split(hsv_image)
    
    # 创建过曝区域的掩模
    overexposed_mask = (v_orig > threshold).astype(np.float32)
    
    # 平滑掩模以创建过渡
    smooth_mask = cv2.GaussianBlur(overexposed_mask, ksize=kernel_size, sigmaX=0)
    
    # 融合亮度通道
    v_nir = (nir_norm * 255).astype(np.uint8)
    v_new_float = v_orig.astype(np.float32) * (1 - smooth_mask) + v_nir.astype(np.float32) * smooth_mask
    v_new = np.clip(v_new_float, 0, 255).astype(np.uint8)
    
    # 合并通道并转换回 RGB
    fused_hsv = cv2.merge([h, s, v_new])
    fused_rgb = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2RGB)
    
    return fused_rgb


def bands_to_rgb_uint8(
    bands: List[np.ndarray],
    nir_fusion: bool = False,
    clahe_clip_limit: float = 3.0
) -> np.ndarray:
    """多波段转 RGB uint8，可选 NIR 过曝修复
    
    Args:
        bands: 波段列表 [B, G, R] 或 [B, G, R, NIR]
        nir_fusion: 是否使用 NIR 修复过曝区域
        clahe_clip_limit: CLAHE 裁剪限制
        
    Returns:
        RGB 或 RGBA uint8 图像 (H, W, 3) 或 (H, W, 4)
    """
    if len(bands) < 3:
        raise ValueError("至少需要 3 个波段 (B, G, R)")
    
    # 处理无效值
    bands = [np.nan_to_num(b, nan=0.0, posinf=65535.0, neginf=0.0) for b in bands]
    
    # 归一化 RGB 波段
    b_norm = clahe_normalize(bands[0], clip_limit=clahe_clip_limit)
    g_norm = clahe_normalize(bands[1], clip_limit=clahe_clip_limit)
    r_norm = clahe_normalize(bands[2], clip_limit=clahe_clip_limit)
    
    # 堆叠为 RGB
    rgb_norm = np.stack([r_norm, g_norm, b_norm], axis=-1)
    
    # 处理 NIR 波段
    if len(bands) >= 4:
        nir_norm = clahe_normalize(bands[3], clip_limit=clahe_clip_limit)
        
        if nir_fusion:
            # 使用 NIR 修复过曝
            rgb_8bit = _nir_overexposure_fusion(rgb_norm, nir_norm)
        else:
            rgb_8bit = (rgb_norm * 255).astype(np.uint8)
        
        # 添加 NIR 作为第四通道
        nir_8bit = (nir_norm * 255).astype(np.uint8)
        return np.dstack([rgb_8bit, nir_8bit])
    else:
        # 仅 RGB
        rgb_8bit = (rgb_norm * 255).astype(np.uint8)
        return rgb_8bit
