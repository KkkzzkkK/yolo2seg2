"""
全色锐化模块 - Gram-Schmidt 算法

提供全色锐化功能，将高分辨率 PAN 与低分辨率 MSS 融合。
仅保留 Gram-Schmidt 算法，光谱保真度最高。
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2


def calculate_band_correlations(
    pan: np.ndarray,
    mss_bands: List[np.ndarray],
    sample_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """计算多光谱波段与全色波段的相关系数
    
    用于自动确定 Gram-Schmidt 全色锐化的最佳权重。
    
    Args:
        pan: 全色波段 (H, W)
        mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]
        sample_ratio: 采样率 (0-1)，用于加速计算
        
    Returns:
        correlations: 相关系数数组 [corr_B, corr_G, corr_R, ...]
        weights: 归一化后的权重数组（相关系数的归一化值，和为 1）
    """
    # 确保尺寸一致
    if pan.shape != mss_bands[0].shape:
        mss_bands_resized = [
            cv2.resize(band, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    else:
        mss_bands_resized = mss_bands
    
    # 采样以加速计算
    step = max(1, int(1.0 / np.sqrt(sample_ratio)))
    pan_sampled = pan[::step, ::step].flatten().astype(np.float64)
    
    correlations = []
    for band in mss_bands_resized:
        band_sampled = band[::step, ::step].flatten().astype(np.float64)
        
        # 计算皮尔逊相关系数
        if len(pan_sampled) > 1 and np.std(pan_sampled) > 0 and np.std(band_sampled) > 0:
            corr = np.corrcoef(pan_sampled, band_sampled)[0, 1]
        else:
            corr = 0.0
        
        # 处理 NaN 值
        if np.isnan(corr):
            corr = 0.0
        
        correlations.append(corr)
    
    correlations = np.array(correlations)
    
    # 将相关系数转换为权重（使用绝对值并归一化）
    abs_correlations = np.abs(correlations)
    corr_sum = np.sum(abs_correlations)
    
    if corr_sum > 0:
        weights = abs_correlations / corr_sum
    else:
        # 如果所有相关系数都为0，使用均等权重
        weights = np.ones(len(correlations)) / len(correlations)
    
    return correlations, weights


def gram_schmidt_sharpen(
    pan: np.ndarray,
    mss_bands: List[np.ndarray],
    weights: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """使用 Gram-Schmidt Adaptive 算法进行全色锐化
    
    光谱保真度优于 IHS 和 Brovey。
    
    Args:
        pan: 全色波段 (H, W)
        mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]，每个为 (H, W)
        weights: 波段权重列表，用于模拟低分辨率全色影像
                 - None: 自动使用均值权重 (1/n)
                 - [w1, w2, w3]: 3波段权重
                 - [w1, w2, w3, w4]: 4波段权重（包括NIR）
                 权重应该归一化（总和=1），如未归一化会自动归一化
    
    Returns:
        锐化后的波段列表（保持与输入相同的波段数）
    """
    # 确保尺寸一致（重采样多光谱到全色分辨率）
    if pan.shape != mss_bands[0].shape:
        mss_bands = [
            cv2.resize(band, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    
    pan_float = pan.astype(np.float32)
    mss_stack = np.stack([b.astype(np.float32) for b in mss_bands], axis=0)
    num_bands = len(mss_bands)
    
    # 1. 处理权重参数
    if weights is None:
        # 默认均值权重
        weights = np.ones(num_bands) / num_bands
    else:
        weights = np.array(weights, dtype=np.float32)
        # 检查权重数量是否匹配波段数
        if len(weights) != num_bands:
            print(f"警告: 权重数量({len(weights)})与波段数({num_bands})不匹配，使用均值权重")
            weights = np.ones(num_bands) / num_bands
        else:
            # 归一化权重
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights = weights / weights_sum
            else:
                print("警告: 权重总和为0，使用均值权重")
                weights = np.ones(num_bands) / num_bands
    
    # 2. 从多光谱影像模拟低分辨率的 PAN（使用加权平均）
    mss_mean = np.zeros_like(mss_stack[0], dtype=np.float32)
    for i in range(num_bands):
        mss_mean += mss_stack[i] * weights[i]
    
    # 3. Gram-Schmidt 正交化
    H, W = mss_stack.shape[1], mss_stack.shape[2]
    mss_vectors = mss_stack.reshape(num_bands, H * W)
    mean_vector = mss_mean.flatten()
    
    # 计算每个波段与模拟PAN的协方差
    cov_matrix = np.cov(np.vstack((mean_vector, mss_vectors)))
    g_coeffs = cov_matrix[0, 1:] / (cov_matrix[0, 0] + 1e-10)
    
    # 4. 用高分辨率 PAN 替换模拟的低分辨率 PAN
    pan_std = pan_float.std()
    if pan_std > 0:
        pan_adjusted = (pan_float - pan_float.mean()) * (mss_mean.std() / pan_std) + mss_mean.mean()
    else:
        pan_adjusted = pan_float
    pan_adjusted_flat = pan_adjusted.flatten()
    
    # 5. 逆变换，生成锐化后的影像
    sharpened_vectors = np.zeros_like(mss_vectors)
    for i in range(num_bands):
        sharpened_vectors[i, :] = mss_vectors[i, :] + g_coeffs[i] * (pan_adjusted_flat - mean_vector)
    
    sharpened_stack = sharpened_vectors.reshape(num_bands, H, W)
    
    sharpened_bands = []
    for i in range(num_bands):
        band = np.clip(sharpened_stack[i], 0, 65535)
        sharpened_bands.append(band)
    
    return sharpened_bands
