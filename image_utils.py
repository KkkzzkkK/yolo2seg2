# -*- coding: utf-8 -*-
"""
遥感影像处理通用工具模块
包含 RPC 处理、影像归一化、全色锐化等通用功能
"""

import numpy as np
import cv2
from pyproj import Transformer
import rasterio

# --- RPC 相关函数 ---

# 解析 .rpb 文件，提取 RPC 参数
def parse_rpb_file(rpb_path):
    """解析 .rpb 文件，提取 RPC 参数"""
    rpc_params = {}
    
    with open(rpb_path, 'r') as f:
        lines = f.readlines()
    
    # 解析 RPC 参数
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().rstrip(';')
            
            if key in ['lineOffset', 'sampOffset', 'latOffset', 'longOffset', 'heightOffset',
                      'lineScale', 'sampScale', 'latScale', 'longScale', 'heightScale']:
                rpc_params[key] = float(value)
    
    # 解析系数数组
    coef_names = ['lineNumCoef', 'lineDenCoef', 'sampNumCoef', 'sampDenCoef']
    for coef_name in coef_names:
        coef_values = []
        in_coef = False
        for line in lines:
            line = line.strip()
            if coef_name + ' = (' in line:
                in_coef = True
                first_val = line.split('(')[1].strip()
                if first_val and first_val != '':
                    if ',' in first_val:
                        first_val = first_val.rstrip(',')
                    coef_values.append(float(first_val))
            elif in_coef:
                if ');' in line:
                    last_val = line.replace(');', '').strip()
                    if last_val:
                        coef_values.append(float(last_val))
                    in_coef = False
                elif line:
                    val = line.rstrip(',').strip()
                    if val and val not in ['+', '-']:
                        try:
                            coef_values.append(float(val))
                        except:
                            pass
        
        if coef_values:
            rpc_params[coef_name] = np.array(coef_values)
    
    return rpc_params

# 使用 RPC 模型将地理坐标转换为像素坐标
def ground_to_image_rpc(lon, lat, height, rpc_params):
    """使用 RPC 模型将地理坐标转换为像素坐标"""
    P = (lat - rpc_params['latOffset']) / rpc_params['latScale']
    L = (lon - rpc_params['longOffset']) / rpc_params['longScale']
    H = (height - rpc_params['heightOffset']) / rpc_params['heightScale']
    
    terms = np.array([
        1.0,
        L, P, H,
        L*P, L*H, P*H,
        L*L, P*P, H*H,
        P*L*H,
        L*L*L, L*L*P, L*L*H, L*P*P, L*P*H,
        L*H*H, P*P*P, P*P*H, P*H*H
    ])
    
    line_num = np.dot(rpc_params['lineNumCoef'], terms)
    line_den = np.dot(rpc_params['lineDenCoef'], terms)
    line_normalized = line_num / line_den
    
    samp_num = np.dot(rpc_params['sampNumCoef'], terms)
    samp_den = np.dot(rpc_params['sampDenCoef'], terms)
    samp_normalized = samp_num / samp_den
    
    row = line_normalized * rpc_params['lineScale'] + rpc_params['lineOffset']
    col = samp_normalized * rpc_params['sampScale'] + rpc_params['sampOffset']
    
    return col, row

# 根据 RPC 参数计算影像的地理范围
def get_image_geo_bounds(rpc_params):
    """根据 RPC 参数计算影像的地理范围"""
    lat_offset = rpc_params.get('latOffset', 0)
    lon_offset = rpc_params.get('longOffset', 0)
    lat_scale = rpc_params.get('latScale', 1)
    lon_scale = rpc_params.get('longScale', 1)
    
    min_lat = lat_offset - lat_scale
    max_lat = lat_offset + lat_scale
    min_lon = lon_offset - lon_scale
    max_lon = lon_offset + lon_scale
    
    return (min_lon, min_lat, max_lon, max_lat)

# 使用 RPC 模型将像素坐标转换为地理坐标
def image_to_ground_rpc(col, row, rpc_params, initial_height=None, iterations=5):
    """
    使用 RPC 模型将像素坐标迭代反演为地理坐标.
    """
    if initial_height is None:
        initial_height = rpc_params.get('heightOffset', 0)

    # 初始猜测值 (使用 RPC 中心)
    lat = rpc_params.get('latOffset', 0)
    lon = rpc_params.get('longOffset', 0)
    height = initial_height

    for _ in range(iterations):
        # 正向投影，得到当前地理坐标猜测值对应的像素坐标
        est_col, est_row = ground_to_image_rpc(lon, lat, height, rpc_params)

        # 计算误差
        d_col = col - est_col
        d_row = row - est_row
        
        if abs(d_col) < 1e-6 and abs(d_row) < 1e-6:
            break
            
        # 建立线性方程求解地理坐标的修正量
        # (这里使用简化的偏导数近似，实际需要RPC的偏导数项，但多数RPC文件不提供)
        # 我们假设地理坐标的小变化与像素坐标的变化是线性的

        # 估算像素坐标相对于地理坐标的变化率 (雅可比矩阵的近似)
        # 扰动一个很小的值来估算偏导
        delta = 1e-5 
        col_lat, _ = ground_to_image_rpc(lon, lat + delta, height, rpc_params)
        col_lon, _ = ground_to_image_rpc(lon + delta, lat, height, rpc_params)
        _, row_lat = ground_to_image_rpc(lon, lat + delta, height, rpc_params)
        _, row_lon = ground_to_image_rpc(lon + delta, lat, height, rpc_params)

        dcol_dlat = (col_lat - est_col) / delta
        dcol_dlon = (col_lon - est_col) / delta
        drow_dlat = (row_lat - est_row) / delta
        drow_dlon = (row_lon - est_row) / delta

        # 解线性方程 [d_col, d_row]' = J * [d_lon, d_lat]'
        J = np.array([[dcol_dlon, dcol_dlat], [drow_dlon, drow_dlat]])
        
        try:
            # 使用伪逆求解
            inv_J = np.linalg.pinv(J)
            corrections = np.dot(inv_J, np.array([d_col, d_row]))
            d_lon, d_lat = corrections
            
            # 更新地理坐标 (增加阻尼因子避免发散)
            lon += d_lon * 0.8
            lat += d_lat * 0.8

        except np.linalg.LinAlgError:
            # 如果矩阵奇异，则用一个小的步长代替
            lon += d_col * delta * 0.1
            lat += d_row * delta * 0.1

    return lon, lat

# 使用 RPC 将多边形从投影坐标转换为像素坐标
def convert_polygon_with_rpc(polygon, rpc_params, transformer, default_height=0):
    """使用 RPC 将多边形从投影坐标转换为像素坐标"""
    pixel_coords = []
    
    coords = list(polygon.exterior.coords)
    if len(coords) >= 2 and (coords[0][0] == coords[-1][0]) and (coords[0][1] == coords[-1][1]):
        coords = coords[:-1]
    
    for coord in coords:
        lon, lat = transformer.transform(coord[0], coord[1])
        px, py = ground_to_image_rpc(lon, lat, default_height, rpc_params)
        pixel_coords.append([float(px), float(py)])
    
    return pixel_coords

# --- 图像归一化函数 ---

# 使用百分位拉伸进行归一化
def percentile_normalize(band, lower=2, upper=95):
    """使用百分位拉伸进行归一化"""
    # 计算百分位值
    p_low, p_high = np.percentile(band, (lower, upper))
    # 防止除以零
    if p_high == p_low:
        return np.zeros_like(band, dtype=np.float32)
    # 线性拉伸到[0,1]范围
    normalized = np.clip((band - p_low) / (p_high - p_low), 0, 1)
    return normalized.astype(np.float32)
    
# 使用 CLAHE 进行归一化和局部对比度增强（基于单个分块统计）
def clahe_normalize(band, clipLimit=3.0, tileGridSize=(8, 8)):
    """使用CLAHE进行归一化和局部对比度增强（基于单个分块统计）"""
    band = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl1 = clahe.apply(band)
    normalized = cl1.astype(np.float32) / 255.0
    return normalized

# 使用全局统计信息进行 CLAHE 归一化
def clahe_normalize_with_global_stats(band, global_min, global_max, clipLimit=3.0, tileGridSize=(8, 8)):
    """
    使用全局统计信息进行CLAHE归一化
    保证不同分块之间的一致性
    
    参数:
        band: 输入波段 (H, W)
        global_min: 全局最小值（从整个大图统计）
        global_max: 全局最大值（从整个大图统计）
        clipLimit: CLAHE裁剪限制
        tileGridSize: CLAHE瓦片网格大小
    
    返回: 归一化后的波段 [0, 1]
    """
    # 使用全局统计进行归一化
    if global_max > global_min:
        band_norm = np.clip((band.astype(np.float32) - global_min) / (global_max - global_min), 0, 1)
    else:
        band_norm = np.zeros_like(band, dtype=np.float32)
    
    # 转换为 uint8 用于 CLAHE
    band_uint8 = (band_norm * 255).astype(np.uint8)
    
    # 应用 CLAHE
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    cl1 = clahe.apply(band_uint8)
    
    # 转换回 [0, 1]
    normalized = cl1.astype(np.float32) / 255.0
    return normalized


# 使用 NIR 波段修复 RGB 过曝区域
def nir_overexposure_fusion(rgb_norm, nir_norm, threshold=235, kernel_size=(7, 7)):
    """
    使用 NIR 波段修复 RGB 过曝区域
    
    参数:
        rgb_norm: RGB 三通道归一化图像 [0,1] (H, W, 3) 或列表 [R, G, B]
        nir_norm: NIR 波段归一化图像 [0,1] (H, W)
        threshold: 过曝阈值 (0-255)
        kernel_size: 高斯模糊核大小
    
    返回:
        融合后的 RGB uint8 图像 (H, W, 3)
    """
    # 准备 RGB 图像
    if isinstance(rgb_norm, list):
        rgb_image = np.stack(rgb_norm, axis=-1)
    else:
        rgb_image = rgb_norm
    
    # 转换到 HSV 颜色空间
    rgb_8bit = (rgb_image * 255).astype(np.uint8)
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

# --- 全色锐化函数 ---
# 使用简单均值法进行全色锐化
def simple_mean_pan_sharpening(pan_band, mss_bands):
    """
    使用简单均值法进行全色锐化
    pan_band: 全色波段 (H, W)
    mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]，每个为 (H, W)
    返回: 锐化后的波段列表（保持与输入相同的波段数）
    """
    # 确保尺寸一致（重采样多光谱到全色分辨率）
    if pan_band.shape != mss_bands[0].shape:
        mss_bands = [
            cv2.resize(band, (pan_band.shape[1], pan_band.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    
    # 计算多光谱影像的平均强度（使用所有波段，包括 NIR）
    intensity = np.mean(mss_bands, axis=0).astype(np.float32)
    intensity = np.where(intensity == 0, 1, intensity)  # 避免除零
    
    # 计算比例因子
    ratio = pan_band.astype(np.float32) / intensity
    
    # 应用比例因子到每个波段
    sharpened_bands = []
    for band in mss_bands:
        sharpened = np.clip(band.astype(np.float32) * ratio, 0, 65535)
        sharpened_bands.append(sharpened)
    
    return sharpened_bands

# 使用 Brovey 变换进行全色锐化 - 色彩保真度更好
def brovey_pan_sharpening(pan_band, mss_bands):
    """
    使用 Brovey 变换进行全色锐化 - 色彩保真度更好
    pan_band: 全色波段 (H, W)
    mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]，每个为 (H, W)
    返回: 锐化后的波段列表（保持与输入相同的波段数）
    """
    # 确保尺寸一致（重采样多光谱到全色分辨率）
    if pan_band.shape != mss_bands[0].shape:
        mss_bands = [
            cv2.resize(band, (pan_band.shape[1], pan_band.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    
    # 计算多光谱影像的总强度（使用所有波段，包括 NIR）
    mss_sum = np.zeros_like(mss_bands[0], dtype=np.float32)
    for band in mss_bands:
        mss_sum += band.astype(np.float32)
    
    # 避免除零
    mss_sum = np.where(mss_sum == 0, 1, mss_sum)
    
    # Brovey 变换
    pan_float = pan_band.astype(np.float32)
    sharpened_bands = []
    for band in mss_bands:
        # 每个波段乘以 (全色 / 多光谱总和)
        sharpened = (band.astype(np.float32) / mss_sum) * pan_float
        sharpened = np.clip(sharpened, 0, 65535)
        sharpened_bands.append(sharpened)
    
    return sharpened_bands

# 使用 IHS (Intensity-Hue-Saturation) 变换进行全色锐化
def ihs_pan_sharpening(pan_band, mss_bands):
    """
    使用 IHS (Intensity-Hue-Saturation) 变换进行全色锐化
    色彩保真度高，细节增强明显，适合语义分割
    pan_band: 全色波段 (H, W)
    mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]，每个为 (H, W)
    返回: 锐化后的波段列表（保持与输入相同的波段数）
    """
    # 确保尺寸一致（重采样多光谱到全色分辨率）
    if pan_band.shape != mss_bands[0].shape:
        mss_bands = [
            cv2.resize(band, (pan_band.shape[1], pan_band.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    
    # 转换为 float32
    pan_float = pan_band.astype(np.float32)
    mss_float = [band.astype(np.float32) for band in mss_bands]
    
    # 计算原始强度（使用所有波段，包括 NIR，模拟全色波段的光谱范围）
    intensity_orig = np.mean(mss_float, axis=0)
    intensity_orig = np.where(intensity_orig == 0, 1, intensity_orig)
    
    # 归一化 PAN 到与 MSS 强度相同的范围
    pan_mean = np.mean(pan_float)
    pan_std = np.std(pan_float)
    intensity_mean = np.mean(intensity_orig)
    intensity_std = np.std(intensity_orig)
    
    if pan_std > 0:
        pan_normalized = (pan_float - pan_mean) / pan_std * intensity_std + intensity_mean
    else:
        pan_normalized = pan_float
    
    # 用归一化的 PAN 替换强度分量
    ratio = pan_normalized / intensity_orig
    
    # 应用到所有波段
    sharpened_bands = []
    for band in mss_float:
        sharpened = band * ratio
        sharpened = np.clip(sharpened, 0, 65535)
        sharpened_bands.append(sharpened)
    
    return sharpened_bands

# 计算多光谱波段与全色波段的相关系数（用于确定 Gram-Schmidt 权重）
def calculate_band_correlations(pan_band, mss_bands, sample_ratio=0.1):
    """
    计算多光谱波段与全色波段的相关系数
    用于自动确定 Gram-Schmidt 全色锐化的最佳权重
    
    参数:
        pan_band: 全色波段 (H, W)
        mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]
        sample_ratio: 采样率 (0-1)，用于加速计算
    
    返回:
        correlations: 相关系数数组 [corr_B, corr_G, corr_R, ...]
        weights: 归一化后的权重数组（相关系数的归一化值）
    """
    # 确保尺寸一致
    if pan_band.shape != mss_bands[0].shape:
        mss_bands_resized = [
            cv2.resize(band, (pan_band.shape[1], pan_band.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
    else:
        mss_bands_resized = mss_bands
    
    # 采样以加速计算
    step = max(1, int(1.0 / np.sqrt(sample_ratio)))
    pan_sampled = pan_band[::step, ::step].flatten().astype(np.float64)
    
    # 检查 PAN 波段的标准差，避免除零
    pan_std = np.std(pan_sampled)
    if pan_std < 1e-10:
        # PAN 波段是常值，无法计算有意义的相关系数，使用均等权重
        num_bands = len(mss_bands_resized)
        return np.zeros(num_bands), np.ones(num_bands) / num_bands
    
    correlations = []
    for band in mss_bands_resized:
        band_sampled = band[::step, ::step].flatten().astype(np.float64)
        
        # 检查波段的标准差，避免除零
        band_std = np.std(band_sampled)
        if band_std < 1e-10:
            # 该波段是常值，相关系数设为 0
            correlations.append(0.0)
            continue
        
        # 手动计算皮尔逊相关系数，避免 np.corrcoef 的除零警告
        pan_centered = pan_sampled - np.mean(pan_sampled)
        band_centered = band_sampled - np.mean(band_sampled)
        
        numerator = np.sum(pan_centered * band_centered)
        denominator = np.sqrt(np.sum(pan_centered**2) * np.sum(band_centered**2))
        
        if denominator > 1e-10:
            corr = numerator / denominator
        else:
            corr = 0.0
        
        # 处理 NaN 值
        if not np.isfinite(corr):
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

# 使用 Gram-Schmidt Adaptive 算法进行全色锐化
def gram_schmidt_pan_sharpening(pan_band, mss_bands, weights=None):
    """
    使用 Gram-Schmidt Adaptive 算法进行全色锐化.
    光谱保真度优于 IHS 和 Brovey.
    
    参数:
        pan_band: 全色波段 (H, W)
        mss_bands: 多光谱波段列表 [B, G, R] 或 [B, G, R, NIR]，每个为 (H, W)
        weights: 波段权重列表，用于模拟低分辨率全色影像
                 - None: 自动使用均值权重 (1/n)
                 - [w1, w2, w3]: 3波段权重
                 - [w1, w2, w3, w4]: 4波段权重（包括NIR）
                 权重应该归一化（总和=1），如未归一化会自动归一化
    
    返回: 锐化后的波段列表（保持与输入相同的波段数）
    """
    # 确保尺寸一致
    if pan_band.shape != mss_bands[0].shape:
        mss_bands = [
            cv2.resize(band, (pan_band.shape[1], pan_band.shape[0]), interpolation=cv2.INTER_CUBIC)
            for band in mss_bands
        ]
        
    pan_float = pan_band.astype(np.float32)
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
    #    这是关键改进：全色波段的光谱范围通常包括可见光和近红外
    #    使用权重来反映每个波段对全色波段的贡献
    mss_mean = np.zeros_like(mss_stack[0], dtype=np.float32)
    for i in range(num_bands):
        mss_mean += mss_stack[i] * weights[i]
    
    # 3. Gram-Schmidt 正交化
    #    第一个基 G1 是模拟的 PAN (mss_mean)
    #    后续的基 Gi 通过减去在前序基上的投影得到
    
    # 将 mss_stack 和 mss_mean 展平为向量进行计算
    H, W = mss_stack.shape[1], mss_stack.shape[2]
    mss_vectors = mss_stack.reshape(num_bands, H * W)
    mean_vector = mss_mean.flatten()
    
    # 计算每个波段与模拟PAN的协方差（点积）
    cov_matrix = np.cov(np.vstack((mean_vector, mss_vectors)))
    g_coeffs = cov_matrix[0, 1:] / (cov_matrix[0, 0] + 1e-10)  # 添加小值避免除零
    
    # 4. 用高分辨率 PAN 替换模拟的低分辨率 PAN
    #    调整高分PAN的均值和方差，使其与模拟PAN匹配
    pan_std = pan_float.std()
    if pan_std > 0:
        pan_adjusted = (pan_float - pan_float.mean()) * (mss_mean.std() / pan_std) + mss_mean.mean()
    else:
        pan_adjusted = pan_float
    pan_adjusted_flat = pan_adjusted.flatten()
    
    # 5. 逆变换，生成锐化后的影像
    sharpened_vectors = np.zeros_like(mss_vectors)
    for i in range(num_bands):
        # Fused_i = MSS_i + g_coeff_i * (HighRes_PAN - LowRes_PAN)
        sharpened_vectors[i, :] = mss_vectors[i, :] + g_coeffs[i] * (pan_adjusted_flat - mean_vector)
        
    sharpened_stack = sharpened_vectors.reshape(num_bands, H, W)
    
    sharpened_bands = []
    for i in range(num_bands):
        band = np.clip(sharpened_stack[i], 0, 65535)
        sharpened_bands.append(band)
        
    return sharpened_bands


# 将 RGB 三个波段归一化并合成为 8 位图像
def process_rgb_to_8bit(red_band, green_band, blue_band, method='clahe'):
    """
    将 RGB 三个波段归一化并合成为 8 位图像
    method: 归一化方法
      - 'clahe': CLAHE 独立处理每个波段（对比度强，但可能色彩失真）
      - 'percentile_rgb': RGB 联合百分位拉伸（保持波段比例，最佳色彩保真）
    返回: (H, W, 3) 的 uint8 数组
    """
    if method == 'clahe':
        red_norm = clahe_normalize(red_band)
        green_norm = clahe_normalize(green_band)
        blue_norm = clahe_normalize(blue_band)
    
    elif method == 'percentile_rgb':
        # RGB 联合归一化 - 保持波段间的相对关系
        rgb_stack = np.stack([red_band, green_band, blue_band], axis=0).astype(np.float32)
        
        # 使用所有波段的联合百分位
        p_low, p_high = np.percentile(rgb_stack, (2, 98))
        
        if p_high > p_low:
            rgb_stack = (rgb_stack - p_low) / (p_high - p_low)
            rgb_stack = np.clip(rgb_stack, 0, 1)
        else:
            rgb_stack = np.zeros_like(rgb_stack)
        
        red_norm = rgb_stack[0]
        green_norm = rgb_stack[1]
        blue_norm = rgb_stack[2]
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 堆叠为 RGB 图像
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
    
    # 安全转换：移除 NaN 和 inf，然后裁剪到 [0, 1]
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # 转换为 8 位
    image_8bit = (rgb_image * 255).astype(np.uint8)
    
    return image_8bit


# 使用全局统计信息将 RGB 三个波段归一化并合成为 8 位图像
def process_rgb_to_8bit_with_global_stats(red_band, green_band, blue_band, global_stats, method='clahe_global'):
    """
    使用全局统计信息将 RGB 三个波段归一化并合成为 8 位图像
    保证不同分块之间的一致性（强烈推荐用于分块处理）
    
    参数:
        red_band, green_band, blue_band: RGB 波段数据
        global_stats: 全局统计信息字典，包含：
            - 'clahe_global': {'r_min', 'r_max', 'g_min', 'g_max', 'b_min', 'b_max'}
            - 'percentile_global': {'r_low', 'r_high', 'g_low', 'g_high', 'b_low', 'b_high'}
            - 'percentile_rgb_global': {'p_low', 'p_high'}
        method: 归一化方法
            - 'clahe_global': 使用全局范围+CLAHE
            - 'percentile_global': 使用全局百分位拉伸（推荐）
            - 'percentile_rgb_global': RGB联合全局百分位拉伸（最推荐）
    
    返回: (H, W, 3) 的 uint8 数组
    """
    if method == 'clahe_global':
        stats = global_stats.get('clahe_global', {})
        red_norm = clahe_normalize_with_global_stats(
            red_band, stats.get('r_min', 0), stats.get('r_max', 65535)
        )
        green_norm = clahe_normalize_with_global_stats(
            green_band, stats.get('g_min', 0), stats.get('g_max', 65535)
        )
        blue_norm = clahe_normalize_with_global_stats(
            blue_band, stats.get('b_min', 0), stats.get('b_max', 65535)
        )
    
    elif method == 'percentile_global':
        stats = global_stats.get('percentile_global', {})
        r_low = stats.get('r_low', 0)
        r_high = stats.get('r_high', 65535)
        g_low = stats.get('g_low', 0)
        g_high = stats.get('g_high', 65535)
        b_low = stats.get('b_low', 0)
        b_high = stats.get('b_high', 65535)
        
        red_norm = np.clip((red_band.astype(np.float32) - r_low) / (r_high - r_low + 1e-6), 0, 1)
        green_norm = np.clip((green_band.astype(np.float32) - g_low) / (g_high - g_low + 1e-6), 0, 1)
        blue_norm = np.clip((blue_band.astype(np.float32) - b_low) / (b_high - b_low + 1e-6), 0, 1)
    
    elif method == 'percentile_rgb_global':
        stats = global_stats.get('percentile_rgb_global', {})
        p_low = stats.get('p_low', 0)
        p_high = stats.get('p_high', 65535)
        
        if p_high > p_low:
            red_norm = np.clip((red_band.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
            green_norm = np.clip((green_band.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
            blue_norm = np.clip((blue_band.astype(np.float32) - p_low) / (p_high - p_low), 0, 1)
        else:
            red_norm = np.zeros_like(red_band, dtype=np.float32)
            green_norm = np.zeros_like(green_band, dtype=np.float32)
            blue_norm = np.zeros_like(blue_band, dtype=np.float32)
    
    else:
        raise ValueError(f"Unknown global normalization method: {method}")
    
    # 堆叠为 RGB 图像
    rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
    
    # 安全转换
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)
    rgb_image = np.clip(rgb_image, 0, 1)
    
    # 转换为 8 位
    image_8bit = (rgb_image * 255).astype(np.uint8)
    
    return image_8bit


def process_multispectral_to_8bit(bands, global_stats=None, method='clahe'):
    """
    Convert fused multispectral bands to uint8.
    - Accepts [B, G, R] or [B, G, R, NIR]
    - Applies RGB normalization with existing pipeline (local/global)
    - If NIR is available, normalize it and fuse it to a 4th channel while
      also using it to gently pull back overexposed highlights for better display.
    """
    if len(bands) < 3:
        raise ValueError("At least 3 bands (B, G, R) are required to build an RGB image.")

    # Use global stats when available, otherwise fall back to local normalization.
    use_global = global_stats and method in ['clahe_global', 'percentile_global', 'percentile_rgb_global']
    if use_global:
        rgb_8bit = process_rgb_to_8bit_with_global_stats(
            bands[2], bands[1], bands[0],
            global_stats=global_stats,
            method=method
        )
    else:
        # If a global-only method is selected without stats, degrade gracefully to CLAHE.
        local_method = method if method not in ['clahe_global', 'percentile_global', 'percentile_rgb_global'] else 'clahe'
        rgb_8bit = process_rgb_to_8bit(
            bands[2], bands[1], bands[0],
            method=local_method
        )

    # No NIR channel -> return RGB directly.
    if len(bands) == 3:
        return rgb_8bit

    # Normalize NIR band to [0, 1].
    nir_band = bands[3]
    if use_global:
        nir_stats = global_stats.get('nir_global', {})
        nir_min = nir_stats.get('nir_min', float(np.min(nir_band)))
        nir_max = nir_stats.get('nir_max', float(np.max(nir_band)))
        if nir_stats.get('nir_low') is not None and nir_stats.get('nir_high') is not None:
            nir_low = nir_stats.get('nir_low')
            nir_high = nir_stats.get('nir_high')
            nir_norm = np.clip((nir_band.astype(np.float32) - nir_low) / (nir_high - nir_low + 1e-6), 0, 1)
        else:
            nir_norm = clahe_normalize_with_global_stats(nir_band, nir_min, nir_max)
    else:
        nir_norm = percentile_normalize(nir_band)

    # Use NIR to gently recover highlights for better visualization.
    rgb_enhanced = nir_overexposure_fusion(rgb_8bit.astype(np.float32) / 255.0, nir_norm)
    nir_8bit = (nir_norm * 255).astype(np.uint8)

    return np.dstack([rgb_enhanced, nir_8bit])

# 使用 PAN 和 MSS 的 RPC 文件在影像中心估计全局偏移
def estimate_offset_from_rpcs(pan_rpc, mss_rpc, pan_size, mss_size):
    """
    使用 PAN 和 MSS 的 RPC 文件在影像中心估计全局偏移.
    返回: (dx_pan, dy_pan) - 以 PAN 像素为单位的偏移.
           正值表示 MSS 内容相对 PAN 向右/下偏移.
    """
    # 使用 RPC 中心点作为参考地标
    lon = pan_rpc.get('longOffset', 0)
    lat = pan_rpc.get('latOffset', 0)
    h = pan_rpc.get('heightOffset', 0)

    # 将地标投影到 PAN 和 MSS 影像
    pan_x, pan_y = ground_to_image_rpc(lon, lat, h, pan_rpc)
    mss_x, mss_y = ground_to_image_rpc(lon, lat, h, mss_rpc)

    # 将 MSS 坐标缩放到 PAN 像素空间
    pan_w, pan_h = pan_size
    mss_w, mss_h = mss_size
    scale_x = pan_w / mss_w
    scale_y = pan_h / mss_h
    mss_x_in_pan_space = mss_x * scale_x
    mss_y_in_pan_space = mss_y * scale_y
    
    # 计算偏移量 (以 PAN 像素为单位)
    dx_pan = mss_x_in_pan_space - pan_x
    dy_pan = mss_y_in_pan_space - pan_y
    
    return dx_pan, dy_pan


# --- 智能裁剪相关函数 ---
def calculate_smart_crop_window(polygon_center_px, target_size, img_width, img_height, 
                                polygon_bounds=None, min_coverage_ratio=0.1):
    """
    智能计算裁剪窗口，避免大片空白并自适应调整
    
    参数:
        polygon_center_px: 多边形中心像素坐标 (x, y)
        target_size: 目标裁剪尺寸 (width, height) 或单个值（正方形）
        img_width: 影像宽度
        img_height: 影像高度
        polygon_bounds: 多边形边界 (min_x, min_y, max_x, max_y)，可选
        min_coverage_ratio: 最小覆盖率，如果多边形占比太小则缩小窗口
    
    返回:
        window: (x, y, width, height) 裁剪窗口
        adjusted: 是否进行了调整
    """
    center_x, center_y = polygon_center_px
    
    # 处理target_size参数
    if isinstance(target_size, (int, float)):
        target_w = target_h = int(target_size)
    else:
        target_w, target_h = int(target_size[0]), int(target_size[1])
    
    # 初始窗口（以中心为基准）
    half_w = target_w // 2
    half_h = target_h // 2
    x = center_x - half_w
    y = center_y - half_h
    
    adjusted = False
    
    # 策略1：检查并调整窗口位置（保持尺寸）
    original_w, original_h = target_w, target_h
    
    # 左边界检查
    if x < 0:
        x = 0
        adjusted = True
    
    # 上边界检查
    if y < 0:
        y = 0
        adjusted = True
    
    # 右边界检查
    if x + target_w > img_width:
        x = max(0, img_width - target_w)
        adjusted = True
    
    # 下边界检查
    if y + target_h > img_height:
        y = max(0, img_height - target_h)
        adjusted = True
    
    # 策略2：如果移动后仍然超出，缩小窗口尺寸
    if x + target_w > img_width:
        target_w = img_width - x
        adjusted = True
    
    if y + target_h > img_height:
        target_h = img_height - y
        adjusted = True
    
    # 策略3：如果有多边形边界信息，检查覆盖率
    if polygon_bounds is not None:
        poly_min_x, poly_min_y, poly_max_x, poly_max_y = polygon_bounds
        poly_width = poly_max_x - poly_min_x
        poly_height = poly_max_y - poly_min_y
        poly_area = poly_width * poly_height
        window_area = target_w * target_h
        
        if window_area > 0:
            coverage_ratio = poly_area / window_area
            
            # 如果多边形太小，尝试缩小窗口以提高覆盖率
            if coverage_ratio < min_coverage_ratio:
                # 计算合适的窗口尺寸（多边形占30%左右）
                scale_factor = np.sqrt(coverage_ratio / 0.3)
                new_w = int(target_w * scale_factor)
                new_h = int(target_h * scale_factor)
                
                # 确保不小于多边形的1.5倍
                new_w = max(new_w, int(poly_width * 1.5))
                new_h = max(new_h, int(poly_height * 1.5))
                
                # 确保不超过原始尺寸
                new_w = min(new_w, original_w)
                new_h = min(new_h, original_h)
                
                # 重新计算窗口位置
                x = int(center_x - new_w // 2)
                y = int(center_y - new_h // 2)
                
                # 边界检查
                x = max(0, min(x, img_width - new_w))
                y = max(0, min(y, img_height - new_h))
                
                target_w = new_w
                target_h = new_h
                adjusted = True
    
    # 最终边界限制
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    target_w = max(1, min(target_w, img_width - x))
    target_h = max(1, min(target_h, img_height - y))
    
    return (int(x), int(y), int(target_w), int(target_h)), adjusted


def get_polygon_pixel_bounds(pixel_coords):
    """
    获取多边形的像素边界
    
    参数:
        pixel_coords: 像素坐标列表 [[x1, y1], [x2, y2], ...]
    
    返回:
        (min_x, min_y, max_x, max_y)
    """
    xs = [c[0] for c in pixel_coords]
    ys = [c[1] for c in pixel_coords]
    return (min(xs), min(ys), max(xs), max(ys))
