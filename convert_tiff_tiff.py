import os
import numpy as np
import cv2
from osgeo import gdal

# 注册 GDAL 驱动
gdal.AllRegister()
gdal.UseExceptions()

# 设置 OpenCV 读取大图限制
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '5368709120'

# ================= 1. 我们自己的图像增强方法 (保持不变) =================

def log_normalize(band, lower=5, upper=95, adjust_factor=0.85):
    """对数变换归一化"""
    band = band.astype(np.float32)
    p_low, p_high = np.percentile(band, (lower, upper))
    
    if p_high == p_low:
        return np.zeros_like(band, dtype=np.uint8)
        
    normalized = (np.log1p(band) - np.log1p(p_low)) / (np.log1p(p_high) - np.log1p(p_low))
    normalized = np.clip(normalized * adjust_factor, 0, 1) * 255.0
    return normalized.astype(np.uint8)

def clahe_normalize(band, clipLimit=3.0, tileGridSize=(8, 8)):
    """CLAHE 归一化"""
    band_norm = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
    band_uint8 = band_norm.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(band_uint8)

def enhance_and_fuse(blue, green, red, nir=None):
    """
    核心增强逻辑：CLAHE + HSV融合 (如果有NIR)
    """
    # 1. 基础增强
    b_norm = clahe_normalize(blue)
    g_norm = clahe_normalize(green)
    r_norm = clahe_normalize(red)
    
    rgb_8bit = np.dstack((r_norm, g_norm, b_norm))
    
    if nir is None:
        return rgb_8bit
        
    # 2. 如果有 NIR，执行 HSV 融合
    nir_norm = clahe_normalize(nir)
    
    hsv_image = cv2.cvtColor(rgb_8bit, cv2.COLOR_RGB2HSV)
    h, s, v_orig = cv2.split(hsv_image)
    
    # 简单的过曝融合策略
    overexposed_mask = (v_orig > 235).astype(np.float32)
    smooth_mask = cv2.GaussianBlur(overexposed_mask, (7, 7), 0)
    
    v_nir = nir_norm.astype(np.float32)
    v_new_float = v_orig.astype(np.float32) * (1 - smooth_mask) + v_nir * smooth_mask
    v_new = np.clip(v_new_float, 0, 255).astype(np.uint8)
    
    fused_hsv = cv2.merge([h, s, v_new])
    fused_rgb = cv2.cvtColor(fused_hsv, cv2.COLOR_HSV2RGB)
    
    return fused_rgb

# ================= 2. GDAL 处理流程 (仅复制，不校正) =================

def process_single_image(input_path, output_dir):
    """
    读取 -> 增强 -> 写入 (复制元数据，不进行几何校正)
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_tif = os.path.join(output_dir, f"{base_name}.tif")
    
    print(f"[{base_name}] 开始处理...")
    
    # --- Step 1: 读取 ---
    ds = gdal.Open(input_path)
    if ds is None:
        print(f"  错误: 无法打开文件 {input_path}")
        return
    alpha = None
    width = ds.RasterXSize
    height = ds.RasterYSize
    num_bands = ds.RasterCount
    
    # 获取所有关键元数据 (这是保留坐标信息的关键)
    rpcs = ds.GetMetadata("RPC")          # 获取 RPC
    geo_transform = ds.GetGeoTransform()  # 获取仿射变换
    projection = ds.GetProjection()       # 获取投影信息
    
    image_8bit = None

    # --- Step 2: 增强 (使用我们的方法) ---
    if num_bands >= 3:
        # 假设波段顺序: 1=Blue, 2=Green, 3=Red (根据实际情况调整)
        b_blue = ds.GetRasterBand(1).ReadAsArray()
        b_green = ds.GetRasterBand(2).ReadAsArray()
        b_red = ds.GetRasterBand(3).ReadAsArray()
        
        b_nir = None
        if num_bands >= 4:
            b_nir = ds.GetRasterBand(4).ReadAsArray()
        if num_bands == 5 or num_bands == 9:
            alpha = ds.GetRasterBand(num_bands).ReadAsArray()
            
        image_8bit = enhance_and_fuse(b_blue, b_green, b_red, b_nir)
        
    elif num_bands <= 2:
        # 全色
        b_pan = ds.GetRasterBand(1).ReadAsArray()
        pan_norm = log_normalize(b_pan)
        image_8bit = np.dstack((pan_norm, pan_norm, pan_norm))
        if num_bands == 2:
            alpha = ds.GetRasterBand(num_bands).ReadAsArray()
        
    else:
        print(f"  跳过: 波段数 {num_bands} 不支持")
        ds = None
        return

    # --- Step 3: 写入 (直接保存为 TIF) ---
    driver = gdal.GetDriverByName('GTiff')
    if alpha is not None:
        out_ds = driver.Create(output_tif, width, height, 4, gdal.GDT_Byte)
    else:
        out_ds = driver.Create(output_tif, width, height, 3, gdal.GDT_Byte)
    
    # 写入增强后的像素
    out_ds.GetRasterBand(1).WriteArray(image_8bit[:, :, 0])
    out_ds.GetRasterBand(2).WriteArray(image_8bit[:, :, 1])
    out_ds.GetRasterBand(3).WriteArray(image_8bit[:, :, 2])
    if alpha is not None:
        out_ds.GetRasterBand(4).WriteArray(alpha)
        out_ds.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)
    # --- 关键：原样复制元数据 ---
    # 如果不做这一步，输出的图就没有坐标信息了
    out_ds.GetRasterBand(1).SetNoDataValue(0)
    out_ds.GetRasterBand(2).SetNoDataValue(0)
    out_ds.GetRasterBand(3).SetNoDataValue(0)
    if alpha is not None:
        out_ds.GetRasterBand(4).SetNoDataValue(0)
        
    if rpcs:
        # 如果原图是 RPC 定位，复制 RPC
        out_ds.SetMetadata(rpcs, "RPC")
        print("  已保留原始 RPC 信息。")
    
    # 无论有没有 RPC，都建议复制 GeoTransform 和 Projection (有些图二者都有)
    if geo_transform:
        out_ds.SetGeoTransform(geo_transform)
    if projection:
        out_ds.SetProjection(projection)
        
    out_ds.FlushCache()
    out_ds = None # 关闭文件，写入磁盘
    ds = None     # 释放源文件
    
    print(f"  处理完成: {output_tif}")

# --- 3. 批量处理入口 ---
if __name__ == "__main__":
    # 主文件夹路径
    input_dir = r"/home/MXJ/data/sample_data_proj"
    output_dir = r"/home/MXJ/data/sample_tiff1"
    if not os.path.exists(input_dir):
        print("输入目录不存在")

    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_nir_dir, exist_ok=True)

    extensions = ("-NAD.tiff", "-MSS1.tiff", "-BWDMUX.tiff", "-MSS2.tiff","-MUX.tiff","-MSS.tiff")
    # extensions = ()
    # 遍历主文件夹及其所有子文件夹
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            # 检查文件名是否符合模式
            if filename.endswith(extensions):  # -NAD.tiff -BWDMUX.tiff
                # 构造完整路径
                full_path = os.path.join(dirpath, filename)
                # 调用函数进行转换
                process_single_image(full_path, output_dir)