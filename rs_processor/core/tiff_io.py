"""
TIFF 读写模块

提供 TIFF 文件读写和元数据保留功能。
支持窗口化读取以处理大图。
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image

from .rpc_utils import RPCParams, parse_rpb_file


class TiffIOError(Exception):
    """TIFF 读写错误"""
    pass


@dataclass
class ImageMetadata:
    """影像元数据
    
    包含 TIFF 文件的地理参考信息和基本属性。
    """
    width: int
    height: int
    bands: int
    dtype: str = 'uint16'
    rpc: Optional[RPCParams] = None
    geo_transform: Optional[Tuple[float, ...]] = None
    projection: Optional[str] = None
    crs: Optional[str] = None
    nodata: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'width': self.width,
            'height': self.height,
            'bands': self.bands,
            'dtype': self.dtype,
            'rpc': self.rpc.to_dict() if self.rpc else None,
            'geo_transform': self.geo_transform,
            'projection': self.projection,
            'crs': self.crs,
            'nodata': self.nodata,
        }


def _find_rpb_file(tiff_path: str) -> Optional[str]:
    """查找与 TIFF 文件对应的 RPB 文件"""
    base_path = os.path.splitext(tiff_path)[0]
    
    # 尝试不同的扩展名
    for ext in ['.rpb', '.RPB', '.RPC', '.rpc']:
        rpb_path = base_path + ext
        if os.path.exists(rpb_path):
            return rpb_path
    
    return None


def read_tiff(
    path: str,
    window: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, ImageMetadata]:
    """读取 TIFF 文件，支持窗口化读取
    
    Args:
        path: TIFF 文件路径
        window: 读取窗口 (col_off, row_off, width, height)，None 表示读取整个文件
        
    Returns:
        data: 影像数据 (bands, H, W) 或 (H, W) 单波段
        metadata: 影像元数据
        
    Raises:
        TiffIOError: 文件不存在或格式错误
    """
    if not os.path.exists(path):
        raise TiffIOError(f"TIFF 文件不存在: {path}")
    
    try:
        with rasterio.open(path) as src:
            # 构建读取窗口
            if window is not None:
                col_off, row_off, width, height = window
                # 裁剪到有效范围
                col_off = max(0, min(col_off, src.width - 1))
                row_off = max(0, min(row_off, src.height - 1))
                width = min(width, src.width - col_off)
                height = min(height, src.height - row_off)
                rio_window = Window(col_off, row_off, width, height)
            else:
                rio_window = None
                width = src.width
                height = src.height
            
            # 读取数据
            data = src.read(window=rio_window)
            
            # 如果是单波段，去掉第一个维度
            if data.shape[0] == 1:
                data = data[0]
            
            # 提取元数据
            geo_transform = src.transform.to_gdal() if src.transform else None
            
            # 尝试加载 RPC 参数
            rpc = None
            rpb_path = _find_rpb_file(path)
            if rpb_path:
                try:
                    rpc = parse_rpb_file(rpb_path)
                except Exception:
                    pass  # 忽略 RPC 解析错误
            
            metadata = ImageMetadata(
                width=width,
                height=height,
                bands=src.count,
                dtype=str(src.dtypes[0]),
                rpc=rpc,
                geo_transform=geo_transform,
                projection=src.crs.to_wkt() if src.crs else None,
                crs=str(src.crs) if src.crs else None,
                nodata=src.nodata,
            )
            
            return data, metadata
            
    except rasterio.errors.RasterioIOError as e:
        raise TiffIOError(f"无法读取 TIFF 文件: {e}")


def write_tiff(
    path: str,
    data: np.ndarray,
    metadata: Optional[ImageMetadata] = None
) -> None:
    """写入 TIFF 文件，保留元数据
    
    Args:
        path: 输出文件路径
        data: 影像数据 (bands, H, W) 或 (H, W) 单波段
        metadata: 影像元数据，用于保留地理参考信息
        
    Raises:
        TiffIOError: 写入失败
    """
    # 确保数据是 3D 的
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    bands, height, width = data.shape
    
    # 确定数据类型
    dtype_map = {
        'uint8': rasterio.uint8,
        'uint16': rasterio.uint16,
        'int16': rasterio.int16,
        'uint32': rasterio.uint32,
        'int32': rasterio.int32,
        'float32': rasterio.float32,
        'float64': rasterio.float64,
    }
    
    if data.dtype.name in dtype_map:
        rio_dtype = dtype_map[data.dtype.name]
    else:
        rio_dtype = rasterio.float32
        data = data.astype(np.float32)
    
    # 构建写入参数
    profile = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': bands,
        'dtype': rio_dtype,
        'compress': 'lzw',
    }
    
    # 添加元数据
    if metadata:
        if metadata.geo_transform:
            from rasterio.transform import Affine
            # GDAL 格式: (x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size)
            gt = metadata.geo_transform
            profile['transform'] = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        
        if metadata.crs:
            from rasterio.crs import CRS
            try:
                profile['crs'] = CRS.from_string(metadata.crs)
            except Exception:
                pass
        
        if metadata.nodata is not None:
            profile['nodata'] = metadata.nodata
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data)
            
    except Exception as e:
        raise TiffIOError(f"无法写入 TIFF 文件: {e}")


def write_png(path: str, data: np.ndarray) -> None:
    """写入 PNG 文件
    
    Args:
        path: 输出文件路径
        data: 影像数据 (H, W) 灰度、(H, W, 3) RGB 或 (H, W, 4) RGBA
              数据类型应为 uint8
              
    Raises:
        TiffIOError: 写入失败
    """
    # 确保数据是 uint8
    if data.dtype != np.uint8:
        if data.max() > 255:
            # 假设是 16 位数据，归一化到 8 位
            data = (data / data.max() * 255).astype(np.uint8)
        else:
            data = data.astype(np.uint8)
    
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        # 使用 PIL 保存
        if data.ndim == 2:
            mode = 'L'
        elif data.shape[2] == 3:
            mode = 'RGB'
        elif data.shape[2] == 4:
            mode = 'RGBA'
        else:
            raise TiffIOError(f"不支持的通道数: {data.shape[2]}")
        
        img = Image.fromarray(data, mode=mode)
        img.save(path, 'PNG')
        
    except Exception as e:
        raise TiffIOError(f"无法写入 PNG 文件: {e}")
