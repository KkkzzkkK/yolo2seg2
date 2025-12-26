"""
Core Layer - 基础图像处理功能

模块：
- rpc_utils: RPC 坐标转换
- pan_sharpen: 全色锐化（Gram-Schmidt）
- normalize: 图像归一化（CLAHE）
- tiff_io: TIFF 读写与元数据保留
"""

from .rpc_utils import (
    RPCParams,
    RPCParseError,
    parse_rpb_file,
    ground_to_image,
    image_to_ground,
    get_image_geo_bounds,
    estimate_offset_from_rpcs,
)
from .pan_sharpen import calculate_band_correlations, gram_schmidt_sharpen
from .normalize import clahe_normalize, bands_to_rgb_uint8
from .tiff_io import ImageMetadata, TiffIOError, read_tiff, write_tiff, write_png

__all__ = [
    # rpc_utils
    "RPCParams",
    "RPCParseError",
    "parse_rpb_file",
    "ground_to_image",
    "image_to_ground",
    "get_image_geo_bounds",
    "estimate_offset_from_rpcs",
    # pan_sharpen
    "calculate_band_correlations",
    "gram_schmidt_sharpen",
    # normalize
    "clahe_normalize",
    "bands_to_rgb_uint8",
    # tiff_io
    "ImageMetadata",
    "TiffIOError",
    "read_tiff",
    "write_tiff",
    "write_png",
]
