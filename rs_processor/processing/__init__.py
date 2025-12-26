"""
Processing Layer - 业务处理逻辑

模块：
- registration: 影像配准（RPC + 特征点）
- label_processor: YOLO 标签处理
- coord_converter: 坐标转换
- tile_processor: 分块处理
- image_merger: 分块合成
- crop_exporter: 裁剪导出
"""

from .registration import RegistrationProcessor, RegistrationResult, RegistrationOffset
from .label_processor import LabelProcessor, Detection
from .coord_converter import CoordConverter, FusionMetadata, CropMetadata
from .tile_processor import TileProcessor, TileInfo
from .image_merger import ImageMerger
from .crop_exporter import CropExporter

__all__ = [
    # registration
    "RegistrationProcessor",
    "RegistrationResult",
    "RegistrationOffset",
    # label_processor
    "LabelProcessor",
    "Detection",
    # coord_converter
    "CoordConverter",
    "FusionMetadata",
    "CropMetadata",
    # tile_processor
    "TileProcessor",
    "TileInfo",
    # image_merger
    "ImageMerger",
    # crop_exporter
    "CropExporter",
]
