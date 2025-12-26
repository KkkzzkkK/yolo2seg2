"""
分块处理模块

提供大尺寸影像的分块处理功能。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np

from ..core.rpc_utils import RPCParams
from ..core.pan_sharpen import gram_schmidt_sharpen
from .registration import RegistrationProcessor, RegistrationOffset


@dataclass
class TileInfo:
    """分块信息"""
    x: int
    y: int
    width: int
    height: int
    tile_idx: int
    contains_annotations: List[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height,
            'tile_idx': self.tile_idx,
            'contains_annotations': self.contains_annotations,
        }


class TileProcessor:
    """分块处理器"""
    
    def __init__(
        self,
        tile_size: int = 8000,
        overlap: int = 200,
        min_annotation_margin: int = 100
    ):
        """初始化分块处理器"""
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_annotation_margin = min_annotation_margin
    
    def generate_tiles(
        self,
        img_width: int,
        img_height: int,
        annotations: Optional[List[Dict]] = None
    ) -> List[TileInfo]:
        """生成分块列表
        
        Args:
            img_width: 影像宽度
            img_height: 影像高度
            annotations: 标注列表（可选，用于标注优先策略）
            
        Returns:
            分块信息列表
        """
        tiles = []
        tile_idx = 0
        
        # 计算步长（考虑重叠）
        step = self.tile_size - self.overlap
        
        y = 0
        while y < img_height:
            x = 0
            while x < img_width:
                # 计算分块尺寸
                width = min(self.tile_size, img_width - x)
                height = min(self.tile_size, img_height - y)
                
                # 查找包含的标注
                contained_annotations = []
                if annotations:
                    for ann_idx, ann in enumerate(annotations):
                        if self._annotation_in_tile(ann, x, y, width, height):
                            contained_annotations.append(ann_idx)
                
                tiles.append(TileInfo(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    tile_idx=tile_idx,
                    contains_annotations=contained_annotations,
                ))
                
                tile_idx += 1
                x += step
            
            y += step
        
        return tiles
    
    def _annotation_in_tile(
        self,
        annotation: Dict,
        tile_x: int,
        tile_y: int,
        tile_width: int,
        tile_height: int
    ) -> bool:
        """检查标注是否在分块内"""
        # 假设标注有 bbox 或 polygon 字段
        bbox = annotation.get('bbox')
        if bbox:
            ax, ay, aw, ah = bbox
            # 检查是否有交集
            return not (ax + aw < tile_x or ax > tile_x + tile_width or
                       ay + ah < tile_y or ay > tile_y + tile_height)
        return False
    
    def filter_annotations_in_tile(
        self,
        annotations: List[Dict],
        tile: TileInfo
    ) -> List[Dict]:
        """过滤分块内的标注，转换为相对坐标"""
        filtered = []
        
        for ann in annotations:
            bbox = ann.get('bbox')
            if bbox:
                ax, ay, aw, ah = bbox
                
                # 检查是否在分块内
                if self._annotation_in_tile(ann, tile.x, tile.y, tile.width, tile.height):
                    # 转换为相对坐标
                    new_ann = ann.copy()
                    new_ann['bbox'] = (ax - tile.x, ay - tile.y, aw, ah)
                    filtered.append(new_ann)
        
        return filtered
    
    def process_tile(
        self,
        pan_data: np.ndarray,
        mss_data: np.ndarray,
        tile: TileInfo,
        pan_rpc: RPCParams,
        mss_rpc: RPCParams,
        registration_processor: RegistrationProcessor
    ) -> Tuple[np.ndarray, RegistrationOffset]:
        """处理单个分块：配准 + 锐化
        
        Args:
            pan_data: PAN 影像数据 (H, W)
            mss_data: MSS 影像数据 (bands, H, W)
            tile: 分块信息
            pan_rpc: PAN 的 RPC 参数
            mss_rpc: MSS 的 RPC 参数
            registration_processor: 配准处理器
            
        Returns:
            fused_data: 融合后的数据 (bands, H, W)
            offset_info: 配准偏移信息
        """
        # 配准
        pan_window = (tile.x, tile.y, tile.width, tile.height)
        result = registration_processor.register(
            pan_data, pan_rpc, mss_data, mss_rpc, pan_window
        )
        
        # 提取 PAN 分块
        pan_tile = pan_data[tile.y:tile.y+tile.height, tile.x:tile.x+tile.width]
        
        # 全色锐化
        mss_bands = [result.aligned_mss[i] for i in range(result.aligned_mss.shape[0])]
        sharpened = gram_schmidt_sharpen(pan_tile, mss_bands)
        
        # 堆叠为数组
        fused_data = np.stack(sharpened, axis=0)
        
        return fused_data, result.offset_info
