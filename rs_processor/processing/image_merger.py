"""
分块合成模块

提供分块图像的合成功能。
"""

from typing import Tuple, Optional, List
import numpy as np

from ..core.tiff_io import ImageMetadata, write_tiff, write_png
from .tile_processor import TileInfo


class ImageMerger:
    """图像合成器"""
    
    def __init__(
        self,
        output_size: Tuple[int, int],
        num_bands: int = 3,
        overlap: int = 200,
        blend_mode: str = 'linear'
    ):
        """初始化图像合成器
        
        Args:
            output_size: 输出图像尺寸 (width, height)
            num_bands: 波段数量
            overlap: 重叠区域大小
            blend_mode: 混合模式 ('linear', 'max', 'average')
        """
        self.output_size = output_size
        self.num_bands = num_bands
        self.overlap = overlap
        self.blend_mode = blend_mode
        
        width, height = output_size
        self._canvas = np.zeros((num_bands, height, width), dtype=np.float32)
        self._weight = np.zeros((height, width), dtype=np.float32)
        self._tiles: List[Tuple[np.ndarray, TileInfo]] = []
    
    def add_tile(self, tile_data: np.ndarray, tile_info: TileInfo) -> None:
        """添加一个分块到合成器"""
        self._tiles.append((tile_data, tile_info))
        
        # 直接合成到画布
        x, y = tile_info.x, tile_info.y
        h, w = tile_data.shape[1], tile_data.shape[2]
        
        # 创建权重掩膜（边缘渐变）
        weight = self._create_weight_mask(w, h)
        
        # 累加到画布
        for b in range(min(self.num_bands, tile_data.shape[0])):
            self._canvas[b, y:y+h, x:x+w] += tile_data[b] * weight
        
        self._weight[y:y+h, x:x+w] += weight
    
    def _create_weight_mask(self, width: int, height: int) -> np.ndarray:
        """创建权重掩膜（边缘渐变）"""
        if self.blend_mode == 'linear':
            # 创建边缘渐变
            mask = np.ones((height, width), dtype=np.float32)
            
            # 边缘渐变区域
            fade = min(self.overlap, width // 4, height // 4)
            
            if fade > 0:
                # 左边缘
                for i in range(fade):
                    mask[:, i] *= i / fade
                # 右边缘
                for i in range(fade):
                    mask[:, width - 1 - i] *= i / fade
                # 上边缘
                for i in range(fade):
                    mask[i, :] *= i / fade
                # 下边缘
                for i in range(fade):
                    mask[height - 1 - i, :] *= i / fade
            
            return mask
        else:
            return np.ones((height, width), dtype=np.float32)
    
    def merge(self) -> np.ndarray:
        """合成所有分块，返回完整图像"""
        # 归一化
        weight_safe = np.where(self._weight > 0, self._weight, 1.0)
        
        result = np.zeros_like(self._canvas)
        for b in range(self.num_bands):
            result[b] = self._canvas[b] / weight_safe
        
        return result
    
    def save(
        self,
        output_path: str,
        metadata: Optional[ImageMetadata] = None
    ) -> None:
        """保存合成结果（支持 TIFF/PNG）"""
        merged = self.merge()
        
        if output_path.lower().endswith('.png'):
            # 转换为 uint8
            merged_uint8 = np.clip(merged / 65535 * 255, 0, 255).astype(np.uint8)
            # 转换为 (H, W, C) 格式
            if merged_uint8.shape[0] <= 4:
                merged_uint8 = np.transpose(merged_uint8, (1, 2, 0))
            write_png(output_path, merged_uint8)
        else:
            write_tiff(output_path, merged.astype(np.uint16), metadata)
