"""
裁剪导出模块

提供基于检测框的裁剪和导出功能。
"""

import os
import json
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from .label_processor import Detection
from .coord_converter import CropMetadata

logger = logging.getLogger(__name__)


class CropExporter:
    """裁剪导出器"""
    
    def __init__(
        self,
        crop_multiple: int = 64,
        min_crop_size: int = 256
    ):
        """初始化裁剪导出器
        
        Args:
            crop_multiple: 裁剪尺寸的倍数（用于对齐）
            min_crop_size: 最小裁剪尺寸
        """
        self.crop_multiple = crop_multiple
        self.min_crop_size = min_crop_size
    
    def compute_crop_window(
        self,
        detection: Detection,
        image_size: Tuple[int, int],
        scale: float = 1.3
    ) -> Tuple[int, int, int, int]:
        """计算裁剪窗口
        
        Args:
            detection: 检测框
            image_size: 影像尺寸 (width, height)
            scale: 扩展比例
            
        Returns:
            (x, y, width, height): 裁剪窗口
        """
        img_width, img_height = image_size
        
        # 计算检测框的像素坐标
        pixel_coords = [
            (x * img_width, y * img_height)
            for x, y in detection.polygon_norm
        ]
        
        # 计算边界框
        xs = [p[0] for p in pixel_coords]
        ys = [p[1] for p in pixel_coords]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # 计算中心和尺寸
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        box_width = max_x - min_x
        box_height = max_y - min_y
        
        # 扩展尺寸
        crop_width = max(box_width * scale, self.min_crop_size)
        crop_height = max(box_height * scale, self.min_crop_size)
        
        # 对齐到倍数
        crop_width = int(np.ceil(crop_width / self.crop_multiple) * self.crop_multiple)
        crop_height = int(np.ceil(crop_height / self.crop_multiple) * self.crop_multiple)
        
        # 计算裁剪窗口
        x = int(center_x - crop_width / 2)
        y = int(center_y - crop_height / 2)
        
        # 边界检查
        x = max(0, min(x, img_width - crop_width))
        y = max(0, min(y, img_height - crop_height))
        
        # 确保不超出边界
        crop_width = min(crop_width, img_width - x)
        crop_height = min(crop_height, img_height - y)
        
        return (x, y, crop_width, crop_height)
    
    def crop_and_save(
        self,
        image_path: str,
        detection: Detection,
        output_dir: str,
        det_idx: int,
        source_image_size: Optional[Tuple[int, int]] = None
    ) -> Optional[CropMetadata]:
        """裁剪并保存
        
        Args:
            image_path: 源图像路径
            detection: 检测框
            output_dir: 输出目录
            det_idx: 检测框索引
            source_image_size: 源图像尺寸（可选）
            
        Returns:
            裁剪元数据，如果失败返回 None
        """
        try:
            # 打开图像
            img = Image.open(image_path)
            img_width, img_height = img.size
            
            if source_image_size is None:
                source_image_size = (img_width, img_height)
            
            # 计算裁剪窗口
            x, y, crop_width, crop_height = self.compute_crop_window(
                detection, (img_width, img_height)
            )
            
            # 检查有效性
            if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                logger.warning(f"检测框 {det_idx} 裁剪区域过小，跳过")
                return None
            
            # 裁剪
            cropped = img.crop((x, y, x + crop_width, y + crop_height))
            
            # 保存
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"crop_{det_idx}.png")
            cropped.save(output_path, 'PNG')
            
            # 创建元数据
            metadata = CropMetadata(
                det_idx=det_idx,
                global_offset_x=x,
                global_offset_y=y,
                crop_width=crop_width,
                crop_height=crop_height,
                image_size=source_image_size,
                class_id=detection.class_id,
                score=detection.score,
                source_image=image_path,
            )
            
            # 保存元数据
            metadata_path = os.path.join(output_dir, f"crop_{det_idx}_info.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
            return metadata
            
        except Exception as e:
            logger.error(f"裁剪检测框 {det_idx} 失败: {e}")
            return None
    
    def batch_crop(
        self,
        image_path: str,
        detections: List[Detection],
        output_dir: str
    ) -> List[CropMetadata]:
        """批量裁剪
        
        Args:
            image_path: 源图像路径
            detections: 检测框列表
            output_dir: 输出目录
            
        Returns:
            成功裁剪的元数据列表
        """
        results = []
        
        # 获取图像尺寸
        try:
            img = Image.open(image_path)
            source_size = img.size
            img.close()
        except Exception as e:
            logger.error(f"无法打开图像 {image_path}: {e}")
            return results
        
        for det_idx, detection in enumerate(detections):
            metadata = self.crop_and_save(
                image_path, detection, output_dir, det_idx, source_size
            )
            if metadata:
                results.append(metadata)
        
        return results
