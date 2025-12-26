"""
标签处理模块

提供 YOLO-OBB 标签读取和坐标转换功能。
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """检测框
    
    存储 YOLO-OBB 格式的检测结果。
    """
    class_id: str
    polygon_norm: List[Tuple[float, float]]  # 归一化坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    score: Optional[float] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'class_id': self.class_id,
            'polygon_norm': [list(p) for p in self.polygon_norm],
            'score': self.score,
        }


class LabelProcessor:
    """标签处理器
    
    提供 YOLO-OBB 标签的读取和坐标转换功能。
    """
    
    @staticmethod
    def read_yolo_obb(label_path: str) -> List[Detection]:
        """读取 YOLO-OBB 标签文件
        
        YOLO-OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 [score]
        坐标为归一化坐标 [0, 1]
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            检测框列表
        """
        detections = []
        
        if not os.path.exists(label_path):
            logger.warning(f"标签文件不存在: {label_path}")
            return detections
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            logger.warning(f"无法读取标签文件 {label_path}: {e}")
            return detections
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # 至少需要 class_id + 8 个坐标值
            if len(parts) < 9:
                logger.warning(f"第 {line_num} 行格式错误，跳过: {line}")
                continue
            
            try:
                class_id = parts[0]
                
                # 解析 8 个坐标值
                coords = [float(parts[i]) for i in range(1, 9)]
                
                # 转换为点列表
                polygon_norm = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
                
                # 可选的置信度
                score = float(parts[9]) if len(parts) > 9 else None
                
                detections.append(Detection(
                    class_id=class_id,
                    polygon_norm=polygon_norm,
                    score=score,
                ))
                
            except (ValueError, IndexError) as e:
                logger.warning(f"第 {line_num} 行解析错误，跳过: {e}")
                continue
        
        return detections
    
    @staticmethod
    def norm_to_pixel(
        polygon_norm: List[Tuple[float, float]],
        width: int,
        height: int
    ) -> List[Tuple[float, float]]:
        """归一化坐标 -> 像素坐标
        
        Args:
            polygon_norm: 归一化坐标列表 [(x1,y1), ...]
            width: 影像宽度
            height: 影像高度
            
        Returns:
            像素坐标列表
        """
        return [(x * width, y * height) for x, y in polygon_norm]
    
    @staticmethod
    def pixel_to_norm(
        polygon_pixel: List[Tuple[float, float]],
        width: int,
        height: int
    ) -> List[Tuple[float, float]]:
        """像素坐标 -> 归一化坐标
        
        Args:
            polygon_pixel: 像素坐标列表 [(x1,y1), ...]
            width: 影像宽度
            height: 影像高度
            
        Returns:
            归一化坐标列表
        """
        if width <= 0 or height <= 0:
            return [(0.0, 0.0) for _ in polygon_pixel]
        
        return [(x / width, y / height) for x, y in polygon_pixel]
    
    @staticmethod
    def local_to_global(
        polygon_local: List[Tuple[float, float]],
        offset_x: int,
        offset_y: int
    ) -> List[Tuple[float, float]]:
        """局部坐标 -> 全局坐标
        
        Args:
            polygon_local: 局部像素坐标列表
            offset_x: X 方向偏移
            offset_y: Y 方向偏移
            
        Returns:
            全局像素坐标列表
        """
        return [(x + offset_x, y + offset_y) for x, y in polygon_local]
    
    @staticmethod
    def write_yolo_obb(
        detections: List[Detection],
        output_path: str,
        include_score: bool = False
    ) -> None:
        """写入 YOLO-OBB 标签文件
        
        Args:
            detections: 检测框列表
            output_path: 输出文件路径
            include_score: 是否包含置信度
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for det in detections:
                coords = []
                for x, y in det.polygon_norm:
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])
                
                line = f"{det.class_id} " + " ".join(coords)
                
                if include_score and det.score is not None:
                    line += f" {det.score:.4f}"
                
                f.write(line + "\n")
