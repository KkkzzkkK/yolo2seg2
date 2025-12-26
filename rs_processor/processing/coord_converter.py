"""
坐标转换模块

提供局部坐标到全局坐标的转换，以及像素坐标到经纬度的转换。
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from ..core.rpc_utils import RPCParams, image_to_ground, parse_rpb_file

logger = logging.getLogger(__name__)


@dataclass
class FusionMetadata:
    """融合元数据（从整图融合输出的 JSON 加载）"""
    pan_path: str
    mss_path: str
    pan_rpb_path: Optional[str]
    mss_rpb_path: Optional[str]
    output_path: str
    output_size: Tuple[int, int]
    coordinate_system: str  # 'pan' 或 'mss'
    pan_rpc: Optional[RPCParams] = None
    mss_rpc: Optional[RPCParams] = None
    registration_info: Dict = field(default_factory=dict)


@dataclass
class CropMetadata:
    """裁剪元数据"""
    det_idx: int
    global_offset_x: int
    global_offset_y: int
    crop_width: int
    crop_height: int
    image_size: Tuple[int, int]
    class_id: str
    score: Optional[float]
    source_image: str
    coordinate_system: str = 'pan'
    fusion_metadata_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'det_idx': self.det_idx,
            'global_offset_x': self.global_offset_x,
            'global_offset_y': self.global_offset_y,
            'crop_width': self.crop_width,
            'crop_height': self.crop_height,
            'image_size': list(self.image_size),
            'class_id': self.class_id,
            'score': self.score,
            'source_image': self.source_image,
            'coordinate_system': self.coordinate_system,
            'fusion_metadata_path': self.fusion_metadata_path,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CropMetadata':
        """从字典创建"""
        return cls(
            det_idx=d['det_idx'],
            global_offset_x=d['global_offset_x'],
            global_offset_y=d['global_offset_y'],
            crop_width=d['crop_width'],
            crop_height=d['crop_height'],
            image_size=tuple(d['image_size']),
            class_id=d['class_id'],
            score=d.get('score'),
            source_image=d['source_image'],
            coordinate_system=d.get('coordinate_system', 'pan'),
            fusion_metadata_path=d.get('fusion_metadata_path'),
        )


class CoordConverter:
    """坐标转换器"""
    
    def __init__(self, fusion_metadata: Optional[FusionMetadata] = None):
        """初始化坐标转换器"""
        self.fusion_metadata = fusion_metadata
        self._rpc: Optional[RPCParams] = None
        
        if fusion_metadata:
            # 根据 coordinate_system 选择 RPC
            if fusion_metadata.coordinate_system == 'pan' and fusion_metadata.pan_rpc:
                self._rpc = fusion_metadata.pan_rpc
            elif fusion_metadata.mss_rpc:
                self._rpc = fusion_metadata.mss_rpc
    
    @classmethod
    def from_metadata_file(cls, metadata_path: str) -> 'CoordConverter':
        """从融合元数据 JSON 文件创建转换器"""
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 解析 RPC 参数
        pan_rpc = None
        mss_rpc = None
        
        source = data.get('source', {})
        pan_rpb_path = source.get('pan_rpb_path')
        mss_rpb_path = source.get('mss_rpb_path')
        
        if pan_rpb_path and os.path.exists(pan_rpb_path):
            try:
                pan_rpc = parse_rpb_file(pan_rpb_path)
            except Exception as e:
                logger.warning(f"无法解析 PAN RPB: {e}")
        
        if mss_rpb_path and os.path.exists(mss_rpb_path):
            try:
                mss_rpc = parse_rpb_file(mss_rpb_path)
            except Exception as e:
                logger.warning(f"无法解析 MSS RPB: {e}")
        
        output = data.get('output', {})
        
        fusion_metadata = FusionMetadata(
            pan_path=source.get('pan_path', ''),
            mss_path=source.get('mss_path', ''),
            pan_rpb_path=pan_rpb_path,
            mss_rpb_path=mss_rpb_path,
            output_path=output.get('path', ''),
            output_size=(output.get('width', 0), output.get('height', 0)),
            coordinate_system=output.get('coordinate_system', 'pan'),
            pan_rpc=pan_rpc,
            mss_rpc=mss_rpc,
            registration_info=data.get('registration', {}),
        )
        
        return cls(fusion_metadata)
    
    def local_to_global_pixel(
        self,
        local_coords: List[Tuple[float, float]],
        crop_metadata: CropMetadata
    ) -> List[Tuple[float, float]]:
        """局部像素坐标 -> 全局像素坐标"""
        return [
            (x + crop_metadata.global_offset_x, y + crop_metadata.global_offset_y)
            for x, y in local_coords
        ]
    
    def pixel_to_geo(
        self,
        pixel_coords: List[Tuple[float, float]],
        height: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """像素坐标 -> 经纬度坐标"""
        if self._rpc is None:
            logger.warning("RPC 参数不可用，返回 (0, 0)")
            return [(0.0, 0.0) for _ in pixel_coords]
        
        if height is None:
            height = self._rpc.height_offset
        
        geo_coords = []
        for col, row in pixel_coords:
            try:
                lon, lat = image_to_ground(col, row, self._rpc, initial_height=height)
                geo_coords.append((lon, lat))
            except Exception:
                geo_coords.append((0.0, 0.0))
        
        return geo_coords
    
    def convert_json_to_global(
        self,
        json_data: Dict,
        crop_metadata: CropMetadata
    ) -> Dict:
        """转换 JSON 分割结果，添加 polygon、geo_polygon 和 coordinate_system"""
        result = json_data.copy()
        
        # 获取局部坐标
        local_polygon = json_data.get('polygon', [])
        
        if local_polygon:
            # 转换为像素坐标
            local_pixel = [
                (x * crop_metadata.crop_width, y * crop_metadata.crop_height)
                for x, y in local_polygon
            ]
            
            # 转换为全局像素坐标
            global_pixel = self.local_to_global_pixel(local_pixel, crop_metadata)
            
            # 归一化到全局图像
            img_w, img_h = crop_metadata.image_size
            global_norm = [(x / img_w, y / img_h) for x, y in global_pixel]
            
            result['polygon'] = global_norm
            
            # 转换为经纬度
            result['geo_polygon'] = self.pixel_to_geo(global_pixel)
        
        result['coordinate_system'] = crop_metadata.coordinate_system
        result['global_offset'] = {
            'x': crop_metadata.global_offset_x,
            'y': crop_metadata.global_offset_y,
        }
        
        return result
    
    def save_global_yolo_label(
        self,
        crop_metadata: CropMetadata,
        local_polygon_norm: List[Tuple[float, float]],
        output_dir: str
    ) -> str:
        """保存全局 YOLO 标签"""
        # 转换为像素坐标
        local_pixel = [
            (x * crop_metadata.crop_width, y * crop_metadata.crop_height)
            for x, y in local_polygon_norm
        ]
        
        # 转换为全局像素坐标
        global_pixel = self.local_to_global_pixel(local_pixel, crop_metadata)
        
        # 归一化到全局图像
        img_w, img_h = crop_metadata.image_size
        global_norm = [(x / img_w, y / img_h) for x, y in global_pixel]
        
        # 写入文件
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"det_{crop_metadata.det_idx}.txt")
        
        coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in global_norm])
        line = f"{crop_metadata.class_id} {coords_str}"
        
        if crop_metadata.score is not None:
            line += f" {crop_metadata.score:.4f}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(line + "\n")
        
        return output_path
