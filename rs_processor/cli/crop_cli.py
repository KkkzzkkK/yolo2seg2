"""
裁剪导出命令行入口

提供裁剪导出处理的命令行接口。
"""

import os
import json
import argparse
import logging
from typing import List

from ..processing.label_processor import LabelProcessor
from ..processing.crop_exporter import CropExporter
from ..processing.coord_converter import CropMetadata

logger = logging.getLogger(__name__)


class CropCLI:
    """裁剪导出命令行：读取融合图 + YOLO标签 → 裁剪 → 保存"""
    
    def __init__(
        self,
        crop_multiple: int = 64,
        scale: float = 1.3
    ):
        """初始化裁剪导出器"""
        self.crop_exporter = CropExporter(crop_multiple)
        self.scale = scale
    
    def process(
        self,
        image_path: str,
        label_path: str,
        output_dir: str,
        fusion_metadata_path: str = None
    ) -> List[CropMetadata]:
        """执行裁剪导出流程
        
        Args:
            image_path: 融合后图像路径
            label_path: YOLO 标签路径
            output_dir: 输出目录
            fusion_metadata_path: 融合元数据路径（可选）
            
        Returns:
            裁剪元数据列表
        """
        logger.info(f"读取标签: {label_path}")
        detections = LabelProcessor.read_yolo_obb(label_path)
        
        if not detections:
            logger.warning("未找到有效的检测框")
            return []
        
        logger.info(f"找到 {len(detections)} 个检测框")
        
        # 批量裁剪
        results = self.crop_exporter.batch_crop(image_path, detections, output_dir)
        
        # 更新融合元数据路径
        if fusion_metadata_path:
            for meta in results:
                meta.fusion_metadata_path = fusion_metadata_path
                # 更新保存的 JSON
                meta_path = os.path.join(output_dir, f"crop_{meta.det_idx}_info.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"成功裁剪 {len(results)} 个区域")
        return results


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='裁剪导出')
    parser.add_argument('--image', required=True, help='融合后图像路径')
    parser.add_argument('--labels', required=True, help='YOLO 标签路径')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--fusion-metadata', help='融合元数据路径')
    parser.add_argument('--crop-multiple', type=int, default=64)
    parser.add_argument('--scale', type=float, default=1.3)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cli = CropCLI(
        crop_multiple=args.crop_multiple,
        scale=args.scale
    )
    
    results = cli.process(
        args.image,
        args.labels,
        args.output,
        args.fusion_metadata
    )
    
    print(f"裁剪完成，共 {len(results)} 个区域")


if __name__ == '__main__':
    main()
