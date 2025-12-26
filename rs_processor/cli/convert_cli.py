"""
坐标转换命令行入口

提供坐标转换处理的命令行接口。
"""

import os
import json
import argparse
import logging
from typing import Optional

from ..processing.coord_converter import CoordConverter, CropMetadata

logger = logging.getLogger(__name__)


class ConvertCLI:
    """坐标转换命令行：局部坐标 → 全局坐标 + 经纬度"""
    
    def __init__(self):
        """初始化坐标转换器"""
        self.coord_converter = None
    
    def process(
        self,
        metadata_dir: str,
        json_dir: Optional[str],
        output_dir: str,
        fusion_metadata_path: Optional[str] = None
    ) -> None:
        """执行坐标转换流程
        
        Args:
            metadata_dir: 裁剪元数据目录
            json_dir: 分割结果 JSON 目录（可选）
            output_dir: 输出目录
            fusion_metadata_path: 融合元数据路径（用于 RPC 转换）
        """
        # 加载融合元数据
        if fusion_metadata_path and os.path.exists(fusion_metadata_path):
            self.coord_converter = CoordConverter.from_metadata_file(fusion_metadata_path)
            logger.info(f"加载融合元数据: {fusion_metadata_path}")
        else:
            self.coord_converter = CoordConverter()
            logger.warning("未提供融合元数据，经纬度转换将不可用")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 扫描裁剪元数据
        metadata_files = [f for f in os.listdir(metadata_dir) if f.endswith('_info.json')]
        logger.info(f"找到 {len(metadata_files)} 个裁剪元数据文件")
        
        for meta_file in metadata_files:
            meta_path = os.path.join(metadata_dir, meta_file)
            
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_dict = json.load(f)
                
                crop_metadata = CropMetadata.from_dict(meta_dict)
                
                # 查找对应的分割结果
                if json_dir:
                    json_file = meta_file.replace('_info.json', '.json')
                    json_path = os.path.join(json_dir, json_file)
                    
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # 转换坐标
                        global_json = self.coord_converter.convert_json_to_global(
                            json_data, crop_metadata
                        )
                        
                        # 保存全局 JSON
                        output_json_path = os.path.join(output_dir, f"global_{json_file}")
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(global_json, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"转换完成: {json_file}")
                
                # 保存全局 YOLO 标签（如果有原始多边形）
                if 'polygon_norm' in meta_dict:
                    self.coord_converter.save_global_yolo_label(
                        crop_metadata,
                        meta_dict['polygon_norm'],
                        output_dir
                    )
                
            except Exception as e:
                logger.error(f"处理 {meta_file} 失败: {e}")
                continue
        
        logger.info(f"坐标转换完成，输出目录: {output_dir}")


def main():
    """CLI 入口"""
    parser = argparse.ArgumentParser(description='坐标转换')
    parser.add_argument('--metadata', required=True, help='裁剪元数据目录')
    parser.add_argument('--json', help='分割结果 JSON 目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--fusion-metadata', help='融合元数据路径（用于经纬度转换）')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cli = ConvertCLI()
    cli.process(args.metadata, args.json, args.output, args.fusion_metadata)
    print("坐标转换完成")


if __name__ == '__main__':
    main()
