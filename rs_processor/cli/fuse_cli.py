"""
整图融合命令行入口

提供整图融合处理的命令行接口。
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional

from ..core.rpc_utils import parse_rpb_file, RPCParams
from ..core.tiff_io import read_tiff, write_png, ImageMetadata
from ..core.normalize import bands_to_rgb_uint8
from ..processing.registration import RegistrationProcessor
from ..processing.tile_processor import TileProcessor
from ..processing.image_merger import ImageMerger

logger = logging.getLogger(__name__)


def _find_rpb_file(tiff_path: str) -> Optional[str]:
    """查找与 TIFF 文件对应的 RPB 文件"""
    base_path = os.path.splitext(tiff_path)[0]
    for ext in ['.rpb', '.RPB', '.rpc', '.RPC']:
        rpb_path = base_path + ext
        if os.path.exists(rpb_path):
            return rpb_path
    return None


class FuseProcessor:
    """整图融合处理器：配准 → 分块锐化 → 合成 → 输出"""
    
    def __init__(
        self,
        tile_size: int = 8000,
        overlap: int = 200,
        enable_feature_refine: bool = True
    ):
        """初始化融合处理器"""
        self.registration_processor = RegistrationProcessor(enable_feature_refine)
        self.tile_processor = TileProcessor(tile_size, overlap)
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process(
        self,
        pan_path: str,
        mss_path: str,
        output_path: str,
        output_format: str = 'png'
    ) -> Dict:
        """执行整图融合流程
        
        Args:
            pan_path: PAN 影像路径
            mss_path: MSS 影像路径
            output_path: 输出路径
            output_format: 输出格式 ('png' 或 'tiff')
            
        Returns:
            处理结果信息
        """
        logger.info(f"开始处理: PAN={pan_path}, MSS={mss_path}")
        
        # 读取影像
        pan_data, pan_meta = read_tiff(pan_path)
        mss_data, mss_meta = read_tiff(mss_path)
        
        # 确保 MSS 是 3D 的
        if mss_data.ndim == 2:
            mss_data = mss_data[None, :, :]
        
        # 加载 RPC 参数
        pan_rpb_path = _find_rpb_file(pan_path)
        mss_rpb_path = _find_rpb_file(mss_path)
        
        pan_rpc = parse_rpb_file(pan_rpb_path) if pan_rpb_path else None
        mss_rpc = parse_rpb_file(mss_rpb_path) if mss_rpb_path else None
        
        if pan_rpc is None or mss_rpc is None:
            logger.warning("RPC 参数不完整，使用简化配准")
        
        # 生成分块
        tiles = self.tile_processor.generate_tiles(pan_meta.width, pan_meta.height)
        logger.info(f"生成 {len(tiles)} 个分块")
        
        # 创建合成器
        num_bands = mss_data.shape[0]
        merger = ImageMerger(
            output_size=(pan_meta.width, pan_meta.height),
            num_bands=num_bands,
            overlap=self.overlap
        )
        
        # 处理每个分块
        tile_offsets = []
        for tile in tiles:
            logger.info(f"处理分块 {tile.tile_idx + 1}/{len(tiles)}")
            
            try:
                if pan_rpc and mss_rpc:
                    fused_data, offset_info = self.tile_processor.process_tile(
                        pan_data, mss_data, tile,
                        pan_rpc, mss_rpc,
                        self.registration_processor
                    )
                    tile_offsets.append({
                        'tile_idx': tile.tile_idx,
                        **tile.to_dict(),
                        **offset_info.to_dict()
                    })
                else:
                    # 简化处理：直接重采样
                    import cv2
                    fused_bands = []
                    for b in range(num_bands):
                        band = cv2.resize(
                            mss_data[b],
                            (tile.width, tile.height),
                            interpolation=cv2.INTER_CUBIC
                        )
                        fused_bands.append(band)
                    fused_data = np.stack(fused_bands, axis=0)
                
                merger.add_tile(fused_data, tile)
                
            except Exception as e:
                logger.error(f"分块 {tile.tile_idx} 处理失败: {e}")
                continue
        
        # 合成并保存
        logger.info("合成分块...")
        merged = merger.merge()
        
        # 归一化并保存
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        if output_format == 'png':
            # 转换为 RGB
            bands_list = [merged[i] for i in range(min(num_bands, 4))]
            rgb_image = bands_to_rgb_uint8(bands_list)
            write_png(output_path, rgb_image)
        else:
            from ..core.tiff_io import write_tiff
            write_tiff(output_path, merged.astype(np.uint16), pan_meta)
        
        logger.info(f"输出保存到: {output_path}")
        
        # 保存元数据
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        metadata = {
            'source': {
                'pan_path': pan_path,
                'mss_path': mss_path,
                'pan_rpb_path': pan_rpb_path,
                'mss_rpb_path': mss_rpb_path,
            },
            'output': {
                'path': output_path,
                'format': output_format,
                'width': pan_meta.width,
                'height': pan_meta.height,
                'coordinate_system': 'pan',
            },
            'processing': {
                'tile_size': self.tile_size,
                'overlap': self.overlap,
                'sharpen_method': 'gram_schmidt',
                'normalize_method': 'clahe',
            },
            'registration': {
                'tile_offsets': tile_offsets,
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    def process_batch(self, input_dir: str, output_dir: str) -> List[Dict]:
        """批量处理目录下的所有影像对"""
        results = []
        
        # 扫描 PAN 文件
        pan_files = []
        for f in os.listdir(input_dir):
            if f.lower().endswith(('.tif', '.tiff')):
                if 'pan' in f.lower():
                    pan_files.append(f)
        
        for pan_file in pan_files:
            # 查找对应的 MSS 文件
            base_name = pan_file.lower().replace('pan', '')
            mss_file = None
            
            for f in os.listdir(input_dir):
                if f.lower().endswith(('.tif', '.tiff')):
                    if 'mss' in f.lower() or 'mul' in f.lower():
                        if base_name in f.lower() or f.lower().replace('mss', '').replace('mul', '') == base_name:
                            mss_file = f
                            break
            
            if mss_file:
                pan_path = os.path.join(input_dir, pan_file)
                mss_path = os.path.join(input_dir, mss_file)
                output_name = os.path.splitext(pan_file)[0].replace('PAN', 'FUSED').replace('pan', 'fused')
                output_path = os.path.join(output_dir, output_name + '.png')
                
                try:
                    result = self.process(pan_path, mss_path, output_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"处理 {pan_file} 失败: {e}")
                    results.append({'error': str(e), 'pan_path': pan_path})
        
        return results


def main():
    """CLI 入口"""
    import numpy as np
    
    parser = argparse.ArgumentParser(description='整图融合处理')
    parser.add_argument('--pan', required=True, help='PAN 影像路径')
    parser.add_argument('--mss', required=True, help='MSS 影像路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--format', choices=['png', 'tiff'], default='png')
    parser.add_argument('--tile-size', type=int, default=8000)
    parser.add_argument('--overlap', type=int, default=200)
    parser.add_argument('--no-feature-refine', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    processor = FuseProcessor(
        tile_size=args.tile_size,
        overlap=args.overlap,
        enable_feature_refine=not args.no_feature_refine
    )
    
    result = processor.process(args.pan, args.mss, args.output, args.format)
    print(f"处理完成: {result['output']['path']}")


if __name__ == '__main__':
    main()
