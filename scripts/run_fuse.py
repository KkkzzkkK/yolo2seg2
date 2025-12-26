# -*- coding: utf-8 -*-
"""
一键运行脚本：整图融合

功能：
- 扫描输入目录，自动配对 PAN/MSS 影像
- 执行 RPC 粗配准 + 特征点精配准
- Gram-Schmidt 全色锐化
- 输出融合后的 TIFF/PNG 和元数据 JSON

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_fuse.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_processor.cli.fuse_cli import FuseProcessor, main as fuse_main

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"           # 影像根目录（每个场景一个子文件夹）
OUTPUT_DIR = r"F:\output\fused"      # 输出目录
TILE_SIZE = 4096                     # 分块大小
OVERLAP = 256                        # 分块重叠
SHARPEN_METHOD = "gram_schmidt"      # 锐化方法: gram_schmidt, brovey, ihs
REFINE_METHOD = "auto"               # 精配准方法: auto, orb, arosics
OUTPUT_FORMAT = "tiff"               # 输出格式: tiff, png, both


def run_batch():
    """批量处理所有场景"""
    processor = FuseProcessor(
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        sharpen_method=SHARPEN_METHOD,
        refine_method=REFINE_METHOD,
        output_format=OUTPUT_FORMAT,
    )
    
    results = processor.process_batch(INPUT_DIR, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"成功: {sum(1 for r in results if r.get('success', False))}")
    print(f"失败: {sum(1 for r in results if not r.get('success', False))}")
    print(f"输出目录: {OUTPUT_DIR}")


def run_single(pan_path: str, mss_path: str, output_path: str):
    """处理单个场景"""
    processor = FuseProcessor(
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        sharpen_method=SHARPEN_METHOD,
        refine_method=REFINE_METHOD,
        output_format=OUTPUT_FORMAT,
    )
    
    result = processor.process(pan_path, mss_path, output_path)
    
    if result.get('success'):
        print(f"融合成功: {output_path}")
    else:
        print(f"融合失败: {result.get('error')}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="整图融合一键运行脚本")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--pan", help="PAN 影像路径（单文件模式）")
    parser.add_argument("--mss", help="MSS 影像路径（单文件模式）")
    parser.add_argument("--output", help="输出路径（单文件模式）")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="输入目录（批量模式）")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录（批量模式）")
    
    args = parser.parse_args()
    
    if args.batch or (not args.pan and not args.mss):
        # 批量模式
        INPUT_DIR = args.input_dir
        OUTPUT_DIR = args.output_dir
        run_batch()
    elif args.pan and args.mss and args.output:
        # 单文件模式
        run_single(args.pan, args.mss, args.output)
    else:
        parser.print_help()
