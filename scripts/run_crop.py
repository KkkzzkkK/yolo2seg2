# -*- coding: utf-8 -*-
"""
一键运行脚本：YOLO 检测框裁剪导出

功能：
- 读取 YOLO OBB 标签文件
- 根据检测框裁剪融合后的影像
- 输出裁剪图片和元数据 JSON

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_crop.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_processor.cli.crop_cli import CropCLI

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"                    # 影像根目录
LABEL_DIR = r"F:\labels_export"               # YOLO 标签目录
FUSED_DIR = r"F:\output\fused"                # 融合后影像目录
OUTPUT_DIR = r"F:\output\crops"               # 裁剪输出目录
BOX_SCALE = 1.3                               # 检测框放大倍数
CROP_SIZE = 512                               # 裁剪尺寸（正方形）
CROP_MULTIPLE = 64                            # 裁剪尺寸对齐倍数
MAX_DETECTIONS = 500                          # 最大检测数量


def run_batch():
    """批量处理所有场景"""
    cli = CropCLI(
        box_scale=BOX_SCALE,
        crop_size=CROP_SIZE,
        crop_multiple=CROP_MULTIPLE,
        max_detections=MAX_DETECTIONS,
    )
    
    # 查找所有标签文件
    label_files = [
        os.path.join(LABEL_DIR, f) 
        for f in os.listdir(LABEL_DIR) 
        if f.lower().endswith(".txt")
    ]
    
    total_success = 0
    total_failed = 0
    
    for label_path in label_files:
        label_stem = os.path.splitext(os.path.basename(label_path))[0]
        if "-" in label_stem:
            scene_name = label_stem.rsplit("-", 1)[0]
        else:
            scene_name = label_stem
        
        print(f"\n{'='*60}")
        print(f"处理场景: {scene_name}")
        print(f"{'='*60}")
        
        # 查找融合后的影像
        fused_path = None
        for ext in ['.tif', '.tiff', '.png']:
            candidate = os.path.join(FUSED_DIR, scene_name, f"fused{ext}")
            if os.path.exists(candidate):
                fused_path = candidate
                break
        
        if not fused_path:
            print(f"[warn] 未找到融合影像: {scene_name}")
            total_failed += 1
            continue
        
        # 查找融合元数据
        metadata_path = os.path.join(FUSED_DIR, scene_name, "fusion_metadata.json")
        
        scene_output = os.path.join(OUTPUT_DIR, scene_name)
        
        try:
            results = cli.process(
                image_path=fused_path,
                label_path=label_path,
                output_dir=scene_output,
                fusion_metadata_path=metadata_path if os.path.exists(metadata_path) else None,
            )
            
            success_count = sum(1 for r in results if r.get('success', False))
            print(f"裁剪完成: {success_count}/{len(results)}")
            total_success += success_count
            total_failed += len(results) - success_count
            
        except Exception as e:
            print(f"[error] {scene_name}: {e}")
            total_failed += 1
    
    print(f"\n{'='*60}")
    print(f"全部完成！")
    print(f"成功: {total_success}")
    print(f"失败: {total_failed}")
    print(f"输出目录: {OUTPUT_DIR}")


def run_single(image_path: str, label_path: str, output_dir: str):
    """处理单个场景"""
    cli = CropCLI(
        box_scale=BOX_SCALE,
        crop_size=CROP_SIZE,
        crop_multiple=CROP_MULTIPLE,
        max_detections=MAX_DETECTIONS,
    )
    
    results = cli.process(
        image_path=image_path,
        label_path=label_path,
        output_dir=output_dir,
    )
    
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"裁剪完成: {success_count}/{len(results)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO 检测框裁剪一键运行脚本")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--image", help="影像路径（单文件模式）")
    parser.add_argument("--label", help="标签路径（单文件模式）")
    parser.add_argument("--output", help="输出目录（单文件模式）")
    parser.add_argument("--label-dir", default=LABEL_DIR, help="标签目录（批量模式）")
    parser.add_argument("--fused-dir", default=FUSED_DIR, help="融合影像目录（批量模式）")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出目录（批量模式）")
    
    args = parser.parse_args()
    
    if args.batch or (not args.image and not args.label):
        # 批量模式
        LABEL_DIR = args.label_dir
        FUSED_DIR = args.fused_dir
        OUTPUT_DIR = args.output_dir
        run_batch()
    elif args.image and args.label and args.output:
        # 单文件模式
        run_single(args.image, args.label, args.output)
    else:
        parser.print_help()
