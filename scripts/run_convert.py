# -*- coding: utf-8 -*-
"""
一键运行脚本：坐标转换（分块坐标 -> 全局坐标）

功能：
- 读取裁剪元数据
- 将分块图的检测坐标转换为原始大图坐标
- 输出全局 YOLO 标签和 JSON（含经纬度）

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_convert.py
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rs_processor.cli.convert_cli import ConvertCLI

# ============================================================================
# 用户配置区
# ============================================================================
CROP_DIR = r"F:\output\crops"                 # 裁剪输出目录（含元数据）
RESULTS_JSON_DIR = r"F:\results_json"         # 检测结果 JSON 目录
OUTPUT_DIR = r"F:\output\global_coords"       # 全局坐标输出目录


def run_batch():
    """批量处理所有场景"""
    cli = ConvertCLI()
    
    # 查找所有场景目录
    scene_dirs = [
        d for d in os.listdir(CROP_DIR)
        if os.path.isdir(os.path.join(CROP_DIR, d))
    ]
    
    total_success = 0
    total_failed = 0
    
    for scene_name in scene_dirs:
        print(f"\n{'='*60}")
        print(f"处理场景: {scene_name}")
        print(f"{'='*60}")
        
        scene_crop_dir = os.path.join(CROP_DIR, scene_name)
        scene_json_dir = os.path.join(RESULTS_JSON_DIR, scene_name)
        scene_output_dir = os.path.join(OUTPUT_DIR, scene_name)
        
        try:
            results = cli.process(
                crop_metadata_dir=scene_crop_dir,
                results_json_dir=scene_json_dir if os.path.isdir(scene_json_dir) else None,
                output_dir=scene_output_dir,
            )
            
            success_count = sum(1 for r in results if r.get('success', False))
            print(f"转换完成: {success_count}/{len(results)}")
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


def run_single(crop_metadata_dir: str, results_json_dir: str, output_dir: str):
    """处理单个场景"""
    cli = ConvertCLI()
    
    results = cli.process(
        crop_metadata_dir=crop_metadata_dir,
        results_json_dir=results_json_dir,
        output_dir=output_dir,
    )
    
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"转换完成: {success_count}/{len(results)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="坐标转换一键运行脚本")
    parser.add_argument("--batch", action="store_true", help="批量处理模式")
    parser.add_argument("--crop-dir", help="裁剪元数据目录（单场景模式）")
    parser.add_argument("--json-dir", help="检测结果 JSON 目录（单场景模式）")
    parser.add_argument("--output", help="输出目录（单场景模式）")
    parser.add_argument("--crops-root", default=CROP_DIR, help="裁剪根目录（批量模式）")
    parser.add_argument("--results-root", default=RESULTS_JSON_DIR, help="结果 JSON 根目录（批量模式）")
    parser.add_argument("--output-root", default=OUTPUT_DIR, help="输出根目录（批量模式）")
    
    args = parser.parse_args()
    
    if args.batch or (not args.crop_dir):
        # 批量模式
        CROP_DIR = args.crops_root
        RESULTS_JSON_DIR = args.results_root
        OUTPUT_DIR = args.output_root
        run_batch()
    elif args.crop_dir and args.output:
        # 单场景模式
        run_single(args.crop_dir, args.json_dir, args.output)
    else:
        parser.print_help()
