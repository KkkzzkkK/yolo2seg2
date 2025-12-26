# -*- coding: utf-8 -*-
"""
一键运行脚本：完整流水线

功能：
1. YOLO 检测框裁剪 + 配准 + 融合
2. 坐标转换到全局

使用方法：
1. 修改下方配置区的路径
2. 运行: python scripts/run_pipeline.py
"""

import os
import sys
import subprocess

# ============================================================================
# 用户配置区
# ============================================================================
INPUT_DIR = r"F:\code\pic"              # 影像根目录
LABEL_DIR = r"F:\labels_export"         # YOLO 标签目录
OUTPUT_DIR = r"F:\output"               # 输出根目录
RESULTS_JSON_DIR = r"F:\results_json"   # 检测结果 JSON 目录（可选）

# 流水线步骤开关
RUN_CLIP_FUSE = True                    # 是否运行裁剪+融合
RUN_CONVERT = True                      # 是否运行坐标转换


def main():
    """运行完整流水线"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    crop_output = os.path.join(OUTPUT_DIR, 'yolo_crops')
    global_output = os.path.join(OUTPUT_DIR, 'global_coords')
    
    # Step 1: 裁剪 + 配准 + 融合
    if RUN_CLIP_FUSE:
        print("\n" + "="*60)
        print("Step 1: YOLO 检测框裁剪 + 配准 + 融合")
        print("="*60)
        
        clip_fuse_script = os.path.join(script_dir, 'run_yolo_clip_fuse.py')
        cmd = [
            sys.executable, clip_fuse_script,
            '--input-dir', INPUT_DIR,
            '--label-dir', LABEL_DIR,
            '--output-dir', crop_output,
        ]
        
        result = subprocess.run(cmd, cwd=os.path.dirname(script_dir))
        if result.returncode != 0:
            print("[error] 裁剪+融合步骤失败")
            return
    
    # Step 2: 坐标转换
    if RUN_CONVERT:
        print("\n" + "="*60)
        print("Step 2: 坐标转换到全局")
        print("="*60)
        
        convert_script = os.path.join(script_dir, 'run_convert_global.py')
        cmd = [
            sys.executable, convert_script,
            '--input-dir', INPUT_DIR,
            '--crop-dir', crop_output,
            '--output-dir', global_output,
        ]
        
        if RESULTS_JSON_DIR and os.path.isdir(RESULTS_JSON_DIR):
            cmd.extend(['--results-json-dir', RESULTS_JSON_DIR])
        
        result = subprocess.run(cmd, cwd=os.path.dirname(script_dir))
        if result.returncode != 0:
            print("[error] 坐标转换步骤失败")
            return
    
    print("\n" + "="*60)
    print("流水线完成！")
    print("="*60)
    print(f"裁剪输出: {crop_output}")
    print(f"全局坐标输出: {global_output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="完整流水线一键运行脚本")
    parser.add_argument("--input-dir", default=INPUT_DIR, help="影像根目录")
    parser.add_argument("--label-dir", default=LABEL_DIR, help="标签目录")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="输出根目录")
    parser.add_argument("--results-json-dir", default=RESULTS_JSON_DIR, help="检测结果 JSON 目录")
    parser.add_argument("--skip-fuse", action="store_true", help="跳过裁剪+融合步骤")
    parser.add_argument("--skip-convert", action="store_true", help="跳过坐标转换步骤")
    
    args = parser.parse_args()
    
    INPUT_DIR = args.input_dir
    LABEL_DIR = args.label_dir
    OUTPUT_DIR = args.output_dir
    RESULTS_JSON_DIR = args.results_json_dir
    RUN_CLIP_FUSE = not args.skip_fuse
    RUN_CONVERT = not args.skip_convert
    
    main()
