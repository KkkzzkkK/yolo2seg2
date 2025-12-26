"""
CLI 属性测试
"""

import os
import tempfile
import json
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from PIL import Image

from rs_processor.cli.fuse_cli import FuseProcessor, _find_rpb_file
from rs_processor.cli.crop_cli import CropCLI
from rs_processor.cli.convert_cli import ConvertCLI


# **Feature: image-processing-refactor, Property 22: PAN/MSS 配对正确性**
def test_find_rpb_file_with_lowercase():
    """应该能找到小写扩展名的 RPB 文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_path = os.path.join(tmpdir, 'test.tif')
        rpb_path = os.path.join(tmpdir, 'test.rpb')
        with open(tiff_path, 'w') as f:
            f.write('dummy')
        with open(rpb_path, 'w') as f:
            f.write('dummy')
        result = _find_rpb_file(tiff_path)
        assert result == rpb_path


def test_find_rpb_file_with_rpc_extension():
    """应该能找到 .rpc 扩展名的文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_path = os.path.join(tmpdir, 'test.tif')
        rpc_path = os.path.join(tmpdir, 'test.rpc')
        with open(tiff_path, 'w') as f:
            f.write('dummy')
        with open(rpc_path, 'w') as f:
            f.write('dummy')
        result = _find_rpb_file(tiff_path)
        assert result is not None
        assert os.path.exists(result)


def test_find_rpb_file_returns_none_when_missing():
    """没有 RPB 文件时应该返回 None"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tiff_path = os.path.join(tmpdir, 'test.tif')
        with open(tiff_path, 'w') as f:
            f.write('dummy')
        result = _find_rpb_file(tiff_path)
        assert result is None


def test_fuse_processor_initialization():
    """FuseProcessor 应该正确初始化"""
    processor = FuseProcessor(tile_size=4000, overlap=100, enable_feature_refine=False)
    assert processor.tile_size == 4000
    assert processor.overlap == 100
    assert processor.registration_processor is not None
    assert processor.tile_processor is not None


# **Feature: image-processing-refactor, Property 23: 批量处理错误隔离**
def test_crop_cli_handles_empty_labels():
    """CropCLI 应该正确处理空标签文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new('RGB', (1000, 1000), color='white')
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)
        label_path = os.path.join(tmpdir, 'test.txt')
        with open(label_path, 'w') as f:
            f.write('')
        cli = CropCLI()
        results = cli.process(img_path, label_path, os.path.join(tmpdir, 'output'))
        assert results == []


def test_crop_cli_handles_invalid_image():
    """CropCLI 应该正确处理无效图像"""
    with tempfile.TemporaryDirectory() as tmpdir:
        label_path = os.path.join(tmpdir, 'test.txt')
        with open(label_path, 'w') as f:
            f.write('0 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n')
        cli = CropCLI()
        results = cli.process('/nonexistent/image.png', label_path, os.path.join(tmpdir, 'output'))
        assert results == []


def test_convert_cli_handles_missing_fusion_metadata():
    """ConvertCLI 应该在没有融合元数据时正常工作"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_dir = os.path.join(tmpdir, 'metadata')
        os.makedirs(metadata_dir)
        meta = {
            'det_idx': 0, 'global_offset_x': 100, 'global_offset_y': 100,
            'crop_width': 512, 'crop_height': 512, 'image_size': [10000, 10000],
            'class_id': '1', 'score': 0.9, 'source_image': '/test.png',
        }
        with open(os.path.join(metadata_dir, 'crop_0_info.json'), 'w') as f:
            json.dump(meta, f)
        output_dir = os.path.join(tmpdir, 'output')
        cli = ConvertCLI()
        cli.process(metadata_dir, None, output_dir, None)
        assert os.path.exists(output_dir)


def test_convert_cli_handles_empty_metadata_dir():
    """ConvertCLI 应该正确处理空元数据目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_dir = os.path.join(tmpdir, 'metadata')
        os.makedirs(metadata_dir)
        output_dir = os.path.join(tmpdir, 'output')
        cli = ConvertCLI()
        cli.process(metadata_dir, None, output_dir, None)
        assert os.path.exists(output_dir)


@given(tile_size=st.integers(min_value=1000, max_value=10000), overlap=st.integers(min_value=0, max_value=500))
@settings(max_examples=50)
def test_fuse_processor_accepts_valid_params(tile_size, overlap):
    """FuseProcessor 应该接受有效的参数"""
    processor = FuseProcessor(tile_size=tile_size, overlap=overlap)
    assert processor.tile_size == tile_size
    assert processor.overlap == overlap


def test_crop_cli_with_valid_detections():
    """CropCLI 应该正确处理有效的检测框"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new('RGB', (1000, 1000), color='white')
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)
        label_path = os.path.join(tmpdir, 'test.txt')
        with open(label_path, 'w') as f:
            f.write('0 0.3 0.3 0.5 0.3 0.5 0.5 0.3 0.5\n')
            f.write('1 0.6 0.6 0.8 0.6 0.8 0.8 0.6 0.8\n')
        output_dir = os.path.join(tmpdir, 'output')
        cli = CropCLI()
        results = cli.process(img_path, label_path, output_dir)
        assert len(results) == 2
        assert os.path.exists(os.path.join(output_dir, 'crop_0.png'))
        assert os.path.exists(os.path.join(output_dir, 'crop_1.png'))
