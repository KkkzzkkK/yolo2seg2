"""
裁剪导出属性测试
"""

import os
import tempfile
import pytest
from hypothesis import given, strategies as st, settings
import numpy as np
from PIL import Image

from rs_processor.processing.crop_exporter import CropExporter
from rs_processor.processing.label_processor import Detection


def make_detection(polygon_norm, class_id='1', score=0.9):
    """创建测试用检测框"""
    return Detection(class_id=class_id, polygon_norm=polygon_norm, score=score)


# **Feature: image-processing-refactor, Property 19: 裁剪窗口包含检测框**
@given(
    cx=st.floats(min_value=0.1, max_value=0.9),
    cy=st.floats(min_value=0.1, max_value=0.9),
    half_w=st.floats(min_value=0.01, max_value=0.1),
    half_h=st.floats(min_value=0.01, max_value=0.1),
)
@settings(max_examples=100)
def test_crop_window_contains_detection_box(cx, cy, half_w, half_h):
    """Property 19: 裁剪窗口包含检测框"""
    polygon_norm = [(cx - half_w, cy - half_h), (cx + half_w, cy - half_h), (cx + half_w, cy + half_h), (cx - half_w, cy + half_h)]
    detection = make_detection(polygon_norm)
    image_size = (10000, 10000)
    
    exporter = CropExporter(crop_multiple=64, min_crop_size=256)
    x, y, crop_w, crop_h = exporter.compute_crop_window(detection, image_size)
    
    for nx, ny in polygon_norm:
        px, py = nx * image_size[0], ny * image_size[1]
        assert x <= px <= x + crop_w
        assert y <= py <= y + crop_h


@given(scale=st.floats(min_value=1.1, max_value=2.0))
@settings(max_examples=50)
def test_crop_window_respects_scale(scale):
    """裁剪窗口应该按比例扩展"""
    polygon_norm = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]
    detection = make_detection(polygon_norm)
    exporter = CropExporter(crop_multiple=64, min_crop_size=256)
    x, y, crop_w, crop_h = exporter.compute_crop_window(detection, (10000, 10000), scale=scale)
    assert crop_w >= 2000 * scale * 0.9
    assert crop_h >= 2000 * scale * 0.9


# **Feature: image-processing-refactor, Property 20: 裁剪元数据完整性**
def test_crop_metadata_has_all_required_fields():
    """Property 20: 裁剪元数据完整性"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new('RGB', (1000, 1000), color='white')
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)
        
        detection = make_detection([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], class_id='2', score=0.85)
        exporter = CropExporter(crop_multiple=64, min_crop_size=256)
        metadata = exporter.crop_and_save(img_path, detection, os.path.join(tmpdir, 'output'), det_idx=3)
        
        assert metadata is not None
        assert metadata.det_idx == 3
        assert metadata.global_offset_x >= 0
        assert metadata.crop_width > 0
        assert metadata.image_size == (1000, 1000)
        assert metadata.class_id == '2'
        assert metadata.score == 0.85


def test_crop_saves_metadata_json():
    """裁剪应该保存元数据 JSON 文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new('RGB', (1000, 1000), color='white')
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)
        
        detection = make_detection([(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)])
        exporter = CropExporter()
        output_dir = os.path.join(tmpdir, 'output')
        exporter.crop_and_save(img_path, detection, output_dir, det_idx=0)
        
        assert os.path.exists(os.path.join(output_dir, 'crop_0.png'))
        assert os.path.exists(os.path.join(output_dir, 'crop_0_info.json'))


# **Feature: image-processing-refactor, Property 21: 无效检测框跳过**
def test_invalid_image_path_returns_none():
    """Property 21: 无效检测框跳过"""
    detection = make_detection([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    exporter = CropExporter()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = exporter.crop_and_save('/nonexistent/path/image.png', detection, tmpdir, det_idx=0)
        assert result is None


def test_batch_crop_continues_on_error():
    """批量裁剪应该在单个失败时继续处理其他"""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.new('RGB', (1000, 1000), color='white')
        img_path = os.path.join(tmpdir, 'test.png')
        img.save(img_path)
        
        detections = [
            make_detection([(0.2, 0.2), (0.4, 0.2), (0.4, 0.4), (0.2, 0.4)]),
            make_detection([(0.5, 0.5), (0.7, 0.5), (0.7, 0.7), (0.5, 0.7)]),
            make_detection([(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]),
        ]
        exporter = CropExporter()
        results = exporter.batch_crop(img_path, detections, os.path.join(tmpdir, 'output'))
        assert len(results) == 3


@given(crop_multiple=st.integers(min_value=16, max_value=128))
@settings(max_examples=50)
def test_crop_size_aligned_to_multiple(crop_multiple):
    """裁剪尺寸应该对齐到指定倍数"""
    detection = make_detection([(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)])
    exporter = CropExporter(crop_multiple=crop_multiple, min_crop_size=256)
    x, y, crop_w, crop_h = exporter.compute_crop_window(detection, (10000, 10000))
    assert crop_w % crop_multiple == 0
    assert crop_h % crop_multiple == 0


def test_crop_window_stays_within_image_bounds():
    """裁剪窗口应该在影像边界内"""
    detection = make_detection([(0.9, 0.9), (0.99, 0.9), (0.99, 0.99), (0.9, 0.99)])
    exporter = CropExporter(crop_multiple=64, min_crop_size=256)
    x, y, crop_w, crop_h = exporter.compute_crop_window(detection, (1000, 1000))
    assert x >= 0 and y >= 0
    assert x + crop_w <= 1000 and y + crop_h <= 1000
