"""
标签处理属性测试
"""

import os
import tempfile
import pytest
from hypothesis import given, settings, strategies as st

from rs_processor.processing.label_processor import LabelProcessor, Detection


# **Feature: image-processing-refactor, Property 12: YOLO 标签解析正确性**
def test_yolo_obb_parsing_valid_line():
    """Property 12: YOLO 标签解析正确性"""
    with tempfile.TemporaryDirectory() as tmpdir:
        label_path = os.path.join(tmpdir, 'test.txt')
        with open(label_path, 'w') as f:
            f.write("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
            f.write("1 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6 0.95\n")
        
        detections = LabelProcessor.read_yolo_obb(label_path)
        
        assert len(detections) == 2
        assert detections[0].class_id == "0"
        assert len(detections[0].polygon_norm) == 4
        assert detections[0].score is None
        assert detections[1].class_id == "1"
        assert detections[1].score == pytest.approx(0.95)


# **Feature: image-processing-refactor, Property 13: 坐标转换 Round-Trip**
@given(
    coords=st.lists(st.tuples(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ), min_size=4, max_size=4),
    width=st.integers(min_value=100, max_value=10000),
    height=st.integers(min_value=100, max_value=10000)
)
@settings(max_examples=100, deadline=None)
def test_coord_conversion_round_trip(coords, width, height):
    """Property 13: 坐标转换 Round-Trip"""
    pixel_coords = LabelProcessor.norm_to_pixel(coords, width, height)
    norm_back = LabelProcessor.pixel_to_norm(pixel_coords, width, height)
    
    for (x_orig, y_orig), (x_back, y_back) in zip(coords, norm_back):
        assert abs(x_back - x_orig) < 1e-6
        assert abs(y_back - y_orig) < 1e-6


# **Feature: image-processing-refactor, Property 14: 无效标签行跳过**
def test_invalid_label_lines_skipped():
    """Property 14: 无效标签行跳过"""
    with tempfile.TemporaryDirectory() as tmpdir:
        label_path = os.path.join(tmpdir, 'test.txt')
        with open(label_path, 'w') as f:
            f.write("0 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
            f.write("invalid line\n")
            f.write("1 0.5 0.5\n")
            f.write("\n")
            f.write("2 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n")
        
        detections = LabelProcessor.read_yolo_obb(label_path)
        assert len(detections) == 2
        assert detections[0].class_id == "0"
        assert detections[1].class_id == "2"


def test_nonexistent_file_returns_empty():
    """测试不存在的文件返回空列表"""
    detections = LabelProcessor.read_yolo_obb('/nonexistent/path/file.txt')
    assert detections == []


def test_local_to_global():
    """测试局部到全局坐标转换"""
    local_coords = [(10.0, 20.0), (30.0, 20.0), (30.0, 40.0), (10.0, 40.0)]
    global_coords = LabelProcessor.local_to_global(local_coords, 100, 200)
    expected = [(110.0, 220.0), (130.0, 220.0), (130.0, 240.0), (110.0, 240.0)]
    for (gx, gy), (ex, ey) in zip(global_coords, expected):
        assert gx == ex
        assert gy == ey


def test_detection_to_dict():
    """测试 Detection 转字典"""
    det = Detection(class_id="1", polygon_norm=[(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)], score=0.95)
    d = det.to_dict()
    assert d['class_id'] == "1"
    assert len(d['polygon_norm']) == 4
    assert d['score'] == 0.95


def test_write_yolo_obb():
    """测试写入 YOLO-OBB 标签"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, 'output.txt')
        detections = [Detection(class_id="0", polygon_norm=[(0.1, 0.2), (0.3, 0.2), (0.3, 0.4), (0.1, 0.4)], score=0.95)]
        LabelProcessor.write_yolo_obb(detections, output_path, include_score=True)
        read_back = LabelProcessor.read_yolo_obb(output_path)
        assert len(read_back) == 1
        assert read_back[0].class_id == "0"
        assert read_back[0].score == pytest.approx(0.95, rel=1e-3)
