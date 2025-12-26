"""
TIFF 读写属性测试
"""

import os
import tempfile
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from rs_processor.core.tiff_io import ImageMetadata, read_tiff, write_tiff, write_png, TiffIOError


@st.composite
def image_data(draw, min_size=32, max_size=64):
    """生成测试用的影像数据"""
    h = draw(st.integers(min_value=min_size, max_value=max_size))
    w = draw(st.integers(min_value=min_size, max_value=max_size))
    bands = draw(st.integers(min_value=1, max_value=4))
    return np.random.randint(0, 65535, size=(bands, h, w), dtype=np.uint16)


# **Feature: image-processing-refactor, Property 10: TIFF 元数据 Round-Trip**
@given(data=image_data())
@settings(max_examples=20, deadline=None)
def test_tiff_data_round_trip(data):
    """Property 10: TIFF 数据 Round-Trip"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.tif')
        write_tiff(path, data)
        data_back, metadata = read_tiff(path)
        
        if data.shape[0] == 1:
            data = data[0]
        
        assert data_back.shape == data.shape
        assert np.allclose(data_back, data)


def test_tiff_metadata_round_trip():
    """Property 10: TIFF 元数据 Round-Trip"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.tif')
        data = np.random.randint(0, 65535, size=(3, 64, 64), dtype=np.uint16)
        metadata = ImageMetadata(width=64, height=64, bands=3, dtype='uint16',
                                  geo_transform=(100.0, 0.5, 0.0, 40.0, 0.0, -0.5), crs='EPSG:4326')
        write_tiff(path, data, metadata)
        data_back, metadata_back = read_tiff(path)
        
        assert metadata_back.width == metadata.width
        assert metadata_back.height == metadata.height
        assert metadata_back.bands == metadata.bands


# **Feature: image-processing-refactor, Property 11: 窗口化读取尺寸正确性**
@given(window_params=st.tuples(
    st.integers(min_value=0, max_value=32),
    st.integers(min_value=0, max_value=32),
    st.integers(min_value=8, max_value=32),
    st.integers(min_value=8, max_value=32),
))
@settings(max_examples=20, deadline=None)
def test_windowed_read_size(window_params):
    """Property 11: 窗口化读取尺寸正确性"""
    col_off, row_off, req_width, req_height = window_params
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.tif')
        full_data = np.random.randint(0, 65535, size=(1, 64, 64), dtype=np.uint16)
        write_tiff(path, full_data)
        
        window = (col_off, row_off, req_width, req_height)
        data, metadata = read_tiff(path, window=window)
        
        expected_width = min(req_width, 64 - col_off)
        expected_height = min(req_height, 64 - row_off)
        
        if data.ndim == 2:
            actual_height, actual_width = data.shape
        else:
            _, actual_height, actual_width = data.shape
        
        assert actual_width == expected_width
        assert actual_height == expected_height


def test_read_nonexistent_file():
    """测试读取不存在的文件"""
    with pytest.raises(TiffIOError, match="不存在"):
        read_tiff('/nonexistent/path/file.tif')


def test_write_png_rgb():
    """测试写入 RGB PNG"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.png')
        data = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
        write_png(path, data)
        assert os.path.exists(path)


def test_write_png_grayscale():
    """测试写入灰度 PNG"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.png')
        data = np.random.randint(0, 255, size=(64, 64), dtype=np.uint8)
        write_png(path, data)
        assert os.path.exists(path)


def test_write_png_rgba():
    """测试写入 RGBA PNG"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test.png')
        data = np.random.randint(0, 255, size=(64, 64, 4), dtype=np.uint8)
        write_png(path, data)
        assert os.path.exists(path)


def test_image_metadata_to_dict():
    """测试元数据转字典"""
    metadata = ImageMetadata(width=100, height=200, bands=3, dtype='uint16')
    d = metadata.to_dict()
    assert d['width'] == 100
    assert d['height'] == 200
    assert d['bands'] == 3
