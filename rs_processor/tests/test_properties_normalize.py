"""
图像归一化属性测试
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from rs_processor.core.normalize import clahe_normalize, bands_to_rgb_uint8


@st.composite
def image_band(draw, min_size=32, max_size=128):
    """生成测试用的单波段影像"""
    h = draw(st.integers(min_value=min_size, max_value=max_size))
    w = draw(st.integers(min_value=min_size, max_value=max_size))
    base_value = draw(st.floats(min_value=100, max_value=50000, allow_nan=False, allow_infinity=False))
    noise_scale = draw(st.floats(min_value=10, max_value=5000, allow_nan=False, allow_infinity=False))
    band = base_value + np.random.randn(h, w).astype(np.float32) * noise_scale
    return np.clip(band, 0, 65535)


# **Feature: image-processing-refactor, Property 8: CLAHE 输出类型和范围**
@given(num_bands=st.integers(min_value=3, max_value=4))
@settings(max_examples=50, deadline=None)
def test_bands_to_rgb_output_type_and_shape(num_bands):
    """Property 8: CLAHE 输出类型和范围"""
    h, w = 64, 64
    bands = [np.random.rand(h, w).astype(np.float32) * 50000 + 100 for _ in range(num_bands)]
    result = bands_to_rgb_uint8(bands)
    
    assert result.dtype == np.uint8
    if num_bands == 3:
        assert result.shape == (h, w, 3)
    else:
        assert result.shape == (h, w, 4)
    assert np.all(result >= 0) and np.all(result <= 255)


@given(band=image_band())
@settings(max_examples=100, deadline=None)
def test_clahe_output_range(band):
    """Property 8 变体: CLAHE 单波段输出范围"""
    result = clahe_normalize(band)
    assert result.dtype == np.float32
    assert np.all(result >= 0)
    assert np.all(result <= 1)


# **Feature: image-processing-refactor, Property 9: 无效值处理**
def test_clahe_handles_nan():
    """Property 9: 无效值处理 - NaN"""
    band = np.random.rand(64, 64).astype(np.float32) * 50000
    band[10:20, 10:20] = np.nan
    result = clahe_normalize(band)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_clahe_handles_inf():
    """Property 9: 无效值处理 - Inf"""
    band = np.random.rand(64, 64).astype(np.float32) * 50000
    band[10:20, 10:20] = np.inf
    band[30:40, 30:40] = -np.inf
    result = clahe_normalize(band)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_bands_to_rgb_handles_invalid_values():
    """Property 9: 多波段无效值处理"""
    bands = [np.random.rand(64, 64).astype(np.float32) * 50000 for _ in range(3)]
    bands[0][10:20, 10:20] = np.nan
    bands[1][20:30, 20:30] = np.inf
    bands[2][30:40, 30:40] = -np.inf
    result = bands_to_rgb_uint8(bands)
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_bands_to_rgb_with_nir_fusion():
    """测试 NIR 融合功能"""
    bands = [np.random.rand(64, 64).astype(np.float32) * 50000 for _ in range(4)]
    result = bands_to_rgb_uint8(bands, nir_fusion=True)
    assert result.shape == (64, 64, 4)
    assert result.dtype == np.uint8


def test_bands_to_rgb_requires_minimum_bands():
    """测试最少波段数要求"""
    bands = [np.random.rand(64, 64).astype(np.float32) * 50000 for _ in range(2)]
    with pytest.raises(ValueError, match="至少需要 3 个波段"):
        bands_to_rgb_uint8(bands)


def test_clahe_preserves_shape():
    """测试 CLAHE 保持形状"""
    for shape in [(32, 32), (64, 128), (100, 50)]:
        band = np.random.rand(*shape).astype(np.float32) * 50000
        result = clahe_normalize(band)
        assert result.shape == shape
