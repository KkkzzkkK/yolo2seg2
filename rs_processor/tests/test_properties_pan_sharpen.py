"""
全色锐化属性测试
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from rs_processor.core.pan_sharpen import calculate_band_correlations, gram_schmidt_sharpen


@st.composite
def simple_image_bands(draw):
    """生成简单的测试影像"""
    h, w = 64, 64
    base_value = draw(st.floats(min_value=1000, max_value=50000, allow_nan=False, allow_infinity=False))
    noise_scale = draw(st.floats(min_value=100, max_value=5000, allow_nan=False, allow_infinity=False))
    
    pan = base_value + np.random.randn(h, w).astype(np.float32) * noise_scale
    pan = np.clip(pan, 0, 65535)
    
    num_bands = draw(st.integers(min_value=3, max_value=4))
    mss_bands = []
    for _ in range(num_bands):
        band = base_value * draw(st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False))
        band = band + np.random.randn(h, w).astype(np.float32) * noise_scale
        band = np.clip(band, 0, 65535)
        mss_bands.append(band)
    
    return pan, mss_bands


# **Feature: image-processing-refactor, Property 5: Gram-Schmidt 波段数量不变量**
@given(data=simple_image_bands())
@settings(max_examples=100, deadline=None)
def test_gram_schmidt_band_count_invariant(data):
    """Property 5: Gram-Schmidt 波段数量不变量"""
    pan, mss_bands = data
    sharpened = gram_schmidt_sharpen(pan, mss_bands)
    assert len(sharpened) == len(mss_bands)


# **Feature: image-processing-refactor, Property 6: Gram-Schmidt 权重归一化**
@given(data=simple_image_bands())
@settings(max_examples=100, deadline=None)
def test_band_correlations_weights_normalized(data):
    """Property 6: Gram-Schmidt 权重归一化"""
    pan, mss_bands = data
    correlations, weights = calculate_band_correlations(pan, mss_bands)
    
    assert len(weights) == len(mss_bands)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6)
    assert np.all(weights >= 0)


# **Feature: image-processing-refactor, Property 7: 锐化输出尺寸一致性**
@given(
    pan_size=st.tuples(st.integers(min_value=64, max_value=128), st.integers(min_value=64, max_value=128)),
    mss_scale=st.floats(min_value=0.25, max_value=0.5, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=50, deadline=None)
def test_sharpen_output_size_matches_pan(pan_size, mss_scale):
    """Property 7: 锐化输出尺寸一致性"""
    pan_h, pan_w = pan_size
    mss_h = max(8, int(pan_h * mss_scale))
    mss_w = max(8, int(pan_w * mss_scale))
    
    pan = np.random.rand(pan_h, pan_w).astype(np.float32) * 50000 + 1000
    mss_bands = [np.random.rand(mss_h, mss_w).astype(np.float32) * 50000 + 1000 for _ in range(3)]
    
    sharpened = gram_schmidt_sharpen(pan, mss_bands)
    
    for i, band in enumerate(sharpened):
        assert band.shape == pan.shape


def test_gram_schmidt_with_custom_weights():
    """测试自定义权重"""
    pan = np.random.rand(64, 64).astype(np.float32) * 50000
    mss_bands = [np.random.rand(64, 64).astype(np.float32) * 50000 for _ in range(3)]
    weights = np.array([0.3, 0.5, 0.2])
    sharpened = gram_schmidt_sharpen(pan, mss_bands, weights=weights)
    assert len(sharpened) == 3


def test_gram_schmidt_output_range():
    """测试输出值范围"""
    pan = np.random.rand(64, 64).astype(np.float32) * 50000 + 1000
    mss_bands = [np.random.rand(64, 64).astype(np.float32) * 50000 + 1000 for _ in range(3)]
    sharpened = gram_schmidt_sharpen(pan, mss_bands)
    for band in sharpened:
        assert np.all(band >= 0)
        assert np.all(band <= 65535)


def test_calculate_correlations_returns_correct_length():
    """测试相关系数数组长度"""
    pan = np.random.rand(64, 64).astype(np.float32) * 50000
    for num_bands in [3, 4]:
        mss_bands = [np.random.rand(64, 64).astype(np.float32) * 50000 for _ in range(num_bands)]
        correlations, weights = calculate_band_correlations(pan, mss_bands)
        assert len(correlations) == num_bands
        assert len(weights) == num_bands
