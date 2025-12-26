"""
影像配准属性测试
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from rs_processor.core.rpc_utils import RPCParams
from rs_processor.processing.registration import RegistrationProcessor, RegistrationResult, RegistrationOffset


def create_test_rpc() -> RPCParams:
    """创建测试用的 RPC 参数"""
    return RPCParams(
        line_offset=10000, samp_offset=10000,
        lat_offset=39.5, long_offset=116.3, height_offset=50,
        line_scale=10000, samp_scale=10000,
        lat_scale=0.5, long_scale=0.5, height_scale=500,
        line_num_coef=np.array([0.0, 0.1, 1.0] + [0.0] * 17),
        line_den_coef=np.array([1.0] + [0.0] * 19),
        samp_num_coef=np.array([0.0, 1.0, 0.1] + [0.0] * 17),
        samp_den_coef=np.array([1.0] + [0.0] * 19),
    )


def create_test_images(pan_size=(256, 256), mss_size=(64, 64), num_bands=4):
    """创建测试用的影像数据"""
    pan_data = np.random.randint(1000, 50000, size=pan_size, dtype=np.uint16)
    mss_data = np.random.randint(1000, 50000, size=(num_bands,) + mss_size, dtype=np.uint16)
    return pan_data, mss_data


# **Feature: image-processing-refactor, Property 3: 配准结果结构完整性**
def test_registration_result_structure():
    """Property 3: 配准结果结构完整性"""
    pan_data, mss_data = create_test_images()
    pan_rpc = create_test_rpc()
    mss_rpc = create_test_rpc()
    
    processor = RegistrationProcessor(enable_feature_refine=False)
    result = processor.register(pan_data, pan_rpc, mss_data, mss_rpc, pan_window=(0, 0, 128, 128))
    
    assert isinstance(result, RegistrationResult)
    assert result.aligned_mss is not None
    assert result.valid_mask is not None
    assert result.offset_info is not None
    
    offset = result.offset_info
    assert isinstance(offset, RegistrationOffset)
    assert hasattr(offset, 'rpc_offset')
    assert hasattr(offset, 'feature_offset')
    assert hasattr(offset, 'transform_matrix')
    assert isinstance(offset.rpc_offset, tuple)
    assert len(offset.rpc_offset) == 2
    assert offset.transform_matrix.shape == (3, 3)


# **Feature: image-processing-refactor, Property 4: 配准窗口边界约束**
@given(window_size=st.tuples(st.integers(min_value=32, max_value=128), st.integers(min_value=32, max_value=128)))
@settings(max_examples=20, deadline=None)
def test_registration_output_size_matches_window(window_size):
    """Property 4: 配准窗口边界约束"""
    width, height = window_size
    pan_data, mss_data = create_test_images(pan_size=(256, 256), mss_size=(64, 64))
    pan_rpc = create_test_rpc()
    mss_rpc = create_test_rpc()
    
    processor = RegistrationProcessor(enable_feature_refine=False)
    result = processor.register(pan_data, pan_rpc, mss_data, mss_rpc, pan_window=(0, 0, width, height))
    
    assert result.aligned_mss.shape[1] == height
    assert result.aligned_mss.shape[2] == width
    assert result.valid_mask.shape == (height, width)


def test_registration_preserves_band_count():
    """测试配准保持波段数量"""
    for num_bands in [3, 4]:
        pan_data, mss_data = create_test_images(num_bands=num_bands)
        pan_rpc = create_test_rpc()
        mss_rpc = create_test_rpc()
        
        processor = RegistrationProcessor(enable_feature_refine=False)
        result = processor.register(pan_data, pan_rpc, mss_data, mss_rpc, pan_window=(0, 0, 128, 128))
        assert result.aligned_mss.shape[0] == num_bands


def test_registration_offset_to_dict():
    """测试偏移信息转字典"""
    offset = RegistrationOffset(
        rpc_offset=(1.5, -2.3), feature_offset=(0.5, 0.2),
        transform_matrix=np.eye(3), feature_match_count=50, feature_refine_success=True,
    )
    d = offset.to_dict()
    assert d['rpc_offset'] == [1.5, -2.3]
    assert d['feature_offset'] == [0.5, 0.2]
    assert d['feature_match_count'] == 50
    assert d['feature_refine_success'] is True


def test_registration_with_feature_refine_disabled():
    """测试禁用特征点精配准"""
    pan_data, mss_data = create_test_images()
    pan_rpc = create_test_rpc()
    mss_rpc = create_test_rpc()
    
    processor = RegistrationProcessor(enable_feature_refine=False)
    result = processor.register(pan_data, pan_rpc, mss_data, mss_rpc, pan_window=(0, 0, 128, 128))
    
    assert result.offset_info.feature_refine_success is False
    assert result.offset_info.feature_match_count == 0
