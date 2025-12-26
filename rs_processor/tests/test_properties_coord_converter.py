"""
坐标转换属性测试
"""

import pytest
from hypothesis import given, strategies as st, settings
import numpy as np

from rs_processor.processing.coord_converter import CoordConverter, FusionMetadata, CropMetadata
from rs_processor.core.rpc_utils import RPCParams


def make_rpc_params():
    """创建测试用 RPC 参数"""
    return RPCParams(
        line_offset=1000.0, samp_offset=1000.0,
        lat_offset=39.5, long_offset=116.3, height_offset=50.0,
        line_scale=1000.0, samp_scale=1000.0,
        lat_scale=0.1, long_scale=0.1, height_scale=100.0,
        line_num_coef=np.array([0.001] + [0.0001] * 19),
        line_den_coef=np.array([1.0] + [0.0001] * 19),
        samp_num_coef=np.array([0.001] + [0.0001] * 19),
        samp_den_coef=np.array([1.0] + [0.0001] * 19),
    )


# **Feature: image-processing-refactor, Property 15: 全局坐标转换输出完整性**
@given(
    local_polygon=st.lists(st.tuples(st.floats(min_value=0.0, max_value=1.0), st.floats(min_value=0.0, max_value=1.0)), min_size=4, max_size=8),
    offset_x=st.integers(min_value=0, max_value=10000),
    offset_y=st.integers(min_value=0, max_value=10000),
    crop_w=st.integers(min_value=256, max_value=2048),
    crop_h=st.integers(min_value=256, max_value=2048),
)
@settings(max_examples=100)
def test_convert_json_output_completeness(local_polygon, offset_x, offset_y, crop_w, crop_h):
    """Property 15: 全局坐标转换输出完整性"""
    fusion_meta = FusionMetadata(
        pan_path='/test/pan.tif', mss_path='/test/mss.tif',
        pan_rpb_path=None, mss_rpb_path=None,
        output_path='/test/output.png', output_size=(20000, 20000),
        coordinate_system='pan', pan_rpc=make_rpc_params(),
    )
    crop_meta = CropMetadata(
        det_idx=0, global_offset_x=offset_x, global_offset_y=offset_y,
        crop_width=crop_w, crop_height=crop_h, image_size=(20000, 20000),
        class_id='1', score=0.9, source_image='/test/output.png',
    )
    
    converter = CoordConverter(fusion_meta)
    result = converter.convert_json_to_global({'polygon': local_polygon}, crop_meta)
    
    assert 'polygon' in result
    assert 'geo_polygon' in result
    assert 'coordinate_system' in result
    assert 'global_offset' in result
    assert len(result['polygon']) == len(local_polygon)
    assert len(result['geo_polygon']) == len(local_polygon)


# **Feature: image-processing-refactor, Property 16: RPC 不可用时的降级处理**
@given(pixel_coords=st.lists(st.tuples(st.floats(min_value=0.0, max_value=10000.0), st.floats(min_value=0.0, max_value=10000.0)), min_size=1, max_size=10))
@settings(max_examples=100)
def test_pixel_to_geo_without_rpc_returns_zeros(pixel_coords):
    """Property 16: RPC 不可用时的降级处理"""
    converter = CoordConverter(fusion_metadata=None)
    result = converter.pixel_to_geo(pixel_coords)
    assert len(result) == len(pixel_coords)
    for lon, lat in result:
        assert lon == 0.0
        assert lat == 0.0


@given(
    offset_x=st.integers(min_value=0, max_value=5000),
    offset_y=st.integers(min_value=0, max_value=5000),
    local_x=st.floats(min_value=0.0, max_value=1000.0),
    local_y=st.floats(min_value=0.0, max_value=1000.0),
)
@settings(max_examples=100)
def test_local_to_global_pixel_adds_offset(offset_x, offset_y, local_x, local_y):
    """局部坐标转全局坐标应该正确添加偏移"""
    crop_meta = CropMetadata(
        det_idx=0, global_offset_x=offset_x, global_offset_y=offset_y,
        crop_width=1024, crop_height=1024, image_size=(10000, 10000),
        class_id='1', score=None, source_image='/test.png',
    )
    converter = CoordConverter()
    result = converter.local_to_global_pixel([(local_x, local_y)], crop_meta)
    assert len(result) == 1
    assert result[0][0] == local_x + offset_x
    assert result[0][1] == local_y + offset_y


def test_crop_metadata_to_dict_from_dict():
    """CropMetadata 序列化和反序列化应该保持一致"""
    original = CropMetadata(
        det_idx=5, global_offset_x=1234, global_offset_y=5678,
        crop_width=512, crop_height=512, image_size=(20000, 18000),
        class_id='2', score=0.85, source_image='/path/to/image.png',
        coordinate_system='mss', fusion_metadata_path='/path/to/meta.json',
    )
    d = original.to_dict()
    restored = CropMetadata.from_dict(d)
    
    assert restored.det_idx == original.det_idx
    assert restored.global_offset_x == original.global_offset_x
    assert restored.crop_width == original.crop_width
    assert restored.image_size == original.image_size
    assert restored.class_id == original.class_id
    assert restored.score == original.score
