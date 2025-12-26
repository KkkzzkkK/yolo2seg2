"""
RPC 坐标转换属性测试

使用 hypothesis 进行属性测试，验证 RPC 坐标转换的正确性。
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from rs_processor.core.rpc_utils import (
    RPCParams,
    RPCParseError,
    ground_to_image,
    image_to_ground,
    get_image_geo_bounds,
)


# --- 测试数据生成策略 ---

@st.composite
def valid_rpc_params(draw):
    """生成有效的 RPC 参数"""
    lat_offset = draw(st.floats(min_value=20, max_value=50, allow_nan=False, allow_infinity=False))
    long_offset = draw(st.floats(min_value=100, max_value=130, allow_nan=False, allow_infinity=False))
    height_offset = draw(st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))
    line_offset = draw(st.floats(min_value=5000, max_value=20000, allow_nan=False, allow_infinity=False))
    samp_offset = draw(st.floats(min_value=5000, max_value=20000, allow_nan=False, allow_infinity=False))
    
    lat_scale = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))
    long_scale = draw(st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False))
    height_scale = draw(st.floats(min_value=100, max_value=1000, allow_nan=False, allow_infinity=False))
    line_scale = draw(st.floats(min_value=5000, max_value=20000, allow_nan=False, allow_infinity=False))
    samp_scale = draw(st.floats(min_value=5000, max_value=20000, allow_nan=False, allow_infinity=False))
    
    line_num_coef = np.zeros(20)
    line_num_coef[0] = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    line_num_coef[1] = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    line_num_coef[2] = draw(st.floats(min_value=0.8, max_value=1.2, allow_nan=False, allow_infinity=False))
    line_num_coef[3] = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    
    samp_num_coef = np.zeros(20)
    samp_num_coef[0] = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    samp_num_coef[1] = draw(st.floats(min_value=0.8, max_value=1.2, allow_nan=False, allow_infinity=False))
    samp_num_coef[2] = draw(st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False))
    samp_num_coef[3] = draw(st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False))
    
    line_den_coef = np.zeros(20)
    line_den_coef[0] = 1.0
    
    samp_den_coef = np.zeros(20)
    samp_den_coef[0] = 1.0
    
    return RPCParams(
        line_offset=line_offset, samp_offset=samp_offset,
        lat_offset=lat_offset, long_offset=long_offset, height_offset=height_offset,
        line_scale=line_scale, samp_scale=samp_scale,
        lat_scale=lat_scale, long_scale=long_scale, height_scale=height_scale,
        line_num_coef=line_num_coef, line_den_coef=line_den_coef,
        samp_num_coef=samp_num_coef, samp_den_coef=samp_den_coef,
    )


# **Feature: image-processing-refactor, Property 1: RPC 坐标转换 Round-Trip**
@given(rpc=valid_rpc_params())
@settings(max_examples=100, deadline=None)
def test_rpc_round_trip(rpc: RPCParams):
    """Property 1: RPC 坐标转换 Round-Trip"""
    col = rpc.samp_offset
    row = rpc.line_offset
    height = rpc.height_offset
    
    lon, lat = image_to_ground(col, row, rpc, initial_height=height, iterations=20)
    col_back, row_back = ground_to_image(lon, lat, height, rpc)
    
    col_error = abs(col_back - col)
    row_error = abs(row_back - row)
    
    assert col_error < 2.0, f"列坐标误差过大: {col_error}"
    assert row_error < 2.0, f"行坐标误差过大: {row_error}"


@given(rpc=valid_rpc_params())
@settings(max_examples=100, deadline=None)
def test_ground_to_image_then_back(rpc: RPCParams):
    """Property 1 变体: 地理 -> 像素 -> 地理 Round-Trip"""
    lon = rpc.long_offset
    lat = rpc.lat_offset
    height = rpc.height_offset
    
    col, row = ground_to_image(lon, lat, height, rpc)
    lon_back, lat_back = image_to_ground(col, row, rpc, initial_height=height, iterations=15)
    
    lon_error = abs(lon_back - lon)
    lat_error = abs(lat_back - lat)
    
    assert lon_error < 0.001, f"经度误差过大: {lon_error}"
    assert lat_error < 0.001, f"纬度误差过大: {lat_error}"


# **Feature: image-processing-refactor, Property 2: RPC 参数完整性验证**
def test_rpc_params_missing_coefs_raises():
    """Property 2: 缺少系数时应该抛出 ValueError"""
    with pytest.raises(ValueError, match="必须包含 20 个系数"):
        RPCParams(
            line_offset=0, samp_offset=0,
            lat_offset=0, long_offset=0, height_offset=0,
            line_scale=1, samp_scale=1,
            lat_scale=1, long_scale=1, height_scale=1,
            line_num_coef=np.zeros(19),
            line_den_coef=np.zeros(20),
            samp_num_coef=np.zeros(20),
            samp_den_coef=np.zeros(20),
        )


def test_rpc_params_zero_scale_raises():
    """Property 2: 缩放因子为零时应该抛出 ValueError"""
    with pytest.raises(ValueError, match="不能为零"):
        RPCParams(
            line_offset=0, samp_offset=0,
            lat_offset=0, long_offset=0, height_offset=0,
            line_scale=0, samp_scale=1,
            lat_scale=1, long_scale=1, height_scale=1,
            line_num_coef=np.zeros(20),
            line_den_coef=np.zeros(20),
            samp_num_coef=np.zeros(20),
            samp_den_coef=np.zeros(20),
        )


@given(
    missing_coef=st.sampled_from(['line_num_coef', 'line_den_coef', 'samp_num_coef', 'samp_den_coef']),
    wrong_size=st.integers(min_value=0, max_value=19)
)
@settings(max_examples=100)
def test_rpc_params_wrong_coef_size_raises(missing_coef: str, wrong_size: int):
    """Property 2: 任意系数数量不正确时应该抛出 ValueError"""
    coefs = {
        'line_num_coef': np.zeros(20),
        'line_den_coef': np.zeros(20),
        'samp_num_coef': np.zeros(20),
        'samp_den_coef': np.zeros(20),
    }
    coefs[missing_coef] = np.zeros(wrong_size)
    
    with pytest.raises(ValueError, match="必须包含 20 个系数"):
        RPCParams(
            line_offset=0, samp_offset=0,
            lat_offset=0, long_offset=0, height_offset=0,
            line_scale=1, samp_scale=1,
            lat_scale=1, long_scale=1, height_scale=1,
            **coefs
        )


def test_get_image_geo_bounds():
    """测试地理范围计算"""
    rpc = RPCParams(
        line_offset=1000, samp_offset=1000,
        lat_offset=39.5, long_offset=116.3, height_offset=50,
        line_scale=1000, samp_scale=1000,
        lat_scale=0.5, long_scale=0.5, height_scale=100,
        line_num_coef=np.zeros(20),
        line_den_coef=np.ones(20),
        samp_num_coef=np.zeros(20),
        samp_den_coef=np.ones(20),
    )
    
    min_lon, min_lat, max_lon, max_lat = get_image_geo_bounds(rpc)
    
    assert min_lon == pytest.approx(115.8)
    assert max_lon == pytest.approx(116.8)
    assert min_lat == pytest.approx(39.0)
    assert max_lat == pytest.approx(40.0)


def test_rpc_params_to_dict_from_dict():
    """测试字典转换的 round-trip"""
    rpc = RPCParams(
        line_offset=1000, samp_offset=2000,
        lat_offset=39.5, long_offset=116.3, height_offset=50,
        line_scale=1000, samp_scale=1000,
        lat_scale=0.5, long_scale=0.5, height_scale=100,
        line_num_coef=np.arange(20, dtype=float),
        line_den_coef=np.ones(20),
        samp_num_coef=np.arange(20, dtype=float) * 0.1,
        samp_den_coef=np.ones(20),
    )
    
    d = rpc.to_dict()
    rpc_back = RPCParams.from_dict(d)
    
    assert rpc_back.line_offset == rpc.line_offset
    assert rpc_back.samp_offset == rpc.samp_offset
    assert np.allclose(rpc_back.line_num_coef, rpc.line_num_coef)
