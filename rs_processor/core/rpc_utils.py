"""
RPC 坐标转换模块

提供 RPC 参数解析和坐标转换功能。
基于 Rational Polynomial Coefficients (RPC) 模型实现像素坐标与地理坐标的相互转换。
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import numpy as np


class RPCParseError(Exception):
    """RPC 文件解析错误"""
    pass


@dataclass
class RPCParams:
    """RPC 参数数据类
    
    包含 RPC 模型所需的所有参数：
    - 偏移量 (offset): 用于归一化坐标
    - 缩放因子 (scale): 用于归一化坐标
    - 多项式系数 (coef): 20 个系数用于计算行/列坐标
    """
    # 偏移量
    line_offset: float
    samp_offset: float
    lat_offset: float
    long_offset: float
    height_offset: float
    
    # 缩放因子
    line_scale: float
    samp_scale: float
    lat_scale: float
    long_scale: float
    height_scale: float
    
    # 多项式系数 (每个 20 个)
    line_num_coef: np.ndarray = field(default_factory=lambda: np.zeros(20))
    line_den_coef: np.ndarray = field(default_factory=lambda: np.zeros(20))
    samp_num_coef: np.ndarray = field(default_factory=lambda: np.zeros(20))
    samp_den_coef: np.ndarray = field(default_factory=lambda: np.zeros(20))
    
    def __post_init__(self):
        """验证参数完整性"""
        # 确保系数是 numpy 数组
        self.line_num_coef = np.asarray(self.line_num_coef)
        self.line_den_coef = np.asarray(self.line_den_coef)
        self.samp_num_coef = np.asarray(self.samp_num_coef)
        self.samp_den_coef = np.asarray(self.samp_den_coef)
        
        # 验证系数数量
        for name, coef in [
            ('line_num_coef', self.line_num_coef),
            ('line_den_coef', self.line_den_coef),
            ('samp_num_coef', self.samp_num_coef),
            ('samp_den_coef', self.samp_den_coef)
        ]:
            if len(coef) != 20:
                raise ValueError(f"RPC 参数 {name} 必须包含 20 个系数，当前为 {len(coef)}")
        
        # 验证缩放因子不为零
        for name, scale in [
            ('line_scale', self.line_scale),
            ('samp_scale', self.samp_scale),
            ('lat_scale', self.lat_scale),
            ('long_scale', self.long_scale),
            ('height_scale', self.height_scale)
        ]:
            if scale == 0:
                raise ValueError(f"RPC 参数 {name} 不能为零")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容旧接口）"""
        return {
            'lineOffset': self.line_offset,
            'sampOffset': self.samp_offset,
            'latOffset': self.lat_offset,
            'longOffset': self.long_offset,
            'heightOffset': self.height_offset,
            'lineScale': self.line_scale,
            'sampScale': self.samp_scale,
            'latScale': self.lat_scale,
            'longScale': self.long_scale,
            'heightScale': self.height_scale,
            'lineNumCoef': self.line_num_coef,
            'lineDenCoef': self.line_den_coef,
            'sampNumCoef': self.samp_num_coef,
            'sampDenCoef': self.samp_den_coef,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RPCParams':
        """从字典创建 RPCParams（兼容旧接口）"""
        return cls(
            line_offset=d['lineOffset'],
            samp_offset=d['sampOffset'],
            lat_offset=d['latOffset'],
            long_offset=d['longOffset'],
            height_offset=d['heightOffset'],
            line_scale=d['lineScale'],
            samp_scale=d['sampScale'],
            lat_scale=d['latScale'],
            long_scale=d['longScale'],
            height_scale=d['heightScale'],
            line_num_coef=np.asarray(d['lineNumCoef']),
            line_den_coef=np.asarray(d['lineDenCoef']),
            samp_num_coef=np.asarray(d['sampNumCoef']),
            samp_den_coef=np.asarray(d['sampDenCoef']),
        )


def parse_rpb_file(rpb_path: str) -> RPCParams:
    """解析 RPB 文件，返回 RPC 参数
    
    Args:
        rpb_path: RPB 文件路径
        
    Returns:
        RPCParams: 解析后的 RPC 参数
        
    Raises:
        RPCParseError: 文件不存在、格式错误或缺少必要字段
    """
    import os
    
    if not os.path.exists(rpb_path):
        raise RPCParseError(f"RPB 文件不存在: {rpb_path}")
    
    try:
        with open(rpb_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        raise RPCParseError(f"无法读取 RPB 文件: {e}")
    
    rpc_dict: Dict[str, Any] = {}
    
    # 解析标量参数
    scalar_keys = {
        'lineOffset': 'line_offset',
        'sampOffset': 'samp_offset', 
        'latOffset': 'lat_offset',
        'longOffset': 'long_offset',
        'heightOffset': 'height_offset',
        'lineScale': 'line_scale',
        'sampScale': 'samp_scale',
        'latScale': 'lat_scale',
        'longScale': 'long_scale',
        'heightScale': 'height_scale',
    }
    
    for line in lines:
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().rstrip(';')
            
            if key in scalar_keys:
                try:
                    rpc_dict[scalar_keys[key]] = float(value)
                except ValueError:
                    raise RPCParseError(f"无法解析参数 {key}: {value}")
    
    # 解析系数数组
    coef_mapping = {
        'lineNumCoef': 'line_num_coef',
        'lineDenCoef': 'line_den_coef',
        'sampNumCoef': 'samp_num_coef',
        'sampDenCoef': 'samp_den_coef',
    }
    
    for coef_name, param_name in coef_mapping.items():
        coef_values = []
        in_coef = False
        
        for line in lines:
            line = line.strip()
            if coef_name + ' = (' in line:
                in_coef = True
                # 提取第一行的值
                first_val = line.split('(')[1].strip()
                if first_val and first_val != '':
                    if ',' in first_val:
                        first_val = first_val.rstrip(',')
                    try:
                        coef_values.append(float(first_val))
                    except ValueError:
                        pass
            elif in_coef:
                if ');' in line:
                    # 最后一行
                    last_val = line.replace(');', '').strip()
                    if last_val:
                        try:
                            coef_values.append(float(last_val))
                        except ValueError:
                            pass
                    in_coef = False
                elif line:
                    # 中间行
                    val = line.rstrip(',').strip()
                    if val and val not in ['+', '-']:
                        try:
                            coef_values.append(float(val))
                        except ValueError:
                            pass
        
        if coef_values:
            rpc_dict[param_name] = np.array(coef_values)
    
    # 验证必要字段
    required_scalars = list(scalar_keys.values())
    required_coefs = list(coef_mapping.values())
    
    missing_scalars = [k for k in required_scalars if k not in rpc_dict]
    missing_coefs = [k for k in required_coefs if k not in rpc_dict]
    
    if missing_scalars:
        raise RPCParseError(f"RPB 文件缺少必要的标量参数: {missing_scalars}")
    
    if missing_coefs:
        raise RPCParseError(f"RPB 文件缺少必要的系数数组: {missing_coefs}")
    
    # 验证系数数量
    for param_name in required_coefs:
        if len(rpc_dict[param_name]) != 20:
            raise RPCParseError(
                f"系数 {param_name} 数量错误: 期望 20，实际 {len(rpc_dict[param_name])}"
            )
    
    return RPCParams(**rpc_dict)


def _compute_rpc_terms(P: float, L: float, H: float) -> np.ndarray:
    """计算 RPC 多项式的 20 个项
    
    Args:
        P: 归一化纬度
        L: 归一化经度
        H: 归一化高程
        
    Returns:
        20 个多项式项的数组
    """
    return np.array([
        1.0,
        L, P, H,
        L*P, L*H, P*H,
        L*L, P*P, H*H,
        P*L*H,
        L*L*L, L*L*P, L*L*H, L*P*P, L*P*H,
        L*H*H, P*P*P, P*P*H, P*H*H
    ])


def ground_to_image(lon: float, lat: float, height: float, 
                    rpc: RPCParams) -> Tuple[float, float]:
    """地理坐标 -> 像素坐标 (col, row)
    
    使用 RPC 模型将地理坐标（经度、纬度、高程）转换为像素坐标。
    
    Args:
        lon: 经度（度）
        lat: 纬度（度）
        height: 高程（米）
        rpc: RPC 参数
        
    Returns:
        (col, row): 像素坐标，col 为列（x），row 为行（y）
    """
    # 归一化地理坐标
    P = (lat - rpc.lat_offset) / rpc.lat_scale
    L = (lon - rpc.long_offset) / rpc.long_scale
    H = (height - rpc.height_offset) / rpc.height_scale
    
    # 计算多项式项
    terms = _compute_rpc_terms(P, L, H)
    
    # 计算归一化行坐标
    line_num = np.dot(rpc.line_num_coef, terms)
    line_den = np.dot(rpc.line_den_coef, terms)
    line_normalized = line_num / line_den
    
    # 计算归一化列坐标
    samp_num = np.dot(rpc.samp_num_coef, terms)
    samp_den = np.dot(rpc.samp_den_coef, terms)
    samp_normalized = samp_num / samp_den
    
    # 反归一化得到像素坐标
    row = line_normalized * rpc.line_scale + rpc.line_offset
    col = samp_normalized * rpc.samp_scale + rpc.samp_offset
    
    return col, row


def image_to_ground(col: float, row: float, rpc: RPCParams,
                    initial_height: Optional[float] = None, 
                    iterations: int = 10,
                    tolerance: float = 1e-6) -> Tuple[float, float]:
    """像素坐标 -> 地理坐标 (lon, lat)，迭代反演
    
    使用牛顿迭代法将像素坐标反演为地理坐标。
    由于 RPC 模型是非线性的，需要通过迭代求解。
    
    Args:
        col: 列坐标（x）
        row: 行坐标（y）
        rpc: RPC 参数
        initial_height: 初始高程猜测值，默认使用 RPC 中的 height_offset
        iterations: 最大迭代次数
        tolerance: 收敛容差（像素）
        
    Returns:
        (lon, lat): 地理坐标（经度、纬度）
    """
    if initial_height is None:
        initial_height = rpc.height_offset
    
    # 初始猜测值（使用 RPC 中心）
    lat = rpc.lat_offset
    lon = rpc.long_offset
    height = initial_height
    
    # 用于数值微分的扰动量
    delta = 1e-7
    
    for _ in range(iterations):
        # 正向投影，得到当前地理坐标猜测值对应的像素坐标
        est_col, est_row = ground_to_image(lon, lat, height, rpc)
        
        # 计算误差
        d_col = col - est_col
        d_row = row - est_row
        
        # 检查收敛
        if abs(d_col) < tolerance and abs(d_row) < tolerance:
            break
        
        # 使用数值微分估算雅可比矩阵
        col_lon_plus, row_lon_plus = ground_to_image(lon + delta, lat, height, rpc)
        col_lat_plus, row_lat_plus = ground_to_image(lon, lat + delta, height, rpc)
        
        dcol_dlon = (col_lon_plus - est_col) / delta
        dcol_dlat = (col_lat_plus - est_col) / delta
        drow_dlon = (row_lon_plus - est_row) / delta
        drow_dlat = (row_lat_plus - est_row) / delta
        
        # 构建雅可比矩阵
        J = np.array([
            [dcol_dlon, dcol_dlat],
            [drow_dlon, drow_dlat]
        ])
        
        try:
            # 使用伪逆求解线性方程
            inv_J = np.linalg.pinv(J)
            corrections = np.dot(inv_J, np.array([d_col, d_row]))
            d_lon, d_lat = corrections
            
            # 更新地理坐标（使用阻尼因子避免发散）
            damping = 0.8
            lon += d_lon * damping
            lat += d_lat * damping
            
        except np.linalg.LinAlgError:
            # 矩阵奇异时使用小步长
            lon += d_col * delta * 0.1
            lat += d_row * delta * 0.1
    
    return lon, lat


def get_image_geo_bounds(rpc: RPCParams) -> Tuple[float, float, float, float]:
    """获取影像地理范围
    
    根据 RPC 参数估算影像覆盖的地理范围。
    使用 offset ± scale 作为近似边界。
    
    Args:
        rpc: RPC 参数
        
    Returns:
        (min_lon, min_lat, max_lon, max_lat): 地理范围边界
    """
    min_lat = rpc.lat_offset - rpc.lat_scale
    max_lat = rpc.lat_offset + rpc.lat_scale
    min_lon = rpc.long_offset - rpc.long_scale
    max_lon = rpc.long_offset + rpc.long_scale
    
    return (min_lon, min_lat, max_lon, max_lat)


def estimate_offset_from_rpcs(pan_rpc: RPCParams, mss_rpc: RPCParams,
                               pan_size: Tuple[int, int], 
                               mss_size: Tuple[int, int]) -> Tuple[float, float]:
    """使用 PAN 和 MSS 的 RPC 参数估计全局偏移
    
    在影像中心点估计 MSS 相对于 PAN 的像素偏移量。
    
    Args:
        pan_rpc: PAN 影像的 RPC 参数
        mss_rpc: MSS 影像的 RPC 参数
        pan_size: PAN 影像尺寸 (width, height)
        mss_size: MSS 影像尺寸 (width, height)
        
    Returns:
        (dx_pan, dy_pan): 以 PAN 像素为单位的偏移量
                         正值表示 MSS 内容相对 PAN 向右/下偏移
    """
    # 使用 RPC 中心点作为参考地标
    lon = pan_rpc.long_offset
    lat = pan_rpc.lat_offset
    h = pan_rpc.height_offset
    
    # 将地标投影到 PAN 和 MSS 影像
    pan_x, pan_y = ground_to_image(lon, lat, h, pan_rpc)
    mss_x, mss_y = ground_to_image(lon, lat, h, mss_rpc)
    
    # 将 MSS 坐标缩放到 PAN 像素空间
    pan_w, pan_h = pan_size
    mss_w, mss_h = mss_size
    scale_x = pan_w / mss_w
    scale_y = pan_h / mss_h
    mss_x_in_pan_space = mss_x * scale_x
    mss_y_in_pan_space = mss_y * scale_y
    
    # 计算偏移量（以 PAN 像素为单位）
    dx_pan = mss_x_in_pan_space - pan_x
    dy_pan = mss_y_in_pan_space - pan_y
    
    return dx_pan, dy_pan
