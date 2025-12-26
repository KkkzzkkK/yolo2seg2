"""
影像配准模块

提供两阶段配准：RPC 粗配准 + 特征点精配准。

支持两种精配准方法：
1. ORB 特征点匹配（默认）
2. Arosics 自动配准（可选，需安装 arosics 库）
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Literal
import logging
import numpy as np
import cv2

from ..core.rpc_utils import RPCParams, ground_to_image

logger = logging.getLogger(__name__)

# 尝试导入 arosics
try:
    from arosics import COREG
    AROSICS_AVAILABLE = True
except ImportError:
    AROSICS_AVAILABLE = False
    logger.debug("arosics 库未安装，将使用 ORB 特征点匹配")


@dataclass
class RegistrationOffset:
    """配准偏移信息"""
    rpc_offset: Tuple[float, float]      # RPC 粗配准偏移 (dx, dy)
    feature_offset: Tuple[float, float]  # 特征点精配准偏移 (dx, dy)
    transform_matrix: np.ndarray         # 最终变换矩阵 (3x3)
    feature_match_count: int             # 特征点匹配数量
    feature_refine_success: bool         # 特征精配准是否成功
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'rpc_offset': list(self.rpc_offset),
            'feature_offset': list(self.feature_offset),
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'feature_match_count': self.feature_match_count,
            'feature_refine_success': self.feature_refine_success,
        }


@dataclass
class RegistrationResult:
    """配准结果"""
    aligned_mss: np.ndarray          # 配准后的 MSS 数据 (bands, H, W)
    valid_mask: np.ndarray           # 有效区域掩膜 (H, W)
    offset_info: RegistrationOffset  # 配准偏移信息


class RegistrationProcessor:
    """影像配准处理器
    
    实现两阶段配准：
    1. RPC 粗配准：使用 RPC 参数计算初始对齐（整体平移）
    2. 特征点精配准：使用 ORB 或 Arosics 进行精细化调整
    
    Arosics 优势：
    - 专为遥感影像设计
    - 自动检测位移，不需要手动找点
    - 支持亚像素精度配准
    - 处理"同源但有位移"的场景非常合适
    """
    
    def __init__(
        self,
        enable_feature_refine: bool = True,
        feature_max: int = 1200,
        feature_min_match: int = 18,
        refine_method: Literal['orb', 'arosics', 'auto'] = 'auto'
    ):
        """初始化配准处理器
        
        Args:
            enable_feature_refine: 是否启用特征点精配准
            feature_max: 最大特征点数量（ORB 方法）
            feature_min_match: 最小匹配点数量（ORB 方法）
            refine_method: 精配准方法
                - 'orb': 使用 ORB 特征点匹配
                - 'arosics': 使用 Arosics 自动配准
                - 'auto': 优先使用 arosics，不可用时回退到 orb
        """
        self.enable_feature_refine = enable_feature_refine
        self.feature_max = feature_max
        self.feature_min_match = feature_min_match
        
        # 确定实际使用的方法
        if refine_method == 'auto':
            self.refine_method = 'arosics' if AROSICS_AVAILABLE else 'orb'
        elif refine_method == 'arosics' and not AROSICS_AVAILABLE:
            logger.warning("arosics 不可用，回退到 ORB 方法")
            self.refine_method = 'orb'
        else:
            self.refine_method = refine_method
        
        logger.info(f"配准方法: RPC 粗配准 + {self.refine_method} 精配准")
    
    def _rpc_coarse_register(
        self,
        pan_data: np.ndarray,
        pan_rpc: RPCParams,
        mss_data: np.ndarray,
        mss_rpc: RPCParams,
        pan_window: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        """RPC 粗配准
        
        使用 RPC 参数计算 MSS 到 PAN 的初始对齐。
        对于同源 PAN/MSS（如 GF1B、GF2），偏移通常很小（几个像素）。
        
        Args:
            pan_data: PAN 影像数据 (H, W)
            pan_rpc: PAN 的 RPC 参数
            mss_data: MSS 影像数据 (bands, H, W)
            mss_rpc: MSS 的 RPC 参数
            pan_window: PAN 窗口 (col_off, row_off, width, height)
            
        Returns:
            aligned_mss: 对齐后的 MSS 数据 (bands, H, W)
            valid_mask: 有效区域掩膜 (H, W)
            rpc_offset: RPC 偏移量 (dx, dy) - 以 PAN 像素为单位
        """
        col_off, row_off, pan_width, pan_height = pan_window
        mss_bands, mss_height, mss_width = mss_data.shape
        
        # 计算 MSS 到 PAN 的缩放比例
        scale_x = pan_data.shape[1] / mss_width if mss_width > 0 else 1.0
        scale_y = pan_data.shape[0] / mss_height if mss_height > 0 else 1.0
        
        # 使用 RPC 中心点作为参考地标
        lon = pan_rpc.long_offset
        lat = pan_rpc.lat_offset
        h = pan_rpc.height_offset
        
        # 将地标投影到 PAN 和 MSS 影像
        pan_col, pan_row = ground_to_image(lon, lat, h, pan_rpc)
        mss_col, mss_row = ground_to_image(lon, lat, h, mss_rpc)
        
        # 将 MSS 坐标缩放到 PAN 像素空间
        mss_col_in_pan = mss_col * scale_x
        mss_row_in_pan = mss_row * scale_y
        
        # 计算偏移量（以 PAN 像素为单位）
        # 正值表示 MSS 内容相对 PAN 向右/下偏移
        dx = mss_col_in_pan - pan_col
        dy = mss_row_in_pan - pan_row
        
        # 创建输出数组
        aligned_mss = np.zeros((mss_bands, pan_height, pan_width), dtype=mss_data.dtype)
        valid_mask = np.zeros((pan_height, pan_width), dtype=np.uint8)
        
        # 对每个波段进行重采样（简单缩放，不应用偏移，偏移在后续分块处理中应用）
        for b in range(mss_bands):
            # 直接重采样到 PAN 分辨率
            resampled = cv2.resize(
                mss_data[b].astype(np.float32),
                (pan_width, pan_height),
                interpolation=cv2.INTER_CUBIC
            )
            aligned_mss[b] = resampled.astype(mss_data.dtype)
        
        # 设置有效区域掩膜
        valid_mask[:] = 255
        
        return aligned_mss, valid_mask, (dx, dy)

    
    def _feature_refine(
        self,
        pan_uint8: np.ndarray,
        mss_uint8: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, Tuple[float, float]]:
        """特征点精配准
        
        根据配置使用 ORB 或 Arosics 进行精细化调整。
        
        Args:
            pan_uint8: PAN 影像 uint8 (H, W)
            mss_uint8: MSS 影像 uint8 (H, W)
            
        Returns:
            transform_matrix: 变换矩阵 (3x3) 或 None
            match_count: 匹配点数量
            feature_offset: 特征点偏移 (dx, dy)
        """
        if self.refine_method == 'arosics':
            return self._arosics_refine(pan_uint8, mss_uint8)
        else:
            return self._orb_refine(pan_uint8, mss_uint8)
    
    def _arosics_refine(
        self,
        pan_uint8: np.ndarray,
        mss_uint8: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, Tuple[float, float]]:
        """使用 Arosics 进行精配准
        
        Arosics 专门解决"同源但有位移"的遥感影像配准问题，
        自动找到两张图的共同特征，计算变换矩阵。
        
        Args:
            pan_uint8: PAN 影像 uint8 (H, W) - 参考影像
            mss_uint8: MSS 影像 uint8 (H, W) - 待配准影像
            
        Returns:
            transform_matrix: 变换矩阵 (3x3) 或 None
            match_count: 匹配点数量（arosics 返回 1 表示成功）
            feature_offset: 特征点偏移 (dx, dy)
        """
        if not AROSICS_AVAILABLE:
            logger.warning("arosics 不可用，回退到 ORB")
            return self._orb_refine(pan_uint8, mss_uint8)
        
        try:
            import tempfile
            import os
            import warnings
            
            # Arosics 需要文件路径，创建临时文件
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_path = os.path.join(tmpdir, 'ref.tif')
                tgt_path = os.path.join(tmpdir, 'tgt.tif')
                
                # 保存为临时 TIFF（使用虚拟坐标，仅用于像素级配准）
                import rasterio
                from rasterio.transform import Affine
                
                h, w = pan_uint8.shape
                # 使用非单位变换避免警告（1像素=1米的虚拟坐标）
                transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, h)
                
                for path, data in [(ref_path, pan_uint8), (tgt_path, mss_uint8)]:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)
                        with rasterio.open(
                            path, 'w',
                            driver='GTiff',
                            height=data.shape[0],
                            width=data.shape[1],
                            count=1,
                            dtype=data.dtype,
                            transform=transform,
                        ) as dst:
                            dst.write(data, 1)
                
                # 使用 Arosics COREG 进行配准
                CR = COREG(
                    ref_path, tgt_path,
                    ws=(256, 256),  # 窗口大小
                    max_shift=100,  # 最大位移（像素）
                    max_iter=10,
                    nodata=(0, 0),
                )
                
                CR.calculate_spatial_shifts()
                
                # 获取位移结果
                if CR.success:
                    dx = CR.coreg_info.get('corrected_shifts_px', {}).get('x', 0)
                    dy = CR.coreg_info.get('corrected_shifts_px', {}).get('y', 0)
                    
                    # 构建平移变换矩阵
                    transform_matrix = np.array([
                        [1.0, 0.0, dx],
                        [0.0, 1.0, dy],
                        [0.0, 0.0, 1.0]
                    ], dtype=np.float64)
                    
                    logger.info(f"Arosics 配准成功: dx={dx:.2f}, dy={dy:.2f}")
                    return transform_matrix, 1, (dx, dy)
                else:
                    logger.warning("Arosics 配准失败，回退到 ORB")
                    return self._orb_refine(pan_uint8, mss_uint8)
                    
        except Exception as e:
            logger.warning(f"Arosics 配准异常: {e}，回退到 ORB")
            return self._orb_refine(pan_uint8, mss_uint8)
    
    def _orb_refine(
        self,
        pan_uint8: np.ndarray,
        mss_uint8: np.ndarray
    ) -> Tuple[Optional[np.ndarray], int, Tuple[float, float]]:
        """使用 ORB 特征点进行精配准
        
        Args:
            pan_uint8: PAN 影像 uint8 (H, W)
            mss_uint8: MSS 影像 uint8 (H, W)
            
        Returns:
            transform_matrix: 变换矩阵 (3x3) 或 None
            match_count: 匹配点数量
            feature_offset: 特征点偏移 (dx, dy)
        """
        # 创建 ORB 检测器
        orb = cv2.ORB_create(nfeatures=self.feature_max)
        
        # 检测特征点和描述符
        kp1, des1 = orb.detectAndCompute(pan_uint8, None)
        kp2, des2 = orb.detectAndCompute(mss_uint8, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0, (0.0, 0.0)
        
        # 使用 BFMatcher 进行匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # 应用比率测试
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.feature_min_match:
            return None, len(good_matches), (0.0, 0.0)
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # 计算仿射变换矩阵
        try:
            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
            
            if M is None:
                return None, len(good_matches), (0.0, 0.0)
            
            # 转换为 3x3 矩阵
            transform_matrix = np.eye(3, dtype=np.float64)
            transform_matrix[:2, :] = M
            
            # 计算平均偏移
            inliers = mask.ravel() == 1
            if np.sum(inliers) > 0:
                src_inliers = src_pts[inliers].reshape(-1, 2)
                dst_inliers = dst_pts[inliers].reshape(-1, 2)
                dx = np.mean(src_inliers[:, 0] - dst_inliers[:, 0])
                dy = np.mean(src_inliers[:, 1] - dst_inliers[:, 1])
            else:
                dx, dy = 0.0, 0.0
            
            return transform_matrix, len(good_matches), (dx, dy)
            
        except Exception:
            return None, len(good_matches), (0.0, 0.0)

    
    def _to_uint8(self, data: np.ndarray) -> np.ndarray:
        """将影像数据转换为 uint8"""
        if data.dtype == np.uint8:
            return data
        
        # 归一化到 0-255
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min) * 255
        else:
            normalized = np.zeros_like(data)
        
        return normalized.astype(np.uint8)
    
    def register(
        self,
        pan_data: np.ndarray,
        pan_rpc: RPCParams,
        mss_data: np.ndarray,
        mss_rpc: RPCParams,
        pan_window: Tuple[int, int, int, int]
    ) -> RegistrationResult:
        """执行两阶段配准：RPC 粗配准 + 特征点精配准
        
        Args:
            pan_data: PAN 影像数据 (H, W)
            pan_rpc: PAN 的 RPC 参数
            mss_data: MSS 影像数据 (bands, H, W)
            mss_rpc: MSS 的 RPC 参数
            pan_window: PAN 窗口 (col_off, row_off, width, height)
            
        Returns:
            RegistrationResult: 配准结果
        """
        col_off, row_off, pan_width, pan_height = pan_window
        
        # 1. RPC 粗配准
        aligned_mss, valid_mask, rpc_offset = self._rpc_coarse_register(
            pan_data, pan_rpc, mss_data, mss_rpc, pan_window
        )
        
        # 初始化偏移信息
        feature_offset = (0.0, 0.0)
        transform_matrix = np.eye(3, dtype=np.float64)
        feature_match_count = 0
        feature_refine_success = False
        
        # 2. 特征点精配准（可选）
        if self.enable_feature_refine:
            # 提取 PAN 窗口
            pan_window_data = pan_data[row_off:row_off+pan_height, col_off:col_off+pan_width]
            
            # 转换为 uint8
            pan_uint8 = self._to_uint8(pan_window_data)
            
            # 使用第一个波段或平均值
            if aligned_mss.shape[0] == 1:
                mss_for_match = aligned_mss[0]
            else:
                mss_for_match = np.mean(aligned_mss[:3], axis=0)
            
            mss_uint8 = self._to_uint8(mss_for_match)
            
            # 特征点精配准
            M, match_count, feat_offset = self._feature_refine(pan_uint8, mss_uint8)
            
            feature_match_count = match_count
            
            if M is not None:
                transform_matrix = M
                feature_offset = feat_offset
                feature_refine_success = True
                
                # 应用变换到所有波段
                for b in range(aligned_mss.shape[0]):
                    aligned_mss[b] = cv2.warpAffine(
                        aligned_mss[b].astype(np.float32),
                        M[:2, :],
                        (pan_width, pan_height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    ).astype(aligned_mss.dtype)
                
                # 更新有效区域掩膜
                valid_mask = cv2.warpAffine(
                    valid_mask,
                    M[:2, :],
                    (pan_width, pan_height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
        
        # 构建偏移信息
        offset_info = RegistrationOffset(
            rpc_offset=rpc_offset,
            feature_offset=feature_offset,
            transform_matrix=transform_matrix,
            feature_match_count=feature_match_count,
            feature_refine_success=feature_refine_success,
        )
        
        return RegistrationResult(
            aligned_mss=aligned_mss,
            valid_mask=valid_mask,
            offset_info=offset_info,
        )
