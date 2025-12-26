"""
影像配准模块

提供两阶段配准：RPC 粗配准 + 特征点精配准。

支持两种精配准方法：
1. ORB 特征点匹配（默认）
2. Arosics 自动配准（可选，需安装 arosics 库）
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import logging
import numpy as np
import cv2

from ..core.rpc_utils import RPCParams, estimate_offset_from_rpcs

logger = logging.getLogger(__name__)


@dataclass
class RegistrationOffset:
    """配准偏移信息"""
    rpc_offset: Tuple[float, float]      # RPC 粗配准偏移 (dx, dy)
    feature_offset: Tuple[float, float]  # 特征点精配准偏移 (dx, dy)
    transform_matrix: np.ndarray         # 最终变换矩阵 (3x3)
    feature_match_count: int             # 特征点匹配数量
    feature_refine_success: bool         # 特征精配准是否成功
    
    @property
    def total_offset(self) -> Tuple[float, float]:
        """总偏移 = RPC偏移 + 特征偏移"""
        return (
            self.rpc_offset[0] + self.feature_offset[0],
            self.rpc_offset[1] + self.feature_offset[1]
        )
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'rpc_offset': list(self.rpc_offset),
            'feature_offset': list(self.feature_offset),
            'total_offset': list(self.total_offset),
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'feature_match_count': self.feature_match_count,
            'feature_refine_success': self.feature_refine_success,
        }


class RegistrationProcessor:
    """影像配准处理器
    
    实现两阶段配准：
    1. RPC 粗配准：使用 RPC 参数计算初始对齐（整体平移）
    2. 特征点精配准：使用 ORB 进行精细化调整
    """
    
    def __init__(
        self,
        enable_feature_refine: bool = True,
        feature_max: int = 2000,
        feature_min_match: int = 10,
    ):
        """初始化配准处理器
        
        Args:
            enable_feature_refine: 是否启用特征点精配准
            feature_max: 最大特征点数量
            feature_min_match: 最小匹配点数量
        """
        self.enable_feature_refine = enable_feature_refine
        self.feature_max = feature_max
        self.feature_min_match = feature_min_match
    
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
        orb = cv2.ORB_create(nfeatures=self.feature_max)
        
        kp1, des1 = orb.detectAndCompute(pan_uint8, None)
        kp2, des2 = orb.detectAndCompute(mss_uint8, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, 0, (0.0, 0.0)
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < self.feature_min_match:
            return None, len(good_matches), (0.0, 0.0)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        try:
            M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC)
            
            if M is None:
                return None, len(good_matches), (0.0, 0.0)
            
            transform_matrix = np.eye(3, dtype=np.float64)
            transform_matrix[:2, :] = M
            
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
        
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min) * 255
        else:
            normalized = np.zeros_like(data)
        
        return normalized.astype(np.uint8)
    
    def register_sampled(
        self,
        pan_sample: np.ndarray,
        mss_sample: np.ndarray,
        pan_rpc: RPCParams,
        mss_rpc: RPCParams,
        pan_full_size: Tuple[int, int],
        mss_full_size: Tuple[int, int],
        sample_step: int = 1,
    ) -> RegistrationOffset:
        """在采样数据上执行两阶段配准
        
        专为大图设计：
        1. RPC 粗配准使用原始尺寸计算偏移
        2. 特征点精配准在采样数据上进行，然后缩放回原始空间
        
        Args:
            pan_sample: PAN 采样数据 (H, W)
            mss_sample: MSS 采样数据 (bands, H, W)，已重采样到与 pan_sample 相同尺寸
            pan_rpc: PAN 的 RPC 参数
            mss_rpc: MSS 的 RPC 参数
            pan_full_size: PAN 原始尺寸 (width, height)
            mss_full_size: MSS 原始尺寸 (width, height)
            sample_step: 采样步长（用于将特征偏移缩放回原始空间）
            
        Returns:
            RegistrationOffset: 配准偏移信息（在原始 PAN 像素空间）
        """
        # 1. RPC 粗配准（使用原始尺寸）
        rpc_offset = estimate_offset_from_rpcs(
            pan_rpc, mss_rpc,
            pan_size=pan_full_size,
            mss_size=mss_full_size
        )
        
        # 初始化
        feature_offset = (0.0, 0.0)
        transform_matrix = np.eye(3, dtype=np.float64)
        feature_match_count = 0
        feature_refine_success = False
        
        # 2. 特征点精配准（在采样数据上）
        if self.enable_feature_refine:
            pan_uint8 = self._to_uint8(pan_sample)
            
            if mss_sample.ndim == 3:
                if mss_sample.shape[0] >= 3:
                    mss_for_match = np.mean(mss_sample[:3], axis=0)
                else:
                    mss_for_match = mss_sample[0]
            else:
                mss_for_match = mss_sample
            
            mss_uint8 = self._to_uint8(mss_for_match)
            
            M, match_count, feat_offset = self._orb_refine(pan_uint8, mss_uint8)
            
            feature_match_count = match_count
            
            if M is not None:
                transform_matrix = M
                feature_offset = (
                    feat_offset[0] * sample_step,
                    feat_offset[1] * sample_step
                )
                feature_refine_success = True
        
        return RegistrationOffset(
            rpc_offset=rpc_offset,
            feature_offset=feature_offset,
            transform_matrix=transform_matrix,
            feature_match_count=feature_match_count,
            feature_refine_success=feature_refine_success,
        )
