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
        """转换为字典（确保 JSON 可序列化）"""
        return {
            'rpc_offset': [float(x) for x in self.rpc_offset],
            'feature_offset': [float(x) for x in self.feature_offset],
            'total_offset': [float(x) for x in self.total_offset],
            'transform_matrix': self.transform_matrix.tolist() if self.transform_matrix is not None else None,
            'feature_match_count': int(self.feature_match_count),
            'feature_refine_success': bool(self.feature_refine_success),
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
    2. 精配准：支持 ORB 特征点匹配或相位相关
    """
    
    def __init__(
        self,
        enable_feature_refine: bool = True,
        feature_max: int = 2000,
        feature_min_match: int = 10,
        refine_method: Literal["orb", "phase", "both"] = "both",
    ):
        """初始化配准处理器
        
        Args:
            enable_feature_refine: 是否启用精配准
            feature_max: 最大特征点数量（ORB）
            feature_min_match: 最小匹配点数量（ORB）
            refine_method: 精配准方法
                - "orb": 仅使用 ORB 特征点
                - "phase": 仅使用相位相关
                - "both": 先相位相关粗调，再 ORB 精调（默认）
        """
        self.enable_feature_refine = enable_feature_refine
        self.feature_max = feature_max
        self.feature_min_match = feature_min_match
        self.refine_method = refine_method
    
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

    def _phase_correlation_refine(
        self,
        pan_data: np.ndarray,
        mss_data: np.ndarray
    ) -> Tuple[float, float, float]:
        """使用相位相关计算亚像素级全局偏移
        
        相位相关利用 FFT 在频域计算两幅图像的平移偏移，
        对所有像素都有贡献，比特征点匹配更稳定。
        
        Args:
            pan_data: PAN 影像 (H, W)
            mss_data: MSS 影像 (H, W)，需与 PAN 尺寸相同
            
        Returns:
            (dx, dy, confidence): 偏移量和置信度
            dx > 0 表示 MSS 需要向右移动才能对齐 PAN
            dy > 0 表示 MSS 需要向下移动才能对齐 PAN
        """
        # 转换为 float64
        pan_f64 = pan_data.astype(np.float64)
        mss_f64 = mss_data.astype(np.float64)
        
        # 归一化到 [0, 1]
        pan_min, pan_max = pan_f64.min(), pan_f64.max()
        mss_min, mss_max = mss_f64.min(), mss_f64.max()
        
        if pan_max > pan_min:
            pan_f64 = (pan_f64 - pan_min) / (pan_max - pan_min)
        if mss_max > mss_min:
            mss_f64 = (mss_f64 - mss_min) / (mss_max - mss_min)
        
        # 应用汉宁窗减少边缘效应
        h, w = pan_f64.shape
        hann_y = np.hanning(h)
        hann_x = np.hanning(w)
        hann_2d = np.outer(hann_y, hann_x)
        
        pan_windowed = pan_f64 * hann_2d
        mss_windowed = mss_f64 * hann_2d
        
        # 使用 OpenCV 的相位相关（支持亚像素精度）
        try:
            shift, response = cv2.phaseCorrelate(mss_windowed, pan_windowed)
            # shift 返回的是 (x, y)，表示 mss 相对于 pan 的偏移
            # 正值表示 mss 在 pan 的右/下方
            dx, dy = shift
            confidence = response
            return dx, dy, confidence
        except Exception as e:
            logger.warning(f"相位相关失败: {e}")
            return 0.0, 0.0, 0.0

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
        enable_rpc_coarse: bool = False,
    ) -> RegistrationOffset:
        """在采样数据上执行配准
        
        专为大图设计：
        1. RPC 粗配准（可选）：使用原始尺寸计算偏移
        2. 精配准：相位相关和/或 ORB 特征点
        
        Args:
            pan_sample: PAN 采样数据 (H, W)
            mss_sample: MSS 采样数据 (bands, H, W)，已重采样到与 pan_sample 相同尺寸
            pan_rpc: PAN 的 RPC 参数
            mss_rpc: MSS 的 RPC 参数
            pan_full_size: PAN 原始尺寸 (width, height)
            mss_full_size: MSS 原始尺寸 (width, height)
            sample_step: 采样步长（用于将特征偏移缩放回原始空间）
            enable_rpc_coarse: 是否启用 RPC 粗配准（同源 PAN/MSS 建议关闭）
            
        Returns:
            RegistrationOffset: 配准偏移信息（在原始 PAN 像素空间）
        """
        # 1. RPC 粗配准（可选）
        if enable_rpc_coarse:
            rpc_offset = estimate_offset_from_rpcs(
                pan_rpc, mss_rpc,
                pan_size=pan_full_size,
                mss_size=mss_full_size
            )
        else:
            rpc_offset = (0.0, 0.0)
        
        # 初始化
        feature_offset = (0.0, 0.0)
        transform_matrix = np.eye(3, dtype=np.float64)
        feature_match_count = 0
        feature_refine_success = False
        phase_offset = (0.0, 0.0)
        phase_confidence = 0.0
        
        # 2. 精配准（在采样数据上）
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
            
            # 相位相关（全局偏移，亚像素精度）
            if self.refine_method in ("phase", "both"):
                dx_phase, dy_phase, phase_confidence = self._phase_correlation_refine(
                    pan_sample, mss_for_match
                )
                phase_offset = (dx_phase * sample_step, dy_phase * sample_step)
                logger.info(f"相位相关偏移: dx={phase_offset[0]:.2f}, dy={phase_offset[1]:.2f}, 置信度={phase_confidence:.4f}")
            
            # ORB 特征点匹配
            orb_offset = (0.0, 0.0)
            if self.refine_method in ("orb", "both"):
                M, match_count, feat_offset = self._orb_refine(pan_uint8, mss_uint8)
                feature_match_count = match_count
                
                if M is not None:
                    transform_matrix = M
                    orb_offset = (
                        feat_offset[0] * sample_step,
                        feat_offset[1] * sample_step
                    )
                    feature_refine_success = True
            
            # 合并偏移结果
            if self.refine_method == "phase":
                feature_offset = phase_offset
                feature_refine_success = phase_confidence > 0.1
            elif self.refine_method == "orb":
                feature_offset = orb_offset
            else:  # both
                # 优先使用相位相关（更稳定），ORB 作为验证
                if phase_confidence > 0.1:
                    feature_offset = phase_offset
                    feature_refine_success = True
                elif feature_refine_success:
                    feature_offset = orb_offset
        
        return RegistrationOffset(
            rpc_offset=rpc_offset,
            feature_offset=feature_offset,
            transform_matrix=transform_matrix,
            feature_match_count=feature_match_count,
            feature_refine_success=feature_refine_success,
        )
