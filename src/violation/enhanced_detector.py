# src/violation/enhanced_detector.py
"""
增强版逆行检测器
支持: 多摄像头、分叉车道、多层验证、误检过滤
"""
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import json

from .lane_matcher import LaneMatcher, LaneInfo
from .direction_checker import DirectionChecker


@dataclass
class EnhancedViolationResult:
    """增强版逆行检测结果"""
    track_id: int
    is_violation: bool
    violation_confidence: float  # 0-1，逆行置信度
    vehicle_angle: float
    lane_angle: float
    lane_id: str
    lane_direction: str
    direction: str
    reason: str
    verification_level: int  # 0-3，验证等级
    is_branch: bool  # 是否在分叉车道


class EnhancedViolationDetector:
    """
    增强版逆行检测器
    
    新增功能:
    1. 多摄像头独立配置
    2. 分叉车道检测
    3. 多层验证机制
    4. 误检过滤
    5. 置信度评分
    """
    
    def __init__(
        self,
        config_path: str = "config/camera_lane_map.json",
        min_trajectory_length: int = 5,
        angle_threshold: float = 90,
        min_confidence: float = 0.5,
        verification_frames: int = 3,
    ):
        """
        初始化增强版检测器
        
        Args:
            config_path: 摄像头车道配置文件路径
            min_trajectory_length: 最小轨迹长度
            angle_threshold: 角度阈值
            min_confidence: 最小置信度
            verification_frames: 验证帧数（连续几帧判定为逆行才触发）
        """
        self.config_path = config_path
        self.min_trajectory_length = min_trajectory_length
        self.angle_threshold = angle_threshold
        self.min_confidence = min_confidence
        self.verification_frames = verification_frames
        
        # 车道匹配器
        self.lane_matcher = LaneMatcher(config_path)
        
        # 方向计算器
        self.direction_checker = DirectionChecker(min_trajectory_length)
        
        # 车辆状态缓存（用于多帧验证）
        self.vehicle_states = {}  # track_id -> deque of bool (是否判定为逆行)
        
        print(f"✅ 增强版逆行检测器初始化成功")
        print(f"   📌 最小轨迹长度: {min_trajectory_length}")
        print(f"   📌 角度阈值: {angle_threshold}°")
        print(f"   📌 最小置信度: {min_confidence}")
        print(f"   📌 验证帧数: {verification_frames}")
    
    def detect(
        self,
        track_id: int,
        trajectory: List[Tuple[int, int]],
        camera_id: str,
        intersection_id: Optional[str] = None,
        vehicle_type: str = "car",
    ) -> EnhancedViolationResult:
        """
        增强版逆行检测
        
        Args:
            track_id: 车辆ID
            trajectory: 轨迹点列表
            camera_id: 摄像头ID
            intersection_id: 路口ID
            vehicle_type: 车辆类型
        
        Returns:
            EnhancedViolationResult: 检测结果
        """
        # ============================================================
        # 第一层：轨迹验证
        # ============================================================
        if len(trajectory) < self.min_trajectory_length:
            return EnhancedViolationResult(
                track_id=track_id,
                is_violation=False,
                violation_confidence=0.0,
                vehicle_angle=0,
                lane_angle=0,
                lane_id="unknown",
                lane_direction="unknown",
                direction="UNKNOWN",
                reason="轨迹长度不足",
                verification_level=0,
                is_branch=False,
            )
        
        # ============================================================
        # 第二层：方向计算
        # ============================================================
        direction_result = self.direction_checker.analyze_trajectory(trajectory)
        
        if not direction_result.is_valid:
            return EnhancedViolationResult(
                track_id=track_id,
                is_violation=False,
                violation_confidence=0.0,
                vehicle_angle=0,
                lane_angle=0,
                lane_id="unknown",
                lane_direction="unknown",
                direction="UNKNOWN",
                reason="方向计算失败",
                verification_level=0,
                is_branch=False,
            )
        
        vehicle_angle = direction_result.angle
        
        # ============================================================
        # 第三层：车道匹配
        # ============================================================
        lane_info, lane_confidence = self.lane_matcher.match_lane(
            camera_id, trajectory, intersection_id
        )
        
        if lane_info is None:
            # 使用默认车道角度
            lane_angle = self.lane_matcher.get_lane_angle(camera_id, trajectory)
            lane_id = "default"
            lane_direction = "default"
        else:
            lane_angle = lane_info.angle
            lane_id = lane_info.lane_id
            lane_direction = lane_info.direction
        
        # 判断是否分叉车道
        is_branch = self.lane_matcher.is_branch_lane(camera_id, lane_id)
        
        # ============================================================
        # 第四层：逆行判定
        # ============================================================
        is_reverse = self.direction_checker.is_reverse(
            vehicle_angle, lane_angle, self.angle_threshold
        )
        
        # 计算逆行置信度
        reverse_confidence = self.direction_checker.get_reverse_confidence(
            vehicle_angle, lane_angle
        )
        
        # 综合置信度（车道匹配置信度 + 方向置信度）
        combined_confidence = reverse_confidence * direction_result.confidence
        
        # ============================================================
        # 第五层：多帧验证（防止误检）
        # ============================================================
        if track_id not in self.vehicle_states:
            self.vehicle_states[track_id] = deque(maxlen=self.verification_frames)
        
        self.vehicle_states[track_id].append(is_reverse)
        
        # 检查最近 N 帧的判定结果
        recent_states = list(self.vehicle_states[track_id])
        if len(recent_states) >= self.verification_frames:
            # 需要连续 N 帧都判定为逆行才触发
            consecutive_reverse = all(recent_states[-self.verification_frames:])
        else:
            # 轨迹不够长，使用当前判定
            consecutive_reverse = is_reverse
        
        # ============================================================
        # 第六层：最终判定
        # ============================================================
        final_is_violation = (
            consecutive_reverse 
            and combined_confidence >= self.min_confidence
            and len(trajectory) >= self.min_trajectory_length
        )
        
        # 生成原因
        if not final_is_violation:
            if combined_confidence < self.min_confidence:
                reason = f"置信度过低 ({combined_confidence:.2f} < {self.min_confidence})"
            elif not consecutive_reverse:
                reason = f"未通过多帧验证 (需要连续 {self.verification_frames} 帧)"
            else:
                reason = "正常行驶"
        else:
            reason = f"车辆方向 {direction_result.direction} 与车道方向 {lane_direction} 相反"
        
        return EnhancedViolationResult(
            track_id=track_id,
            is_violation=final_is_violation,
            violation_confidence=combined_confidence,
            vehicle_angle=vehicle_angle,
            lane_angle=lane_angle,
            lane_id=lane_id,
            lane_direction=lane_direction,
            direction=direction_result.direction,
            reason=reason,
            verification_level=self._get_verification_level(combined_confidence, lane_confidence),
            is_branch=is_branch,
        )
    
    def _get_verification_level(self, confidence: float, lane_confidence: float) -> int:
        """获取验证等级"""
        if confidence > 0.8 and lane_confidence > 0.7:
            return 3  # 高置信度
        elif confidence > 0.6 and lane_confidence > 0.5:
            return 2  # 中置信度
        elif confidence > self.min_confidence:
            return 1  # 低置信度
        else:
            return 0  # 无效
    
    def detect_batch(
        self,
        trajectories: Dict[int, List[Tuple[int, int]]],
        camera_id: str,
        vehicle_types: Optional[Dict[int, str]] = None,
    ) -> List[EnhancedViolationResult]:
        """批量检测"""
        results = []
        
        for track_id, trajectory in trajectories.items():
            vehicle_type = vehicle_types.get(track_id, "car") if vehicle_types else "car"
            result = self.detect(track_id, trajectory, camera_id, vehicle_type=vehicle_type)
            results.append(result)
        
        return results
    
    def get_violation_summary(self, results: List[EnhancedViolationResult]) -> Dict:
        """获取违规摘要"""
        violations = [r for r in results if r.is_violation]
        
        return {
            "total": len(results),
            "violations": len(violations),
            "violation_rate": len(violations) / len(results) if results else 0,
            "violation_ids": [r.track_id for r in violations],
            "details": [
                {
                    "track_id": r.track_id,
                    "confidence": r.violation_confidence,
                    "reason": r.reason,
                    "is_branch": r.is_branch,
                }
                for r in violations
            ],
            "by_confidence": {
                "high": len([r for r in violations if r.violation_confidence > 0.8]),
                "medium": len([r for r in violations if 0.6 < r.violation_confidence <= 0.8]),
                "low": len([r for r in violations if r.violation_confidence <= 0.6]),
            }
        }
    
    def clear_vehicle_states(self):
        """清除车辆状态缓存"""
        self.vehicle_states.clear()