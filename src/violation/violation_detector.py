# src/violation/violation_detector.py
"""
逆行检测器（工业级增强版）
整合：
1. 高级方向分析器（多帧滑动窗口 + 圆形平均 + 一致性检查）
2. 车道规则加载（lane_rules.json）
3. 时序稳定判定（过滤抖动误报）
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


# ============================================================
# 第一部分：高级方向分析器（你提供的核心算法）
# ============================================================
class AdvancedDirectionAnalyzer:
    """
    基于滑动窗口的多帧方向分析器
    解决单帧抖动和角度跳变问题
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def _vector(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
        return (p2[0] - p1[0], p2[1] - p1[1])

    def _normalize(self, v: Tuple[int, int]) -> Tuple[float, float]:
        mag = math.sqrt(v[0]**2 + v[1]**2)
        if mag == 0:
            return (0.0, 0.0)
        return (v[0] / mag, v[1] / mag)

    def _angle(self, v: Tuple[int, int]) -> float:
        return math.atan2(v[1], v[0]) * 180 / math.pi

    def trajectory_angles(self, trajectory: List[Tuple[int, int]]) -> List[float]:
        """
        计算轨迹中每一小段的方向角度列表
        """
        angles = []
        for i in range(len(trajectory) - 1):
            v = self._vector(trajectory[i], trajectory[i+1])
            # 过滤掉几乎没有移动的点（防止静止车辆干扰）
            if abs(v[0]) + abs(v[1]) < 2:
                continue
            angles.append(self._angle(v))
        return angles

    def stable_direction(self, trajectory: List[Tuple[int, int]]) -> Tuple[Optional[float], float]:
        """
        计算稳定方向（工业级核心）
        
        Returns:
            avg_angle: 平滑后的平均方向角度（度）
            consistency: 方向一致性（0~1），越接近1表示方向越稳定可信
        """
        angles = self.trajectory_angles(trajectory)

        # 如果轨迹太短，无法计算
        if len(angles) < self.window_size:
            return None, 0.0

        # 取最近的 N 帧（滑动窗口）
        recent = angles[-self.window_size:]

        # 圆形平均（解决 -179° 和 179° 跳变问题）
        sin_sum = sum(math.sin(math.radians(a)) for a in recent)
        cos_sum = sum(math.cos(math.radians(a)) for a in recent)

        avg_angle = math.atan2(sin_sum, cos_sum) * 180 / math.pi

        # 计算一致性（向量长度 / 数量）
        consistency = math.sqrt(sin_sum**2 + cos_sum**2) / len(recent)

        return avg_angle, consistency


# ============================================================
# 第二部分：检测结果数据结构
# ============================================================
@dataclass
class ViolationResult:
    """逆行检测结果（增强版）"""
    track_id: int
    is_violation: bool          # 是否逆行
    vehicle_angle: float         # 车辆行驶角度（平滑后）
    lane_angle: float            # 车道预设角度
    direction: str               # 车辆方向描述
    lane_direction: str          # 车道方向描述
    confidence: float            # 综合置信度
    consistency: float           # 方向一致性（新增）
    trajectory_length: int       # 轨迹长度
    reason: str                  # 判定原因


# ============================================================
# 第三部分：主检测器（整合全部逻辑）
# ============================================================
class ViolationDetector:
    """
    逆行检测器（工业级增强版）
    """

    def __init__(
        self,
        config_path: str = "config/lane_rules.json",
        window_size: int = 5,
        angle_threshold: float = 90.0,
        consistency_threshold: float = 0.5,
    ):
        """
        初始化逆行检测器

        Args:
            config_path: 车道配置文件路径（lane_rules.json）
            window_size: 滑动窗口大小（帧数）
            angle_threshold: 逆行角度阈值（度）
            consistency_threshold: 方向一致性阈值（低于此值不判定）
        """
        self.config_path = Path(config_path)
        self.angle_threshold = angle_threshold
        self.consistency_threshold = consistency_threshold

        # 加载车道配置
        self.lane_config = self._load_config()

        # 高级方向分析器
        self.analyzer = AdvancedDirectionAnalyzer(window_size=window_size)

        # 默认车道角度
        self.default_lane_angle = 90  # N->S

        # 方向名称映射（用于展示）
        self.dir_names = {
            (-45, 45): "E->W",
            (45, 135): "N->S",
            (135, 180): "W->E",
            (-180, -135): "W->E",
            (-135, -45): "S->N",
        }

        print(f"✅ 逆行检测器初始化成功 (增强版)")
        print(f"   📌 滑动窗口: {window_size} 帧")
        print(f"   📌 角度阈值: {angle_threshold}°")
        print(f"   📌 一致性阈值: {consistency_threshold}")
        print(f"   📌 加载路口: {len(self.lane_config.get('intersections', {}))} 个")

    def _load_config(self) -> Dict:
        """加载车道配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"⚠️ 配置文件不存在: {self.config_path}")
            print("   使用默认配置")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置（如果 lane_rules.json 不存在）"""
        return {
            "intersections": {
                "default": {
                    "name": "默认路口",
                    "default_angle": 90,
                    "lanes": [
                        {"id": "lane_1", "direction": "N->S", "angle": 90}
                    ]
                }
            }
        }

    def _get_direction_name(self, angle: float) -> str:
        """根据角度获取方向名称"""
        # 归一化
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360

        for (low, high), name in self.dir_names.items():
            if low <= angle <= high:
                return name
        return "UNKNOWN"

    def get_lane_angle(self, intersection_id: str = "default") -> float:
        """获取路口车道角度"""
        intersections = self.lane_config.get("intersections", {})
        if intersection_id in intersections:
            return intersections[intersection_id].get("default_angle", 90)
        elif "default" in intersections:
            return intersections["default"].get("default_angle", 90)
        return self.default_lane_angle

    def get_lane_info(self, intersection_id: str = "default") -> Dict:
        """获取路口车道信息"""
        intersections = self.lane_config.get("intersections", {})
        if intersection_id in intersections:
            return intersections[intersection_id]
        elif "default" in intersections:
            return intersections["default"]
        return {
            "name": "默认路口",
            "default_angle": 90,
            "lanes": [{"id": "lane_1", "direction": "N->S", "angle": 90}]
        }

    def detect(
        self,
        track_id: int,
        trajectory: List[Tuple[int, int]],
        intersection_id: str = "default",
    ) -> ViolationResult:
        """
        检测单个车辆是否逆行（增强版）

        Args:
            track_id: 车辆跟踪ID
            trajectory: 轨迹点列表
            intersection_id: 路口ID

        Returns:
            ViolationResult: 增强的检测结果
        """
        # ============================================================
        # 1. 使用高级方向分析器计算平滑角度和一致性
        # ============================================================
        vehicle_angle, consistency = self.analyzer.stable_direction(trajectory)

        # 如果轨迹长度不足，直接返回无效
        if len(trajectory) < self.analyzer.window_size:
            return ViolationResult(
                track_id=track_id,
                is_violation=False,
                vehicle_angle=0,
                lane_angle=0,
                direction="UNKNOWN",
                lane_direction="UNKNOWN",
                confidence=0.0,
                consistency=0.0,
                trajectory_length=len(trajectory),
                reason=f"轨迹长度不足 ({len(trajectory)} < {self.analyzer.window_size})"
            )

        # 如果方向不稳定（一致性太低），不判定，避免误报
        if vehicle_angle is None or consistency < self.consistency_threshold:
            return ViolationResult(
                track_id=track_id,
                is_violation=False,
                vehicle_angle=vehicle_angle or 0,
                lane_angle=0,
                direction="UNKNOWN",
                lane_direction="UNKNOWN",
                confidence=0.0,
                consistency=consistency,
                trajectory_length=len(trajectory),
                reason=f"方向不稳定 (一致性: {consistency:.2f} < {self.consistency_threshold})"
            )

        # ============================================================
        # 2. 获取车道角度
        # ============================================================
        lane_angle = self.get_lane_angle(intersection_id)
        lane_info = self.get_lane_info(intersection_id)
        lane_dir = lane_info.get("lanes", [{}])[0].get("direction", "N->S")

        # ============================================================
        # 3. 判断是否逆行（角度差法）
        # ============================================================
        angle_diff = abs(vehicle_angle - lane_angle)
        # 归一化到 0~180
        while angle_diff > 180:
            angle_diff = 360 - angle_diff

        is_reverse = angle_diff > self.angle_threshold

        # ============================================================
        # 4. 计算置信度（结合角度差和一致性）
        # ============================================================
        # 角度差置信度（0~1），差越大越像逆行
        angle_confidence = min(1.0, angle_diff / 180.0)
        # 综合置信度 = 角度差置信度 * 一致性
        final_confidence = angle_confidence * consistency

        # ============================================================
        # 5. 生成结果
        # ============================================================
        vehicle_dir = self._get_direction_name(vehicle_angle)

        if is_reverse:
            reason = f"车辆方向 {vehicle_dir} ({vehicle_angle:.1f}°) 与车道方向 {lane_dir} ({lane_angle}°) 相反 [角度差: {angle_diff:.1f}°]"
        else:
            reason = f"车辆方向 {vehicle_dir} ({vehicle_angle:.1f}°) 与车道方向 {lane_dir} ({lane_angle}°) 一致 [角度差: {angle_diff:.1f}°]"

        return ViolationResult(
            track_id=track_id,
            is_violation=is_reverse,
            vehicle_angle=vehicle_angle,
            lane_angle=lane_angle,
            direction=vehicle_dir,
            lane_direction=lane_dir,
            confidence=final_confidence,
            consistency=consistency,
            trajectory_length=len(trajectory),
            reason=reason
        )

    def detect_batch(
        self,
        trajectories: Dict[int, List[Tuple[int, int]]],
        intersection_id: str = "default",
    ) -> List[ViolationResult]:
        """批量检测多个车辆"""
        results = []
        for track_id, trajectory in trajectories.items():
            result = self.detect(track_id, trajectory, intersection_id)
            results.append(result)
        return results

    def get_violation_tracks(
        self,
        trajectories: Dict[int, List[Tuple[int, int]]],
        intersection_id: str = "default",
    ) -> List[int]:
        """获取所有逆行车辆的ID"""
        results = self.detect_batch(trajectories, intersection_id)
        return [r.track_id for r in results if r.is_violation]