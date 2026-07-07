"""
方向计算器 - 计算车辆行驶方向
"""
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DirectionResult:
    """方向计算结果"""
    angle: float                      # 行驶角度（度）
    direction: str                   # 方向描述 (N->S, S->N, W->E, E->W)
    confidence: float                # 置信度 (0-1)
    trajectory_length: int           # 轨迹长度
    is_valid: bool                   # 是否有效


class DirectionChecker:
    """
    方向计算器（优化版）

    关键优化：
    - 统一图像坐标系与交通坐标系
    - 修正 y轴方向问题（图像坐标 y向下）
    """

    # 方向映射（交通坐标系）
    DIRECTION_MAP = {
        (-45, 45): "E->W",
        (45, 135): "N->S",
        (135, 180): "W->E",
        (-180, -135): "W->E",
        (-135, -45): "S->N",
    }

    def __init__(self, min_trajectory_length: int = 5):
        self.min_trajectory_length = min_trajectory_length

    # ==============================
    # 核心修正：统一坐标系
    # ==============================
    def normalize_angle(self, angle: float) -> float:
        """
        将角度规范化到 [-180, 180]
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def calculate_angle(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        """
        使用首尾点计算方向角度（优化版）
        """
        if len(trajectory) < self.min_trajectory_length:
            return None

        start = trajectory[0]
        end = trajectory[-1]

        dx = end[0] - start[0]

        # ⚠️ 关键修正：图像坐标系 y向下 → 转为数学坐标系
        dy = start[1] - end[1]

        if abs(dx) < 1 and abs(dy) < 1:
            return None

        angle = math.atan2(dy, dx) * 180 / math.pi

        return self.normalize_angle(angle)

    def calculate_angle_smooth(self, trajectory: List[Tuple[int, int]]) -> Optional[float]:
        """
        平滑角度计算（线性回归版本）
        """
        if len(trajectory) < self.min_trajectory_length:
            return None

        points = trajectory[-self.min_trajectory_length:]
        xs = [p[0] for p in points]

        # ⚠️ 同样修正 y轴方向
        ys = [p[1] for p in points][::-1]

        n = len(xs)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)

        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom

        angle = math.atan(slope) * 180 / math.pi

        return self.normalize_angle(angle)

    def get_direction_name(self, angle: float) -> str:
        """
        根据角度获取方向名称
        """
        angle = self.normalize_angle(angle)

        for (low, high), direction in self.DIRECTION_MAP.items():
            if low <= angle <= high:
                return direction

        return "UNKNOWN"

    def analyze_trajectory(self, trajectory: List[Tuple[int, int]]) -> DirectionResult:
        """
        完整轨迹分析
        """
        if len(trajectory) < self.min_trajectory_length:
            return DirectionResult(
                angle=0,
                direction="UNKNOWN",
                confidence=0.0,
                trajectory_length=len(trajectory),
                is_valid=False
            )

        angle = self.calculate_angle_smooth(trajectory)

        if angle is None:
            return DirectionResult(
                angle=0,
                direction="UNKNOWN",
                confidence=0.0,
                trajectory_length=len(trajectory),
                is_valid=False
            )

        direction = self.get_direction_name(angle)
        confidence = self._calculate_confidence(trajectory)

        return DirectionResult(
            angle=angle,
            direction=direction,
            confidence=confidence,
            trajectory_length=len(trajectory),
            is_valid=True
        )

    def _calculate_confidence(self, trajectory: List[Tuple[int, int]]) -> float:
        """
        置信度计算（保持你原逻辑）
        """
        if len(trajectory) < self.min_trajectory_length:
            return 0.0

        length_conf = min(1.0, len(trajectory) / 20)

        if len(trajectory) > 2:
            distances = []
            for i in range(1, len(trajectory)):
                dx = trajectory[i][0] - trajectory[i - 1][0]
                dy = trajectory[i][1] - trajectory[i - 1][1]
                distances.append(math.sqrt(dx * dx + dy * dy))

            avg_dist = sum(distances) / len(distances)

            if avg_dist > 0:
                variance = sum((d - avg_dist) ** 2 for d in distances) / len(distances)
                std = math.sqrt(variance)
                smooth_conf = max(0, 1 - std / avg_dist)
            else:
                smooth_conf = 0.5
        else:
            smooth_conf = 0.5

        confidence = 0.6 * length_conf + 0.4 * smooth_conf

        return min(1.0, confidence)

    def is_reverse(self, vehicle_angle: float, lane_angle: float, threshold: float = 90) -> bool:
        """
        是否逆行判断（保留原逻辑）
        """
        diff = abs(vehicle_angle - lane_angle)

        while diff > 180:
            diff = 360 - diff

        return diff > threshold

    def get_reverse_confidence(self, vehicle_angle: float, lane_angle: float) -> float:
        """
        逆行置信度（保留）
        """
        diff = abs(vehicle_angle - lane_angle)

        while diff > 180:
            diff = 360 - diff

        return min(1.0, diff / 180.0)