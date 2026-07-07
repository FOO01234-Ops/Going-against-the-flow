# src/tracking/trajectory.py
"""
轨迹管理模块
"""
from typing import List, Dict, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class Trajectory:
    """车辆轨迹"""
    track_id: int
    points: deque = field(default_factory=lambda: deque(maxlen=100))
    speeds: List[float] = field(default_factory=list)
    
    def add_point(self, point: Tuple[int, int]):
        """添加轨迹点"""
        if len(self.points) >= 1:
            last_point = self.points[-1]
            # 计算速度
            dx = point[0] - last_point[0]
            dy = point[1] - last_point[1]
            speed = math.sqrt(dx**2 + dy**2)
            self.speeds.append(speed)
        
        self.points.append(point)
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """获取轨迹点列表"""
        return list(self.points)
    
    def get_direction(self) -> Tuple[float, float]:
        """获取行驶方向向量"""
        if len(self.points) < 2:
            return (0, 0)
        
        start = self.points[0]
        end = self.points[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # 归一化
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            return (dx / length, dy / length)
        return (0, 0)
    
    def get_angle(self) -> float:
        """获取行驶角度（度）"""
        dx, dy = self.get_direction()
        return math.atan2(dy, dx) * 180 / math.pi
    
    def get_average_speed(self) -> float:
        """获取平均速度"""
        if not self.speeds:
            return 0
        return sum(self.speeds) / len(self.speeds)
    
    def get_length(self) -> int:
        """获取轨迹长度"""
        return len(self.points)
    
    def is_valid(self, min_length: int = 5) -> bool:
        """检查轨迹是否有效"""
        return len(self.points) >= min_length


class TrajectoryManager:
    """轨迹管理器"""
    
    def __init__(self, max_length: int = 100, min_length: int = 5):
        self.trajectories: Dict[int, Trajectory] = {}
        self.max_length = max_length
        self.min_length = min_length
    
    def add_point(self, track_id: int, point: Tuple[int, int]):
        """添加轨迹点"""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = Trajectory(
                track_id=track_id,
                points=deque(maxlen=self.max_length)
            )
        self.trajectories[track_id].add_point(point)
    
    def get_trajectory(self, track_id: int) -> Optional[Trajectory]:
        """获取轨迹"""
        return self.trajectories.get(track_id)
    
    def remove_trajectory(self, track_id: int):
        """移除轨迹"""
        if track_id in self.trajectories:
            del self.trajectories[track_id]
    
    def get_all_trajectories(self) -> Dict[int, Trajectory]:
        """获取所有轨迹"""
        return self.trajectories
    
    def get_valid_trajectories(self) -> Dict[int, Trajectory]:
        """获取有效的轨迹"""
        return {
            tid: traj
            for tid, traj in self.trajectories.items()
            if traj.is_valid(self.min_length)
        }
    
    def clear(self):
        """清空所有轨迹"""
        self.trajectories.clear()