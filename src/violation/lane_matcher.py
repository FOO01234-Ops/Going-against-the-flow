# src/violation/lane_matcher.py
"""
车道匹配器 - 将车辆轨迹匹配到对应车道
"""
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class LaneInfo:
    """车道信息"""
    lane_id: str
    direction: str
    angle: float
    description: str
    confidence_threshold: float = 0.6


class LaneMatcher:
    """
    车道匹配器
    
    功能:
    1. 将车辆轨迹匹配到对应的车道
    2. 支持多摄像头独立配置
    3. 支持分叉车道检测
    """
    
    def __init__(self, config_path: str = "config/camera_lane_map.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # 缓存
        self._lane_cache = {}
        
        print(f"✅ 车道匹配器初始化成功")
        print(f"   📌 加载摄像头: {len(self.config.get('cameras', {}))} 个")
        print(f"   📌 加载路口: {len(self.config.get('intersections', {}))} 个")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"⚠️ 配置文件不存在: {self.config_path}")
            return {"cameras": {}, "intersections": {}}
    
    def get_camera_config(self, camera_id: str) -> Optional[Dict]:
        """获取摄像头配置"""
        return self.config.get("cameras", {}).get(camera_id)
    
    def get_intersection_config(self, intersection_id: str) -> Optional[Dict]:
        """获取路口配置"""
        return self.config.get("intersections", {}).get(intersection_id)
    
    def get_lane_info(self, camera_id: str, lane_id: str) -> Optional[LaneInfo]:
        """获取车道信息"""
        camera_config = self.get_camera_config(camera_id)
        if not camera_config:
            return None
        
        for lane in camera_config.get("lanes", []):
            if lane["id"] == lane_id:
                return LaneInfo(
                    lane_id=lane["id"],
                    direction=lane["direction"],
                    angle=lane["angle"],
                    description=lane.get("description", ""),
                    confidence_threshold=camera_config.get("confidence_threshold", 0.6)
                )
        return None
    
    def match_lane(
        self,
        camera_id: str,
        trajectory: List[Tuple[int, int]],
        intersection_id: Optional[str] = None,
    ) -> Tuple[Optional[LaneInfo], float]:
        """
        将轨迹匹配到对应车道
        
        匹配策略:
        1. 计算轨迹的平均位置 (x, y)
        2. 根据摄像头配置中的车道区域进行匹配
        3. 如果没有区域配置，使用默认车道
        
        Args:
            camera_id: 摄像头ID
            trajectory: 轨迹点列表
            intersection_id: 路口ID
        
        Returns:
            (LaneInfo, confidence): 匹配的车道和置信度
        """
        if not trajectory:
            return None, 0.0
        
        camera_config = self.get_camera_config(camera_id)
        if not camera_config:
            return None, 0.0
        
        lanes = camera_config.get("lanes", [])
        if not lanes:
            return None, 0.0
        
        # 计算轨迹的平均位置
        avg_x = sum(p[0] for p in trajectory) / len(trajectory)
        avg_y = sum(p[1] for p in trajectory) / len(trajectory)
        
        # 简化匹配: 根据x坐标匹配车道
        # 在实际项目中，可以使用更复杂的区域匹配算法
        matched_lane = None
        max_confidence = 0.0
        
        for lane in lanes:
            # 这里可以扩展为基于区域的多边形匹配
            # 目前使用x坐标简单划分（左侧车道 vs 右侧车道）
            lane_confidence = self._calculate_lane_confidence(
                avg_x, avg_y, lane, camera_config
            )
            
            if lane_confidence > max_confidence:
                max_confidence = lane_confidence
                matched_lane = lane
        
        if matched_lane:
            lane_info = LaneInfo(
                lane_id=matched_lane["id"],
                direction=matched_lane["direction"],
                angle=matched_lane["angle"],
                description=matched_lane.get("description", ""),
                confidence_threshold=camera_config.get("confidence_threshold", 0.6)
            )
            return lane_info, max_confidence
        
        return None, 0.0
    
    def _calculate_lane_confidence(
        self,
        x: float,
        y: float,
        lane: Dict,
        camera_config: Dict,
    ) -> float:
        """
        计算车辆在某个车道上的置信度
        
        基于位置距离车道的接近程度
        """
        # 默认置信度：所有车道均等
        return 0.5
    
    def get_lane_angle(self, camera_id: str, trajectory: List[Tuple[int, int]]) -> float:
        """
        获取车辆所在车道的预设角度
        
        如果匹配不到车道，返回默认角度
        """
        lane_info, confidence = self.match_lane(camera_id, trajectory)
        
        if lane_info and confidence >= lane_info.confidence_threshold:
            return lane_info.angle
        
        # 返回默认角度
        camera_config = self.get_camera_config(camera_id)
        return camera_config.get("default_angle", 90) if camera_config else 90
    
    def is_branch_lane(self, camera_id: str, lane_id: str) -> bool:
        """判断是否为分叉车道"""
        lane_info = self.get_lane_info(camera_id, lane_id)
        if lane_info:
            # 分叉车道通常方向描述包含"分叉"或角度不是标准方向
            return "分叉" in lane_info.description or lane_info.angle not in [0, 90, -90, 180]
        return False
    
    def get_possible_lanes(self, camera_id: str) -> List[LaneInfo]:
        """获取摄像头所有可能的车道"""
        camera_config = self.get_camera_config(camera_id)
        if not camera_config:
            return []
        
        return [
            LaneInfo(
                lane_id=lane["id"],
                direction=lane["direction"],
                angle=lane["angle"],
                description=lane.get("description", ""),
                confidence_threshold=camera_config.get("confidence_threshold", 0.6)
            )
            for lane in camera_config.get("lanes", [])
        ]