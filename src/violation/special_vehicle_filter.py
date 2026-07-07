# src/violation/special_vehicle_filter.py
"""
特殊车辆过滤器 - 豁免救护车、消防车、警车等
"""
from typing import List, Set, Optional


class SpecialVehicleFilter:
    """
    特殊车辆过滤器
    
    对特殊车辆（救护车、消防车、警车）进行豁免
    """
    
    # 特殊车辆类型
    SPECIAL_TYPES = {
        "ambulance": "救护车",
        "fire_truck": "消防车",
        "police": "警车",
        "emergency": "应急车辆",
    }
    
    # 特殊车辆关键词（用于从车型名称中识别）
    SPECIAL_KEYWORDS = ["ambulance", "fire", "police", "emergency"]
    
    def __init__(self, special_types: Optional[List[str]] = None):
        """
        初始化特殊车辆过滤器
        
        Args:
            special_types: 特殊车辆类型列表，默认使用 SPECIAL_TYPES
        """
        if special_types:
            self.special_types = {t: t for t in special_types}
        else:
            self.special_types = self.SPECIAL_TYPES.copy()
    
    def is_special(self, vehicle_type: str) -> bool:
        """
        判断是否为特殊车辆
        
        Args:
            vehicle_type: 车型名称
        
        Returns:
            bool: 是否为特殊车辆
        """
        if not vehicle_type:
            return False
        
        vehicle_type_lower = vehicle_type.lower()
        
        # 检查是否在特殊类型列表中
        if vehicle_type_lower in self.special_types:
            return True
        
        # 检查是否包含特殊关键词
        for keyword in self.SPECIAL_KEYWORDS:
            if keyword in vehicle_type_lower:
                return True
        
        return False
    
    def get_special_type(self, vehicle_type: str) -> Optional[str]:
        """
        获取特殊车辆的具体类型
        
        Args:
            vehicle_type: 车型名称
        
        Returns:
            Optional[str]: 特殊车辆类型名称
        """
        if not self.is_special(vehicle_type):
            return None
        
        vehicle_type_lower = vehicle_type.lower()
        
        # 检查精确匹配
        if vehicle_type_lower in self.special_types:
            return self.special_types[vehicle_type_lower]
        
        # 检查关键词匹配
        for keyword, name in self.SPECIAL_TYPES.items():
            if keyword in vehicle_type_lower:
                return name
        
        return "特殊车辆"
    
    def should_exempt(self, vehicle_type: str) -> bool:
        """
        判断是否应该豁免（不生成工单）
        
        Args:
            vehicle_type: 车型名称
        
        Returns:
            bool: 是否豁免
        """
        return self.is_special(vehicle_type)