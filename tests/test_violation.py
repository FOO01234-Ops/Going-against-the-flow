# tests/test_violation.py
"""
逆行检测测试
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.violation import (
    DirectionChecker,
    ViolationDetector,
    SpecialVehicleFilter
)


def test_direction_checker():
    """测试方向计算器"""
    print("=" * 60)
    print("🧭 测试方向计算器")
    print("=" * 60)
    
    checker = DirectionChecker(min_trajectory_length=3)
    
    # 模拟轨迹：从左上到右下（N->S方向）
    trajectory_ns = [(100, 100), (100, 150), (100, 200), (100, 250)]
    
    result = checker.analyze_trajectory(trajectory_ns)
    print(f"\n📊 N->S 轨迹:")
    print(f"   角度: {result.angle:.1f}°")
    print(f"   方向: {result.direction}")
    print(f"   置信度: {result.confidence:.2f}")
    print(f"   有效: {result.is_valid}")
    
    # 模拟轨迹：从右下到左上（S->N方向）
    trajectory_sn = [(100, 250), (100, 200), (100, 150), (100, 100)]
    
    result = checker.analyze_trajectory(trajectory_sn)
    print(f"\n📊 S->N 轨迹:")
    print(f"   角度: {result.angle:.1f}°")
    print(f"   方向: {result.direction}")
    print(f"   置信度: {result.confidence:.2f}")
    
    # 判断逆行
    lane_angle = 90  # N->S
    is_reverse = checker.is_reverse(0, lane_angle)  # 0度是W->E
    
    print(f"\n📊 逆行判断:")
    print(f"   车道方向: {lane_angle}° (N->S)")
    print(f"   车辆方向: 0° (W->E)")
    print(f"   是否逆行: {is_reverse}")


def test_violation_detector():
    """测试逆行检测器"""
    print("\n" + "=" * 60)
    print("🚨 测试逆行检测器")
    print("=" * 60)
    
    detector = ViolationDetector(
        config_path="config/lane_rules.json",
        min_trajectory_length=3,
        angle_threshold=90
    )
    
    # 模拟多个轨迹
    trajectories = {
        # 正常行驶 (N->S)
        1: [(100, 100), (100, 150), (100, 200), (100, 250)],
        # 逆行 (S->N)
        2: [(100, 250), (100, 200), (100, 150), (100, 100)],
        # 横向行驶 (W->E)
        3: [(100, 150), (150, 150), (200, 150), (250, 150)],
        # 轨迹太短
        4: [(100, 100), (100, 150)],
    }
    
    results = detector.detect_batch(trajectories, "INT_001")
    
    print(f"\n📊 检测结果:")
    for result in results:
        status = "🚨 逆行" if result.is_violation else "✅ 正常"
        print(f"   ID:{result.track_id} | {status} | {result.direction} | "
              f"置信度: {result.confidence:.2f} | {result.reason}")


def test_special_vehicle_filter():
    """测试特殊车辆过滤器"""
    print("\n" + "=" * 60)
    print("🚑 测试特殊车辆过滤器")
    print("=" * 60)
    
    filter = SpecialVehicleFilter()
    
    test_vehicles = [
        "car",
        "ambulance",
        "police",
        "fire_truck",
        "truck",
        "suv",
        "emergency_vehicle",
    ]
    
    print(f"\n📊 特殊车辆检测:")
    for vehicle in test_vehicles:
        is_special = filter.is_special(vehicle)
        special_type = filter.get_special_type(vehicle)
        should_exempt = filter.should_exempt(vehicle)
        
        status = "🚨 特殊" if is_special else "✅ 普通"
        print(f"   {vehicle}: {status} | 类型: {special_type} | 豁免: {should_exempt}")


def test_full_pipeline():
    """测试完整流程（模拟）"""
    print("\n" + "=" * 60)
    print("🔗 测试完整逆行检测流程")
    print("=" * 60)
    
    # 1. 模拟跟踪器输出
    trajectories = {
        1: [(590, 371), (600, 380), (610, 390), (620, 400), (630, 410)],
        2: [(558, 119), (555, 125), (552, 131), (549, 137), (546, 143)],
        3: [(544, 87), (548, 90), (552, 93), (556, 96), (560, 99)],
    }
    
    # 2. 逆行检测
    detector = ViolationDetector()
    results = detector.detect_batch(trajectories, "INT_DETRAC")
    
    # 3. 特殊车辆过滤
    filter = SpecialVehicleFilter()
    
    print(f"\n📊 完整检测结果:")
    for result in results:
        # 模拟车辆类型（实际项目中从检测器获取）
        vehicle_type = "car"
        is_special = filter.is_special(vehicle_type)
        
        status = "🚨 逆行" if result.is_violation else "✅ 正常"
        if is_special and result.is_violation:
            status = "🔵 特殊车辆豁免"
        
        print(f"   ID:{result.track_id} | {status} | {result.direction} | "
              f"置信度: {result.confidence:.2f}")


if __name__ == "__main__":
    test_direction_checker()
    test_violation_detector()
    test_special_vehicle_filter()
    test_full_pipeline()