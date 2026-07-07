# tests/test_tracker.py
"""
DeepSORT 跟踪器测试
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from src.detection import YOLODetector
from src.tracking import DeepSORTTracker


def test_tracker_with_detector():
    """测试 YOLO + DeepSORT 联合跟踪"""
    print("=" * 60)
    print("🔍 测试 YOLO + DeepSORT 联合跟踪")
    print("=" * 60)
    
    # 1. 创建检测器
    detector = YOLODetector(
        model_name="yolo11n.pt",
        conf_threshold=0.4,
        device="cpu"
    )
    
    # 2. 创建跟踪器
    tracker = DeepSORTTracker(
        max_age=30,
        min_hits=3,
        use_appearance=False,
    )
    
    # 3. 测试图片
    test_image_path = Path("data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011/img00001.jpg")
    
    if not test_image_path.exists():
        print(f"⚠️ 测试图片不存在: {test_image_path}")
        return
    
    test_image = cv2.imread(str(test_image_path))
    print(f"\n📸 测试图片: {test_image_path}")
    
    # 4. 检测
    print("\n🔍 执行检测...")
    detections = detector.detect_vehicles_only(test_image)
    
    print(f"✅ 检测到 {len(detections)} 辆车")
    
    # 5. 转换为跟踪格式
    track_input = []
    for det in detections:
        track_input.append({
            "bbox": det.bbox,
            "confidence": det.confidence,
            "class_id": det.class_id,
            "class_name": det.class_name,
        })
    
    # 6. 跟踪
    print("\n🎯 执行跟踪...")
    tracks = tracker.update(track_input, test_image)
    
    print(f"✅ 跟踪到 {len(tracks)} 个目标")
    for track in tracks:
        print(f"   ID:{track.track_id} | {track.class_name} | 置信度: {track.confidence:.2f} | 位置: {track.bbox}")
    
    # 7. 可视化
    annotated = tracker.draw_tracks(test_image, tracks, show_trajectory=True)
    
    output_path = Path("output/tracking_result.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    
    print(f"\n💾 结果保存到: {output_path}")


def test_tracker_single_image():
    """单图测试跟踪（显示ID）"""
    print("\n" + "=" * 60)
    print("🎯 测试单图跟踪（ID分配）")
    print("=" * 60)
    
    # 创建模拟检测（模拟多辆车）
    mock_detections = [
        {"bbox": (100, 100, 150, 180), "confidence": 0.9, "class_id": 2, "class_name": "car"},
        {"bbox": (300, 150, 370, 220), "confidence": 0.85, "class_id": 2, "class_name": "car"},
        {"bbox": (500, 120, 580, 200), "confidence": 0.8, "class_id": 2, "class_name": "car"},
        {"bbox": (700, 180, 780, 260), "confidence": 0.75, "class_id": 2, "class_name": "car"},
    ]
    
    tracker = DeepSORTTracker()
    
    # 创建模拟图片
    frame = np.ones((400, 900, 3), dtype=np.uint8) * 128
    
    # 执行跟踪
    tracks = tracker.update(mock_detections, frame)
    
    print(f"✅ 跟踪到 {len(tracks)} 个目标:")
    for track in tracks:
        print(f"   ID:{track.track_id} | {track.bbox}")
    
    # 可视化
    annotated = tracker.draw_tracks(frame, tracks)
    output_path = Path("output/tracking_single.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"\n💾 保存到: {output_path}")


if __name__ == "__main__":
    # 先测试单图ID分配
    test_tracker_single_image()
    
    # 再测试真实图片
    test_tracker_with_detector()