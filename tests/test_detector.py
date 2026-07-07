# tests/test_detector.py
"""
YOLO11 检测器测试
"""
import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection import YOLODetector


def test_image_detection():
    """测试单张图片检测"""
    print("=" * 60)
    print("🚗 测试 YOLO11 车辆检测")
    print("=" * 60)
    
    # 1. 创建检测器
    detector = YOLODetector(
        model_name="yolo11n.pt",
        conf_threshold=0.5,
        device="cpu"
    )
    
    print(f"\n📦 模型信息: {detector.get_model_info()}")
    
    # 2. 测试图片
    # 使用DETRAC数据集中的一张图片
    test_image_path = Path("data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011/img00001.jpg")
    
    if not test_image_path.exists():
        print(f"\n⚠️ 测试图片不存在: {test_image_path}")
        print("   创建模拟图片进行测试...")
        
        # 创建模拟图片
        test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        # 画一些模拟车辆
        cv2.rectangle(test_image, (100, 200), (200, 400), (128, 128, 128), -1)
        cv2.rectangle(test_image, (500, 150), (650, 380), (128, 128, 128), -1)
        cv2.rectangle(test_image, (800, 250), (950, 450), (128, 128, 128), -1)
    else:
        test_image = cv2.imread(str(test_image_path))
    
    # 3. 检测
    print(f"\n🔍 执行检测...")
    detections = detector.detect_vehicles_only(test_image)
    
    print(f"\n✅ 检测到 {len(detections)} 辆车:")
    for i, det in enumerate(detections):
        print(f"   {i+1}. {det.class_name} | 置信度: {det.confidence:.2f} | 位置: {det.bbox}")
    
    # 4. 保存结果
    output_path = Path("output/detection_result.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    annotated = detector.draw_detections(test_image, detections)
    cv2.imwrite(str(output_path), annotated)
    
    print(f"\n💾 结果保存到: {output_path}")


def test_video_detection():
    """测试视频检测"""
    print("\n" + "=" * 60)
    print("📹 测试视频检测")
    print("=" * 60)
    
    # 检查是否有测试视频
    test_video_path = Path("data/raw/test_video.mp4")
    
    if not test_video_path.exists():
        print("⚠️ 没有测试视频，跳过视频测试")
        print("   可以用摄像头测试: detector.detect_video(0, show=True)")
        return
    
    detector = YOLODetector(model_name="yolo11n.pt", device="cpu")
    
    output_path = Path("output/detection_video.mp4")
    
    detector.detect_video(
        video_path=str(test_video_path),
        output_path=str(output_path),
        show=False,
    )


if __name__ == "__main__":
    test_image_detection()
    test_video_detection()