# tests/test_ocr_mock.py
"""
测试车牌识别（模拟模式）
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from src.ocr import LicenseOCR


def test_mock_mode():
    """测试模拟模式"""
    print("=" * 60)
    print("🚗 测试车牌识别 - 模拟模式")
    print("=" * 60)
    
    # ✅ 创建 OCR 实例，启用模拟模式
    ocr = LicenseOCR(
        use_gpu=False,
        mock_mode=True  # 关键参数
    )
    
    # 创建测试图片
    test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    
    # 模拟车辆区域
    cv2.rectangle(test_image, (400, 300), (650, 500), (0, 0, 200), -1)
    cv2.rectangle(test_image, (400, 300), (650, 500), (0, 0, 255), 3)
    
    # 模拟车牌区域
    cv2.rectangle(test_image, (460, 440), (590, 475), (200, 200, 200), -1)
    cv2.rectangle(test_image, (460, 440), (590, 475), (100, 100, 100), 2)
    
    # 识别
    print("\n🔍 执行车牌识别...")
    results = ocr.detect_plate(test_image)
    
    print(f"\n📊 识别结果: {len(results)} 个车牌")
    for i, result in enumerate(results):
        print(f"   {i+1}. 车牌: {result['plate']} | 置信度: {result['confidence']:.2f}")
    
    # 保存结果
    if results:
        annotated = ocr.draw_plate_result(test_image, results[0])
        output_path = Path("output/ocr_mock_result.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 保存到: {output_path}")
    else:
        print("\n⚠️ 没有识别到车牌（模拟模式有 30% 概率返回空）")
        print("   💡 再次运行可能会检测到")


def test_multiple_times():
    """多次测试，查看模拟数据的多样性"""
    print("\n" + "=" * 60)
    print("🔄 多次测试模拟模式")
    print("=" * 60)
    
    ocr = LicenseOCR(mock_mode=True)
    test_image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    
    plates = []
    for i in range(10):
        results = ocr.detect_plate(test_image)
        if results:
            plates.append(results[0]["plate"])
    
    print(f"\n📊 10 次测试中，检测到 {len(plates)} 次")
    if plates:
        print(f"   生成的车牌示例: {', '.join(plates[:5])}")


if __name__ == "__main__":
    test_mock_mode()
    test_multiple_times()