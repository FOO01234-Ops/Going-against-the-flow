# tests/test_ocr.py
"""
车牌识别测试 - 使用真实图片
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import random

from src.ocr import LicenseOCR


def test_ocr_real_image():
    """使用真实图片测试车牌识别"""
    print("=" * 60)
    print("🚗 测试车牌识别 (真实图片)")
    print("=" * 60)
    
    ocr = LicenseOCR(use_gpu=False)
    
    # 1. 从 DETRAC 数据集中找一张图片
    detrac_path = Path("data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train")
    
    if detrac_path.exists():
        # 随机选一个序列
        sequences = list(detrac_path.glob("MVI_*"))
        if sequences:
            seq = random.choice(sequences)
            images = list(seq.glob("img*.jpg"))
            if images:
                img_path = random.choice(images)
                print(f"\n📸 使用图片: {img_path}")
                test_image = cv2.imread(str(img_path))
            else:
                test_image = None
        else:
            test_image = None
    else:
        test_image = None
    
    if test_image is None:
        print("\n⚠️ 未找到 DETRAC 图片，使用模拟图片")
        test_image = create_mock_image()
    
    # 2. 执行识别
    print("\n🔍 执行车牌识别...")
    results = ocr.detect_plate(test_image)
    
    print(f"\n📊 识别结果: {len(results)} 个车牌")
    for i, result in enumerate(results):
        print(f"   {i+1}. 车牌: {result['plate']} | 置信度: {result['confidence']:.2f}")
    
    # 3. 保存结果
    if results:
        annotated = test_image.copy()
        for result in results:
            annotated = ocr.draw_plate_result(annotated, result)
        
        output_path = Path("output/ocr_result_real.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"\n💾 保存到: {output_path}")
    else:
        print("\n⚠️ 未识别到车牌 (DETRAC 数据集没有车牌标注，这是正常的)")
        print("   💡 DETRAC 是车辆检测数据集，不包含车牌信息")
        print("   💡 车牌识别需要专门的车牌数据集 (如 CCPD)")


def create_mock_image():
    """创建更真实的模拟图片（包含车辆区域）"""
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    
    # 画道路
    cv2.rectangle(frame, (0, 400), (1280, 720), (80, 80, 80), -1)
    cv2.rectangle(frame, (0, 350), (1280, 400), (180, 180, 180), -1)
    
    # 画车道线
    for i in range(0, 1280, 80):
        cv2.line(frame, (i, 370), (i + 40, 380), (255, 255, 255), 2)
    
    # 画一辆车（带车牌区域）
    # 车身
    cv2.rectangle(frame, (400, 300), (650, 480), (0, 0, 200), -1)
    cv2.rectangle(frame, (400, 300), (650, 480), (0, 0, 255), 3)
    
    # 车牌区域（白色方块模拟）
    cv2.rectangle(frame, (470, 440), (580, 470), (200, 200, 200), -1)
    cv2.rectangle(frame, (470, 440), (580, 470), (100, 100, 100), 2)
    
    # 模拟车牌文字
    cv2.putText(frame, "京A12345", (485, 463),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame


if __name__ == "__main__":
    test_ocr_real_image()