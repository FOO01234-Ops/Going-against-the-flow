# scripts/run_pipeline.py
"""
运行完整流水线 - 测试端到端功能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import random

from src.pipeline import VideoPipeline


def test_on_single_image():
    """测试单张图片"""
    print("=" * 70)
    print("🚦 测试完整流水线 - 单张图片")
    print("=" * 70)
    
    # 1. 找一张 DETRAC 图片
    detrac_path = Path("data/raw/DETRAC-test-data/Insight-MVT_Annotation_Test")
    
    if not detrac_path.exists():
        print("❌ DETRAC 数据集不存在")
        return
    
    sequences = list(detrac_path.glob("MVI_*"))
    if not sequences:
        print("❌ 没有找到 DETRAC 序列")
        return
    
    seq = random.choice(sequences)
    images = list(seq.glob("img*.jpg"))
    if not images:
        print("❌ 没有找到图片")
        return
    
    img_path = random.choice(images)
    
    # 2. 创建流水线
    pipeline = VideoPipeline(
        intersection_id="INT_DETRAC",
        camera_id=seq.name,
        mock_plate=True,  # 使用模拟车牌
    )
    
    # 3. 处理图片
    output_dir = Path("output/pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"result_{img_path.stem}.jpg"
    
    result = pipeline.process_image(
        img_path,
        output_path=output_path,
        show=False,
    )


def test_on_video():
    """测试视频"""
    print("\n" + "=" * 70)
    print("🚦 测试完整流水线 - 视频")
    print("=" * 70)
    
    # 检查是否有测试视频
    test_video = Path("data/raw/test_video.mp4")
    
    if not test_video.exists():
        print("⚠️ 没有测试视频，跳过")
        print("   你可以用摄像头测试: pipeline.process_video(0, show=True)")
        return
    
    pipeline = VideoPipeline(mock_plate=True)
    
    output_dir = Path("output/pipeline_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline.process_video(
        video_path=str(test_video),
        output_path=str(output_dir / "output_video.mp4"),
        max_frames=100,
        sample_interval=5,
        show=False,
    )


if __name__ == "__main__":
    test_on_single_image()
    test_on_video()