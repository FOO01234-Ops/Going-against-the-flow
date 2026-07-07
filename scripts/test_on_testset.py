# scripts/test_on_testset.py
"""
在测试集上评估系统性能
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import json
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import random

from src.pipeline import VideoPipeline
from src.detection import YOLODetector
from src.tracking import DeepSORTTracker
from src.violation import EnhancedViolationDetector
from src.ocr import LicenseOCR
from src.ticket import TicketManager


class TestSetEvaluator:
    """
    测试集评估器
    在测试集上评估逆行检测系统性能
    """
    
    def __init__(self):
        self.results = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "violations": 0,
            "tickets": 0,
            "processing_time": 0,
            "per_frame_times": [],
            "detection_stats": defaultdict(int),
            "violation_stats": defaultdict(int),
            "sequence_results": [],
        }
        
        # 创建流水线组件
        self.detector = YOLODetector(
            model_name="yolo11n.pt",
            conf_threshold=0.4,
            device="cpu"
        )
        
        self.tracker = DeepSORTTracker(
            model_path="yolo11n.pt",
            conf_threshold=0.4
        )
        
        self.violation_detector = EnhancedViolationDetector(
            config_path="config/camera_lane_map.json",
            min_trajectory_length=5,
            angle_threshold=90,
            min_confidence=0.5,
            verification_frames=3,
        )
        
        self.license_ocr = LicenseOCR(
            use_gpu=False,
            mock_mode=True
        )
        
        self.ticket_manager = TicketManager("data/database/test_eval.db")
        
        print("✅ 测试集评估器初始化成功")
    
    def find_test_images(self, test_path: str = "data/raw/DETRAC-test-data/Insight-MVT_Annotation_Test") -> List[Path]:
        """查找测试集中的所有图片"""
        test_dir = Path(test_path)
        
        if not test_dir.exists():
            # 如果没有测试集，使用训练集的一部分作为测试
            print(f"⚠️ 测试集不存在: {test_dir}")
            print("   使用训练集的一部分作为测试")
            test_dir = Path("data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train")
            
            if not test_dir.exists():
                print("❌ 没有找到任何数据")
                return []
        
        all_images = []
        for seq_dir in sorted(test_dir.glob("MVI_*")):
            if seq_dir.is_dir():
                images = list(seq_dir.glob("img*.jpg"))
                if images:
                    all_images.extend(images)
                    print(f"   📁 {seq_dir.name}: {len(images)} 张图片")
        
        return sorted(all_images)
    
    def evaluate_single_sequence(self, images: List[Path], max_frames: int = 30):
        """
        评估单个视频序列
        """
        sequence_name = images[0].parent.name if images else "unknown"
        print(f"\n📹 评估序列: {sequence_name} ({len(images)} 张图片)")
        
        # 采样
        sample_images = images[::5][:max_frames]  # 每5帧采样1帧，最多30帧
        
        sequence_results = {
            "sequence": sequence_name,
            "frames": len(sample_images),
            "detections": 0,
            "tracks": 0,
            "violations": 0,
            "tickets": 0,
            "time": 0,
        }
        
        for idx, img_path in enumerate(sample_images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            start_time = time.time()
            
            # 1. 检测
            detections = self.detector.detect_vehicles_only(frame)
            self.results["total_detections"] += len(detections)
            
            # 2. 跟踪
            track_input = []
            for det in detections:
                track_input.append({
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "class_id": det.class_id,
                    "class_name": det.class_name,
                })
            
            tracks = self.tracker.update(frame, detections=track_input)
            self.results["total_tracks"] += len(tracks)
            
            # 3. 逆行检测
            violation_count = 0
            for track in tracks:
                trajectory = self.tracker.get_trajectory(track.track_id)
                if len(trajectory) >= 5:
                    result = self.violation_detector.detect(
                        track.track_id,
                        trajectory,
                        camera_id="CAM_01",
                    )
                    if result.is_violation:
                        violation_count += 1
                        self.results["violations"] += 1
                        self.results["violation_stats"]["total"] += 1
                        
                        # 记录置信度分布
                        if result.violation_confidence > 0.8:
                            self.results["violation_stats"]["high_conf"] += 1
                        elif result.violation_confidence > 0.6:
                            self.results["violation_stats"]["mid_conf"] += 1
                        else:
                            self.results["violation_stats"]["low_conf"] += 1
            
            elapsed = time.time() - start_time
            
            self.results["total_frames"] += 1
            self.results["processing_time"] += elapsed
            self.results["per_frame_times"].append(elapsed)
            
            # 更新序列统计
            sequence_results["detections"] += len(detections)
            sequence_results["tracks"] += len(tracks)
            sequence_results["violations"] += violation_count
            
            # 进度显示
            if (idx + 1) % 10 == 0:
                print(f"   📊 进度: {idx+1}/{len(sample_images)} | "
                      f"违规: {sequence_results['violations']}")
        
        # 保存序列结果
        self.results["sequence_results"].append(sequence_results)
        
        # 打印序列结果
        print(f"\n   📊 {sequence_name} 结果:")
        print(f"      - 处理帧数: {sequence_results['frames']}")
        print(f"      - 检测车辆: {sequence_results['detections']}")
        print(f"      - 跟踪目标: {sequence_results['tracks']}")
        print(f"      - 🚨 逆行: {sequence_results['violations']}")
        print(f"      - ⏱️  平均耗时: {sequence_results['time']/max(1, sequence_results['frames']):.3f}s")
    
    def run_full_evaluation(self, max_sequences: int = 3, max_frames_per_sequence: int = 30):
        """
        运行完整评估
        """
        print("=" * 70)
        print("🚦 测试集评估开始")
        print("=" * 70)
        
        # 1. 查找测试图片
        all_images = self.find_test_images()
        
        if not all_images:
            print("❌ 没有找到测试图片")
            return
        
        print(f"\n📁 总共找到 {len(all_images)} 张测试图片")
        
        # 2. 按序列分组
        sequences = defaultdict(list)
        for img in all_images:
            seq_name = img.parent.name
            sequences[seq_name].append(img)
        
        print(f"📁 共 {len(sequences)} 个视频序列")
        
        # 3. 评估每个序列
        for idx, (seq_name, images) in enumerate(sorted(sequences.items())):
            if idx >= max_sequences:
                break
            self.evaluate_single_sequence(images, max_frames_per_sequence)
        
        # 4. 打印总体结果
        self.print_summary()
        
        # 5. 保存结果
        self.save_results()
    
    def print_summary(self):
        """打印总体结果"""
        print("\n" + "=" * 70)
        print("📊 测试集评估结果汇总")
        print("=" * 70)
        
        total_frames = self.results["total_frames"]
        total_detections = self.results["total_detections"]
        total_tracks = self.results["total_tracks"]
        total_violations = self.results["violations"]
        
        print(f"\n📈 总体统计:")
        print(f"   - 处理帧数: {total_frames}")
        print(f"   - 检测车辆: {total_detections}")
        print(f"   - 跟踪目标: {total_tracks}")
        print(f"   - 🚨 逆行检测: {total_violations}")
        
        if total_frames > 0:
            print(f"\n📊 平均每帧:")
            print(f"   - 车辆数: {total_detections / total_frames:.2f}")
            print(f"   - 跟踪目标: {total_tracks / total_frames:.2f}")
            print(f"   - 逆行数: {total_violations / total_frames:.2f}")
        
        # 处理时间
        avg_time = self.results["processing_time"] / max(1, total_frames)
        print(f"\n⏱️  性能:")
        print(f"   - 总耗时: {self.results['processing_time']:.2f}s")
        print(f"   - 平均每帧: {avg_time:.3f}s")
        print(f"   - 帧率: {1/avg_time if avg_time > 0 else 0:.1f} FPS")
        
        # 违规置信度分布
        if self.results["violation_stats"]:
            print(f"\n📊 违规置信度分布:")
            print(f"   - 高置信度 (>0.8): {self.results['violation_stats'].get('high_conf', 0)}")
            print(f"   - 中置信度 (0.6-0.8): {self.results['violation_stats'].get('mid_conf', 0)}")
            print(f"   - 低置信度 (<0.6): {self.results['violation_stats'].get('low_conf', 0)}")
    
    def save_results(self):
        """保存评估结果"""
        output_dir = Path("output/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存 JSON
        json_path = output_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 结果保存到: {json_path}")
        
        # 生成报告
        report_path = output_dir / f"evaluation_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("🚦 测试集评估报告\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"评估时间: {datetime.now()}\n")
            f.write(f"处理帧数: {self.results['total_frames']}\n")
            f.write(f"检测车辆: {self.results['total_detections']}\n")
            f.write(f"跟踪目标: {self.results['total_tracks']}\n")
            f.write(f"逆行检测: {self.results['violations']}\n\n")
            
            avg_time = self.results["processing_time"] / max(1, self.results["total_frames"])
            f.write(f"平均每帧耗时: {avg_time:.3f}s\n")
            f.write(f"帧率: {1/avg_time if avg_time > 0 else 0:.1f} FPS\n")
        
        print(f"💾 报告保存到: {report_path}")


def run_test():
    """运行测试"""
    evaluator = TestSetEvaluator()
    evaluator.run_full_evaluation(
        max_sequences=3,       # 测试3个序列
        max_frames_per_sequence=30  # 每个序列30帧
    )


if __name__ == "__main__":
    run_test()