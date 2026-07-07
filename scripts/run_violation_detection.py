# scripts/run_violation_detection.py (完整修复版)
"""
完整逆行检测流水线 - 可视化展示
从DETRAC数据集读取图片，运行检测+跟踪+逆行检测，输出标注图片
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detection import YOLODetector
from src.tracking import DeepSORTTracker
from src.violation import ViolationDetector, DirectionChecker


class ViolationVisualizer:
    """
    逆行检测可视化器
    串联检测、跟踪、逆行检测，并输出标注图片
    """
    
    def __init__(self):
        self.detector = YOLODetector(
            model_name="yolo11n.pt",
            conf_threshold=0.4,
            device="cpu"
        )
        
        # ✅ 修正：使用 model_path 参数
        self.tracker = DeepSORTTracker(
            model_path="yolo11n.pt",  # 与 tracker.py 中的参数名一致
        )
        
        self.violation_detector = ViolationDetector(
            min_trajectory_length=3,
            angle_threshold=90
        )
        
        self.direction_checker = DirectionChecker(min_trajectory_length=3)
        
        # 存储所有帧的轨迹
        self.all_trajectories = defaultdict(list)
        
        # 颜色映射
        self.colors = {}
        
        # 统计
        self.stats = {
            "total_frames": 0,
            "total_detections": 0,
            "total_tracks": 0,
            "violations": [],
            "normal": [],
        }
        
        print("✅ 逆行检测可视化器初始化完成")
    
    def process_frame(self, frame: np.ndarray, frame_id: int, sequence_name: str) -> np.ndarray:
        """处理单帧"""
        if frame is None:
            return None
        
        self.stats["total_frames"] += 1
        
        # 1. YOLO检测
        detections = self.detector.detect_vehicles_only(frame)
        self.stats["total_detections"] += len(detections)
        
        # 2. 格式化为跟踪输入
        track_input = []
        for det in detections:
            track_input.append({
                "bbox": det.bbox,
                "confidence": det.confidence,
                "class_id": det.class_id,
                "class_name": det.class_name,
            })
        
        # 3. 跟踪
        tracks = self.tracker.update(frame, detections=track_input)
        self.stats["total_tracks"] += len(tracks)
        
        # 4. 记录轨迹
        for track in tracks:
            centroid = ((track.bbox[0] + track.bbox[2]) // 2,
                       (track.bbox[1] + track.bbox[3]) // 2)
            self.all_trajectories[track.track_id].append(centroid)
        
        # 5. 逆行检测
        violation_results = []
        for track in tracks:
            track_id = track.track_id
            trajectory = self.all_trajectories.get(track_id, [])
            
            if len(trajectory) >= 3:
                result = self.violation_detector.detect(
                    track_id,
                    trajectory,
                    intersection_id="INT_DETRAC"
                )
                violation_results.append(result)
                
                if result.is_violation:
                    self.stats["violations"].append({
                        "frame": frame_id,
                        "track_id": track_id,
                        "bbox": track.bbox,
                        "direction": result.direction,
                        "confidence": result.confidence
                    })
                else:
                    self.stats["normal"].append({
                        "frame": frame_id,
                        "track_id": track_id,
                        "bbox": track.bbox,
                        "direction": result.direction
                    })
        
        # 6. 绘制结果
        annotated = self.draw_results(frame, tracks, violation_results)
        
        return annotated
    
    def draw_results(self, frame: np.ndarray, tracks: List, violation_results: List) -> np.ndarray:
        """绘制检测结果"""
        annotated = frame.copy()
        
        violation_ids = {r.track_id for r in violation_results if r.is_violation}
        
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.bbox
            
            is_violation = track_id in violation_ids
            
            if is_violation:
                color = (0, 0, 255)
                status_text = "🚨 REVERSE"
            else:
                color = (0, 255, 0)
                status_text = "✅ NORMAL"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            label = f"ID:{track_id} {track.class_name} {status_text}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if track_id in self.all_trajectories:
                trajectory = list(self.all_trajectories[track_id])
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(annotated, trajectory[i-1], trajectory[i], color, 2)
                    cv2.circle(annotated, trajectory[-1], 5, color, -1)
            
            if track_id in self.all_trajectories and len(self.all_trajectories[track_id]) >= 2:
                traj = list(self.all_trajectories[track_id])
                if len(traj) >= 2:
                    start = traj[0]
                    end = traj[-1]
                    cv2.arrowedLine(annotated, start, end, (255, 255, 0), 2, tipLength=0.3)
        
        # 添加帧信息
        cv2.putText(annotated, f"Frames: {self.stats['total_frames']}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def process_sequence(self, sequence_path: Path, max_frames: int = 30, sample_interval: int = 2):
        """处理整个视频序列"""
        print(f"\n📹 处理序列: {sequence_path.name}")
        
        images = sorted(sequence_path.glob("img*.jpg"))
        
        if not images:
            print(f"   ⚠️ 没有找到图片")
            return None
        
        sample_images = images[::sample_interval]
        if max_frames > 0:
            sample_images = sample_images[:max_frames]
        
        print(f"   📊 处理 {len(sample_images)} 张图片 (采样间隔: {sample_interval})")
        
        output_dir = Path("output/violation_demo") / sequence_path.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_trajectories.clear()
        self.colors.clear()
        self.stats["violations"] = []
        self.stats["normal"] = []
        self.stats["total_frames"] = 0
        self.stats["total_detections"] = 0
        self.stats["total_tracks"] = 0
        
        for idx, img_path in enumerate(sample_images):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            
            annotated = self.process_frame(frame, idx, sequence_path.name)
            
            if annotated is not None:
                output_path = output_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(output_path), annotated)
                
                if (idx + 1) % 5 == 0 or idx == 0:
                    violation_count = len(self.stats["violations"])
                    normal_count = len(self.stats["normal"])
                    print(f"   📊 进度: {idx+1}/{len(sample_images)} | "
                          f"违规: {violation_count} | 正常: {normal_count}")
        
        self.print_stats(sequence_path.name)
        self.generate_summary(sequence_path.name, output_dir)
        
        return output_dir
    
    def print_stats(self, sequence_name: str):
        """打印统计信息"""
        print(f"\n📊 统计结果: {sequence_name}")
        print(f"   - 处理帧数: {self.stats['total_frames']}")
        print(f"   - 检测到车辆: {self.stats['total_detections']}")
        print(f"   - 跟踪目标: {self.stats['total_tracks']}")
        print(f"   - 🚨 逆行车辆: {len(self.stats['violations'])}")
        print(f"   - ✅ 正常车辆: {len(self.stats['normal'])}")
    
    def generate_summary(self, sequence_name: str, output_dir: Path):
        """生成汇总图"""
        output_files = list(output_dir.glob("frame_*.jpg"))
        if not output_files:
            return
        
        last_frame = sorted(output_files)[-1]
        summary_img = cv2.imread(str(last_frame))
        
        h, w = summary_img.shape[:2]
        
        cv2.rectangle(summary_img, (10, 10), (450, 170), (0, 0, 0), -1)
        cv2.rectangle(summary_img, (10, 10), (450, 170), (255, 255, 255), 1)
        
        y = 40
        cv2.putText(summary_img, f"Sequence: {sequence_name}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(summary_img, f"Frames: {self.stats['total_frames']}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(summary_img, f"🚨 Violations: {len(self.stats['violations'])}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 30
        cv2.putText(summary_img, f"✅ Normal: {len(self.stats['normal'])}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        summary_path = output_dir / "summary.jpg"
        cv2.imwrite(str(summary_path), summary_img)
        print(f"   📊 汇总图保存到: {summary_path}")


def find_detrac_sequences():
    """查找DETRAC数据集中的序列"""
    base_path = Path("data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train")
    
    if not base_path.exists():
        print(f"❌ DETRAC数据集不存在: {base_path}")
        return []
    
    sequences = []
    for seq_dir in sorted(base_path.glob("MVI_*")):
        if seq_dir.is_dir():
            image_count = len(list(seq_dir.glob("img*.jpg")))
            if image_count > 0:
                sequences.append(seq_dir)
    
    return sequences


def main():
    """主函数"""
    print("=" * 70)
    print("🚦 真实逆行检测可视化")
    print("=" * 70)
    
    sequences = find_detrac_sequences()
    
    if not sequences:
        print("❌ 未找到DETRAC数据集")
        print("请确保数据在: data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train/")
        return
    
    print(f"\n📁 找到 {len(sequences)} 个视频序列")
    
    selected_sequences = sequences[:3]
    print(f"📌 处理前 {len(selected_sequences)} 个序列")
    
    visualizer = ViolationVisualizer()
    
    all_outputs = []
    for seq in selected_sequences:
        try:
            output_dir = visualizer.process_sequence(
                seq,
                max_frames=30,
                sample_interval=2
            )
            if output_dir:
                all_outputs.append(output_dir)
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("🎉 处理完成!")
    print("=" * 70)
    print(f"\n📁 输出目录:")
    for out in all_outputs:
        print(f"   - {out}")
    print("\n💡 查看 output/violation_demo/ 目录下的图片")


if __name__ == "__main__":
    main()