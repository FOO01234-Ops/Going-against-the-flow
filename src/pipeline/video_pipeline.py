# src/pipeline/video_pipeline.py
"""
视频处理流水线 - 处理整个视频
"""
import cv2
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from src.detection import YOLODetector
from src.tracking import DeepSORTTracker
from src.violation import ViolationDetector
from src.ocr import LicenseOCR
from src.ticket import TicketManager

from .frame_processor import FrameProcessor


class VideoPipeline:
    """
    视频处理流水线
    
    完整流程:
    视频帧 → YOLO11检测 → DeepSORT跟踪 → 逆行检测 → 车牌识别 → 工单生成
    """
    
    def __init__(
        self,
        detector: Optional[YOLODetector] = None,
        tracker: Optional[DeepSORTTracker] = None,
        violation_detector: Optional[ViolationDetector] = None,
        license_ocr: Optional[LicenseOCR] = None,
        ticket_manager: Optional[TicketManager] = None,
        intersection_id: str = "INT_001",
        camera_id: str = "CAM_01",
        mock_plate: bool = True,
    ):
        """初始化视频流水线"""
        # 创建默认组件
        self.detector = detector or YOLODetector(
            model_name="yolo11n.pt",
            conf_threshold=0.4,
            device="cpu"
        )
        
        self.tracker = tracker or DeepSORTTracker(
            model_path="yolo11n.pt",
            conf_threshold=0.4
        )
        
        self.violation_detector = violation_detector or ViolationDetector(
            min_trajectory_length=5,
            angle_threshold=90
        )
        
        self.license_ocr = license_ocr or LicenseOCR(
            use_gpu=False,
            mock_mode=mock_plate
        )
        
        self.ticket_manager = ticket_manager or TicketManager(
            "data/database/violation.db"
        )
        
        self.intersection_id = intersection_id
        self.camera_id = camera_id
        self.mock_plate = mock_plate
        
        # 创建帧处理器
        self.frame_processor = FrameProcessor(
            detector=self.detector,
            tracker=self.tracker,
            violation_detector=self.violation_detector,
            license_ocr=self.license_ocr,
            ticket_manager=self.ticket_manager,
            intersection_id=intersection_id,
            camera_id=camera_id,
            min_trajectory_length=5,
            mock_plate=mock_plate,
        )
        
        print(f"✅ 视频流水线初始化成功")
        print(f"   📌 路口: {intersection_id}")
        print(f"   📌 摄像头: {camera_id}")
        print(f"   📌 模拟车牌: {mock_plate}")
    
    def process_image(
        self,
        image_path: Path,
        output_path: Optional[Path] = None,
        show: bool = False,
    ) -> Dict:
        """处理单张图片"""
        print(f"\n📸 处理图片: {image_path}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"❌ 无法读取图片: {image_path}")
            return {"error": "无法读取图片"}
        
        start_time = time.time()
        result = self.frame_processor.process_frame(frame)
        elapsed = time.time() - start_time
        
        print(f"   ⏱️  处理时间: {elapsed:.3f}s")
        
        # ✅ 打印统计
        self._print_stats(result)
        
        if output_path and result.get("annotated_frame") is not None:
            cv2.imwrite(str(output_path), result["annotated_frame"])
            print(f"   💾 保存到: {output_path}")
        
        if show and result.get("annotated_frame") is not None:
            cv2.imshow("Result", result["annotated_frame"])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result
    
    def _print_stats(self, result: Dict):
        """打印处理结果统计"""
        violations = [v for v in result.get("violations", []) if v.is_violation]
        tickets = result.get("tickets", [])
        
        print(f"\n   📊 处理结果:")
        print(f"      - 检测车辆: {len(result.get('detections', []))}")
        print(f"      - 跟踪目标: {len(result.get('tracks', []))}")
        print(f"      - 🚨 逆行: {len(violations)}")
        print(f"      - 🚗 车牌识别: {len(result.get('plates', []))}")
        print(f"      - 📋 生成工单: {len(tickets)}")
        
        if tickets:
            print(f"\n   📋 工单详情:")
            for t in tickets:
                print(f"      - {t.ticket_id}: {t.plate_number} | {t.violation_time}")
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        max_frames: int = -1,
        sample_interval: int = 1,
        show: bool = False,
    ) -> Dict:
        """处理视频"""
        video_path = str(video_path)
        print(f"\n📹 处理视频: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return {"error": "无法打开视频"}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   📊 视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"   💾 输出视频: {output_path}")
        
        stats = {
            "frames_processed": 0,
            "detections": 0,
            "tracks": 0,
            "violations": 0,
            "plates": 0,
            "tickets": 0,
            "total_time": 0,
        }
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % sample_interval != 0:
                continue
            
            if max_frames > 0 and stats["frames_processed"] >= max_frames:
                break
            
            result = self.frame_processor.process_frame(frame)
            
            if result.get("annotated_frame") is not None:
                annotated = result["annotated_frame"]
                
                if out:
                    out.write(annotated)
                
                stats["frames_processed"] += 1
                stats["detections"] += len(result.get("detections", []))
                stats["tracks"] += len(result.get("tracks", []))
                stats["violations"] += len([v for v in result.get("violations", []) if v.is_violation])
                stats["plates"] += len(result.get("plates", []))
                stats["tickets"] += len(result.get("tickets", []))
                
                if stats["frames_processed"] % 30 == 0:
                    print(f"   📊 进度: {stats['frames_processed']} 帧 | "
                          f"违规: {stats['violations']} | 工单: {stats['tickets']}")
                
                if show:
                    cv2.imshow("Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        cap.release()
        if out:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        stats["total_time"] = time.time() - start_time
        
        print(f"\n📊 视频处理完成!")
        print(f"   - 处理帧数: {stats['frames_processed']}")
        print(f"   - 检测车辆: {stats['detections']}")
        print(f"   - 跟踪目标: {stats['tracks']}")
        print(f"   - 🚨 逆行: {stats['violations']}")
        print(f"   - 🚗 车牌识别: {stats['plates']}")
        print(f"   - 📋 生成工单: {stats['tickets']}")
        print(f"   - ⏱️  总耗时: {stats['total_time']:.2f}s")
        
        return stats