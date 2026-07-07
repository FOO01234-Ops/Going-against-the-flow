# src/api/dependencies.py
"""
依赖注入
"""
from src.detection import YOLODetector
from src.tracking import DeepSORTTracker
from src.violation import ViolationDetector
from src.ocr import LicenseOCR
from src.ticket import TicketManager
from src.pipeline import FrameProcessor


# 全局单例
_detector = None
_tracker = None
_violation_detector = None
_license_ocr = None
_ticket_manager = None
_frame_processor = None


def get_detector():
    """获取检测器单例"""
    global _detector
    if _detector is None:
        _detector = YOLODetector(
            model_name="yolo11n.pt",
            conf_threshold=0.4,
            device="cpu"
        )
    return _detector


def get_tracker():
    """获取跟踪器单例"""
    global _tracker
    if _tracker is None:
        _tracker = DeepSORTTracker(
            model_path="yolo11n.pt",
            conf_threshold=0.4
        )
    return _tracker


def get_violation_detector():
    """获取逆行检测器单例"""
    global _violation_detector
    if _violation_detector is None:
        _violation_detector = ViolationDetector(
            min_trajectory_length=5,
            angle_threshold=90
        )
    return _violation_detector


def get_license_ocr():
    """获取车牌识别器单例"""
    global _license_ocr
    if _license_ocr is None:
        _license_ocr = LicenseOCR(
            use_gpu=False,
            mock_mode=True
        )
    return _license_ocr


def get_ticket_manager():
    """获取工单管理器单例"""
    global _ticket_manager
    if _ticket_manager is None:
        _ticket_manager = TicketManager("data/database/violation.db")
    return _ticket_manager


def get_frame_processor():
    """获取帧处理器单例"""
    global _frame_processor
    if _frame_processor is None:
        _frame_processor = FrameProcessor(
            detector=get_detector(),
            tracker=get_tracker(),
            violation_detector=get_violation_detector(),
            license_ocr=get_license_ocr(),
            ticket_manager=get_ticket_manager(),
            intersection_id="INT_001",
            camera_id="CAM_01",
            min_trajectory_length=5,
            mock_plate=True,
        )
    return _frame_processor