"""
Frame Processor - YOLO + Tracking + Violation Pipeline
"""
import cv2
import base64
from src.violation.violation_detector import ViolationDetector


class FrameProcessor:
    def __init__(self, yolo_model, tracker, ticket_manager):
        self.model = yolo_model
        self.tracker = tracker
        self.violation = ViolationDetector()
        self.ticket_manager = ticket_manager

    def process(self, frame, camera_id, visualize=False):
        """
        单帧处理入口

        Args:
            frame: 原始图像 (numpy array)
            camera_id: 摄像头ID（用于逆行判定配置）
            visualize: 是否返回标注图像

        Returns:
            events: 违章事件列表
            annotated_frame: 标注后的图像（若 visualize=True），否则 None
        """
        # 复制原图用于绘制（不污染原始帧）
        annotated = frame.copy() if visualize else None

        # =========================
        # 1. YOLO 检测
        # =========================
        results = self.model(frame)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": conf,
                    "cls": cls
                })

        # =========================
        # 2. 跟踪（参数顺序已修正）
        # =========================
        tracks = self.tracker.update(frame, detections)

        # =========================
        # 3. 绘制标注（如果需要）
        # =========================
        if visualize and annotated is not None and tracks:
            annotated = self.tracker.draw_tracks(
                annotated,
                tracks,
                show_trajectory=True,
                traj_length=30
            )

        events = []

        # =========================
        # 4. 逆行检测 & 工单生成
        # =========================
        for t in tracks:
            track_id = t.track_id
            trajectory = self.tracker.get_trajectory(track_id)

            result = self.violation.detect(
                track_id=track_id,
                trajectory=trajectory,
                intersection_id=camera_id
            )

            if result.is_violation:
                ticket = self.ticket_manager.create_ticket(
                    track_id=track_id,
                    result=result
                )
                events.append({
                    "type": "violation",
                    "track_id": track_id,
                    "confidence": result.confidence,
                    "ticket_id": ticket.id
                })

        return events, annotated