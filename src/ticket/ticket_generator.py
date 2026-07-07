"""
工单生成器
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from .models import ViolationTicket, ViolationType, VehicleType, TicketStatus
from .database import TicketDatabase


class TicketGenerator:
    """违规工单生成器"""
    
    def __init__(self, db: TicketDatabase, snapshot_dir: str = "output/snapshots"):
        self.db = db
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._counter = 0
    
    def generate_ticket(
        self,
        plate_number: str,
        vehicle_type: VehicleType,
        intersection_id: str,
        camera_id: str,
        direction: str,
        frame: np.ndarray,
        bbox: tuple,
        violation_type: ViolationType = ViolationType.REVERSE_DRIVING,
        is_special: bool = False,
        special_type: Optional[str] = None,
        confidence: float = 0.0,
        vehicle_color: Optional[str] = None
    ) -> ViolationTicket:
        """
        生成违规工单
        
        Args:
            plate_number: 车牌号码
            vehicle_type: 车型
            intersection_id: 路口ID
            camera_id: 摄像头ID
            direction: 行驶方向
            frame: 视频帧
            bbox: 车辆边界框 (x1, y1, x2, y2)
            violation_type: 违规类型
            is_special: 是否特殊车辆
            special_type: 特殊车辆类型
            confidence: 检测置信度
            vehicle_color: 车辆颜色
        
        Returns:
            ViolationTicket: 生成的工单对象
        """
        # 生成工单ID
        ticket_id = self._generate_ticket_id()
        
        # 保存证据截图
        snapshot_path = self._save_snapshot(frame, bbox, ticket_id)
        
        # 创建工单
        ticket = ViolationTicket(
            ticket_id=ticket_id,
            violation_type=violation_type,
            plate_number=plate_number,
            vehicle_type=vehicle_type,
            vehicle_color=vehicle_color,
            violation_time=datetime.now(),
            intersection_id=intersection_id,
            camera_id=camera_id,
            direction=direction,
            snapshot_path=str(snapshot_path),
            video_clip_path=None,  # 预留视频片段
            status=TicketStatus.PENDING if not is_special else TicketStatus.DISMISSED,
            is_special_vehicle=is_special,
            special_vehicle_type=special_type,
            confidence_score=confidence
        )
        
        # 保存到数据库
        self.db.save_ticket(ticket)
        
        return ticket
    
    def _generate_ticket_id(self) -> str:
        """生成工单编号"""
        self._counter += 1
        date_str = datetime.now().strftime("%Y%m%d")
        return f"T{date_str}{self._counter:06d}"
    
    def _save_snapshot(self, frame: np.ndarray, bbox: tuple, ticket_id: str) -> Path:
        """保存违规截图"""
        x1, y1, x2, y2 = bbox
        
        # 裁剪车辆区域
        h, w = frame.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2))
        
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # 在截图上添加标注
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            annotated,
            f"TICKET: {ticket_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        
        # 保存
        filename = f"{ticket_id}.jpg"
        filepath = self.snapshot_dir / filename
        
        # 创建带标注的完整截图
        cv2.imwrite(str(filepath), annotated)
        
        return filepath