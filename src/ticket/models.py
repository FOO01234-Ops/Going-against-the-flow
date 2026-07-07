# src/ticket/models.py
"""
工单数据模型定义
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import json


class ViolationType(Enum):
    """违规类型枚举"""
    REVERSE_DRIVING = "reverse_driving"
    SPEEDING = "speeding"
    RED_LIGHT = "red_light"


class TicketStatus(Enum):
    """工单状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    PROCESSED = "processed"


class VehicleType(Enum):
    """车型枚举"""
    CAR = "car"
    SUV = "suv"
    VAN = "van"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"
    UNKNOWN = "unknown"


@dataclass
class ViolationTicket:
    """
    违规工单数据模型
    """
    # ============================================
    # 1. 必填字段（没有默认值）—— 必须放在最前面
    # ============================================
    ticket_id: str
    violation_type: ViolationType
    plate_number: str
    vehicle_type: VehicleType
    violation_time: datetime
    intersection_id: str
    camera_id: str
    direction: str
    snapshot_path: str
    status: TicketStatus

    # ============================================
    # 2. 可选字段（有默认值）—— 放在后面
    # ============================================
    vehicle_color: Optional[str] = None
    video_clip_path: Optional[str] = None
    is_special_vehicle: bool = False
    special_vehicle_type: Optional[str] = None
    review_time: Optional[datetime] = None
    reviewer: Optional[str] = None
    review_comment: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "ticket_id": self.ticket_id,
            "violation_type": self.violation_type.value,
            "plate_number": self.plate_number,
            "vehicle_type": self.vehicle_type.value,
            "vehicle_color": self.vehicle_color,
            "violation_time": self.violation_time.isoformat(),
            "intersection_id": self.intersection_id,
            "camera_id": self.camera_id,
            "direction": self.direction,
            "snapshot_path": self.snapshot_path,
            "video_clip_path": self.video_clip_path,
            "status": self.status.value,
            "is_special_vehicle": self.is_special_vehicle,
            "special_vehicle_type": self.special_vehicle_type,
            "review_time": self.review_time.isoformat() if self.review_time else None,
            "reviewer": self.reviewer,
            "review_comment": self.review_comment,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence_score": self.confidence_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ViolationTicket":
        """从字典创建实例"""
        return cls(
            ticket_id=data["ticket_id"],
            violation_type=ViolationType(data["violation_type"]),
            plate_number=data["plate_number"],
            vehicle_type=VehicleType(data["vehicle_type"]),
            vehicle_color=data.get("vehicle_color"),
            violation_time=datetime.fromisoformat(data["violation_time"]),
            intersection_id=data["intersection_id"],
            camera_id=data["camera_id"],
            direction=data["direction"],
            snapshot_path=data["snapshot_path"],
            video_clip_path=data.get("video_clip_path"),
            status=TicketStatus(data.get("status", "pending")),
            is_special_vehicle=data.get("is_special_vehicle", False),
            special_vehicle_type=data.get("special_vehicle_type"),
            review_time=datetime.fromisoformat(data["review_time"]) if data.get("review_time") else None,
            reviewer=data.get("reviewer"),
            review_comment=data.get("review_comment"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            confidence_score=data.get("confidence_score", 0.0)
        )

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)