# src/api/schemas/models.py
"""
API 数据模型 - 带中文描述
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ViolationType(str, Enum):
    """违规类型"""
    REVERSE_DRIVING = "reverse_driving"
    SPEEDING = "speeding"
    RED_LIGHT = "red_light"


class TicketStatus(str, Enum):
    """工单状态"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    PROCESSED = "processed"


class VehicleType(str, Enum):
    """车型"""
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


# ============================================================
# 请求模型
# ============================================================

class DetectRequest(BaseModel):
    """检测请求"""
    image_base64: Optional[str] = Field(None, description="Base64编码的图片")
    image_url: Optional[str] = Field(None, description="图片URL地址")
    intersection_id: str = Field("INT_001", description="路口ID")
    camera_id: str = Field("CAM_01", description="摄像头ID")
    mock_plate: bool = Field(True, description="是否使用模拟车牌识别")


class UpdateTicketRequest(BaseModel):
    """更新工单请求"""
    status: TicketStatus = Field(..., description="新状态")
    reviewer: Optional[str] = Field(None, description="审核人")
    comment: Optional[str] = Field(None, description="审核意见")


# ============================================================
# 响应模型
# ============================================================

class DetectionResult(BaseModel):
    """检测结果"""
    bbox: List[int] = Field(..., description="边界框 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="置信度 (0-1)")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")


class TrackResult(BaseModel):
    """跟踪结果"""
    track_id: int = Field(..., description="跟踪ID")
    bbox: List[int] = Field(..., description="边界框 [x1, y1, x2, y2]")
    confidence: float = Field(..., description="置信度 (0-1)")
    class_id: int = Field(..., description="类别ID")
    class_name: str = Field(..., description="类别名称")
    age: int = Field(..., description="跟踪时长（帧数）")


class ViolationResult(BaseModel):
    """逆行检测结果"""
    track_id: int = Field(..., description="跟踪ID")
    is_violation: bool = Field(..., description="是否逆行")
    vehicle_angle: float = Field(..., description="车辆行驶角度")
    lane_angle: float = Field(..., description="车道预设角度")
    direction: str = Field(..., description="车辆方向")
    lane_direction: str = Field(..., description="车道方向")
    confidence: float = Field(..., description="逆行置信度")
    reason: str = Field(..., description="判定原因")


class PlateResult(BaseModel):
    """车牌识别结果"""
    plate: str = Field(..., description="车牌号码")
    confidence: float = Field(..., description="识别置信度")
    track_id: int = Field(..., description="关联的跟踪ID")


class TicketResponse(BaseModel):
    """工单响应"""
    ticket_id: str = Field(..., description="工单ID")
    violation_type: str = Field(..., description="违规类型")
    plate_number: str = Field(..., description="车牌号码")
    vehicle_type: str = Field(..., description="车型")
    vehicle_color: Optional[str] = Field(None, description="车辆颜色")
    violation_time: datetime = Field(..., description="违规时间")
    intersection_id: str = Field(..., description="路口ID")
    camera_id: str = Field(..., description="摄像头ID")
    direction: str = Field(..., description="行驶方向")
    snapshot_path: str = Field(..., description="截图路径")
    status: str = Field(..., description="工单状态")
    is_special_vehicle: bool = Field(..., description="是否特殊车辆")
    special_vehicle_type: Optional[str] = Field(None, description="特殊车辆类型")
    confidence_score: float = Field(..., description="检测置信度")
    reviewer: Optional[str] = Field(None, description="审核人")
    review_comment: Optional[str] = Field(None, description="审核意见")
    created_at: datetime = Field(..., description="创建时间")
    updated_at: datetime = Field(..., description="更新时间")


class ProcessFrameResponse(BaseModel):
    """单帧处理响应"""
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="消息")
    frame_id: Optional[int] = Field(None, description="帧ID")
    detections: List[DetectionResult] = Field(default_factory=list, description="检测结果")
    tracks: List[TrackResult] = Field(default_factory=list, description="跟踪结果")
    violations: List[ViolationResult] = Field(default_factory=list, description="逆行检测结果")
    plates: List[PlateResult] = Field(default_factory=list, description="车牌识别结果")
    tickets: List[TicketResponse] = Field(default_factory=list, description="生成的工单")
    stats: dict = Field(default_factory=dict, description="统计信息")


class TicketsResponse(BaseModel):
    """工单列表响应"""
    total: int = Field(..., description="总记录数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    tickets: List[TicketResponse] = Field(..., description="工单列表")


class StatisticsResponse(BaseModel):
    """统计响应"""
    total: int = Field(..., description="总工单数")
    today_count: int = Field(..., description="今日违规数")
    status_counts: dict = Field(..., description="状态分布")
    vehicle_counts: dict = Field(..., description="车型分布")