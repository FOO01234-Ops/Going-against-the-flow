# src/api/schemas/__init__.py
"""
API 数据模型
"""
from .models import *

__all__ = [
    "DetectRequest",
    "UpdateTicketRequest",
    "SearchTicketsRequest",
    "DetectionResult",
    "TrackResult",
    "ViolationResult",
    "PlateResult",
    "TicketResponse",
    "ProcessFrameResponse",
    "TicketsResponse",
    "StatisticsResponse",
    "ViolationType",
    "TicketStatus",
    "VehicleType",
]