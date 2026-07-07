# src/api/routes/__init__.py
"""
API 路由
"""
from .detect import router as detect_router
from .tickets import router as tickets_router
from .statistics import router as statistics_router

__all__ = [
    "detect_router",
    "tickets_router",
    "statistics_router",
]