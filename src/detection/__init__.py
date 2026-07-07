# src/detection/__init__.py
"""
车辆检测模块 - YOLO11
"""
from .detector import YOLODetector
from .model_factory import ModelFactory

__all__ = [
    "YOLODetector",
    "ModelFactory"
]