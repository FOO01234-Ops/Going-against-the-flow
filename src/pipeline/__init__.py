# src/pipeline/__init__.py
"""
完整流水线模块
"""
from .video_pipeline import VideoPipeline
from .frame_processor import FrameProcessor

__all__ = [
    "VideoPipeline",
    "FrameProcessor"
]