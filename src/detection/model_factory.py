# src/detection/model_factory.py
"""
模型工厂 - 管理和下载YOLO模型
"""
from pathlib import Path
from typing import Dict, Optional
import os

from ultralytics import YOLO


class ModelFactory:
    """YOLO模型工厂"""
    
    # 可用模型列表
    AVAILABLE_MODELS = {
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo11m": "yolo11m.pt",
        "yolo11l": "yolo11l.pt",
        "yolo11x": "yolo11x.pt",
    }
    
    # 模型大小 (MB)
    MODEL_SIZES = {
        "yolo11n": 5.9,
        "yolo11s": 21.6,
        "yolo11m": 49.8,
        "yolo11l": 83.6,
        "yolo11x": 136.7,
    }
    
    @classmethod
    def get_model(cls, model_name: str, download: bool = True) -> str:
        """
        获取模型路径
        
        Args:
            model_name: 模型名称 (yolo11n, yolo11s, etc.)
            download: 是否自动下载
        
        Returns:
            str: 模型路径
        """
        if model_name in cls.AVAILABLE_MODELS:
            model_path = cls.AVAILABLE_MODELS[model_name]
            
            # 检查本地是否存在
            if not Path(model_path).exists() and download:
                print(f"📦 下载模型: {model_path}")
                model = YOLO(model_path)  # 这会自动下载
                return model_path
            
            return model_path
        else:
            raise ValueError(f"未知模型: {model_name}. 可用: {list(cls.AVAILABLE_MODELS.keys())}")
    
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """列出所有可用模型"""
        return {
            name: {
                "file": path,
                "size_mb": cls.MODEL_SIZES.get(name, "unknown"),
            }
            for name, path in cls.AVAILABLE_MODELS.items()
        }
    
    @classmethod
    def get_recommended_model(cls, device: str = "cpu") -> str:
        """
        根据设备推荐模型
        
        Args:
            device: 设备类型 (cpu, cuda, mps)
        
        Returns:
            str: 推荐的模型名称
        """
        if device == "cuda":
            return "yolo11m"  # GPU推荐中型模型
        elif device == "mps":
            return "yolo11s"  # Apple M系列推荐小型
        else:
            return "yolo11n"  # CPU推荐轻量级