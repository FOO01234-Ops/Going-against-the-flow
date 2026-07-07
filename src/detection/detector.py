# src/detection/detector.py
"""
YOLO11 车辆检测器
支持图片和视频流检测
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

from ultralytics import YOLO


@dataclass
class DetectionResult:
    """检测结果数据类"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                  # 置信度
    class_id: int                      # 类别ID
    class_name: str                    # 类别名称


class YOLODetector:
    """
    YOLO11 车辆检测器
    
    支持:
    - 图片检测
    - 视频流检测
    - 多种YOLO模型 (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)
    """
    
    # COCO数据集车辆相关类别
    VEHICLE_CLASSES = {
        2: "car",       # car
        3: "motorcycle", # motorcycle
        5: "bus",       # bus
        6: "train",     # train
        7: "truck",     # truck
    }
    
    # 车型映射
    VEHICLE_TYPE_MAP = {
        "car": "car",
        "motorcycle": "motorcycle",
        "bus": "bus",
        "train": "bus",
        "truck": "truck",
        "bicycle": "bicycle",
    }
    
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        classes: List[int] = None,
    ):
        """
        初始化YOLO检测器
        
        Args:
            model_name: 模型名称或路径 (yolo11n.pt, yolo11s.pt, etc.)
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IOU阈值
            device: 运行设备 (cpu, cuda, mps)
            classes: 只检测指定类别 (None表示检测所有)
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes or list(self.VEHICLE_CLASSES.keys())
        
        # 加载模型
        self.model = None
        self._load_model()
        
        print(f"✅ YOLO11 检测器初始化成功")
        print(f"   📦 模型: {model_name}")
        print(f"   🎯 置信度阈值: {conf_threshold}")
        print(f"   💻 设备: {device}")
        print(f"   🚗 检测类别: {len(self.classes)} 种车辆")
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            self.model = YOLO(self.model_name)
            # 如果指定了设备，移动模型到对应设备
            if self.device != "cpu":
                self.model.to(self.device)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def detect(
        self,
        image: Union[np.ndarray, str, Path],
        conf_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        检测单张图片中的车辆
        
        Args:
            image: 图片 (numpy数组) 或 图片路径
            conf_threshold: 自定义置信度阈值
        
        Returns:
            List[DetectionResult]: 检测结果列表
        """
        # 如果是路径，读取图片
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                print(f"❌ 无法读取图片: {image}")
                return []
        
        if image is None:
            return []
        
        # 使用指定的阈值或默认阈值
        conf = conf_threshold or self.conf_threshold
        
        # 执行推理
        results = self.model(
            image,
            conf=conf,
            iou=self.iou_threshold,
            classes=self.classes,
            verbose=False,
        )
        
        # 解析结果
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框 (xyxy格式)
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = map(int, xyxy[:4])
                    
                    # 获取置信度
                    confidence = float(box.conf.cpu().numpy().flatten()[0])
                    
                    # 获取类别
                    class_id = int(box.cls.cpu().numpy().flatten()[0])
                    class_name = self.model.names.get(class_id, "unknown")
                    
                    # 只保留车辆类别
                    if class_id in self.classes:
                        detections.append(
                            DetectionResult(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name
                            )
                        )
        
        return detections
    
    def detect_vehicles_only(
        self,
        image: Union[np.ndarray, str, Path],
        conf_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """
        仅检测车辆（过滤掉非车辆）
        
        Args:
            image: 图片或图片路径
            conf_threshold: 置信度阈值
        
        Returns:
            List[DetectionResult]: 车辆检测结果
        """
        all_detections = self.detect(image, conf_threshold)
        
        # 过滤出车辆类别
        vehicle_detections = [
            d for d in all_detections
            if d.class_id in self.classes
        ]
        
        return vehicle_detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: Optional[float] = None,
    ) -> List[List[DetectionResult]]:
        """
        批量检测多张图片
        
        Args:
            images: 图片列表
            conf_threshold: 置信度阈值
        
        Returns:
            List[List[DetectionResult]]: 每张图片的检测结果
        """
        conf = conf_threshold or self.conf_threshold
        
        results = []
        
        for image in images:
            detections = self.detect(image, conf)
            results.append(detections)
        
        return results
    
    def detect_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None,
        show: bool = False,
        on_frame: Optional[callable] = None,
    ):
        """
        处理视频流
        
        Args:
            video_path: 视频文件路径 (或 0 表示摄像头)
            output_path: 输出视频路径 (可选)
            conf_threshold: 置信度阈值
            show: 是否实时显示
            on_frame: 每帧回调函数 (frame, detections) -> processed_frame
        """
        video_path = str(video_path)
        conf = conf_threshold or self.conf_threshold
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 准备输出
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 输出视频: {output_path}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 检测
            detections = self.detect_vehicles_only(frame, conf)
            
            # 绘制结果
            annotated_frame = self.draw_detections(frame, detections)
            
            # 回调处理
            if on_frame:
                annotated_frame = on_frame(annotated_frame, detections)
            
            # 写入输出
            if out:
                out.write(annotated_frame)
            
            # 显示
            if show:
                cv2.imshow("YOLO11 Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 进度
            if frame_count % 30 == 0:
                print(f"   📊 处理帧: {frame_count}/{total_frames}")
        
        cap.release()
        if out:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        print(f"✅ 视频处理完成，共 {frame_count} 帧")
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        colors: Dict[str, tuple] = None,
    ) -> np.ndarray:
        """
        在图片上绘制检测结果
        
        Args:
            image: 原始图片
            detections: 检测结果列表
            colors: 颜色映射 {class_name: (R, G, B)}
        
        Returns:
            np.ndarray: 标注后的图片
        """
        if image is None:
            return image
        
        # 默认颜色
        if colors is None:
            colors = {
                "car": (0, 255, 0),
                "bus": (255, 165, 0),
                "truck": (0, 255, 255),
                "motorcycle": (255, 0, 255),
                "train": (255, 0, 0),
                "bicycle": (0, 0, 255),
            }
        
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (0, 255, 0))
            
            # 画边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "device": self.device,
            "classes": self.classes,
            "num_classes": len(self.classes),
        }


# 便捷函数
def create_detector(
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.5,
    device: str = "cpu",
) -> YOLODetector:
    """
    创建YOLO检测器的便捷函数
    
    Args:
        model_name: 模型名称
        conf_threshold: 置信度阈值
        device: 设备
    
    Returns:
        YOLODetector: 检测器实例
    """
    return YOLODetector(
        model_name=model_name,
        conf_threshold=conf_threshold,
        device=device,
    )