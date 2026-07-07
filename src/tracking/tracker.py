# src/tracking/tracker.py
"""
目标跟踪器 - 使用 Ultralytics 内置跟踪
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from ultralytics import YOLO


@dataclass
class TrackResult:
    """跟踪结果"""
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    age: int = 0
    hits: int = 0
    time_since_update: int = 0


class DeepSORTTracker:
    """
    目标跟踪器（使用 Ultralytics 内置跟踪）
    """
    
    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        conf_threshold: float = 0.4,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        
        # 加载 YOLO 模型（支持跟踪）
        self.model = YOLO(model_path)
        
        # 轨迹历史
        self.trajectories = {}  # track_id -> list of centroids
        
        print(f"✅ 跟踪器初始化成功 (使用 Ultralytics)")
        print(f"   📦 模型: {model_path}")
        print(f"   📌 max_age: {max_age}, min_hits: {min_hits}")
    
    def update(
        self,
        frame: np.ndarray,
        detections: Optional[List[Dict]] = None,
    ) -> List[TrackResult]:
        """
        更新跟踪
        
        Args:
            frame: 当前帧
            detections: 可选，外部检测结果
        
        Returns:
            List[TrackResult]: 跟踪结果
        """
        # 使用 YOLO 的跟踪功能
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker="bytetrack.yaml",
            verbose=False
        )
        
        tracks = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                # ✅ 修复：正确获取数据
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy()
                
                if boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy()
                    
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        track_id = int(track_ids[i])
                        class_id = int(cls_ids[i])
                        class_name = self.model.names.get(class_id, "vehicle")
                        confidence = float(confs[i])
                        
                        # 记录轨迹
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        if track_id not in self.trajectories:
                            self.trajectories[track_id] = deque(maxlen=100)
                        self.trajectories[track_id].append(centroid)
                        
                        tracks.append(
                            TrackResult(
                                track_id=track_id,
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name,
                                age=0,
                                hits=1,
                                time_since_update=0,
                            )
                        )
                else:
                    # 没有跟踪ID，只有检测
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = map(int, xyxy[i])
                        class_id = int(cls_ids[i])
                        class_name = self.model.names.get(class_id, "vehicle")
                        confidence = float(confs[i])
                        
                        # 分配临时ID
                        track_id = hash(f"{x1}_{y1}_{x2}_{y2}") % 10000
                        
                        tracks.append(
                            TrackResult(
                                track_id=track_id,
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                class_id=class_id,
                                class_name=class_name,
                            )
                        )
        
        return tracks
    
    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """获取指定轨迹"""
        if track_id in self.trajectories:
            return list(self.trajectories[track_id])
        return []
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """获取所有轨迹"""
        return {tid: list(trj) for tid, trj in self.trajectories.items()}
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[TrackResult],
        show_trajectory: bool = True,
        traj_length: int = 30,
    ) -> np.ndarray:
        """绘制跟踪结果"""
        annotated = frame.copy()
        colors = {}
        
        for track in tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.bbox
            
            if track_id not in colors:
                colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
            
            color = colors[track_id]
            
            # 画边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # 标签
            label = f"ID:{track_id} {track.class_name}"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # 画轨迹
            if show_trajectory and track_id in self.trajectories:
                trajectory = list(self.trajectories[track_id])[-traj_length:]
                if len(trajectory) > 1:
                    for i in range(1, len(trajectory)):
                        cv2.line(
                            annotated,
                            trajectory[i-1],
                            trajectory[i],
                            color,
                            2
                        )
                    cv2.circle(annotated, trajectory[-1], 5, color, -1)
        
        return annotated