"""
Stream Manager - Production Stable Version
YOLO + DeepSORT + Violation + EventFilter + Safety Guard
"""

import cv2
import time
import threading
from typing import Dict, Callable
from collections import deque, defaultdict

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.tracking.trajectory import TrajectoryManager
from src.violation.violation_detector import ViolationDetector


# ============================================================
# Event Filter（去重+限流）
# ============================================================

class EventFilter:

    def __init__(self, ttl: int = 5):
        self.ttl = ttl
        self.last_event_time = defaultdict(float)
        self.event_queue = deque(maxlen=200)
        self.window_count = defaultdict(int)
        self.window_time = defaultdict(float)

    def allow(self, event: dict) -> bool:

        key = (event["camera_id"], event["track_id"])
        now = time.time()

        # =========================
        # 时间窗口限流（1秒1次）
        # =========================
        if now - self.window_time[key] > 1:
            self.window_time[key] = now
            self.window_count[key] = 0

        self.window_count[key] += 1

        if self.window_count[key] > 1:
            return False

        # =========================
        # TTL去重
        # =========================
        if now - self.last_event_time[key] < self.ttl:
            return False

        self.last_event_time[key] = now
        self.event_queue.append(event)

        return True


# ============================================================
# StreamTask（单摄像头）
# ============================================================

class StreamTask:

    def __init__(self, camera_id: str, source: str):

        self.camera_id = camera_id
        self.source = source
        self.running = False

        self.cap = None
        self.thread = None

        self.frame_buffer = deque(maxlen=10)
        self.event_buffer = deque(maxlen=100)

        self.event_callback: Callable = None

        # AI modules
        self.model = YOLO("best.pt")

        self.deepsort = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.3
        )

        self.trajectory_manager = TrajectoryManager()
        self.violation = ViolationDetector()

        self.event_filter = EventFilter(ttl=5)

    # ============================================================
    # callback
    # ============================================================

    def set_event_callback(self, callback: Callable):
        self.event_callback = callback

    # ============================================================
    # start
    # ============================================================

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source)

        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    # ============================================================
    # stop
    # ============================================================

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    # ============================================================
    # main loop（生产稳定版）
    # ============================================================

    def _run(self):

        while self.running:

            try:

                ret, frame = self.cap.read()

                # =========================
                # 断流重连
                # =========================
                if not ret:
                    print(f"[WARN] {self.camera_id} reconnecting...")
                    self.cap.release()
                    time.sleep(2)
                    self.cap = cv2.VideoCapture(self.source)
                    continue

                # =========================
                # YOLO
                # =========================

                results = self.model(frame)[0]

                detections = []

                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1],
                        conf,
                        cls
                    ))

                # =========================
                # DeepSORT
                # =========================

                tracks = self.deepsort.update_tracks(
                    detections,
                    frame=frame
                )

                track_data = {}

                # =========================
                # trajectory
                # =========================

                for track in tracks:

                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id
                    x1, y1, x2, y2 = track.to_ltrb()

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    self.trajectory_manager.add_point(track_id, (cx, cy))

                    traj = self.trajectory_manager.get_trajectory(track_id)

                    if not traj:
                        continue

                    track_data[track_id] = traj.get_trajectory()

                # =========================
                # violation
                # =========================

                for track_id, trajectory in track_data.items():

                    result = self.violation.detect(
                        track_id=track_id,
                        trajectory=trajectory,
                        intersection_id=self.camera_id
                    )

                    if result.is_violation:

                        event = {
                            "type": "violation",
                            "camera_id": self.camera_id,
                            "track_id": track_id,
                            "confidence": result.confidence,
                            "direction": result.direction,
                            "reason": result.reason,
                            "timestamp": time.time()
                        }

                        self.event_buffer.append(event)

                        # =========================
                        # event filter + callback
                        # =========================
                        if self.event_filter.allow(event):

                            if self.event_callback:
                                self.event_callback(event)

                time.sleep(0.03)

            except Exception as e:
                print(f"[Stream Error] {self.camera_id}: {e}")
                time.sleep(1)


# ============================================================
# StreamManager（全局）
# ============================================================

class StreamManager:

    def __init__(self):
        self.tasks: Dict[str, StreamTask] = {}

    def start_stream(self, camera_id: str, source: str):

        if camera_id in self.tasks:
            return {"status": "already_running"}

        task = StreamTask(camera_id, source)
        task.start()

        self.tasks[camera_id] = task

        return {"status": "started"}

    def stop_stream(self, camera_id: str):

        if camera_id not in self.tasks:
            return {"status": "not_found"}

        self.tasks[camera_id].stop()
        del self.tasks[camera_id]

        return {"status": "stopped"}

    def get_status(self):

        return {
            cam_id: {
                "running": task.running,
                "events": len(task.event_buffer)
            }
            for cam_id, task in self.tasks.items()
        }