"""
Event Filter - 工业级事件去重与限流
"""

import time
from collections import defaultdict, deque


class EventFilter:

    def __init__(self, ttl: int = 5):
        """
        ttl: 同一车辆事件冷却时间（秒）
        """
        self.ttl = ttl
        self.last_event_time = defaultdict(float)
        self.event_queue = deque(maxlen=100)

    def allow(self, event: dict) -> bool:

        key = (event["camera_id"], event["track_id"])
        now = time.time()

        # =========================
        # 1. 冷却机制（防重复报警）
        # =========================
        if now - self.last_event_time[key] < self.ttl:
            return False

        self.last_event_time[key] = now

        # =========================
        # 2. 加入队列（缓冲）
        # =========================
        self.event_queue.append(event)

        return True

    def get_events(self):
        return list(self.event_queue)