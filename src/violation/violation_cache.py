"""
Violation Cache - 工单去重控制（工业版）
防止重复生成工单
"""

import time
from collections import defaultdict


class ViolationCache:

    def __init__(self, ttl: int = 30):
        self.ttl = ttl
        self.cache = defaultdict(float)

    def allow(self, camera_id: str, track_id: int) -> bool:

        key = (camera_id, track_id)
        now = time.time()

        if now - self.cache[key] < self.ttl:
            return False

        self.cache[key] = now
        return True