"""
Event Bus - Redis版本（工业标准）
"""

import json
import redis


class EventBus:

    def __init__(self, host="localhost", port=6379, db=0):

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

        self.channel = "traffic_events"

    def publish(self, event: dict):

        self.client.publish(
            self.channel,
            json.dumps(event)
        )

    def subscribe(self):

        pubsub = self.client.pubsub()
        pubsub.subscribe(self.channel)

        return pubsub