"""
Smart Traffic System - Final Integrated Version
（已整合：Stream / Redis / WebSocket / Tickets / Statistics）
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import asyncio
import json

from src.api.routes import detect, tickets, statistics
from src.api.stream.stream_manager import StreamManager
from src.api.stream.event_bus import EventBus
from src.utils.logger import get_logger


# ============================================================
# Logger
# ============================================================

logger = get_logger("API")

# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="🚦 Smart Traffic System",
    version="3.0.0"
)

# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Core Components
# ============================================================

stream_manager = StreamManager()

event_bus = None
pubsub = None


# ============================================================
# WebSocket Manager
# ============================================================

class ConnectionManager:

    def __init__(self):
        self.connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        logger.info("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)
        logger.info("WebSocket disconnected")

    async def send(self, data: dict):
        for conn in self.connections:
            try:
                await conn.send_json(data)
            except Exception as e:
                logger.error(f"Send error: {e}")


manager = ConnectionManager()


# ============================================================
# System API
# ============================================================

@app.get("/")
def root():
    return {
        "system": "Smart Traffic System",
        "status": "running",
        "time": datetime.now().isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_streams": len(stream_manager.tasks)
    }


# ============================================================
# Stream Control API
# ============================================================

@app.post("/api/stream/start")
def start_stream(camera_id: str, source: str):

    result = stream_manager.start_stream(camera_id, source)
    logger.info(f"Start stream: {camera_id}")
    return result


@app.post("/api/stream/stop")
def stop_stream(camera_id: str):

    result = stream_manager.stop_stream(camera_id)
    logger.info(f"Stop stream: {camera_id}")
    return result


@app.get("/api/stream/status")
def stream_status():
    return stream_manager.status()


# ============================================================
# WebSocket Event Stream
# ============================================================

@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):

    await manager.connect(websocket)

    try:

        while True:

            if pubsub is None:
                pass
            
            msg = pubsub.get_message(ignore_subscribe_messages=True)

            if msg:

                try:
                    data = json.loads(msg["data"])
                    await manager.send(data)

                except Exception as e:
                    logger.error(f"WS parse error: {e}")

            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"WS error: {e}")
        manager.disconnect(websocket)


# ============================================================
# Existing Modules
# ============================================================

app.include_router(detect.router, prefix="/api/detect", tags=["Detection"])
app.include_router(tickets.router, prefix="/api/tickets", tags=["Tickets"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["Statistics"])