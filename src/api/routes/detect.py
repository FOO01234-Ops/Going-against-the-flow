# src/api/routes/detect.py
"""
车辆检测模块（完整版 + 可视化）
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import Dict, Any
import numpy as np
import cv2
import base64
from ultralytics import YOLO

from src.pipeline.frame_processor import FrameProcessor
from src.tracking.tracker import DeepSORTTracker
from src.ticket.ticket_manager import TicketManager
from src.detection.model_factory import ModelFactory

router = APIRouter(tags=["🚗 车辆检测"])

print("🔧 初始化检测系统依赖...")
model_path = ModelFactory.get_model("yolo11n")
yolo_model = YOLO(model_path)
tracker = DeepSORTTracker(model_path=model_path)
ticket_manager = TicketManager()
processor = FrameProcessor(
    yolo_model=yolo_model,
    tracker=tracker,
    ticket_manager=ticket_manager
)
print("✅ 检测系统初始化完成\n")


def read_image(file: UploadFile) -> np.ndarray:
    try:
        contents = file.file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解析失败: {str(e)}")


# ============================================================
# API 1：单帧检测（带可视化开关）
# ============================================================
@router.post("/frame")
async def detect_frame(
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="是否返回标注图片（Base64）")
) -> Dict[str, Any]:
    image = read_image(file)
    if image is None:
        raise HTTPException(status_code=400, detail="无效图片")

    events, annotated = processor.process(image, camera_id="frame", visualize=visualize)

    result_data = {
        "events": events,
        "count": len(events)
    }

    if visualize and annotated is not None:
        # 将标注图像编码为 Base64 字符串
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result_data["image_base64"] = img_base64

    return {
        "status": "success",
        "type": "frame_detection",
        "data": result_data
    }


# ============================================================
# API 2：上传图片检测（别名，同样增加 visualize 参数）
# ============================================================
@router.post("/upload")
async def detect_upload(
    file: UploadFile = File(...),
    visualize: bool = Query(False, description="是否返回标注图片（Base64）")
) -> Dict[str, Any]:
    image = read_image(file)
    if image is None:
        raise HTTPException(status_code=400, detail="无效图片")

    events, annotated = processor.process(image, camera_id="upload", visualize=visualize)

    result_data = {
        "events": events,
        "count": len(events)
    }

    if visualize and annotated is not None:
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        result_data["image_base64"] = img_base64

    return {
        "status": "success",
        "type": "upload_detection",
        "data": result_data
    }


# ============================================================
# API 3：视频检测（预留）
# ============================================================
@router.post("/video")
async def detect_video() -> Dict[str, Any]:
    return {
        "status": "not_implemented",
        "message": "视频检测模块开发中"
    }


# ============================================================
# API 4：健康检查
# ============================================================
@router.get("/health")
async def detect_health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "module": "detection",
        "model": "yolo11n",
        "tracker": "DeepSORT"
    }