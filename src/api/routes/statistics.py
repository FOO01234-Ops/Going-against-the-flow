# src/api/routes/statistics.py
"""
📊 数据统计接口
"""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from ..schemas.models import StatisticsResponse
from ..dependencies import get_ticket_manager


router = APIRouter(
    prefix="/api/statistics",
    tags=["📊 数据统计"],
    responses={
        400: {"description": "请求参数错误"},
    }
)


@router.get(
    "/",
    response_model=StatisticsResponse,
    summary="📊 获取统计数据",
    description="""
## 获取统计数据

返回以下统计信息：

| 字段 | 说明 |
| :--- | :--- |
| `total` | 总工单数 |
| `today_count` | 今日违规数量 |
| `status_counts` | 各状态工单数量分布 |
| `vehicle_counts` | 各车型违规数量分布 |

可选参数：
- `intersection_id`: 按路口筛选统计
    """,
)
async def get_statistics(
    intersection_id: Optional[str] = Query(None, description="路口ID（可选）"),
):
    """获取统计信息"""
    manager = get_ticket_manager()
    stats = manager.get_statistics(intersection_id)
    
    return StatisticsResponse(
        total=stats["total"],
        today_count=stats["today_count"],
        status_counts=stats["status_counts"],
        vehicle_counts=stats["vehicle_counts"],
    )


@router.get(
    "/today",
    summary="📅 获取今日违规数量",
    description="获取今日（当前日期）的违规数量，可按路口筛选",
)
async def get_today_count(
    intersection_id: Optional[str] = Query(None, description="路口ID（可选）"),
):
    """获取今日违规数量"""
    manager = get_ticket_manager()
    count = manager.db.get_today_count(intersection_id)
    
    return {
        "date": datetime.now().date().isoformat(),
        "count": count,
        "intersection_id": intersection_id or "全部路口"
    }


@router.get(
    "/intersections",
    summary="📍 获取路口列表",
    description="获取所有存在工单记录的路口ID列表",
)
async def get_intersections():
    """获取所有路口"""
    manager = get_ticket_manager()
    intersections = manager.get_all_intersections()
    
    return {
        "total": len(intersections),
        "intersections": intersections
    }