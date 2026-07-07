# src/api/routes/tickets.py
"""
📋 工单管理接口
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from ..schemas.models import (
    TicketResponse,
    TicketsResponse,
    UpdateTicketRequest,
    TicketStatus,
)
from ..dependencies import get_ticket_manager


router = APIRouter(
    prefix="/api/tickets",
    tags=["📋 工单管理"],
)


@router.get("/", response_model=TicketsResponse)
async def get_tickets(
    plate_number: Optional[str] = Query(None, description="车牌号"),
    status: Optional[TicketStatus] = Query(None, description="工单状态"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    intersection_id: Optional[str] = Query(None, description="路口ID"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
):
    """查询工单列表"""
    manager = get_ticket_manager()
    result = manager.search_tickets(
        plate_number=plate_number,
        status=status,
        start_time=start_time,
        end_time=end_time,
        intersection_id=intersection_id,
        page=page,
        page_size=page_size,
    )
    return TicketsResponse(
        total=result["total"],
        page=result["page"],
        page_size=result["page_size"],
        tickets=[
            TicketResponse(
                ticket_id=t.ticket_id,
                violation_type=t.violation_type.value,
                plate_number=t.plate_number,
                vehicle_type=t.vehicle_type.value,
                vehicle_color=t.vehicle_color,
                violation_time=t.violation_time,
                intersection_id=t.intersection_id,
                camera_id=t.camera_id,
                direction=t.direction,
                snapshot_path=t.snapshot_path,
                status=t.status.value,
                is_special_vehicle=t.is_special_vehicle,
                special_vehicle_type=t.special_vehicle_type,
                confidence_score=t.confidence_score,
                reviewer=t.reviewer,
                review_comment=t.review_comment,
                created_at=t.created_at,
                updated_at=t.updated_at
            )
            for t in result["tickets"]
        ]
    )


@router.get("/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str):
    """查询工单详情"""
    manager = get_ticket_manager()
    ticket = manager.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="工单不存在")
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        violation_type=ticket.violation_type.value,
        plate_number=ticket.plate_number,
        vehicle_type=ticket.vehicle_type.value,
        vehicle_color=ticket.vehicle_color,
        violation_time=ticket.violation_time,
        intersection_id=ticket.intersection_id,
        camera_id=ticket.camera_id,
        direction=ticket.direction,
        snapshot_path=ticket.snapshot_path,
        status=ticket.status.value,
        is_special_vehicle=ticket.is_special_vehicle,
        special_vehicle_type=ticket.special_vehicle_type,
        confidence_score=ticket.confidence_score,
        reviewer=ticket.reviewer,
        review_comment=ticket.review_comment,
        created_at=ticket.created_at,
        updated_at=ticket.updated_at
    )


@router.put("/{ticket_id}/status")
async def update_ticket_status(
    ticket_id: str,
    request: UpdateTicketRequest,
):
    """更新工单状态"""
    manager = get_ticket_manager()
    success = manager.update_status(
        ticket_id=ticket_id,
        status=request.status,
        reviewer=request.reviewer,
        comment=request.comment,
    )
    if not success:
        raise HTTPException(status_code=404, detail="工单不存在")
    updated = manager.get_ticket(ticket_id)
    return {
        "success": True,
        "message": f"工单 {ticket_id} 状态已更新为 {request.status.value}",
        "ticket": TicketResponse(
            ticket_id=updated.ticket_id,
            violation_type=updated.violation_type.value,
            plate_number=updated.plate_number,
            vehicle_type=updated.vehicle_type.value,
            vehicle_color=updated.vehicle_color,
            violation_time=updated.violation_time,
            intersection_id=updated.intersection_id,
            camera_id=updated.camera_id,
            direction=updated.direction,
            snapshot_path=updated.snapshot_path,
            status=updated.status.value,
            is_special_vehicle=updated.is_special_vehicle,
            special_vehicle_type=updated.special_vehicle_type,
            confidence_score=updated.confidence_score,
            reviewer=updated.reviewer,
            review_comment=updated.review_comment,
            created_at=updated.created_at,
            updated_at=updated.updated_at
        ) if updated else None
    }


@router.get("/export/json")
async def export_tickets_json(
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
):
    """导出工单为 JSON"""
    manager = get_ticket_manager()
    data = manager.export_tickets(start_time, end_time, format="json")
    return {"data": data}


@router.get("/export/csv")
async def export_tickets_csv(
    start_time: datetime = Query(...),
    end_time: datetime = Query(...),
):
    """导出工单为 CSV"""
    manager = get_ticket_manager()
    data = manager.export_tickets(start_time, end_time, format="csv")
    return {"data": data}