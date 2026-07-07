# src/ticket/ticket_manager.py
"""
工单管理器 - 提供统一的工单管理接口
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import ViolationTicket, TicketStatus
from .database import TicketDatabase
from .ticket_generator import TicketGenerator


class TicketManager:
    """工单管理器 - 统一管理工单的创建、查询、更新、统计和导出"""

    def __init__(self, db_path: str = "data/database/violation.db"):
        """
        初始化工单管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db = TicketDatabase(db_path)
        self.generator = TicketGenerator(self.db)

    def create_ticket(self, **kwargs) -> ViolationTicket:
        """
        创建工单
        
        Args:
            **kwargs: 传递给 TicketGenerator.generate_ticket 的参数
                - plate_number: 车牌号码
                - vehicle_type: 车型 (VehicleType)
                - intersection_id: 路口ID
                - camera_id: 摄像头ID
                - direction: 行驶方向
                - frame: 视频帧 (numpy.ndarray)
                - bbox: 车辆边界框 (x1, y1, x2, y2)
                - violation_type: 违规类型 (ViolationType, 默认 REVERSE_DRIVING)
                - is_special: 是否特殊车辆 (bool, 默认 False)
                - special_type: 特殊车辆类型 (str, 可选)
                - confidence: 检测置信度 (float, 默认 0.0)
                - vehicle_color: 车辆颜色 (str, 可选)
        
        Returns:
            ViolationTicket: 生成的工单对象
        """
        return self.generator.generate_ticket(**kwargs)

    def get_ticket(self, ticket_id: str) -> Optional[ViolationTicket]:
        """
        根据工单ID查询工单
        
        Args:
            ticket_id: 工单ID
        
        Returns:
            ViolationTicket: 工单对象，不存在则返回 None
        """
        return self.db.get_ticket(ticket_id)

    def search_tickets(
        self,
        plate_number: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        intersection_id: Optional[str] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        分页查询工单
        
        Args:
            plate_number: 车牌号（模糊匹配）
            status: 工单状态
            start_time: 开始时间
            end_time: 结束时间
            intersection_id: 路口ID
            page: 页码（从1开始）
            page_size: 每页数量
        
        Returns:
            Dict: 包含分页信息和工单列表
        """
        offset = (page - 1) * page_size

        tickets = self.db.get_tickets(
            plate_number=plate_number,
            status=status,
            start_time=start_time,
            end_time=end_time,
            intersection_id=intersection_id,
            limit=page_size,
            offset=offset
        )

        return {
            "page": page,
            "page_size": page_size,
            "total": self._get_total_count(plate_number, status, intersection_id),
            "tickets": tickets
        }

    def update_status(
        self,
        ticket_id: str,
        status: TicketStatus,
        reviewer: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        更新工单状态
        
        Args:
            ticket_id: 工单ID
            status: 新状态
            reviewer: 审核人
            comment: 审核意见
        
        Returns:
            bool: 是否更新成功
        """
        return self.db.update_status(ticket_id, status, reviewer, comment)

    def get_statistics(self, intersection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            intersection_id: 路口ID（可选）
        
        Returns:
            Dict: 统计信息
        """
        return self.db.get_statistics(intersection_id)

    def export_tickets(
        self,
        start_time: datetime,
        end_time: datetime,
        format: str = "json"
    ) -> str:
        """
        导出工单数据
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            format: 导出格式 ('json' 或 'csv')
        
        Returns:
            str: 导出的数据字符串
        """
        tickets = self.db.get_tickets(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )

        if format == "json":
            import json
            return json.dumps([t.to_dict() for t in tickets], ensure_ascii=False, indent=2)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            if tickets:
                writer = csv.DictWriter(output, fieldnames=tickets[0].to_dict().keys())
                writer.writeheader()
                for t in tickets:
                    writer.writerow(t.to_dict())
            return output.getvalue()
        else:
            raise ValueError(f"不支持的导出格式: {format}")

    def get_today_count(self, intersection_id: Optional[str] = None) -> int:
        """
        获取今日违规数量
        
        Args:
            intersection_id: 路口ID（可选）
        
        Returns:
            int: 今日数量
        """
        return self.db.get_today_count(intersection_id)

    def get_tickets_by_plate(self, plate_number: str) -> List[ViolationTicket]:
        """
        按车牌号精确查询
        
        Args:
            plate_number: 完整车牌号
        
        Returns:
            List[ViolationTicket]: 工单列表
        """
        return self.db.get_tickets_by_plate(plate_number)

    def get_all_intersections(self) -> List[str]:
        """
        获取所有有工单的路口
        
        Returns:
            List[str]: 路口ID列表
        """
        return self.db.get_all_intersections()

    def delete_ticket(self, ticket_id: str) -> bool:
        """
        删除工单（物理删除）
        
        注意：建议使用 update_status 改为 PROCESSED 而非物理删除
        
        Args:
            ticket_id: 工单ID
        
        Returns:
            bool: 是否删除成功
        """
        return self.db.delete_ticket(ticket_id)

    def _get_total_count(self, plate_number, status, intersection_id) -> int:
        """
        获取符合条件的工单总数（内部方法）
        
        Args:
            plate_number: 车牌号
            status: 状态
            intersection_id: 路口ID
        
        Returns:
            int: 总数
        """
        tickets = self.db.get_tickets(
            plate_number=plate_number,
            status=status,
            intersection_id=intersection_id,
            limit=100000
        )
        return len(tickets)