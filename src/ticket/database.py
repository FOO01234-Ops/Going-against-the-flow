# src/ticket/database.py
"""
数据库操作模块
使用SQLite存储违规工单
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from .models import (
    ViolationTicket,
    ViolationType,
    VehicleType,
    TicketStatus
)


class TicketDatabase:
    """工单数据库管理类"""

    def __init__(self, db_path: str = "data/database/violation.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """初始化数据库表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 创建工单表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                violation_type TEXT NOT NULL,
                plate_number TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                vehicle_color TEXT,
                violation_time TEXT NOT NULL,
                intersection_id TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                direction TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                video_clip_path TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                is_special_vehicle INTEGER DEFAULT 0,
                special_vehicle_type TEXT,
                review_time TEXT,
                reviewer TEXT,
                review_comment TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                confidence_score REAL DEFAULT 0.0
            )
        ''')

        # 创建索引以提升查询性能
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_plate_number 
            ON tickets(plate_number)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_violation_time 
            ON tickets(violation_time)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_status 
            ON tickets(status)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_intersection 
            ON tickets(intersection_id)
        ''')

        conn.commit()
        conn.close()

    def save_ticket(self, ticket: ViolationTicket) -> bool:
        """
        保存工单到数据库（新增或更新）
        
        Args:
            ticket: 工单对象
        
        Returns:
            bool: 是否保存成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO tickets (
                    ticket_id, violation_type, plate_number, vehicle_type,
                    vehicle_color, violation_time, intersection_id, camera_id,
                    direction, snapshot_path, video_clip_path, status,
                    is_special_vehicle, special_vehicle_type, review_time,
                    reviewer, review_comment, created_at, updated_at,
                    confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket.ticket_id,
                ticket.violation_type.value,
                ticket.plate_number,
                ticket.vehicle_type.value,
                ticket.vehicle_color,
                ticket.violation_time.isoformat(),
                ticket.intersection_id,
                ticket.camera_id,
                ticket.direction,
                ticket.snapshot_path,
                ticket.video_clip_path,
                ticket.status.value,
                1 if ticket.is_special_vehicle else 0,
                ticket.special_vehicle_type,
                ticket.review_time.isoformat() if ticket.review_time else None,
                ticket.reviewer,
                ticket.review_comment,
                ticket.created_at.isoformat(),
                ticket.updated_at.isoformat(),
                ticket.confidence_score
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"保存工单失败: {e}")
            return False
        finally:
            conn.close()

    def get_ticket(self, ticket_id: str) -> Optional[ViolationTicket]:
        """
        根据工单ID查询单个工单
        
        Args:
            ticket_id: 工单ID
        
        Returns:
            ViolationTicket: 工单对象，不存在则返回 None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM tickets WHERE ticket_id = ?', (ticket_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_ticket(row)
        return None

    def get_tickets(
        self,
        plate_number: Optional[str] = None,
        status: Optional[TicketStatus] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        intersection_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ViolationTicket]:
        """
        条件查询工单列表
        
        Args:
            plate_number: 车牌号（模糊匹配）
            status: 工单状态
            start_time: 开始时间
            end_time: 结束时间
            intersection_id: 路口ID
            limit: 每页数量
            offset: 偏移量
        
        Returns:
            List[ViolationTicket]: 工单列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params = []

        if plate_number:
            conditions.append("plate_number LIKE ?")
            params.append(f"%{plate_number}%")

        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if start_time:
            conditions.append("violation_time >= ?")
            params.append(start_time.isoformat())

        if end_time:
            conditions.append("violation_time <= ?")
            params.append(end_time.isoformat())

        if intersection_id:
            conditions.append("intersection_id = ?")
            params.append(intersection_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f'''
            SELECT * FROM tickets 
            {where_clause}
            ORDER BY violation_time DESC
            LIMIT ? OFFSET ?
        '''
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_ticket(row) for row in rows]

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
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE tickets 
                SET status = ?, review_time = ?, reviewer = ?, review_comment = ?, updated_at = ?
                WHERE ticket_id = ?
            ''', (
                status.value,
                datetime.now().isoformat(),
                reviewer,
                comment,
                datetime.now().isoformat(),
                ticket_id
            ))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"更新工单状态失败: {e}")
            return False
        finally:
            conn.close()

    def get_statistics(self, intersection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            intersection_id: 路口ID（可选，不传则统计全部）
        
        Returns:
            Dict: 包含总数、状态分布、车型分布、今日数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 构建 WHERE 子句
        where_clause = ""
        params = []

        if intersection_id:
            where_clause = "WHERE intersection_id = ?"
            params.append(intersection_id)

        # 1. 总工单数
        cursor.execute(f'SELECT COUNT(*) FROM tickets {where_clause}', params)
        total = cursor.fetchone()[0]

        # 2. 各状态统计
        cursor.execute(f'''
            SELECT status, COUNT(*) 
            FROM tickets {where_clause}
            GROUP BY status
        ''', params)
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # 3. 各车型统计
        cursor.execute(f'''
            SELECT vehicle_type, COUNT(*) 
            FROM tickets {where_clause}
            GROUP BY vehicle_type
        ''', params)
        vehicle_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # 4. 今日违规数（独立查询）
        today = datetime.now().date().isoformat()

        if intersection_id:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM tickets 
                WHERE intersection_id = ? AND date(violation_time) = ?
            ''', (intersection_id, today))
        else:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM tickets 
                WHERE date(violation_time) = ?
            ''', (today,))
        today_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total": total,
            "status_counts": status_counts,
            "vehicle_counts": vehicle_counts,
            "today_count": today_count
        }

    def get_max_ticket_number(self, date_str: str) -> int:
        """
        获取指定日期的最大工单序号
        
        用于生成不重复的工单ID。
        例如: 日期 20260705，查询所有 T20260705 开头的工单，
        提取序号部分并取最大值。
        
        Args:
            date_str: 日期字符串，格式 YYYYMMDD
        
        Returns:
            int: 该日期的最大序号，如果没有则返回 0
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # 使用 SUBSTR 提取工单ID中的序号部分
        # 例如: T20260705000001 -> SUBSTR(..., 10) -> 000001 -> CAST -> 1
        cursor.execute('''
            SELECT MAX(CAST(SUBSTR(ticket_id, 10) AS INTEGER)) 
            FROM tickets 
            WHERE ticket_id LIKE ?
        ''', (f'T{date_str}%',))
        
        row = cursor.fetchone()
        conn.close()

        # 如果没有记录，返回 0；否则返回最大序号
        return row[0] if row and row[0] else 0

    def delete_ticket(self, ticket_id: str) -> bool:
        """
        删除工单（物理删除）
        
        注意：一般不建议物理删除，建议使用 update_status 改为 PROCESSED
        
        Args:
            ticket_id: 工单ID
        
        Returns:
            bool: 是否删除成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM tickets WHERE ticket_id = ?', (ticket_id,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            print(f"删除工单失败: {e}")
            return False
        finally:
            conn.close()

    def get_tickets_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        intersection_id: Optional[str] = None
    ) -> List[ViolationTicket]:
        """
        按日期范围查询工单（便捷方法）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            intersection_id: 路口ID（可选）
        
        Returns:
            List[ViolationTicket]: 工单列表
        """
        return self.get_tickets(
            start_time=start_date,
            end_time=end_date,
            intersection_id=intersection_id,
            limit=10000
        )

    def get_tickets_by_plate(self, plate_number: str) -> List[ViolationTicket]:
        """
        按车牌号精确查询工单（便捷方法）
        
        Args:
            plate_number: 完整车牌号
        
        Returns:
            List[ViolationTicket]: 工单列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM tickets 
            WHERE plate_number = ?
            ORDER BY violation_time DESC
        ''', (plate_number,))
        
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_ticket(row) for row in rows]

    def count_by_status(self, status: TicketStatus) -> int:
        """
        统计指定状态的工单数量
        
        Args:
            status: 工单状态
        
        Returns:
            int: 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM tickets WHERE status = ?
        ''', (status.value,))
        
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def count_by_intersection(self, intersection_id: str) -> int:
        """
        统计指定路口的工单数量
        
        Args:
            intersection_id: 路口ID
        
        Returns:
            int: 数量
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM tickets WHERE intersection_id = ?
        ''', (intersection_id,))
        
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def get_all_intersections(self) -> List[str]:
        """
        获取所有有工单的路口ID列表
        
        Returns:
            List[str]: 路口ID列表
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT DISTINCT intersection_id FROM tickets ORDER BY intersection_id
        ''')
        
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_today_count(self, intersection_id: Optional[str] = None) -> int:
        """
        获取今日违规数量
        
        Args:
            intersection_id: 路口ID（可选）
        
        Returns:
            int: 今日数量
        """
        today = datetime.now().date().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()

        if intersection_id:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM tickets 
                WHERE date(violation_time) = ? AND intersection_id = ?
            ''', (today, intersection_id))
        else:
            cursor.execute('''
                SELECT COUNT(*) 
                FROM tickets 
                WHERE date(violation_time) = ?
            ''', (today,))
        
        count = cursor.fetchone()[0]
        conn.close()

        return count

    def _row_to_ticket(self, row) -> ViolationTicket:
        """
        将数据库行转换为工单对象
        
        Args:
            row: sqlite3.Row 对象
        
        Returns:
            ViolationTicket: 工单对象
        """
        return ViolationTicket(
            ticket_id=row["ticket_id"],
            violation_type=ViolationType(row["violation_type"]),
            plate_number=row["plate_number"],
            vehicle_type=VehicleType(row["vehicle_type"]),
            vehicle_color=row["vehicle_color"],
            violation_time=datetime.fromisoformat(row["violation_time"]),
            intersection_id=row["intersection_id"],
            camera_id=row["camera_id"],
            direction=row["direction"],
            snapshot_path=row["snapshot_path"],
            video_clip_path=row["video_clip_path"],
            status=TicketStatus(row["status"]),
            is_special_vehicle=bool(row["is_special_vehicle"]),
            special_vehicle_type=row["special_vehicle_type"],
            review_time=datetime.fromisoformat(row["review_time"]) if row["review_time"] else None,
            reviewer=row["reviewer"],
            review_comment=row["review_comment"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            confidence_score=row["confidence_score"]
        )

    def clear_all_tickets(self) -> bool:
        """
        清空所有工单（危险操作，仅用于测试）
        
        Returns:
            bool: 是否清空成功
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM tickets')
            conn.commit()
            return True
        except Exception as e:
            print(f"清空工单失败: {e}")
            return False
        finally:
            conn.close()