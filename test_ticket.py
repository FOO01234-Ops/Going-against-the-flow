# test_ticket.py 
"""
工单模块完整测试
测试工单的创建、查询、统计和状态更新功能
"""
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入工单模块所有需要的类型
from src.ticket import (
    TicketManager,
    TicketDatabase,
    ViolationTicket,
    TicketStatus,
    VehicleType,
    ViolationType
)


def create_mock_frame():
    # 读取真实图片
    real_frame = cv2.imread("D:/vscoda/xm/Going against the flow/data/raw/DETRAC-train-data/Insight-MVT_Annotation_Train/MVI_20011")
    if real_frame is None:
        # 如果没有真实图片，再使用模拟图片
        real_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # 添加一些彩色条纹或文字，让截图有内容
        cv2.putText(real_frame, "TEST IMAGE", (400, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return real_frame


def test_ticket_creation(manager: TicketManager):
    """测试工单创建和存储"""
    print("=" * 60)
    print("🚗 测试工单创建")
    print("=" * 60)

    # ✅ 使用带内容的模拟帧
    frame = create_mock_frame()

    # 生成工单
    ticket = manager.create_ticket(
        plate_number="京A12345",
        vehicle_type=VehicleType.CAR,
        intersection_id="INT_001",
        camera_id="CAM_01",
        direction="N->S",
        frame=frame,  # ← 现在是有内容的图片了
        bbox=(450, 320, 620, 480),
        confidence=0.92,
        vehicle_color="红色"
    )

    # 验证并打印结果
    print(f"\n✅ 工单创建成功!")
    print(f"  📋 工单ID: {ticket.ticket_id}")
    print(f"  🚗 车牌号: {ticket.plate_number}")
    print(f"  🏷️  车型: {ticket.vehicle_type.value}")
    print(f"  🎨 颜色: {ticket.vehicle_color}")
    print(f"  🕐 违规时间: {ticket.violation_time}")
    print(f"  📍 路口ID: {ticket.intersection_id}")
    print(f"  📷 摄像头: {ticket.camera_id}")
    print(f"  📊 置信度: {ticket.confidence_score}")
    print(f"  📁 截图路径: {ticket.snapshot_path}")
    print(f"  📌 状态: {ticket.status.value}")

    return ticket


def test_special_vehicle_ticket(manager: TicketManager):
    """测试特殊车辆工单（应自动驳回）"""
    print("\n" + "=" * 60)
    print("🚑 测试特殊车辆工单")
    print("=" * 60)

    # ✅ 使用带内容的模拟帧
    frame = create_mock_frame()

    # 创建救护车工单
    ticket = manager.create_ticket(
        plate_number="京B99999",
        vehicle_type=VehicleType.AMBULANCE,
        intersection_id="INT_001",
        camera_id="CAM_01",
        direction="S->N",
        frame=frame,  # ← 现在是有内容的图片了
        bbox=(700, 350, 850, 490),
        confidence=0.95,
        vehicle_color="白色",
        is_special=True,
        special_type="救护车"
    )

    print(f"\n✅ 特殊车辆工单创建成功!")
    print(f"  📋 工单ID: {ticket.ticket_id}")
    print(f"  🚗 车牌号: {ticket.plate_number}")
    print(f"  🏷️  车型: {ticket.vehicle_type.value}")
    print(f"  🚨 特殊车辆: {ticket.special_vehicle_type}")
    print(f"  📌 状态: {ticket.status.value} (自动驳回)")

    return ticket


def test_query(manager: TicketManager):
    """测试查询功能"""
    print("\n" + "=" * 60)
    print("🔍 测试工单查询")
    print("=" * 60)

    result = manager.search_tickets(page=1, page_size=10)
    print(f"📊 总工单数: {result['total']}")

    if result['total'] > 0:
        total_pages = (result['total'] + 9) // 10
        print(f"📄 当前页: {result['page']}/{total_pages}")
        print("\n工单列表:")
        for t in result['tickets']:
            status_icon = "✅" if t.status == TicketStatus.CONFIRMED else "⏳" if t.status == TicketStatus.PENDING else "❌"
            print(f"  {status_icon} [{t.ticket_id}] {t.plate_number} | {t.violation_time.strftime('%Y-%m-%d %H:%M')} | {t.status.value}")
    else:
        print("📭 暂无工单记录")

    return result


def test_search_by_plate(manager: TicketManager):
    """测试按车牌号搜索"""
    print("\n" + "=" * 60)
    print("🔍 测试按车牌号搜索")
    print("=" * 60)

    result = manager.search_tickets(plate_number="京A", page=1, page_size=10)
    print(f"🔎 搜索关键词: '京A'")
    print(f"📊 找到 {result['total']} 条记录")

    for t in result['tickets']:
        print(f"  - {t.ticket_id}: {t.plate_number} | {t.violation_time}")

    return result


def test_statistics(manager: TicketManager):
    """测试统计功能"""
    print("\n" + "=" * 60)
    print("📊 测试统计功能")
    print("=" * 60)

    stats = manager.get_statistics()

    print(f"📈 总工单数: {stats['total']}")

    print(f"\n📊 状态分布:")
    for status, count in stats['status_counts'].items():
        emoji = "✅" if status == "confirmed" else "⏳" if status == "pending" else "❌" if status == "dismissed" else "📌"
        print(f"    {emoji} {status}: {count}")

    print(f"\n🏷️  车型分布:")
    for vtype, count in stats['vehicle_counts'].items():
        print(f"    - {vtype}: {count}")

    print(f"\n📅 今日违规: {stats['today_count']}")


def test_update_status(manager: TicketManager):
    """测试状态更新"""
    print("\n" + "=" * 60)
    print("✏️ 测试状态更新")
    print("=" * 60)

    result = manager.search_tickets(status=TicketStatus.PENDING, page=1, page_size=1)

    if result['tickets']:
        ticket = result['tickets'][0]
        print(f"📋 当前工单: {ticket.ticket_id}")
        print(f"   车牌: {ticket.plate_number}")
        print(f"   状态: {ticket.status.value}")

        success = manager.update_status(
            ticket_id=ticket.ticket_id,
            status=TicketStatus.CONFIRMED,
            reviewer="admin",
            comment="审核通过，违规属实"
        )

        if success:
            print("✅ 状态更新成功!")
            updated = manager.get_ticket(ticket.ticket_id)
            print(f"   新状态: {updated.status.value}")
            print(f"   审核人: {updated.reviewer}")
            print(f"   审核意见: {updated.review_comment}")
        else:
            print("❌ 状态更新失败")
    else:
        print("📭 没有待审核的工单")


def test_export(manager: TicketManager):
    """测试导出功能并保存到文件"""
    print("\n" + "=" * 60)
    print("📤 测试导出功能")
    print("=" * 60)

    end_time = datetime.now()
    start_time = datetime(end_time.year, end_time.month, end_time.day)

    try:
        output_dir = Path("output/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        json_data = manager.export_tickets(start_time, end_time, format="json")
        json_path = output_dir / f"tickets_export_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            f.write(json_data)
        print(f"✅ JSON导出成功! 已保存到: {json_path}")
        print(f"   数据大小: {len(json_data)} 字符")

        csv_data = manager.export_tickets(start_time, end_time, format="csv")
        csv_path = output_dir / f"tickets_export_{timestamp}.csv"
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write(csv_data)
        print(f"✅ CSV导出成功! 已保存到: {csv_path}")
        print(f"   数据大小: {len(csv_data)} 字符")

    except Exception as e:
        print(f"❌ 导出失败: {e}")


def test_delete_ticket(manager: TicketManager):
    """测试删除工单（通过更新状态为已处理）"""
    print("\n" + "=" * 60)
    print("🗑️  测试工单归档")
    print("=" * 60)

    result = manager.search_tickets(status=TicketStatus.CONFIRMED, page=1, page_size=1)

    if result['tickets']:
        ticket = result['tickets'][0]
        print(f"📋 准备归档工单: {ticket.ticket_id}")
        print(f"   车牌: {ticket.plate_number}")

        success = manager.update_status(
            ticket_id=ticket.ticket_id,
            status=TicketStatus.PROCESSED,
            reviewer="admin",
            comment="已处理完成"
        )

        if success:
            print("✅ 工单已归档!")
            updated = manager.get_ticket(ticket.ticket_id)
            print(f"   新状态: {updated.status.value}")
        else:
            print("❌ 归档失败")
    else:
        print("📭 没有已确认的工单需要归档")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🚦 工单模块完整测试开始")
    print("=" * 60)

    Path("data/database").mkdir(parents=True, exist_ok=True)
    Path("output/snapshots").mkdir(parents=True, exist_ok=True)

    try:
        manager = TicketManager("data/database/test.db")
        print("✅ 工单管理器初始化成功\n")

        test_ticket_creation(manager)
        test_special_vehicle_ticket(manager)
        test_query(manager)
        test_search_by_plate(manager)
        test_statistics(manager)
        test_update_status(manager)
        test_export(manager)
        test_delete_ticket(manager)

        print("\n" + "=" * 60)
        print("🎉 所有测试通过!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()