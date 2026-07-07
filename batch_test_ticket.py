# batch_test_ticket.py
"""
批量测试工单模块 - 使用 DETRAC 真实标注数据
"""
import sys
from pathlib import Path
from datetime import datetime
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ticket import (
    TicketManager,
    TicketStatus,
    VehicleType,
)


class DETRACBatchTester:
    """DETRAC 数据集批量测试器（使用真实标注）"""

    def __init__(self, data_path: str = "data/raw"):
        self.data_path = Path(data_path)
        self.manager = TicketManager("data/database/detrac_test.db")
        
        # 统计信息
        self.stats = {
            "total_frames": 0,
            "total_vehicles": 0,
            "tickets_created": 0,
            "special_tickets": 0,
            "errors": 0,
        }
        
        # 车型映射: DETRAC标注 -> 我们的枚举
        self.type_mapping = {
            "sedan": VehicleType.CAR,
            "suv": VehicleType.SUV,
            "van": VehicleType.VAN,
            "truck": VehicleType.TRUCK,
            "bus": VehicleType.BUS,
            "others": VehicleType.UNKNOWN,
        }
        
        # 特殊车辆车牌（模拟）
        self.special_plates = ["京B99999", "京B88888", "京B77777", "京B66666"]

    def load_annotations(self, xml_path: Path) -> Dict[str, List[Dict]]:
        """
        加载 DETRAC XML 标注文件
        
        DETRAC XML 结构:
        <sequence>
          <frame num="1">
            <target_list>
              <target id="1">
                <box left="..." top="..." width="..." height="..."/>
                <attribute vehicle_type="car" orientation="..."/>
              </target>
            </target_list>
          </frame>
        </sequence>
        """
        if not xml_path.exists():
            print(f"   ⚠️ 标注文件不存在: {xml_path}")
            return {}
        
        annotations = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 查找所有 frame 标签
            frames = root.findall("frame")
            
            if not frames:
                print(f"   ⚠️ 未找到 frame 标签")
                return {}
            
            for frame_elem in frames:
                frame_id = frame_elem.get("num")
                if frame_id is None:
                    continue
                
                vehicles = []
                
                # 查找 target_list
                target_list = frame_elem.find("target_list")
                if target_list is None:
                    continue
                
                # 遍历所有 target
                for target in target_list.findall("target"):
                    # 获取边界框
                    box = target.find("box")
                    if box is None:
                        continue
                    
                    try:
                        x = float(box.get("left", 0))
                        y = float(box.get("top", 0))
                        w = float(box.get("width", 0))
                        h = float(box.get("height", 0))
                    except (ValueError, TypeError):
                        continue
                    
                    # 获取属性
                    attr = target.find("attribute")
                    if attr is not None:
                        vehicle_type_str = attr.get("vehicle_type", "others")
                        orientation = attr.get("orientation", "N->S")
                        truncation_ratio = float(attr.get("truncation_ratio", "0"))
                    else:
                        vehicle_type_str = "others"
                        orientation = "N->S"
                        truncation_ratio = 0
                    
                    # 跳过截断严重的车辆（截断比例 > 0.5）
                    if truncation_ratio > 0.5:
                        continue
                    
                    # 只保留合理的边界框
                    if w < 10 or h < 10:
                        continue
                    
                    vehicles.append({
                        "bbox": (int(x), int(y), int(x + w), int(y + h)),
                        "vehicle_type_str": vehicle_type_str,
                        "vehicle_type": self.type_mapping.get(vehicle_type_str, VehicleType.UNKNOWN),
                        "orientation": orientation,
                        "truncation_ratio": truncation_ratio,
                        "confidence": 1.0,
                    })
                
                if vehicles:
                    annotations[frame_id] = vehicles
            
            return annotations
            
        except ET.ParseError as e:
            print(f"   ❌ XML 解析失败: {e}")
            return {}
        except Exception as e:
            print(f"   ❌ 加载标注失败: {e}")
            return {}

    def find_all_sequences(self) -> List[Dict]:
        """查找所有视频序列及其对应的标注文件"""
        sequences = []
        
        img_base = self.data_path / "DETRAC-train-data" / "Insight-MVT_Annotation_Train"
        ann_base = self.data_path / "DETRAC-Train-Annotations-XML"
        
        print(f"🔍 查找图片目录: {img_base}")
        print(f"🔍 查找标注目录: {ann_base}")
        
        if not img_base.exists():
            print(f"❌ 图片目录不存在: {img_base}")
            return []
        
        if not ann_base.exists():
            print(f"❌ 标注目录不存在: {ann_base}")
            return []
        
        # 遍历所有 MVI_ 文件夹
        for video_dir in sorted(img_base.glob("MVI_*")):
            if video_dir.is_dir():
                xml_path = ann_base / f"{video_dir.name}.xml"
                
                if xml_path.exists():
                    image_count = len(list(video_dir.glob("img*.jpg")))
                    sequences.append({
                        "name": video_dir.name,
                        "image_dir": video_dir,
                        "xml_path": xml_path,
                        "image_count": image_count
                    })
                    print(f"   ✅ 找到序列: {video_dir.name} ({image_count} 张图片)")
                else:
                    print(f"   ⚠️ 跳过 {video_dir.name}: 无标注文件")
        
        return sequences

    def get_frame_image(self, sequence_name: str, frame_id: str) -> Optional[np.ndarray]:
        """获取指定帧的图片"""
        img_dir = self.data_path / "DETRAC-train-data" / "Insight-MVT_Annotation_Train" / sequence_name
        img_path = img_dir / f"img{int(frame_id):05d}.jpg"
        
        if img_path.exists():
            return cv2.imread(str(img_path))
        return None

    def simulate_plate_number(self, is_special: bool = False) -> str:
        """生成模拟车牌号"""
        if is_special:
            return random.choice(self.special_plates)
        
        provinces = ["京", "沪", "粤", "苏", "浙", "鲁", "豫", "川", "渝", "鄂"]
        city_codes = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        digits = "0123456789"
        
        province = random.choice(provinces)
        city = random.choice(city_codes)
        
        chars = []
        for _ in range(5):
            if random.random() < 0.3:
                chars.append(random.choice(digits))
            else:
                chars.append(random.choice(city_codes))
        
        return f"{province}{city}{''.join(chars)}"

    def process_sequence(self, sequence: Dict, max_frames: int = 50, sample_interval: int = 10):
        """处理单个视频序列"""
        sequence_name = sequence["name"]
        xml_path = sequence["xml_path"]
        
        print(f"\n📹 处理序列: {sequence_name}")
        
        # 加载标注
        annotations = self.load_annotations(xml_path)
        if not annotations:
            print(f"   ⚠️ 无有效标注数据")
            return
        
        # 获取所有帧ID并排序
        frame_ids = sorted(annotations.keys(), key=lambda x: int(x))
        
        # 采样
        if sample_interval > 1:
            frame_ids = frame_ids[::sample_interval]
        
        # 限制数量
        if max_frames > 0:
            frame_ids = frame_ids[:max_frames]
        
        print(f"   📊 处理帧数: {len(frame_ids)}")
        
        for idx, frame_id in enumerate(frame_ids):
            if (idx + 1) % 5 == 0 or idx == 0:
                print(f"   📊 进度: {idx + 1}/{len(frame_ids)} "
                      f"(已生成 {self.stats['tickets_created']} 张工单)")
            
            frame = self.get_frame_image(sequence_name, frame_id)
            if frame is None:
                continue
            
            self.stats["total_frames"] += 1
            
            vehicles = annotations[frame_id]
            self.stats["total_vehicles"] += len(vehicles)
            
            for vehicle in vehicles:
                # 30%概率逆行
                if random.random() < 0.3:
                    # 5%概率是特殊车辆
                    is_special = random.random() < 0.05
                    
                    plate = self.simulate_plate_number(is_special)
                    special_type = None
                    if is_special:
                        special_type = vehicle["vehicle_type_str"]
                    
                    ticket = self.manager.create_ticket(
                        plate_number=plate,
                        vehicle_type=vehicle["vehicle_type"],
                        intersection_id="INT_DETRAC",
                        camera_id=sequence_name,
                        direction=vehicle["orientation"],
                        frame=frame,
                        bbox=vehicle["bbox"],
                        confidence=vehicle["confidence"],
                        vehicle_color="未知",
                        is_special=is_special,
                        special_type=special_type
                    )
                    self.stats["tickets_created"] += 1
                    
                    if is_special:
                        self.stats["special_tickets"] += 1

    def run_batch_test(
        self, 
        max_sequences: int = 3, 
        max_frames_per_sequence: int = 20,
        sample_interval: int = 5
    ):
        """批量运行测试"""
        print("=" * 70)
        print("🚦 DETRAC 数据集批量测试（使用真实标注）")
        print("=" * 70)
        
        sequences = self.find_all_sequences()
        
        if not sequences:
            print("\n❌ 未找到任何视频序列")
            return
        
        print(f"\n📁 找到 {len(sequences)} 个视频序列")
        
        if max_sequences > 0:
            sequences = sequences[:max_sequences]
            print(f"📌 处理前 {max_sequences} 个序列")
        
        print(f"\n📌 每序列最大帧数: {max_frames_per_sequence}")
        print(f"📌 采样间隔: {sample_interval}")
        print("\n开始处理...\n")
        
        for seq in sequences:
            self.process_sequence(
                seq, 
                max_frames=max_frames_per_sequence, 
                sample_interval=sample_interval
            )
        
        self.print_statistics()

    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 70)
        print("📊 批量测试统计结果")
        print("=" * 70)
        
        print(f"\n📈 处理统计:")
        print(f"   - 处理图片数: {self.stats['total_frames']}")
        print(f"   - 检测到的车辆数: {self.stats['total_vehicles']}")
        print(f"   - 生成工单数: {self.stats['tickets_created']}")
        print(f"   - 特殊车辆工单: {self.stats['special_tickets']}")
        
        if self.stats['total_frames'] > 0:
            print(f"\n📊 分析:")
            print(f"   - 平均每张图片车辆数: {self.stats['total_vehicles'] / self.stats['total_frames']:.2f}")
            print(f"   - 平均每张图片生成工单: {self.stats['tickets_created'] / self.stats['total_frames']:.2f}")
        
        stats = self.manager.get_statistics()
        print(f"\n📊 数据库统计:")
        print(f"   - 总工单数: {stats['total']}")
        print(f"   - 状态分布:")
        for status, count in stats['status_counts'].items():
            emoji = "⏳" if status == "pending" else "❌" if status == "dismissed" else "📌"
            print(f"      {emoji} {status}: {count}")


def run_test():
    """运行测试"""
    tester = DETRACBatchTester()
    tester.run_batch_test(
        max_sequences=3,
        max_frames_per_sequence=20,
        sample_interval=5
    )


if __name__ == "__main__":
    run_test()