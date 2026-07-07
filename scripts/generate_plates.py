# scripts/generate_plates.py
"""
自动生成合成车牌数据
支持：
- 中国标准车牌（蓝牌/绿牌）
- 多种字体、颜色、背景
- 随机噪声和变形，增加真实性
"""
import cv2
import numpy as np
import random
import os
from pathlib import Path
from typing import Tuple, Optional
import argparse
from typing import List, Tuple, Optional

class PlateGenerator:
    """车牌生成器"""
    
    # 省份简称
    PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    
    # 城市代码
    CITY_CODES = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    
    # 字母数字
    CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    
    # 新能源车牌字符
    GREEN_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    
    def __init__(self, output_dir: str = "data/raw/synthetic_plates"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 颜色配置
        self.colors = {
            "blue": {
                "bg": (14, 80, 179),      # 蓝色背景
                "text": (255, 255, 255),   # 白色文字
                "border": (0, 0, 150),
            },
            "green": {
                "bg": (0, 150, 0),         # 绿色背景
                "text": (255, 255, 255),   # 白色文字
                "border": (0, 100, 0),
            },
            "yellow": {
                "bg": (0, 200, 255),       # 黄色背景
                "text": (0, 0, 0),         # 黑色文字
                "border": (0, 150, 200),
            },
            "white": {
                "bg": (255, 255, 255),     # 白色背景
                "text": (0, 0, 0),         # 黑色文字
                "border": (200, 200, 200),
            },
        }
    
    def generate_plate(
        self,
        plate_type: str = "blue",
        text: Optional[str] = None,
        width: int = 240,
        height: int = 80,
        add_noise: bool = True,
        add_rotation: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        生成一张车牌图片
        
        Args:
            plate_type: 车牌类型 (blue, green, yellow, white)
            text: 车牌号码（不指定则随机生成）
            width: 图片宽度
            height: 图片高度
            add_noise: 是否添加噪声
            add_rotation: 是否随机旋转
        
        Returns:
            (图片, 车牌号码)
        """
        # 1. 生成车牌号码
        if text is None:
            text = self._generate_plate_text(plate_type)
        
        # 2. 创建背景
        color = self.colors.get(plate_type, self.colors["blue"])
        plate = np.ones((height, width, 3), dtype=np.uint8) * 255
        plate[:, :] = color["bg"]
        
        # 3. 添加边框
        cv2.rectangle(plate, (2, 2), (width-3, height-3), color["border"], 2)
        
        # 4. 添加文字
        plate = self._draw_text(plate, text, color["text"], plate_type)
        
        # 5. 添加噪点
        if add_noise:
            plate = self._add_noise(plate)
        
        # 6. 随机旋转
        if add_rotation:
            plate = self._add_rotation(plate)
        
        # 7. 随机亮度/对比度调整
        plate = self._adjust_brightness_contrast(plate)
        
        return plate, text
    
    def _generate_plate_text(self, plate_type: str) -> str:
        """生成随机车牌号码"""
        
        province = random.choice(self.PROVINCES)
        city = random.choice(self.CITY_CODES)
        
        if plate_type == "green":
            # 新能源车牌：省份+城市+6位（字母数字混合）
            chars = ''.join(random.choice(self.GREEN_CHARS + "0123456789") for _ in range(6))
            return f"{province}{city}{chars}"
        else:
            # 普通车牌：省份+城市+5位（字母数字混合）
            chars = ''.join(random.choice(self.CHARS) for _ in range(5))
            return f"{province}{city}{chars}"
    
    def _draw_text(self, plate: np.ndarray, text: str, text_color: Tuple[int, int, int], plate_type: str) -> np.ndarray:
        """使用 PIL 绘制中文文字"""
        from PIL import Image, ImageDraw, ImageFont
        
        # 转换为 PIL 图像
        pil_img = Image.fromarray(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # 加载中文字体（优先使用 simhei.ttf）
        font = None
        font_paths = [
            "simhei.ttf",  # 项目根目录
            "C:/Windows/Fonts/simhei.ttf",  # Windows 系统字体
            "C:/Windows/Fonts/msyh.ttc",  # Windows 微软雅黑
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, 40)
                    print(f"   ✅ 使用字体: {path}")
                    break
                except:
                    continue
        
        if font is None:
            print("   ⚠️ 未找到中文字体，使用默认字体")
            font = ImageFont.load_default()
        
        # 计算文字位置（居中）
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (plate.shape[1] - text_width) // 2
        y = (plate.shape[0] - text_height) // 2 - 5
        
        # 绘制文字
        draw.text((x, y), text, fill=text_color, font=font)
        
        # 转回 OpenCV 格式
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _add_noise(self, plate: np.ndarray, noise_level: float = 0.02) -> np.ndarray:
        """添加随机噪点"""
        noise = np.random.randn(*plate.shape) * noise_level * 255
        noisy = plate.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def _add_rotation(self, plate: np.ndarray) -> np.ndarray:
        """随机旋转"""
        angle = random.uniform(-5, 5)
        h, w = plate.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(plate, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        
        return rotated
    
    def _adjust_brightness_contrast(self, plate: np.ndarray) -> np.ndarray:
        """调整亮度和对比度"""
        brightness = random.uniform(0.7, 1.3)
        contrast = random.uniform(0.8, 1.2)
        
        adjusted = cv2.convertScaleAbs(plate, alpha=contrast, beta=(brightness-1)*50)
        return adjusted
    
    def generate_batch(
        self,
        count: int = 1000,
        plate_types: List[str] = None,
        output_prefix: str = "plate",
        save_annotation: bool = True,
    ) -> List[Tuple[str, str]]:
        """
        批量生成车牌
        
        Args:
            count: 生成数量
            plate_types: 车牌类型列表
            output_prefix: 输出文件名前缀
            save_annotation: 是否保存标注文件
        
        Returns:
            List[(filepath, plate_text)]
        """
        if plate_types is None:
            plate_types = ["blue", "green", "yellow", "white"]
        
        results = []
        annotations = []
        
        print(f"🔄 开始生成 {count} 张车牌...")
        
        for i in range(count):
            plate_type = random.choice(plate_types)
            
            # 随机生成参数
            width = random.randint(200, 280)
            height = random.randint(70, 90)
            
            img, text = self.generate_plate(
                plate_type=plate_type,
                width=width,
                height=height,
                add_noise=True,
                add_rotation=True
            )
            
            # 保存图片
            filename = f"{output_prefix}_{i:05d}_{text}.jpg"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), img)
            
            results.append((str(filepath), text))
            annotations.append({
                "filename": filename,
                "plate": text,
                "type": plate_type,
                "width": width,
                "height": height,
            })
            
            # 进度显示
            if (i + 1) % 100 == 0:
                print(f"   📊 已生成: {i+1}/{count}")
        
        # 保存标注文件
        if save_annotation:
            self._save_annotations(annotations)
        
        print(f"\n✅ 生成完成！共 {count} 张车牌")
        print(f"   📁 保存位置: {self.output_dir}")
        
        return results
    
    def _save_annotations(self, annotations: list):
        """保存标注文件"""
        import json
        
        # 保存为 JSON
        json_path = self.output_dir / "annotations.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        # 保存为 CSV
        csv_path = self.output_dir / "annotations.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("filename,plate,type,width,height\n")
            for ann in annotations:
                f.write(f"{ann['filename']},{ann['plate']},{ann['type']},{ann['width']},{ann['height']}\n")
        
        print(f"   📄 标注文件: {json_path}, {csv_path}")


def generate_demo_plates():
    """生成演示用的车牌图片"""
    generator = PlateGenerator("data/raw/synthetic_plates")
    
    # 生成不同类型车牌各一张
    plate_types = ["blue", "green", "yellow", "white"]
    texts = ["京A12345", "京A123456", "沪B88888", "粤C99999"]
    
    for i, (ptype, text) in enumerate(zip(plate_types, texts)):
        img, _ = generator.generate_plate(
            plate_type=ptype,
            text=text,
            width=240,
            height=80,
            add_noise=False,
            add_rotation=False
        )
        cv2.imwrite(str(generator.output_dir / f"demo_{ptype}_{text}.jpg"), img)
        print(f"✅ 生成演示: {ptype} - {text}")
    
    # 生成一批随机车牌
    generator.generate_batch(count=200, plate_types=["blue", "green"])


def main():
    parser = argparse.ArgumentParser(description="生成合成车牌数据")
    parser.add_argument("--count", type=int, default=500, help="生成数量")
    parser.add_argument("--types", nargs="+", default=["blue", "green"], 
                       help="车牌类型 (blue, green, yellow, white)")
    parser.add_argument("--output", type=str, default="data/raw/synthetic_plates", 
                       help="输出目录")
    parser.add_argument("--demo", action="store_true", help="仅生成演示图片")
    
    args = parser.parse_args()
    
    if args.demo:
        generate_demo_plates()
    else:
        generator = PlateGenerator(args.output)
        generator.generate_batch(
            count=args.count,
            plate_types=args.types
        )


if __name__ == "__main__":
    main()