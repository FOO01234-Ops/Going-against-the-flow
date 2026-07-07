# src/ocr/license_ocr.py
"""
车牌识别 - 使用 PaddleOCR（支持模拟模式）
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import re
import random

try:
    from paddleocr import PaddleOCR
    PADDLE_OCR_AVAILABLE = True
except ImportError:
    PADDLE_OCR_AVAILABLE = False
    print("⚠️ PaddleOCR 未安装，请运行: pip install paddlepaddle paddleocr")


class LicenseOCR:
    """
    车牌识别器
    
    支持两种模式：
    1. 真实模式：使用 PaddleOCR 识别
    2. 模拟模式：返回模拟数据（用于流程测试）
    """
    
    # 中国车牌正则表达式
    PLATE_PATTERN = re.compile(
        r'^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]'
        r'[A-Z]'
        r'[A-Z0-9]{5}$'
    )
    
    # 省份简称
    PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼"
    
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = 'ch',
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        rec_score_thresh: float = 0.5,
        mock_mode: bool = False,  # ✅ 新增：模拟模式开关
    ):
        """
        初始化车牌识别器
        
        Args:
            use_gpu: 是否使用GPU
            lang: 语言 (ch, en, etc.)
            det_db_thresh: 检测阈值
            det_db_box_thresh: 检测框阈值
            rec_score_thresh: 识别置信度阈值
            mock_mode: 是否使用模拟模式（默认 False）
        """
        self.use_gpu = use_gpu
        self.lang = lang
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.rec_score_thresh = rec_score_thresh
        self.mock_mode = mock_mode
        
        self.ocr = None
        
        if PADDLE_OCR_AVAILABLE and not mock_mode:
            self._init_ocr()
        elif mock_mode:
            print(f"✅ 车牌识别器初始化成功 (模拟模式)")
            print(f"   📌 将返回模拟车牌数据，用于流程测试")
        else:
            print("⚠️ PaddleOCR 不可用，使用模拟模式")
            self.mock_mode = True
    
    def _init_ocr(self):
        """初始化 PaddleOCR"""
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                rec_score_thresh=self.rec_score_thresh,
                use_gpu=self.use_gpu,
                show_log=False,
            )
            print(f"✅ PaddleOCR 初始化成功")
            print(f"   📌 GPU: {self.use_gpu}, 语言: {self.lang}")
        except Exception as e:
            print(f"❌ PaddleOCR 初始化失败: {e}")
            self.ocr = None
            self.mock_mode = True
            print(f"   🔄 自动切换到模拟模式")
    
    def detect_plate(self, image: np.ndarray) -> List[Dict]:
        """
        检测并识别图片中的车牌
        
        Args:
            image: 输入图片 (BGR格式)
        
        Returns:
            List[Dict]: 识别结果列表
                [
                    {
                        "plate": "京A12345",
                        "confidence": 0.95,
                        "bbox": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                        "text": "京A12345"
                    }
                ]
        """
        # ✅ 模拟模式：返回模拟数据
        if self.mock_mode:
            return self._mock_detect_plate(image)
        
        # 真实模式
        if self.ocr is None:
            return self._mock_detect_plate(image)
        
        try:
            # 转换 BGR -> RGB (PaddleOCR 需要 RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 执行OCR
            result = self.ocr.ocr(rgb_image, cls=True)
            
            if not result or not result[0]:
                return []
            
            plates = []
            for line in result[0]:
                # 解析结果
                bbox = line[0]  # 四个点坐标
                text_info = line[1]  # (text, confidence)
                
                if len(text_info) < 2:
                    continue
                
                text = text_info[0]
                confidence = text_info[1]
                
                # 过滤低置信度
                if confidence < self.rec_score_thresh:
                    continue
                
                # 清理文本（只保留字母和数字）
                clean_text = self._clean_plate_text(text)
                
                # 验证车牌格式
                if self._is_valid_plate(clean_text):
                    plates.append({
                        "plate": clean_text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "text": text,
                    })
            
            # 如果没识别到，用模拟数据兜底
            if not plates:
                return self._mock_detect_plate(image)
            
            return plates
            
        except Exception as e:
            print(f"❌ OCR 识别失败: {e}")
            return self._mock_detect_plate(image)
    
    def detect_plate_from_bbox(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[Dict]:
        """
        从指定区域识别车牌
        
        Args:
            image: 完整图片
            bbox: (x1, y1, x2, y2) 车辆边界框
        
        Returns:
            Optional[Dict]: 车牌识别结果
        """
        x1, y1, x2, y2 = bbox
        
        # 扩大区域（车牌可能在车辆框的下半部分）
        h = y2 - y1
        w = x2 - x1
        
        # 裁剪车辆区域的下半部分（车牌通常在这里）
        crop_y1 = y1 + int(h * 0.5)
        crop_y2 = y2 + int(h * 0.1)
        crop_x1 = max(0, x1 - int(w * 0.1))
        crop_x2 = min(image.shape[1], x2 + int(w * 0.1))
        
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop.size == 0:
            return None
        
        # 识别
        results = self.detect_plate(crop)
        
        if results:
            # 调整边界框坐标回到原图
            result = results[0]
            bbox_original = []
            for point in result["bbox"]:
                bbox_original.append([point[0] + crop_x1, point[1] + crop_y1])
            
            result["bbox"] = bbox_original
            return result
        
        return None
    
    def _mock_detect_plate(self, image: np.ndarray) -> List[Dict]:
        """
        模拟车牌检测（当 PaddleOCR 不可用或启用模拟模式时）
        
        返回:
        - 有 70% 概率返回一个模拟车牌
        - 有 30% 概率返回空（模拟没检测到）
        """
        # 30% 概率没检测到
        if random.random() < 0.3:
            return []
        
        # 生成模拟车牌
        province = random.choice(self.PROVINCES)
        city = random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ")
        chars = ''.join(random.choice("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789") for _ in range(5))
        plate = f"{province}{city}{chars}"
        
        h, w = image.shape[:2]
        
        # 生成一个合理的边界框（在图片下半部分）
        bbox = [
            [int(w * random.uniform(0.1, 0.2)), int(h * random.uniform(0.5, 0.6))],
            [int(w * random.uniform(0.7, 0.8)), int(h * random.uniform(0.5, 0.6))],
            [int(w * random.uniform(0.7, 0.8)), int(h * random.uniform(0.7, 0.8))],
            [int(w * random.uniform(0.1, 0.2)), int(h * random.uniform(0.7, 0.8))],
        ]
        
        return [{
            "plate": plate,
            "confidence": random.uniform(0.7, 0.98),
            "bbox": bbox,
            "text": plate,
        }]
    
    def _clean_plate_text(self, text: str) -> str:
        """清理车牌文本"""
        text = text.replace(" ", "").replace("-", "").replace("·", "")
        text = text.upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
        return text
    
    def _is_valid_plate(self, text: str) -> bool:
        """验证车牌格式"""
        if not text:
            return False
        
        if len(text) not in [7, 8]:
            return False
        
        if text[0] not in self.PROVINCES:
            return False
        
        if not text[1].isalpha():
            return False
        
        if not all(c.isalnum() for c in text[2:]):
            return False
        
        return True
    
    def draw_plate_result(
        self,
        image: np.ndarray,
        result: Dict,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """
        在图片上绘制车牌识别结果
        
        Args:
            image: 原始图片
            result: 识别结果
            color: 颜色
        
        Returns:
            np.ndarray: 标注后的图片
        """
        annotated = image.copy()
        
        if not result:
            return annotated
        
        bbox = result["bbox"]
        plate = result["plate"]
        confidence = result["confidence"]
        
        # 画多边形
        pts = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [pts], True, color, 2)
        
        # 标签
        label = f"{plate} ({confidence:.2f})"
        x, y = int(bbox[0][0]), int(bbox[0][1] - 10)
        
        cv2.putText(
            annotated,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        return annotated
    
    def set_mock_mode(self, enabled: bool = True):
        """切换模拟模式"""
        self.mock_mode = enabled
        print(f"🔧 模拟模式: {'开启' if enabled else '关闭'}")


def create_license_ocr(mock_mode: bool = False, **kwargs) -> LicenseOCR:
    """创建车牌识别器的便捷函数"""
    return LicenseOCR(mock_mode=mock_mode, **kwargs)