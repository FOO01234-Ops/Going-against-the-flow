# convert_to_base64.py
import base64
from pathlib import Path

def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')
    
# 使用示例
img_path = "data/raw/DETRAC-test-data/Insight-MVT_Annotation_Test"
b64 = image_to_base64(img_path)
print(b64)  # 复制这个字符串到 Swagger 的 image_base64 字段