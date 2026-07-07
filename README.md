# 🚦 智能交通路口管理系统

> **基于深度学习的车辆逆行检测与违规工单管理平台**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11-green.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.85+-teal.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 项目简介

本项目是一套完整的**智能交通路口管理系统**，通过对监控视频流进行实时分析，自动检测车辆逆行行为、识别违规车辆车牌，并生成完整的违规工单。系统采用模块化设计，各组件可独立运行或集成部署，适用于城市交通管理、智慧交通建设等场景。

### 核心能力

| 功能 | 说明 |
| :--- | :--- |
| 🚗 **车辆检测** | YOLO11 实时检测视频中的车辆，输出边界框和车型分类 |
| 🎯 **目标跟踪** | DeepSORT 跨帧追踪车辆，为每辆车分配唯一 ID |
| 🚨 **逆行检测** | 基于轨迹方向与车道预设方向对比，智能判定逆行 |
| 📋 **车牌识别** | PaddleOCR 识别逆行车辆车牌（支持真实/模拟模式） |
| 📄 **工单管理** | 完整的违规工单生命周期管理（创建→审核→处理→归档） |
| 🔌 **API 服务** | RESTful API 接口，支持第三方系统集成 |
| 📊 **数据统计** | 多维度违规数据分析与报表导出 |

---

## 🛠️ 技术栈

| 模块 | 技术选型 | 版本 | 说明 |
| :--- | :--- | :--- | :--- |
| **编程语言** | Python | 3.9+ | 主要开发语言 |
| **深度学习框架** | PyTorch / PaddlePaddle | 最新 | 模型推理引擎 |
| **目标检测** | YOLO11 (Ultralytics) | 8.0+ | 车辆检测与分类 |
| **目标跟踪** | DeepSORT (ByteTrack) | - | 车辆跟踪与轨迹记录 |
| **车牌识别** | PaddleOCR | 2.6+ | 文本检测 + 文本识别 |
| **数据库** | SQLite | 内置 | 轻量级本地数据库 |
| **API 框架** | FastAPI | 0.85+ | RESTful API + Swagger 文档 |
| **图像处理** | OpenCV | 4.5+ | 视频帧处理与可视化 |
| **配置管理** | JSON / YAML | - | 灵活配置车道规则 |



## 📁 项目结构

going-against-the-flow/
│
├── 🚀 启动与配置
│ ├── run_api.py # API 服务入口
│ ├── requirements.txt # Python 依赖
│ ├── docker-compose.yml # 容器编排
│ └── config/ # 配置文件
│ ├── lane_rules.json # 车道方向规则（7种路口）
│ └── camera_lane_map.json # 摄像头-车道映射
│
├── 🧠 核心算法引擎（src/）
│ ├── detection/ # YOLO11 车辆检测
│ ├── tracking/ # DeepSORT 多目标跟踪
│ ├── violation/ # 逆行检测（增强版）
│ ├── ocr/ # 车牌识别（PaddleOCR）
│ ├── ticket/ # 工单生成与管理
│ └── pipeline/ # 检测→跟踪→逆行→工单 全流程
│
├── 🌐 API 服务层（src/api/）
│ ├── main.py # FastAPI 主入口
│ ├── routes/ # 路由（detect / tickets / statistics）
│ ├── schemas/ # Pydantic 数据模型
│ └── stream/ # WebSocket 实时推送
│
├── 🛠️ 辅助工具
│ ├── scripts/ # 运行脚本（流水线、测试集评估）
│ ├── tests/ # 单元测试
│ └── utils/ # 日志、可视化工具
│
├── 📁 数据与输出（被 .gitignore 排除）
│ ├── data/ # 原始图片/视频
│ ├── checkpoints/ # 模型权重（*.pt）
│ └── output/ # 运行结果（标注图、报告）
│
└── 📄 文档
└── README.md # 项目说明


---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
| :--- | :--- |
| Python | 3.9 或更高版本 |
| 内存 | 8GB+ (推荐 16GB) |
| 存储 | 10GB+ (含数据集) |
| CUDA | 11.8+ (可选，GPU加速) |

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd "Going against the flow"

# 2. 创建并激活虚拟环境 (Conda)
conda create -n yolo26_env python=3.9
conda activate yolo26_env

# 3. 安装依赖
pip install -r requirements.txt

# 4. 下载 YOLO11 模型 (自动下载)
# 首次运行时会自动下载 yolo11n.pt

# 5. 准备数据集
# 将 DETRAC 数据集放入 data/raw/DETRAC-train-data/

📖 使用指南
1. 运行工单模块测试

python test_ticket.py

2. 运行完整流水线（单张图片检测）

python scripts/run_pipeline.py
输出：output/pipeline_results/result_*.jpg

3. 逆行检测可视化（批量处理序列）

python scripts/run_violation_detection.py
输出：output/violation_demo/

4. 启动 API 服务

python run_api.py
访问：http://127.0.0.1:8000/docs

5. 测试集评估

python scripts/test_on_testset.py
输出：output/evaluation/evaluation_results_*.json

6. 生成合成车牌数据

python scripts/generate_plates.py --demo
python scripts/generate_plates.py --count 500

📊 测试结果
工单模块测试

============================================================
🚦 工单模块完整测试开始
============================================================
✅ 工单管理器初始化成功

🚗 测试工单创建
✅ 工单创建成功!
  📋 工单ID: T20260706000001
  🚗 车牌号: 京A12345
  🏷️  车型: car
  📌 状态: pending

🚑 测试特殊车辆工单
✅ 特殊车辆工单创建成功!
  📋 工单ID: T20260706000002
  🚗 车牌号: 京B99999
  🏷️  车型: ambulance
  🚨 特殊车辆: 救护车
  📌 状态: dismissed (自动驳回)

🔍 测试工单查询
📊 总工单数: 2

✏️ 测试状态更新
✅ 状态更新成功!
   新状态: confirmed

📤 测试导出功能
✅ JSON导出成功!
✅ CSV导出成功!

🎉 所有测试通过!

逆行检测测试

📊 处理结果:
   - 检测车辆: 12
   - 跟踪目标: 8
   - 🚨 逆行: 3
   - 🚗 车牌识别: 3
   - 📋 生成工单: 3

🔮 未来改进方向
方向	    说明
模型优化	使用 YOLO11x 提升检测精度，适配更多场景
车牌识别	使用 CCPD 数据集训练专用车牌识别模型
实时流处理	支持 RTSP 摄像头实时流，实现 7×24 监控
多路口管理	支持多个路口同时监控与管理
Web 管理界面	开发前端可视化平台，提升用户体验
告警推送	支持短信、邮件、Webhook 告警通知
数据大屏	实时数据可视化大屏展示