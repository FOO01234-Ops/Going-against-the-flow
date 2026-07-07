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

Going against the flow/
│
├── README.md # 项目说明文档
├── requirements.txt # 依赖包列表
├── run_api.py # API 服务启动入口
├── test_ticket.py # 工单模块单元测试
│
├── config/ # 配置文件目录
│ ├── lane_rules.json # 车道方向规则
│ └── camera_lane_map.json # 摄像头车道映射
│
├── data/ # 数据目录
│ ├── raw/ # 原始数据 (DETRAC, 合成车牌)
│ │ ├── DETRAC-train-data/ # DETRAC 训练集
│ │ ├── DETRAC-Train-Annotations-XML/ # DETRAC 标注
│ │ └── synthetic_plates/ # 合成车牌数据
│ ├── database/ # SQLite 数据库
│ │ └── violation.db # 工单数据库
│ └── checkpoints/ # 模型权重文件
│ └── yolo11n.pt # YOLO11 模型权重
│
├── src/ # 源代码目录
│ ├── detection/ # 🚗 YOLO11 车辆检测模块
│ │ ├── detector.py # 检测器封装
│ │ └── model_factory.py # 模型工厂
│ │
│ ├── tracking/ # 🎯 DeepSORT 目标跟踪模块
│ │ ├── tracker.py # 跟踪器封装
│ │ └── trajectory.py # 轨迹管理
│ │
│ ├── violation/ # 🚨 逆行检测模块
│ │ ├── direction_checker.py # 方向计算器
│ │ ├── violation_detector.py # 基础检测器
│ │ ├── enhanced_detector.py # 增强版检测器
│ │ ├── lane_matcher.py # 车道匹配器
│ │ └── special_vehicle_filter.py # 特殊车辆豁免
│ │
│ ├── ocr/ # 📋 车牌识别模块
│ │ └── license_ocr.py # PaddleOCR 封装 (支持模拟模式)
│ │
│ ├── ticket/ # 📄 工单管理模块
│ │ ├── models.py # 数据模型 (Enum, Dataclass)
│ │ ├── database.py # SQLite 数据库操作
│ │ ├── ticket_generator.py # 工单生成器
│ │ └── ticket_manager.py # 工单管理器 (统一接口)
│ │
│ ├── pipeline/ # 🔗 完整流水线模块
│ │ ├── video_pipeline.py # 视频处理流水线
│ │ └── frame_processor.py # 单帧处理器
│ │
│ ├── api/ # 🔌 API 服务模块
│ │ ├── main.py # FastAPI 主入口
│ │ ├── dependencies.py # 依赖注入
│ │ ├── routes/ # 路由
│ │ │ ├── detect.py # 检测接口
│ │ │ ├── tickets.py # 工单接口
│ │ │ └── statistics.py # 统计接口
│ │ └── schemas/ # 数据模型
│ │ └── models.py # Pydantic 模型
│ │
│ └── utils/ # 工具函数
│ ├── logger.py # 日志工具
│ └── visualization.py # 可视化工具
│
├── scripts/ # 运行脚本
│ ├── run_pipeline.py # 运行完整流水线
│ ├── run_violation_detection.py # 逆行检测可视化
│ ├── test_on_testset.py # 测试集评估
│ └── generate_plates.py # 合成车牌生成器
│
├── tests/ # 测试目录
│ ├── test_ticket.py # 工单模块测试
│ └── test_violation.py # 逆行检测测试
│
└── output/ # 输出目录
├── pipeline_results/ # 流水线标注结果
├── violation_demo/ # 逆行检测演示
├── snapshots/ # 违规截图
└── reports/ # 导出报告


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