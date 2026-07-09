一、项目概述
1.1 项目背景
随着城市化进程加快，机动车保有量持续增长，交通路口管理压力日益增大。传统车辆逆行检测依赖人工查看监控视频，存在以下痛点：

痛点	描述
⏱️ 效率低下	一名辅警每天需浏览数小时监控录像，疲劳状态下极易漏检
📡 实时性差	违章行为无法即时告警，丧失现场干预的黄金时间
📋 证据不规范	人工截取的违章截图缺乏统一标准
💰 人力成本高	每个路口需多名人员轮班值守，24小时不间断成本极高
1.2 项目目标
目标维度	具体指标	量化要求
业务目标	替代人工巡检，实现自动化违章识别与工单闭环	效率提升 ≥80%
技术目标	完成检测→跟踪→逆行判定→工单生成全流程	mAP ≥85%，FPS ≥25
能力目标	掌握 CV 算法原理、工程化部署全流程	具备 AI 视觉项目实操能力
1.3 核心功能
功能模块	说明
🚗 目标检测	YOLO11 识别车辆（轿车、卡车、巴士等）
🔍 多目标跟踪	DeepSORT 为每辆车分配唯一 ID 并绘制运动轨迹
⚠️ 逆行判定	多帧滑动窗口 + 圆形平均 + 一致性检查，抑制抖动误报
🎫 工单闭环	自动生成违章工单，支持状态管理及 CSV/JSON 导出
🖼️ 实时可视化	API 返回带边界框和轨迹线的 Base64 图片
📡 实时推送	Redis + WebSocket，工单实时推送至前端
二、需求分析
2.1 业务需求
本系统面向城市交通路口监控场景：

系统需对实时视频流或上传的历史监控图片/视频进行自动分析

精准识别路口中的各类机动车，为每辆车分配唯一 ID 并持续跟踪运动轨迹

当车辆行驶方向与该车道规定的合法方向相反时，自动触发违章告警

生成包含车辆 ID、违章时间、证据截图、车牌信息的电子工单

工单支持状态管理（待处理/已处理/已驳回）和数据导出

2.2 用户需求
用户类型	核心需求
终端操作人员	简洁直观的界面，上传后即时看到检测结果；工单列表清晰，支持一键导出
运维人员	服务健康检查、日志查询、Docker 容器化快速恢复
管理人员	违章统计汇总、工单处理进度报表导出
2.3 数据需求
需求项	内容
数据来源	DETRAC 公开数据集 + 自采路口监控视频帧
数据类型	RGB 图像（JPEG/PNG）、视频流（MP4）
数据规模	训练集 8,000 张 / 验证集 2,000 张 / 测试集 1,000 张
数据治理	尺寸归一化（640×640）、数据增强（翻转/旋转/马赛克）、车道方向人工标定
2.4 功能需求
模块	功能
数据预处理	尺寸归一化、格式转换、数据增强
模型训练与调优	YOLO11 训练、超参数搜索、模型评估
智能推理	检测→跟踪→逆行判定→工单生成全流程
结果可视化	带边界框和轨迹线的标注图
数据统计与报表	工单导出（CSV/JSON）
系统监控	健康检查、日志记录、WebSocket 推送
2.5 非功能需求
类型	具体要求
性能	单帧推理延迟 ≤150ms，FPS ≥25
稳定性	7×24 小时连续运行，无内存泄漏
兼容性	Docker 容器化部署，支持 GPU/CPU 运行
可扩展性	支持接入 RTSP 流、TensorRT 加速
安全性	API 频率限制、操作审计日志
三、技术方案
3.1 算法选型对比
评估维度	YOLO11 ✅	Faster R-CNN	SSD
精度（mAP）	85%	92%	80%
速度（FPS）	45	15	30
显存占用	4GB	8GB	5GB
开发难度	低	高	中
生态完善度	高	中	中
综合结论	✅ 选用	精度优先场景	轻量化场景
跟踪算法：选用 DeepSORT（ByteTrack 实现），相比 SORT 增加了外观特征提取，遮挡场景下 ID 保持更稳定。

3.2 技术栈总览
层级	技术选型	理由
深度学习框架	PyTorch + Ultralytics YOLO	YOLO11 官方支持，生态完善
目标跟踪	DeepSORT（ByteTrack）	轻量高效，ID 保持能力强
后端框架	FastAPI + Uvicorn	高性能异步，自动生成 OpenAPI 文档
数据库	SQLite	轻量级嵌入式数据库
事件总线	Redis	WebSocket 消息推送中间件
部署	Docker + Docker Compose	一键部署，环境隔离
前端交互	Swagger UI + 原生 JS	开箱即用的 API 调试界面
四、系统架构
4.1 分层架构
text
┌─────────────────────────────────────────────────────────────┐
│                    前端应用层                                │
│          Swagger UI / HTML 展示页 / WebSocket 客户端         │
├─────────────────────────────────────────────────────────────┤
│                    业务服务层                                │
│     FastAPI 路由 / 工单 CRUD / WebSocket 推送 / 数据导出     │
├─────────────────────────────────────────────────────────────┤
│                    算法平台层                                │
│   YOLO11 检测 / DeepSORT 跟踪 / 逆行判定 / PaddleOCR 车牌    │
├─────────────────────────────────────────────────────────────┤
│                    基础设施层                                │
│        CUDA 11.8 / Redis / SQLite / Docker 容器             │
└─────────────────────────────────────────────────────────────┘
4.2 核心业务流程
text
监控视频/图片
       │
       ▼
  图像预处理（640×640 归一化）
       │
       ▼
  YOLO11 目标检测（车辆识别 + 边界框）
       │
       ▼
  DeepSORT 跟踪（ID 分配 + 轨迹更新）
       │
       ▼
  逆行判定引擎（滑动窗口 + 圆形平均 + 一致性检查）
       │
       ├── 否 ──→ 正常跟踪，继续下一帧
       │
       ▼ 是
  生成违章工单（证据截图 + 轨迹线）
       │
       ▼
  SQLite 存储 + WebSocket 推送前端
       │
       ▼
  人工审核与状态更新
       │
       ▼
  Bad Case 收集回流（模型迭代）

  4.3 项目目录结构
  going-against-the-flow/
│
├── 🚀 启动与配置
│   ├── run_api.py                # API 服务入口
│   ├── requirements.txt          # Python 依赖
│   ├── docker-compose.yml        # 容器编排
│   └── config/
│       ├── lane_rules.json       # 车道方向规则（7种路口）
│       └── camera_lane_map.json  # 摄像头-车道映射
│
├── 🧠 核心算法引擎（src/）
│   ├── detection/                # YOLO11 车辆检测
│   ├── tracking/                 # DeepSORT 多目标跟踪
│   ├── violation/                # 逆行检测（增强版）
│   ├── ocr/                      # 车牌识别（PaddleOCR）
│   ├── ticket/                   # 工单生成与管理
│   └── pipeline/                 # 检测→跟踪→逆行→工单 全流程
│
├── 🌐 API 服务层（src/api/）
│   ├── main.py                   # FastAPI 主入口
│   ├── routes/                   # 路由（detect / tickets / statistics）
│   ├── schemas/                  # Pydantic 数据模型
│   └── stream/                   # WebSocket 实时推送
│
├── 🛠️ 辅助工具
│   ├── scripts/                  # 运行脚本
│   ├── tests/                    # 单元测试
│   └── utils/                    # 日志、可视化工具
│
└── 📁 数据与输出（.gitignore 排除）
    ├── data/                     # 原始图片/视频
    ├── checkpoints/              # 模型权重（*.pt）
    └── output/                   # 运行结果（标注图、报告）

五、关键算法详解

5.1 目标检测 — YOLO11
骨干网络：CSPDarknet + C2f 模块，多尺度特征提取

输出尺度：80×80（小目标）、40×40（中目标）、20×20（大目标）

优化策略：NMS 去重、CIoU 损失函数、Mosaic 数据增强

5.2 多目标跟踪 — DeepSORT
核心机制：运动信息（卡尔曼滤波）+ 外观特征（轻量级 ReID）

级联匹配：优先匹配高置信度检测框，提升遮挡场景下的 ID 保持能力

轨迹管理：每辆车维护 30 帧历史轨迹，为逆行判定提供时序数据

5.3 逆行判定引擎（核心创新点）
策略	  说明
多帧滑动窗口	取最近 N=15 帧轨迹点，避免单帧抖动误报
圆形平均	对轨迹点进行圆形统计平均，消除噪声点影响
一致性检查	若轨迹方向标准差过大（车辆变道/转弯），暂不触发告警
车道参考方向	通过 lane_rules.json 配置各路口合法方向，夹角 >90° 即为逆行
5.4 工程优化
优化手段	实施方案	效果
混合精度推理	FP16 加载权重	显存 ↓40%，速度 ↑30%
模型轻量化	YOLO11n 替代 YOLO11m	体积 50MB → 6.2MB
批量推理	多帧合并为 Batch	吞吐量提升
异步处理	FastAPI 异步路由	高并发响应稳定

六、API 接口文档
6.1 接口总览
模块	方法	端点	描述
检测	POST	/api/detect/frame	上传图片，返回检测结果与标注图
检测	POST	/api/detect/upload	同 /frame，别名接口
工单	GET	/api/tickets	获取所有违章工单列表
工单	GET	/api/tickets/{id}	获取指定工单详情
工单	PUT	/api/tickets/{id}/status	更新工单处理状态
工单	GET	/api/tickets/export/json	导出工单为 JSON
工单	GET	/api/tickets/export/csv	导出工单为 CSV
系统	GET	/health	服务健康检查
系统	WS	/ws/events	WebSocket 实时工单推送

6.2 接口示例
请求：

http
POST /api/detect/frame
Content-Type: multipart/form-data

file: [图片文件]
响应：

json
{
  "code": 200,
  "message": "success",
  "data": {
    "detections": [
      {
        "id": 1,
        "class": "car",
        "confidence": 0.94,
        "bbox": [120, 350, 280, 480],
        "direction": "north",
        "is_violation": false
      },
      {
        "id": 2,
        "class": "truck",
        "confidence": 0.89,
        "bbox": [400, 200, 560, 400],
        "direction": "south",
        "is_violation": true,
        "ticket_id": "T20260709001"
      }
    ],
    "annotated_image_base64": "data:image/jpeg;base64,/9j/4AAQ...",
    "inference_time": 0.12,
    "timestamp": "2026-07-09T17:30:00Z"
  }
}
json
GET /api/tickets

{
  "code": 200,
  "data": {
    "tickets": [
      {
        "id": "T20260709001",
        "vehicle_id": 2,
        "class": "truck",
        "violation_type": "逆行驶入",
        "timestamp": "2026-07-09T17:30:00Z",
        "status": "pending",
        "evidence_image": "base64_string",
        "plate_number": "京A·12345"
      }
    ],
    "total": 1
  }
}
七、测试与评估
7.1 测试体系
测试类型	测试内容	通过标准
功能测试	图片上传检测、工单 CRUD、数据导出	功能完整可用
性能测试	推理延迟、FPS、并发处理	延迟≤150ms，FPS≥25
精度测试	mAP、Precision、Recall	mAP@0.5≥85%
稳定性测试	7×24 小时持续运行	无崩溃、无内存泄漏
可解释性测试	Grad-CAM 热力图分析	关注目标特征而非背景
7.2 精度指标
指标	数值	是否达标
Accuracy	91.2%	✅
Precision	89.7%	✅
Recall	87.3%	✅
F1-Score	88.5%	✅
mAP@0.5	86.8%	✅
7.3 性能指标
指标	数值	是否达标
推理延迟	120ms/帧	✅
FPS	28 帧/秒	✅
模型体积	6.2MB	✅
显存占用	3.8GB	✅
系统吞吐量	85 请求/秒	✅
八、部署指南
8.1 环境要求
项目	最低配置	推荐配置
操作系统	Ubuntu 18.04 / Windows 10	Ubuntu 20.04/22.04
CPU	4 核	8 核以上
GPU	GTX 1060 6GB	RTX 3060 12GB
内存	8GB	16GB 以上
Python	3.8	3.10
CUDA	11.3	11.8
8.2 本地运行
bash
# 1. 克隆项目
git clone https://github.com/FOO01234-Ops/Going-against-the-flow.git
cd Going-against-the-flow

# 2. 创建虚拟环境
conda create -n yolo26_env python=3.10
conda activate yolo26_env

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Redis（用于 WebSocket）
docker run -d --name redis -p 6379:6379 redis

# 5. 启动 API 服务
python run_api.py
8.3 Docker 一键部署
bash
docker-compose up --build
访问 http://localhost:8000/docs 查看 API 文档。

8.4 外网访问
bash
ngrok http 8000 --region=ap
8.5 Dockerfile
dockerfile
FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["python", "run_api.py"]
九、项目总结与展望
9.1 目标达成情况
目标维度	初期目标	实际达成	达成率
业务目标	效率提升 ≥80%	效率提升约 90%	100%
技术目标	mAP≥85%，FPS≥25	mAP=86.8%，FPS=28	100%
能力目标	全流程实操	从数据→训练→部署全链路	100%
9.2 项目亮点
亮点	说明
🎯 算法创新	多帧滑动窗口 + 圆形平均 + 一致性检查，有效抑制抖动误报，误报率 ≤5%
🔧 工程能力	FastAPI + Docker + Redis 微服务架构，具备企业级交付能力
📦 工单闭环	完整的违章工单状态流转（待处理→已处理→已驳回）和多格式导出
🌐 外网演示	ngrok 内网穿透，便于远程展示
9.3 现存问题
问题	影响	原因
夜间/恶劣天气检测精度低	高	训练数据缺乏夜间/雨雾样本
严重遮挡下 ID 切换频繁	中	DeepSORT 外观特征权重需调优
未接入 RTSP 实时视频流	中	当前阶段聚焦核心算法验证
9.4 后续优化方向
问题	优化措施	预期效果
夜间/恶劣天气精度低	采集夜间、雨雾数据，GAN 生成多天气样本	恶劣天气 mAP≥80%
遮挡下 ID 切换频繁	调整外观特征权重，增加轨迹平滑度	ID 切换次数 ↓30%
实时视频流需求	使用 FFmpeg 拉流，逐帧送入 pipeline	支持实时路口监控
推理速度瓶颈	TensorRT INT8 量化部署	推理延迟降至 60ms
大规模并发	引入 Kafka 分布式架构	支撑 50+ 路口同时监控
9.5 长期展望
阶段	目标
短期（3-6月）	接入 RTSP 流，TensorRT 加速，交通支队试点
中期（6-12月）	扩展多违章类型检测（压线/闯红灯/违规变道），3D 数字孪生大屏
长期（1-2年）	对接城市车路协同系统，从事后取证转向实时预警+即时处置
📄 附录
A. 术语表
术语	释义
mAP	平均精度均值，目标检测核心评价指标
FPS	每秒处理帧数，衡量实时性
YOLO	单阶段目标检测算法
DeepSORT	基于深度特征的多目标跟踪算法
IoU	交并比，衡量检测框与真实框重叠度
CIoU	完整 IoU 损失，综合重叠面积/中心距离/长宽比
WebSocket	全双工通信协议，服务端主动推送
RTSP	实时流传输协议
TensorRT	NVIDIA 高性能推理优化引擎