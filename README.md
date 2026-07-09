# 🚦 智能交通路口管理系统

基于 **YOLO11 + DeepSORT + FastAPI** 的智慧交通解决方案。系统通过分析路口监控视频或图片，实时检测车辆逆行行为，自动生成违章工单，并提供可视化标注输出和 WebSocket 实时推送。

---

## ✨ 核心功能

- 🚗 **目标检测**：YOLO11 识别车辆（轿车、卡车、巴士等）
- 🔍 **多目标跟踪**：DeepSORT 为每辆车分配唯一 ID 并绘制运动轨迹
- ⚠️ **逆行判定**：多帧滑动窗口 + 圆形平均 + 一致性检查，抑制抖动误报
- 🎫 **工单闭环**：自动生成违章工单，支持状态管理及 CSV/JSON 导出
- 🖼️ **实时可视化**：API 返回带边界框和轨迹线的 Base64 图片
- 📡 **实时推送**：Redis + WebSocket，工单实时推送至前端

---

## 📂 项目结构

```text
going-against-the-flow/
│
├── 🚀 启动与配置
│   ├── run_api.py                # API 服务入口
│   ├── requirements.txt          # Python 依赖
│   ├── docker-compose.yml        # 容器编排
│   └── config/                   # 配置文件
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
│   ├── scripts/                  # 运行脚本（流水线、测试集评估）
│   ├── tests/                    # 单元测试
│   └── utils/                    # 日志、可视化工具
│
├── 📁 数据与输出（被 .gitignore 排除）
│   ├── data/                     # 原始图片/视频
│   ├── checkpoints/              # 模型权重（*.pt）
│   └── output/                   # 运行结果（标注图、报告）
│
└── 📄 文档
    └── README.md                 # 项目说明
```

---

## 🚀 快速开始

### 环境要求

| 项目 | 要求 |
| :--- | :--- |
| Python | 3.9 或更高版本 |
| 内存 | 8GB+（推荐 16GB） |
| 存储 | 16GB+（含数据集） |
| CUDA | 11.8+（可选，GPU加速） |

### 本地运行

```bash
# 1. 克隆项目
git clone https://github.com/FOO01234-Ops/Going-against-the-flow.git
cd Going-against-the-flow

# 2. 创建并激活虚拟环境（以 conda 为例）
conda create -n yolo26_env python=3.10
conda activate yolo26_env

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动 Redis（可选，用于 WebSocket 推送）
docker run -d --name redis -p 6379:6379 redis

# 5. 启动 API 服务
python run_api.py
```

### Docker 一键部署

```bash
docker-compose up --build
```

访问 `http://localhost:8000/docs` 查看 API 文档。

### 外网访问（公网演示）

```bash
ngrok http 8000 --region=ap
```

生成公网链接后，访问 `https://xxxx.ngrok-free.dev/docs` 即可。

---

## 📖 API 接口概览

| 模块 | 方法 | 端点 | 描述 |
| :--- | :--- | :--- | :--- |
| 检测 | POST | `/api/detect/frame` | 上传图片，返回检测结果与标注图 |
| 检测 | POST | `/api/detect/upload` | 同 `/frame`，别名接口 |
| 工单 | GET | `/api/tickets` | 获取所有违章工单列表 |
| 工单 | GET | `/api/tickets/{id}` | 获取指定工单详情 |
| 工单 | PUT | `/api/tickets/{id}/status` | 更新工单处理状态 |
| 工单 | GET | `/api/tickets/export/json` | 导出工单为 JSON |
| 工单 | GET | `/api/tickets/export/csv` | 导出工单为 CSV |
| 系统 | GET | `/health` | 服务健康检查 |
| 系统 | WS | `/ws/events` | WebSocket 实时工单推送 |

---

## 🧠 技术栈

| 领域 | 技术选型 |
| :--- | :--- |
| 目标检测 | Ultralytics YOLO11 |
| 多目标跟踪 | DeepSORT (ByteTrack) |
| 逆行判定 | 滑动窗口 + 圆形平均 + 一致性检查 |
| 后端框架 | FastAPI + Uvicorn |
| 数据库 | SQLite |
| 事件总线 | Redis（可选） |
| 部署 | Docker & Docker Compose |
| 前端交互 | Swagger UI / HTML + JavaScript |

---

## 📝 待优化方向

- [ ] 接入 RTSP 实时视频流
- [ ] TensorRT GPU 加速
- [ ] Kafka 分布式事件系统
- [ ] 3D 数字孪生大屏前端

---

## 📄 许可证

MIT License
