# run_api.py (项目根目录)
"""
启动 API 服务
"""
import sys
from pathlib import Path

# ✅ 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn


if __name__ == "__main__":
    print("=" * 60)
    print("🚦 智能交通路口管理系统 API")
    print("=" * 60)
    print("")
    print("📖 API 文档: http://127.0.0.1:8000/docs")
    print("📖 ReDoc:    http://127.0.0.1:8000/redoc")
    print("🏠 根路径:   http://127.0.0.1:8000/")
    print("💚 健康检查: http://127.0.0.1:8000/health")
    print("")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print("=" * 60)
    print("")
    
    uvicorn.run(
        "src.api.main:app",  # ✅ 从 src 导入
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

# ngrok http 8000 --region=ap