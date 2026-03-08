#!/bin/bash
# 足球分析 AI 系统启动脚本 (Git Bash 兼容)

echo "============================================================"
echo "  Football Analysis AI System"
echo "============================================================"
echo ""

# 检查是否在 backend 目录
if [ ! -f "main.py" ]; then
    echo "[ERROR] Please run this script from the backend directory"
    echo "Usage: cd backend && ./start.sh"
    exit 1
fi

# 检查 API Key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "[WARNING] DEEPSEEK_API_KEY not configured"
    echo "[WARNING] AI chat function will be unavailable"
    echo ""
    echo "To configure, run:"
    echo "  export DEEPSEEK_API_KEY='sk-your-key-here'"
    echo ""
else
    echo "[OK] DEEPSEEK_API_KEY configured"
fi

# 设置编码环境变量
export PYTHONIOENCODING=utf-8

echo ""
echo "[INFO] Starting server..."
echo "[INFO] Access at: http://127.0.0.1:9999"
echo ""
echo "------------------------------------------------------------"
echo ""

# 启动服务器
python start_server.py
