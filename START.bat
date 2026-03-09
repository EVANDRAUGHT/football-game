@echo off
chcp 65001 >nul
title GoalScope - 足球智能视频分析系统
cd /d "%~dp0backend"

echo ============================================================
echo   GoalScope - 足球智能视频分析系统
echo   Football Intelligent Video Analysis System
echo ============================================================
echo.
echo [INFO] 正在启动服务器，端口: 9999 ...
echo.
echo [INFO] 服务启动后，请在浏览器中访问:
echo        http://127.0.0.1:9999
echo.
echo [INFO] 按 Ctrl+C 可停止服务器
echo ============================================================
echo.

python main.py

pause
