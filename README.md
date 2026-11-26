# GoalScope · 足球视频分析系统

GoalScope 面向足球比赛和训练录像，支持球员与足球跟踪、动态战术可视化和精彩集锦生成。

## 主要功能

- **两阶段球员分析**。第一阶段跟踪视频中的球员并缓存结果，第二阶段根据用户选定的球员继续分析。
- **动态战术板**。随视频播放查看所选球员的位置和移动轨迹。
- **精彩集锦**。通过多模态事件检测识别候选片段，去重后生成集锦视频。

## 本地运行

1. 创建 Python 3.10 环境，安装适合当前平台的 PyTorch 和 torchvision。
2. 在项目根目录执行 `python -m pip install -r backend/requirements.txt`。
3. 使用自有或获授权的模型权重。球员检测权重放在 `backend/football_player.pt`，也可在项目根目录放置 `yolov8n.pt`。专用足球检测权重 `football-ball-detection.pt` 和可选的 OSNet 权重均放在项目根目录。
4. Windows 下运行 `START.bat`，或进入 `backend` 目录执行 `python main.py`。
5. 打开 `http://127.0.0.1:9999`，注册账号或选择游客体验。

系统面向本地运行。SQLite 数据库、上传的视频和分析缓存默认保存在 `GoalScope_data/` 下。
