# GoalScope · 足球智能视频分析系统

> 基于 YOLOv8 目标检测与 DeepSeek AI 大模型的足球视频智能分析平台

---

## 项目简介

GoalScope 是一套面向足球训练场景的 AI 视频分析系统，支持对比赛/训练录像进行全自动的球员追踪、队伍识别、能力评估和精彩片段提取。系统采用前后端分离架构，提供可视化 Web 界面，无需专业技术背景即可操作。

### 核心功能

| 功能模块 | 说明 |
|---------|------|
| **视频上传与分析** | 拖拽或点击上传足球视频，YOLOv8 自动检测场内球员，输出标注视频 |
| **队伍自动识别** | 基于 HSV 颜色聚类，自动区分主客队，用不同颜色边框标注 |
| **球员能力画像** | 生成防守/射门/传球/速度/体能五维雷达图及 AI 评估报告 |
| **球衣号码识别** | YOLOv8 Pose 姿态估计精确定位躯干区域，EasyOCR 多策略识别球衣背号 |
| **精彩片段生成** | Farneback 光流分析 + 人群密度检测，自动剪辑比赛精彩时刻 |
| **AI 智能助手** | 接入 DeepSeek 大模型，支持自然语言问答球员技战术数据 |
| **历史记录管理** | 查看历史分析记录，支持搜索/排序/下载 |
| **管理后台** | 管理员专属后台，支持用户管理、数据清理、系统监控 |

---

## 技术栈

### 后端
- **Python 3.10+**
- **FastAPI 0.104** — 高性能异步 Web 框架
- **SQLAlchemy 2.0 + SQLite** — 数据库 ORM
- **YOLOv8n (ultralytics)** — 实时目标检测与追踪
- **YOLOv8n-Pose (ultralytics)** — 姿态估计，用于精确定位球员躯干关键点
- **OpenCV 4.8** — 视频处理与图像分析（光流、颜色聚类、号码裁剪）
- **imageio-ffmpeg** — 视频编解码（内置 ffmpeg，无需单独安装）
- **EasyOCR** — 球衣背号 OCR 识别（多策略 + 多阈值）
- **DeepSeek API (OpenAI 兼容)** — AI 智能对话

### 前端
- **HTML5 / CSS3 / JavaScript** — 零框架依赖
- **Chart.js 3.9** — 雷达图/数据可视化
- **Font Awesome 6.1** — 图标库

---

## 环境要求

| 组件 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.9 | 3.10 / 3.11 |
| pip | 21.0+ | 最新版 |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 3 GB 可用空间 | 6 GB+（含两个模型文件） |
| 操作系统 | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 |
| 浏览器 | Chrome 90+ / Edge 90+ | Chrome 最新版 |

> **注意**：无需单独安装 ffmpeg，系统通过 `imageio-ffmpeg` 自动获取内置版本。

---

## 快速开始

### 第一步：安装依赖

打开命令提示符（cmd）或 PowerShell，进入项目根目录：

```bash
cd backend
pip install -r requirements.txt
```

> 首次安装约需 5～15 分钟（包含 PyTorch、YOLOv8、EasyOCR 等大型库）。
> 如网络较慢，可使用国内镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 第二步：配置环境变量（可选）

将项目根目录的 `.env.example` 复制为 `.env`，并填写所需配置：

```bash
copy .env.example .env
```

主要配置项说明：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（AI 功能必填） | 演示模式 |
| `HOST` | 服务监听地址 | `127.0.0.1` |
| `PORT` | 服务端口 | `9999` |
| `MAX_CONCURRENT_REQUESTS` | AI 最大并发请求数 | `100` |
| `CACHE_MAX_SIZE` | AI 缓存容量 | `500` |

### 第三步：启动服务器

**方式一**：双击项目根目录的 `START.bat`（Windows）

**方式二**：命令行启动

```bash
cd backend
python main.py
```

服务启动成功后，控制台将显示：
```
Application Started Successfully
```

### 第四步：打开浏览器

访问：**http://127.0.0.1:9999**

系统会自动跳转至登录页面。

---

## 登录说明

系统提供三种登录方式：

| 方式 | 账号 | 密码 | 权限 |
|------|------|------|------|
| 管理员账号 | `admin` | `123456` | 全功能 + 管理后台 |
| 普通用户 | `user` | `123456` | 全功能 |
| 游客体验 | 点击"游客体验"按钮 | 无需密码 | 全功能（演示模式） |

> **演示建议**：推荐使用"游客体验"或 `admin/123456` 快速进入。

---

## 功能演示指南

#### 1. 视频上传与 YOLO 分析
1. 登录后进入主页面
2. 将足球视频文件拖入上传区域（支持 mp4/avi/mov 格式，建议视频时长 15 分钟以内）
3. 系统自动启动分析，进度条实时更新
4. 分析完成后（约 5 ~ 10 分钟）自动显示：
   - **标注视频**：不同队伍球员用不同颜色边框标注（主队橙色/客队红色）
   - **球员能力雷达图**：防守/射门/传球/速度/体能五维可视化
   - **AI 评估报告**：针对每名球员的技战术分析
   - **球衣号码**：基于姿态估计精确裁剪躯干，EasyOCR 多策略识别

#### 2. 精彩片段生成
1. 在上传界面勾选"生成精彩视频"选项
2. 系统使用 Farneback 光流算法自动检测运动强度高的精彩时刻，约 15 分钟生成完毕
3. 生成后可在线预览或下载

#### 3. AI 智能助手
1. 点击右上角"AI 智能助手"进入对话界面
2. 选择已分析的视频，直接以自然语言提问，例如：
   - "分析最近上传视频中球员的防守能力"
   - "哪名球员最适合担任前锋？"
   - "综合分析本场比赛两队整体表现"
3. 支持 6 个预设问题快速发送
4. 若已配置 `DEEPSEEK_API_KEY`，将使用真实大模型；否则为演示模式

#### 4. 历史记录
1. 点击右下角悬浮按钮（时钟图标）展开历史面板
2. 支持按时间排序、关键词搜索
3. 可查看历史分析报告和下载精彩视频

#### 5. 管理后台
1. 使用 `admin/123456` 登录后访问管理后台
2. 支持用户管理、视频记录查看、系统数据清理

---

## AI 功能配置（可选）

AI 智能助手默认为演示模式，若需接入真实 DeepSeek 大模型：

1. 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请 API Key
2. 在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

或通过环境变量临时设置：

**Windows CMD：**
```bat
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
cd backend
python main.py
```

**Windows PowerShell：**
```powershell
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
cd backend
python main.py
```

---

## 项目结构

```
GoalScope/
├── frontend/                    # 前端页面
│   ├── index.html               # 主页面（视频上传与分析）
│   ├── login.html               # 用户登录页
│   ├── ai-chat.html             # AI 智能助手
│   ├── admin.html               # 管理后台
│   ├── admin-login.html         # 管理员登录
│   ├── app.js                   # 主应用逻辑（视频上传/分析/可视化）
│   ├── highlight.js             # 精彩视频功能
│   ├── logo-football.svg        # 项目 Logo
│   └── style.css                # 全局样式
│
├── backend/                     # 后端服务
│   ├── main.py                  # FastAPI 主服务（入口）
│   ├── analysis.py              # 视频分析引擎（YOLO + 姿态估计 + 标注）
│   ├── highlight_generator.py   # 精彩视频生成器（光流分析）
│   ├── jersey_number_recognition.py  # 球衣号码识别（EasyOCR 多策略）
│   ├── ai_agent.py              # DeepSeek AI 代理
│   ├── database.py              # 数据库模型（SQLAlchemy）
│   ├── auto_cleaner.py          # 自动清理过期任务
│   ├── create_demo_data.py      # 演示数据生成
│   ├── convert_videos.py        # 视频格式转换工具
│   ├── patch_ultralytics.py     # PyTorch 兼容性补丁
│   ├── start_server.py          # 带编码修复的启动脚本
│   ├── requirements.txt         # Python 依赖清单
│   ├── yolov8n.pt               # YOLOv8n 目标检测预训练模型
│   ├── yolov8n-pose.pt          # YOLOv8n-Pose 姿态估计预训练模型
│   ├── football_demo.db         # SQLite 数据库
│   ├── Dockerfile               # Docker 容器化配置
│   ├── start.sh                 # Linux/macOS 启动脚本
│   └── uploads/                 # 视频上传目录
│       └── exports/             # 标注视频导出目录
│
├── uploads/                     # 根目录上传缓存
├── START.bat                    # Windows 一键启动脚本
├── init.sql                     # 数据库初始化 SQL
├── football_demo.db             # SQLite 数据库（根目录副本）
├── .env.example                 # 环境变量配置示例
└── README.md                    # 本文件
```

---

## 常见问题

**Q: 启动时提示 "ModuleNotFoundError"**
```bash
pip install -r backend/requirements.txt
```

**Q: 分析进度卡在某个百分比不动**

等待 2 分钟后若仍无响应，刷新页面重新上传。首次运行 YOLOv8 时需下载模型权重，可能较慢。

**Q: 标注视频生成后无法播放**

系统已内置 ffmpeg 自动转码为 H.264 格式，若仍无法播放，请更换 Chrome/Edge 浏览器。

**Q: AI 助手无响应**

未配置 `DEEPSEEK_API_KEY` 时，AI 助手工作在演示模式，仅返回预设回答。配置后即可使用完整功能。

**Q: 提示 "YOLO 模型加载失败"**

确认 `backend/yolov8n.pt` 和 `backend/yolov8n-pose.pt` 文件存在，且 `ultralytics` 已正确安装：
```bash
pip install ultralytics
```

**Q: 球衣号码识别为空或不准确**

号码识别依赖视频分辨率与拍摄角度。低分辨率或背部遮挡时识别率会下降，属正常现象。系统已集成多策略 OCR（5 种阈值处理 + 姿态关键点精确裁剪）以提升准确率。

---

## 版本信息

- **版本**：v2.0.0
- **服务端口**：9999
- **数据库**：SQLite（`backend/football_demo.db`）
- **检测模型**：YOLOv8n（轻量版，适合 CPU 推理）
- **姿态模型**：YOLOv8n-Pose（轻量版）
- **作者**：韩欣涛 · 汤宇飞 · 彭勃元
