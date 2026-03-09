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
| **精彩片段生成** | 光流分析 + 人群密度检测，自动剪辑比赛精彩时刻 |
| **AI 智能助手** | 接入 DeepSeek 大模型，支持自然语言问答球员技战术数据 |
| **历史记录管理** | 查看历史分析记录，支持搜索/排序/下载 |

---

## 技术栈

### 后端
- **Python 3.10+**
- **FastAPI 0.104** — 高性能异步 Web 框架
- **SQLAlchemy 2.0 + SQLite** — 数据库
- **YOLOv8 (ultralytics)** — 实时目标检测与追踪
- **OpenCV 4.8** — 视频处理与图像分析
- **imageio-ffmpeg** — 视频编解码（内置 ffmpeg，无需单独安装）
- **EasyOCR** — 球衣背号 OCR 识别
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
| 磁盘 | 2 GB 可用空间 | 5 GB+ |
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

### 第二步：启动服务器

**方式一**：双击项目根目录的 `START.bat`

**方式二**：命令行启动

```bash
cd backend
python main.py
```

服务启动成功后，控制台将显示：
```
Application Started Successfully
```

### 第三步：打开浏览器

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
2. 将足球视频文件拖入上传区域（支持 mp4/avi/mov格式，建议视频时长 15 分钟以内）
3. 系统自动启动分析，进度条实时更新
4. 分析完成后（约 5 ~ 10 分钟）自动显示：
   - **标注视频**：不同队伍球员用不同颜色边框标注（主队橙色/客队红色）
   - **球员能力雷达图**：五维能力可视化
   - **AI 评估报告**：针对每名球员的粗略分析

#### 2. 精彩片段生成
1. 在上传界面勾选"生成精彩视频"按钮
2. 系统自动检测运动强度高的精彩时刻，约 15 分钟生成完毕
3. 生成后可在线预览或下载

#### 3. AI 智能助手
1. 点击右上角"AI 智能助手"进入对话界面
2. 可直接提问，例如：
   - "分析最近上传视频中球员的防守能力"
   - "哪名球员最适合担任前锋？"
3. 若已配置 `DEEPSEEK_API_KEY`，将使用真实大模型；否则为演示模式

#### 4. 历史记录
1. 点击右下角悬浮按钮（时钟图标）展开历史面板
2. 支持按时间排序、关键词搜索
3. 可查看历史分析报告和下载精彩视频

---

## AI 功能配置（可选）

AI 智能助手默认为演示模式，若需接入真实 DeepSeek 大模型：

1. 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 申请 API Key
2. 设置环境变量后启动服务器：

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
│   ├── app.js                   # 主应用逻辑
│   ├── highlight.js             # 精彩视频功能
│   └── style.css                # 全局样式
│
├── backend/                     # 后端服务
│   ├── main.py                  # FastAPI 主服务（入口）
│   ├── analysis.py              # 视频分析引擎（YOLO + 标注）
│   ├── highlight_generator.py   # 精彩视频生成器
│   ├── jersey_number_recognition.py  # 球衣号码识别
│   ├── ai_agent.py              # DeepSeek AI 代理
│   ├── database.py              # 数据库模型
│   ├── auto_cleaner.py          # 自动清理任务
│   ├── create_demo_data.py      # 演示数据生成
│   ├── patch_ultralytics.py     # PyTorch 兼容性补丁
│   ├── start_server.py          # 带编码修复的启动脚本
│   ├── requirements.txt         # Python 依赖清单
│   ├── yolov8n.pt               # YOLOv8 预训练模型
│   ├── football_demo.db         # SQLite 数据库
│   └── uploads/                 # 视频上传目录
│       └── exports/             # 标注视频导出目录
│
├── START.bat                    # Windows 一键启动脚本
├── init.sql                     # 数据库初始化 SQL
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

确认 `backend/yolov8n.pt` 文件存在，且 `ultralytics` 已正确安装：
```bash
pip install ultralytics==8.0.206
```

---

## 版本信息

- **版本**：v2.0.0
- **服务端口**：9999
- **数据库**：SQLite（`backend/football_demo.db`）
- **模型**：YOLOv8n（轻量版，适合 CPU 推理）
