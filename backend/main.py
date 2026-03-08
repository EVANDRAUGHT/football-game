from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import os
import sys
import uuid
import shutil
import json
import asyncio
import datetime
import database
from database import VideoModel, AthleteProfile, AIChatSession, AIVideoCache, UserModel, get_db, SessionLocal
import analysis
from ai_agent import get_ai_manager, DeepSeekAgent
from auto_cleaner import get_cleaner, auto_clean_on_startup, periodic_clean_task

# Windows 编码修复：强制使用 UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

app = FastAPI()

# 配置 CORS（必须在路由之前）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化数据库
database.init_db()

# 初始化 AI 管理器
ai_manager = get_ai_manager()

# 后台任务存储
_background_tasks = {}

# 🔥 新增：任务锁机制，防止重复启动
highlight_task_locks = set()  # 存储正在执行的 video_id

# 启动时异步初始化 AI Agent 和自动清理
@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    print("\n" + "="*60)
    print("🚀 Application Starting...")
    print("="*60)
    
    # 1. 启动时自动清理数据库
    # 🔥 临时禁用：避免删除正在使用的视频记录
    print("\n[STARTUP] Step 1: Database cleanup")
    print("[INFO] Startup cleanup DISABLED (to preserve video records)")
    # await auto_clean_on_startup()  # 注释掉

    # 1b. 初始化内置账号（admin/demo）到 users 表
    try:
        _ensure_default_users()
        print("[INFO] [OK] Default users initialized")
    except Exception as e:
        print(f"[WARNING] Default users init failed: {e}")
    
    # 2. 初始化 AI 代理
    print("\n[STARTUP] Step 2: Initialize AI agent")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if api_key:
        try:
            await ai_manager.initialize(api_key)
            print("[INFO] [OK] DeepSeek AI agent initialized successfully")
        except Exception as e:
            print(f"[WARNING] AI agent initialization failed: {e}")
            print("[WARNING] AI chat function unavailable, other functions normal")
    else:
        print("[WARNING] DEEPSEEK_API_KEY not configured, AI chat unavailable")
    
    # 3. 启动后台定期清理任务
    print("\n[STARTUP] Step 3: Start periodic cleanup task")
    _background_tasks['cleaner'] = asyncio.create_task(periodic_clean_task())
    print("[INFO] Periodic cleanup enabled (interval: 60 minutes)")  # 🔑 修改: 延长到60分钟
    
    print("\n" + "="*60)
    print("✅ Application Started Successfully")
    print("="*60 + "\n")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    print("\n[SHUTDOWN] Stopping background tasks...")
    
    # 取消后台任务
    if 'cleaner' in _background_tasks:
        _background_tasks['cleaner'].cancel()
        try:
            await _background_tasks['cleaner']
        except asyncio.CancelledError:
            print("[INFO] Cleanup task stopped")
    
    print("[SHUTDOWN] Application closed")

# ===== Pydantic 数据模型 =====
class ChatRequest(BaseModel):
    message: str
    session_id: str
    video_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    answer: str
    response_time: float
    tokens_used: int
    confidence: float
    session_id: str
    from_cache: bool = False

# 目录配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # 项目根目录
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
EXPORT_DIR = os.path.join(UPLOAD_DIR, "exports")
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)

# 挂载静态文件目录（前端页面）
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
app.mount("/exports", StaticFiles(directory=EXPORT_DIR), name="exports")

print(f"[INFO] 导出目录: {EXPORT_DIR}")
print(f"[INFO] 导出目录存在: {os.path.exists(EXPORT_DIR)}")
print(f"[INFO] 导出目录内容: {os.listdir(EXPORT_DIR) if os.path.exists(EXPORT_DIR) else '无'}")

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    owner: Optional[str] = None,   # 通过查询参数传递当前登录用户名
):
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1]
    saved_filename = f"{file_id}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    # 使用异步迭代写入，防止 1GB 大文件保存时阻塞 FastAPI 主线程
    async def save_file():
        with open(file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # 1MB chunk
                buffer.write(content)

    await save_file()
    
    # 存入数据库（绑定 owner，游客统一记为 "guest"）
    new_video = VideoModel(
        video_uuid=file_id,
        filename=saved_filename,
        owner=owner or "guest",
    )
    db.add(new_video)
    db.commit()
    
    # 在后台启动分析任务
    abs_file_path = os.path.abspath(file_path)
    # 确保清除旧的进度，防止重新上传时状态混淆
    analysis.progress_store[file_id] = 0
    background_tasks.add_task(analysis.process_video, abs_file_path, file_id)


    
    return {"status": "success", "video_id": file_id, "filename": file.filename}

@app.get("/results/{video_id}")
async def get_results(video_id: str, db: Session = Depends(get_db)):
    # 尝试从数据库获取结果
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    
    # 🔑 修复4: 增加健壮性检查
    if not video:
        # 检查内存中是否有正在进行的分析
        current_progress = analysis.progress_store.get(video_id, None)
        
        if current_progress is None:
            return {
                "error": "task_not_found",
                "message": "分析任务不存在或已被清理",
                "progress": -1,
                "detailed_analysis": {
                    "error": True,
                    "message": "请重新上传视频开始新的分析任务"
                }
            }
        else:
            return {
                "video_id": video_id,
                "progress": current_progress,
                "decision_summary": f"正在分析... (当前进度: {current_progress}%)",
                "detailed_analysis": None  # 明确返回None
            }
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    
    if profile:
        # 验证数据完整性
        if not profile.detailed_analysis:
            return {
                "video_id": video_id,
                "progress": -1,
                "error": "data_incomplete",
                "message": "分析结果数据不完整",
                "detailed_analysis": {
                    "error": True,
                    "message": "数据库记录存在但内容不完整，请重新分析"
                }
            }
        
        # 验证athletes字段
        athletes = profile.detailed_analysis.get('athletes', [])
        if not isinstance(athletes, list) or len(athletes) == 0:
            return {
                "video_id": video_id,
                "progress": -1,
                "error": "invalid_data",
                "message": "分析结果中没有球员数据",
                "detailed_analysis": profile.detailed_analysis
            }
        
        return {
            "video_id": video_id,
            "posture_score": profile.overall_score,
            "decision_summary": profile.decision_summary,
            "detailed_analysis": profile.detailed_analysis,
            "progress": 100,
            "key_frames": []
        }
    
    else:
        # video存在但profile不存在 → 正在分析中
        current_progress = analysis.progress_store.get(video_id, 0)
        
        # 特殊处理：progress=100但数据库无记录（竞态条件）
        if current_progress == 100:
            import time
            time.sleep(0.5)  # 等待500ms让数据库提交完成
            # 重新查询
            profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
            if profile:
                return {
                    "video_id": video_id,
                    "posture_score": profile.overall_score,
                    "decision_summary": profile.decision_summary,
                    "detailed_analysis": profile.detailed_analysis,
                    "progress": 100,
                    "key_frames": []
                }
        
        status_msg = "正在扫描全场主力球员并生成 YOLO 标注视频，请稍候..."
        if current_progress == -1:
            status_msg = "分析过程中发生错误，请检查视频格式或后端日志"
        
        return {
            "video_id": video_id,
            "posture_score": 0,
            "progress": current_progress,
            "decision_summary": f"{status_msg} (当前进度: {current_progress}%)",
            "detailed_analysis": None,  # 明确返回None
            "key_frames": []
        }

# 在main.py中添加这个函数
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return HTMLResponse('''<!DOCTYPE html>
<html>
<head>
    <title>足球分析AI系统 - API文档</title>
    <!-- 使用国内可访问的CDN -->
    <link rel="stylesheet" href="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui.css">
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
        .swagger-ui .topbar { display: none; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui-bundle.js"></script>
    <script src="https://cdn.staticfile.org/swagger-ui/4.15.5/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: '/openapi.json',
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout",
                displayRequestDuration: true
            });
        }
    </script>
</body>
</html>''')

@app.api_route("/download/{video_id}", methods=["GET", "HEAD", "OPTIONS"])
async def download_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """
    专用下载端点，设置正确的响应头确保视频可下载和播放
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        return {"error": "Video not found"}
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    if not profile or not profile.detailed_analysis.get("export_url"):
        return {"error": "Annotated video not ready"}
    
    # 构建文件路径
    filename = profile.detailed_analysis["export_url"].replace("/exports/", "")
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}
    
    # 返回文件响应，设置正确的 MIME 类型和下载头
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"足球分析_{video_id[:8]}.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="football_analysis_{video_id[:8]}.mp4"',
            "Cache-Control": "no-cache",
            "Accept-Ranges": "bytes",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.api_route("/preview/{video_id}", methods=["GET", "HEAD", "OPTIONS"])
async def preview_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """
    在线预览端点，支持 Range 请求的视频流播放
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
    if not video:
        return {"error": "Video not found"}
    
    profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
    if not profile or not profile.detailed_analysis.get("export_url"):
        return {"error": "Annotated video not ready"}
    
    # 构建文件路径
    filename = profile.detailed_analysis["export_url"].replace("/exports/", "")
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found", "path": file_path}
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    
    # 处理 HEAD 请求（只返回头信息，不返回文件内容）
    if request.method == "HEAD":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    # 处理 Range 请求（视频播放必需）
    range_header = request.headers.get("range")
    
    if range_header:
        # 解析 Range 头: "bytes=start-end"
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
        
        # 限制范围
        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        # 读取指定范围的数据
        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)  # 1MB chunks
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,  # Partial Content
            headers=headers
        )
    
    # 无 Range 请求，返回完整文件
    def iter_full_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):  # 1MB chunks
                yield chunk
    
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*"
    }
    
    return StreamingResponse(
        iter_full_file(),
        headers=headers
    )


# 根路由：重定向到前端主页面
@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/index.html")

# 添加健康检查
@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "football-ai", "timestamp": datetime.datetime.now().isoformat()}


# ===== 自定义静态文件端点（支持 CORS + Range 请求）=====
@app.api_route("/exports/{filename:path}", methods=["GET", "HEAD", "OPTIONS"])
async def serve_export_file(filename: str, request: Request):
    """
    自定义静态文件服务，支持 CORS 和 Range 请求
    """
    # 处理 OPTIONS 预检请求
    if request.method == "OPTIONS":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    file_path = os.path.join(EXPORT_DIR, filename)
    
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    
    # 处理 HEAD 请求
    if request.method == "HEAD":
        from fastapi.responses import Response
        return Response(
            status_code=200,
            headers={
                "Content-Type": "video/mp4",
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache"
            }
        )
    
    # 处理 Range 请求
    range_header = request.headers.get("range")
    
    if range_header:
        # 解析 Range 头
        range_match = range_header.replace("bytes=", "").split("-")
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if len(range_match) > 1 and range_match[1] else file_size - 1
        
        start = max(0, start)
        end = min(end, file_size - 1)
        content_length = end - start + 1
        
        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(1024 * 1024, remaining)
                    data = f.read(chunk_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
        
        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers=headers
        )
    
    # 完整文件响应
    def iter_full_file():
        with open(file_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
    
    headers = {
        "Content-Length": str(file_size),
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-cache",
        "Access-Control-Allow-Origin": "*"
    }
    
    return StreamingResponse(
        iter_full_file(),
        headers=headers
    )

print(f"[INFO] 静态文件路由已配置: /exports -> {EXPORT_DIR}")


# ===== 缓存同步辅助函数 =====
async def sync_ai_cache_with_db(db: Session):
    """
    同步AI缓存与数据库状态
    每次AI请求前自动调用，确保缓存数据一致性

    同步内容：
      1. 内存中的 video_knowledge_base —— 与 videos 表比对，移除幽灵条目
      2. DB 中的 ai_video_cache      —— 标记 is_ghost=True（video_uuid 不在 videos 表）
      3. DB 中的 ai_chat_sessions    —— 标记 is_ghost=True（video_uuid 不在 videos 表）

    返回：同步结果字典
    """
    try:
        # 获取数据库中所有视频的UUID
        db_videos = db.query(VideoModel.video_uuid).all()
        db_video_ids = {video.video_uuid for video in db_videos}

        # ── 1. 同步内存缓存 ───────────────────────────────────────────────────
        sync_result = {"removed_count": 0, "removed_videos": [], "cleaned_sessions": 0,
                       "status": "skipped", "sync_time": ""}
        if ai_manager.agent is not None:
            sync_result = await ai_manager.sync_cache_with_database(db_video_ids)

        # ── 2. 标记 DB ai_video_cache 中的幽灵记录 ──────────────────────────
        try:
            ghost_caches = db.query(AIVideoCache).filter(
                AIVideoCache.is_ghost == False  # noqa: E712
            ).all()
            ghost_cache_count = 0
            for cache in ghost_caches:
                if cache.video_uuid not in db_video_ids:
                    cache.is_ghost = True
                    ghost_cache_count += 1
            if ghost_cache_count:
                db.commit()
                logger.info(f"[CACHE_SYNC] Marked {ghost_cache_count} ghost video caches in DB")
        except Exception as e:
            db.rollback()
            logger.warning(f"[CACHE_SYNC] Failed to mark ghost video caches: {e}")

        # ── 3. 标记 DB ai_chat_sessions 中的幽灵消息 ────────────────────────
        try:
            ghost_sessions = db.query(AIChatSession).filter(
                AIChatSession.video_uuid != None,  # noqa: E711
                AIChatSession.is_ghost == False    # noqa: E712
            ).all()
            ghost_session_count = 0
            for session in ghost_sessions:
                if session.video_uuid not in db_video_ids:
                    session.is_ghost = True
                    ghost_session_count += 1
            if ghost_session_count:
                db.commit()
                logger.info(f"[CACHE_SYNC] Marked {ghost_session_count} ghost chat messages in DB")
        except Exception as e:
            db.rollback()
            logger.warning(f"[CACHE_SYNC] Failed to mark ghost chat sessions: {e}")

        return sync_result

    except Exception as e:
        print(f"[WARNING] Cache sync failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "removed_count": 0
        }


# ===== AI 缓存管理端点 =====
@app.post("/ai/cache/sync")
async def sync_cache(db: Session = Depends(get_db)):
    """
    手动触发缓存同步
    
    功能：
    1. 检查数据库中的视频列表
    2. 与AI缓存进行比对
    3. 删除已不存在的视频缓存
    4. 返回同步结果
    
    响应：
    {
        "status": "success",
        "removed_count": 3,
        "cached_count": 12,
        "removed_videos": ["uuid1", "uuid2", "uuid3"],
        "cleaned_sessions": 2,
        "sync_time": "2026-01-26T20:30:00"
    }
    """
    try:
        sync_result = await sync_ai_cache_with_db(db)
        return {
            "status": "success",
            **sync_result
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] Cache sync endpoint failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ai/cache/stats")
async def get_cache_stats():
    """
    获取缓存统计信息
    
    响应：
    {
        "total_cached_videos": 15,
        "total_sessions": 8,
        "cache_size_mb": 2.5,
        "video_ids": ["uuid1", "uuid2", ...],
        "response_cache_size": 120,
        "response_cache_usage_percent": 24.0
    }
    """
    if ai_manager.agent is None:
        return {
            "status": "unavailable",
            "message": "AI agent not initialized"
        }
    
    try:
        stats = ai_manager.get_cache_stats()
        return {
            "status": "success",
            **stats
        }
    except Exception as e:
        print(f"[ERROR] Get cache stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ai/cache/video/{video_id}")
async def clear_video_cache(video_id: str):
    """
    清除指定视频的缓存
    
    参数：
        video_id: 视频UUID
    
    响应：
    {
        "status": "success",
        "video_cache_cleared": true,
        "response_cache_cleared": 5,
        "video_id": "abc-def-ghi"
    }
    """
    if ai_manager.agent is None:
        return {
            "status": "unavailable",
            "message": "AI agent not initialized"
        }
    
    try:
        result = ai_manager.clear_video_cache(video_id)
        return result
    except Exception as e:
        print(f"[ERROR] Clear video cache failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== 数据库自动清理端点 =====
@app.post("/api/database/clean")
async def clean_database(
    db: Session = Depends(get_db),
    clean_type: str = "full"  # full, duplicates, orphaned, failed
):
    """
    手动触发数据库清理
    
    参数：
        clean_type: 清理类型
            - "full": 完整清理（默认）
            - "duplicates": 只清理重复记录
            - "orphaned": 只清理孤立视频
            - "failed": 只清理失败记录
    
    响应：
    {
        "status": "success",
        "total_removed": 5,
        "details": {
            "duplicates": {"removed_count": 2, ...},
            "orphaned": {"removed_count": 1, ...},
            "failed": {"removed_count": 2, ...}
        }
    }
    """
    try:
        cleaner = get_cleaner()
        
        if clean_type == "duplicates":
            result = cleaner.clean_duplicates(db)
        elif clean_type == "orphaned":
            result = cleaner.clean_orphaned_videos(db)
        elif clean_type == "failed":
            result = cleaner.clean_failed_videos(db)
        else:  # full
            result = cleaner.full_clean(db)
        
        return result
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Database cleanup failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/database/status")
async def get_database_status(db: Session = Depends(get_db)):
    """
    获取数据库状态信息
    
    响应：
    {
        "total_videos": 15,
        "total_profiles": 15,
        "duplicates": 0,
        "orphaned": 0,
        "failed": 0,
        "health": "good"
    }
    """
    try:
        from collections import defaultdict
        
        # 统计总数
        total_videos = db.query(VideoModel).count()
        total_profiles = db.query(AthleteProfile).count()
        
        # 检查重复
        all_videos = db.query(VideoModel).all()
        uuid_count = defaultdict(int)
        for video in all_videos:
            uuid_count[video.video_uuid] += 1
        
        duplicate_count = sum(1 for count in uuid_count.values() if count > 1)
        
        # 检查孤立
        orphaned_count = db.query(VideoModel).outerjoin(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).filter(AthleteProfile.id == None).count()
        
        # 检查失败
        failed_count = 0
        videos_with_profiles = db.query(VideoModel).join(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).all()
        
        for video in videos_with_profiles:
            profile = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == video.id
            ).first()
            
            if profile and profile.detailed_analysis:
                detailed = profile.detailed_analysis
                if isinstance(detailed, dict) and detailed.get('error'):
                    failed_count += 1
        
        # 评估健康状态
        issues = duplicate_count + orphaned_count + failed_count
        if issues == 0:
            health = "excellent"
        elif issues <= 3:
            health = "good"
        elif issues <= 10:
            health = "fair"
        else:
            health = "poor"
        
        # 检测幽灵档案（AthleteProfile 无对应 Video 记录）
        ghost_profiles = db.query(AthleteProfile).outerjoin(
            VideoModel, AthleteProfile.video_id == VideoModel.id
        ).filter(VideoModel.id == None).count()

        return {
            "status": "success",
            "total_videos": total_videos,
            "total_profiles": total_profiles,
            "duplicates": duplicate_count,
            "orphaned": orphaned_count,
            "failed": failed_count,
            "ghost_profiles": ghost_profiles,
            "issues_total": issues + ghost_profiles,
            "health": health,
            "auto_clean_enabled": True,
            "auto_clean_interval": "10 minutes"
        }

    except Exception as e:
        import traceback
        print(f"[ERROR] Get database status failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/database/clean-ghosts")
async def clean_ghost_records(db: Session = Depends(get_db)):
    """
    清理所有幽灵数据记录

    幽灵数据定义：
      1. 幽灵档案（Ghost Profiles）  — AthleteProfile 记录对应的 video_id
                                      在 videos 表中已不存在
      2. 孤立视频（Orphaned Videos） — VideoModel 无任何 AthleteProfile，
                                      且上传文件也不存在于磁盘
      3. 文件孤儿（File Orphans）    — uploads/ 目录中存在 mp4 文件但
                                      数据库中无对应 video_uuid 记录

    返回每类幽灵数据的清理数量，不影响正常分析记录。
    """
    report = {
        "ghost_profiles_removed": 0,
        "orphaned_videos_removed": 0,
        "file_orphans_removed": 0,
        "details": [],
    }

    try:
        # ── 1. 清理幽灵档案 ───────────────────────────────────────────────────
        ghost_profiles = db.query(AthleteProfile).outerjoin(
            VideoModel, AthleteProfile.video_id == VideoModel.id
        ).filter(VideoModel.id == None).all()

        for gp in ghost_profiles:
            report["details"].append({
                "type": "ghost_profile",
                "athlete_profile_id": gp.id,
                "video_id_ref": gp.video_id,
            })
            db.delete(gp)

        report["ghost_profiles_removed"] = len(ghost_profiles)

        # ── 2. 清理孤立视频（无档案 + 文件不存在） ────────────────────────────
        orphaned_videos = db.query(VideoModel).outerjoin(
            AthleteProfile, VideoModel.id == AthleteProfile.video_id
        ).filter(AthleteProfile.id == None).all()

        orphan_removed = 0
        for v in orphaned_videos:
            # 排除正在分析中的视频
            if str(v.id) in highlight_task_locks or v.video_uuid in highlight_task_locks:
                continue
            # 只清理磁盘文件也不存在的记录（文件存在说明可能正在分析）
            video_file = os.path.join(UPLOAD_DIR, f"{v.video_uuid}.mp4")
            if not os.path.exists(video_file):
                report["details"].append({
                    "type": "orphaned_video",
                    "video_id": v.id,
                    "video_uuid": v.video_uuid,
                    "filename": v.filename,
                })
                db.delete(v)
                orphan_removed += 1

        report["orphaned_videos_removed"] = orphan_removed

        # ── 3. 清理文件孤儿（磁盘文件无数据库记录） ──────────────────────────
        if os.path.exists(UPLOAD_DIR):
            # 获取数据库中所有 video_uuid
            db_uuids = {row.video_uuid for row in db.query(VideoModel.video_uuid).all()}

            file_orphans_removed = 0
            for fname in os.listdir(UPLOAD_DIR):
                if not fname.endswith(".mp4"):
                    continue
                # _highlights.mp4 是生成的精彩视频，不是源文件
                if fname.endswith("_highlights.mp4"):
                    continue
                uuid_part = fname.replace(".mp4", "")
                if uuid_part not in db_uuids:
                    file_path = os.path.join(UPLOAD_DIR, fname)
                    try:
                        os.remove(file_path)
                        report["details"].append({
                            "type": "file_orphan",
                            "filename": fname,
                            "path": file_path,
                        })
                        file_orphans_removed += 1
                    except OSError as e:
                        report["details"].append({
                            "type": "file_orphan_error",
                            "filename": fname,
                            "error": str(e),
                        })

            report["file_orphans_removed"] = file_orphans_removed

        db.commit()

        total = (
            report["ghost_profiles_removed"]
            + report["orphaned_videos_removed"]
            + report["file_orphans_removed"]
        )
        report["total_removed"] = total
        report["status"] = "success"
        report["message"] = (
            f"清理完成：幽灵档案 {report['ghost_profiles_removed']} 条，"
            f"孤立视频 {report['orphaned_videos_removed']} 条，"
            f"文件孤儿 {report['file_orphans_removed']} 个"
        )

        print(f"[CLEAN_GHOSTS] {report['message']}")
        return report

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ===== AI 智能对话端点 =====
@app.post("/ai/chat")
async def ai_chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    AI 智能对话接口（带自动缓存同步）
    
    请求体：
    {
        "message": "3号球员的传球能力如何？",
        "session_id": "user_123_session_1",
        "video_id": "abc-def-ghi",
        "stream": false
    }
    
    响应：
    {
        "answer": "根据分析数据，3号球员...",
        "response_time": 0.85,
        "tokens_used": 150,
        "confidence": 0.96,
        "session_id": "user_123_session_1"
    }
    """
    if ai_manager.agent is None:
        raise HTTPException(
            status_code=503, 
            detail="AI 代理未初始化，请检查 DEEPSEEK_API_KEY 配置"
        )
    
    try:
        # 🔄 自动缓存同步：每次对话前确保缓存与数据库一致
        sync_result = await sync_ai_cache_with_db(db)
        removed_count = sync_result.get("removed_count", 0)
        if isinstance(removed_count, int) and removed_count > 0:
            print(f"[CACHE_SYNC] Auto-synced before chat: removed {removed_count} orphaned videos")
        
        if request.stream:
            # 流式输出（SSE）
            result = await ai_manager.agent.chat(
                user_message=request.message,
                session_id=request.session_id,
                video_id=request.video_id,
                stream=True
            )
            
            async def event_generator():
                async for chunk in result["stream"]:
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            # 标准响应
            response = await ai_manager.submit_request(
                user_message=request.message,
                session_id=request.session_id,
                video_id=request.video_id
            )
            
            if "error" in response:
                raise HTTPException(status_code=500, detail=response["error"])
            
            return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/analyze-video")
async def ai_analyze_video(video_id: str, db: Session = Depends(get_db)):
    """
    将视频分析结果转换为 AI 可理解的语义知识库
    在视频分析完成后自动调用，或手动触发
    """
    if ai_manager.agent is None:
        raise HTTPException(status_code=503, detail="AI 代理未初始化")
    
    try:
        # 🔄 分析前先同步缓存
        await sync_ai_cache_with_db(db)
        
        # 从数据库获取分析结果
        video = db.query(VideoModel).filter(VideoModel.video_uuid == video_id).first()
        if not video:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        profile = db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).first()
        if not profile:
            raise HTTPException(status_code=404, detail="视频分析尚未完成")
        
        # 构建分析数据
        video_analysis = {
            "video_id": video_id,
            "detailed_analysis": profile.detailed_analysis
        }
        
        # 转换为语义知识
        semantic_info = await ai_manager.agent.analyze_video_content(video_analysis)
        
        return {
            "status": "success",
            "video_id": video_id,
            "semantic_info": semantic_info,
            "message": "视频内容已成功转换为 AI 知识库"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] AI analyze video failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/ai/session/{session_id}/history")
async def get_session_history(session_id: str):
    """获取指定会话的对话历史"""
    if ai_manager.agent is None:
        raise HTTPException(status_code=503, detail="AI 代理未初始化")
    
    history = ai_manager.agent.get_session_history(session_id)
    return {
        "session_id": session_id,
        "message_count": len(history),
        "history": history
    }


@app.delete("/ai/session/{session_id}")
async def clear_session(session_id: str):
    """清空指定会话的历史记录"""
    if ai_manager.agent is None:
        raise HTTPException(status_code=503, detail="AI 代理未初始化")
    
    ai_manager.agent.clear_session(session_id)
    return {
        "status": "success",
        "message": f"会话 {session_id} 已清空"
    }


@app.get("/ai/metrics")
async def get_ai_metrics():
    """获取 AI 系统性能指标"""
    if ai_manager.agent is None:
        return {
            "status": "unavailable",
            "message": "AI 代理未初始化"
        }
    
    stats = ai_manager.get_stats()
    return {
        "status": "operational",
        "metrics": stats,
        "sla": {
            "target_response_time": 1.0,
            "current_response_time": stats["avg_response_time"],
            "target_success_rate": 0.95,
            "current_success_rate": stats["success_rate"],
            "meets_sla": stats["avg_response_time"] < 1.0 and stats["success_rate"] > 0.95
        }
    }


# ===== AI 幽灵数据管理端点 =====

@app.get("/ai/ghosts/stats")
async def get_ai_ghost_stats(db: Session = Depends(get_db)):
    """
    统计 AI 相关的幽灵数据数量

    幽灵数据定义：
      - 幽灵视频缓存：ai_video_cache 中 video_uuid 在 videos 表已不存在
      - 幽灵对话消息：ai_chat_sessions 中 video_uuid 在 videos 表已不存在
      - 内存孤立缓存：video_knowledge_base 中存在但 DB 无对应有效视频

    返回：
    {
        "ghost_video_caches":    3,
        "ghost_chat_messages":  15,
        "memory_orphan_caches":  2,
        "total_valid_caches":   12,
        "total_chat_messages":  80
    }
    """
    try:
        db_video_ids = {v.video_uuid for v in db.query(VideoModel.video_uuid).all()}

        ghost_video_caches  = db.query(AIVideoCache).filter(AIVideoCache.is_ghost == True).count()   # noqa: E712
        ghost_chat_messages = db.query(AIChatSession).filter(AIChatSession.is_ghost == True).count()  # noqa: E712
        total_valid_caches  = db.query(AIVideoCache).filter(AIVideoCache.is_ghost == False).count()  # noqa: E712
        total_chat_messages = db.query(AIChatSession).count()

        memory_orphan_caches = 0
        if ai_manager.agent is not None:
            memory_orphan_caches = sum(
                1 for vid in ai_manager.agent.video_knowledge_base
                if vid not in db_video_ids
            )

        return {
            "status": "success",
            "ghost_video_caches":   ghost_video_caches,
            "ghost_chat_messages":  ghost_chat_messages,
            "memory_orphan_caches": memory_orphan_caches,
            "total_valid_caches":   total_valid_caches,
            "total_chat_messages":  total_chat_messages,
            "total_ghosts":         ghost_video_caches + ghost_chat_messages + memory_orphan_caches,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ai/ghosts/clean")
async def clean_ai_ghost_data(
    purge: bool = False,
    db: Session = Depends(get_db),
):
    """
    彻底清理 AI 智能助手页面的所有幽灵数据

    参数：
      purge=false（默认）：标记幽灵数据（is_ghost=True），不物理删除
      purge=true         ：永久删除所有幽灵记录（不可恢复）

    清理范围：
      1. 内存中的孤立视频知识库缓存（video_knowledge_base）
      2. 内存中的孤立响应缓存（response_cache）
      3. DB ai_video_cache  中幽灵记录
      4. DB ai_chat_sessions 中幽灵消息
      5. DB ai_video_cache  中 video_uuid 不在 videos 表的未标记记录（补充同步）

    返回：
    {
        "status": "success",
        "memory_caches_removed":    2,
        "memory_sessions_cleaned":  1,
        "db_ghost_caches_removed":  3,
        "db_ghost_messages_removed":15,
        "total_removed":           21,
        "purge_mode":            false
    }
    """
    report = {
        "status": "success",
        "purge_mode": purge,
        "memory_caches_removed": 0,
        "memory_sessions_cleaned": 0,
        "db_ghost_caches_removed": 0,
        "db_ghost_messages_removed": 0,
        "total_removed": 0,
        "details": [],
    }

    try:
        # ── 步骤1：先执行全量缓存同步，标记所有幽灵记录 ─────────────────────
        await sync_ai_cache_with_db(db)

        # ── 步骤2：清理内存幽灵数据 ──────────────────────────────────────────
        if ai_manager.agent is not None:
            db_video_ids = {v.video_uuid for v in db.query(VideoModel.video_uuid).all()}

            # 2a. 清理 video_knowledge_base
            orphan_vids = [
                vid for vid in list(ai_manager.agent.video_knowledge_base.keys())
                if vid not in db_video_ids
            ]
            for vid in orphan_vids:
                del ai_manager.agent.video_knowledge_base[vid]
                report["details"].append({"type": "memory_video_cache", "video_uuid": vid})
            report["memory_caches_removed"] = len(orphan_vids)

            # 2b. 清理 response_cache（包含孤立视频ID的响应）
            orphan_set = set(orphan_vids)
            removed_resp = 0
            for cache_key in list(ai_manager.response_cache.keys()):
                resp_str = json.dumps(ai_manager.response_cache.get(cache_key, {}))
                if any(vid in resp_str for vid in orphan_set):
                    del ai_manager.response_cache[cache_key]
                    removed_resp += 1
            report["memory_sessions_cleaned"] = removed_resp

        # ── 步骤3：清理 DB 幽灵记录 ──────────────────────────────────────────
        try:
            if purge:
                # 物理删除
                n_caches = db.query(AIVideoCache).filter(
                    AIVideoCache.is_ghost == True  # noqa: E712
                ).delete(synchronize_session=False)
                n_msgs = db.query(AIChatSession).filter(
                    AIChatSession.is_ghost == True  # noqa: E712
                ).delete(synchronize_session=False)
            else:
                # 仅计数（标记已在 sync_ai_cache_with_db 中完成）
                n_caches = db.query(AIVideoCache).filter(
                    AIVideoCache.is_ghost == True  # noqa: E712
                ).count()
                n_msgs = db.query(AIChatSession).filter(
                    AIChatSession.is_ghost == True  # noqa: E712
                ).count()

            db.commit()
            report["db_ghost_caches_removed"] = n_caches
            report["db_ghost_messages_removed"] = n_msgs

        except Exception as e:
            db.rollback()
            raise e

        report["total_removed"] = (
            report["memory_caches_removed"]
            + report["memory_sessions_cleaned"]
            + report["db_ghost_caches_removed"]
            + report["db_ghost_messages_removed"]
        )

        action = "已永久删除" if purge else "已标记为幽灵数据"
        report["message"] = (
            f"清理完成（{action}）："
            f"内存视频缓存 {report['memory_caches_removed']} 条，"
            f"内存响应缓存 {report['memory_sessions_cleaned']} 条，"
            f"DB视频缓存 {report['db_ghost_caches_removed']} 条，"
            f"DB对话消息 {report['db_ghost_messages_removed']} 条"
        )
        print(f"[GHOST_CLEAN] {report['message']}")
        return report

    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/videos")
async def get_analyzed_videos(db: Session = Depends(get_db), owner: Optional[str] = None):
    """
    获取所有已完成分析的视频列表
    返回格式：[{id: video_uuid, name: filename, analyzed_at: timestamp}]
    
    参数：
      owner: 登录用户名（不传则返回全部，传入则只返回该用户的视频；
             admin 用户返回全部）
    
    🔄 自动功能：
    1. 查询数据库中的有效视频
    2. 自动同步AI缓存（清理已删除视频的缓存）
    3. 去重（同一个video_uuid只保留最新记录）
    """
    try:
        print("[INFO] 正在查询已分析的视频...")
        
        # 🔄 步骤1: 自动清理AI缓存（确保缓存与数据库一致）
        try:
            sync_result = await sync_ai_cache_with_db(db)
            removed_count = sync_result.get("removed_count", 0)
            if isinstance(removed_count, int) and removed_count > 0:
                print(f"[CACHE_SYNC] Auto-cleaned {removed_count} orphaned video caches")
        except Exception as cache_error:
            print(f"[WARNING] Cache sync failed (non-critical): {cache_error}")
        
        # 步骤2: 查询所有有关联 AthleteProfile 的视频（即已完成分析的）
        query = db.query(VideoModel).join(
            AthleteProfile,
            VideoModel.id == AthleteProfile.video_id
        )

        # 按 owner 过滤：admin 看全部，其他用户只看自己的
        if owner and owner != "admin":
            query = query.filter(VideoModel.owner == owner)

        videos_with_analysis = query.all()
        
        print(f"[INFO] 数据库中找到 {len(videos_with_analysis)} 个已分析的视频")
        
        # 步骤3: 去重处理（使用字典确保每个video_uuid只保留一条）
        video_dict = {}
        
        for video in videos_with_analysis:
            # 获取关联的分析数据
            profile = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == video.id
            ).first()
            
            # 检查是否分析失败
            if profile and profile.detailed_analysis:
                detailed = profile.detailed_analysis
                # 跳过分析失败的视频
                if isinstance(detailed, dict) and detailed.get('error'):
                    print(f"[INFO] Skip failed video: {video.filename} (error: {detailed.get('message', 'Unknown')})")
                    continue
                
                video_info = {
                    "id": video.video_uuid,
                    "name": video.filename or f"Video_{video.video_uuid[:8]}",
                    "analyzed_at": database.to_beijing_time(video.upload_time).isoformat() if hasattr(video, 'upload_time') and video.upload_time else None,
                    "db_id": video.id,
                    # 历史记录额外字段
                    "has_highlight": os.path.exists(
                        os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")
                    ),
                    "highlight_url": f"/api/download-highlight/{video.video_uuid}" if os.path.exists(
                        os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")
                    ) else None,
                    "highlight_size_mb": round(
                        os.path.getsize(os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")) / 1048576, 1
                    ) if os.path.exists(os.path.join(UPLOAD_DIR, f"{video.video_uuid}_highlights.mp4")) else None,
                    "athlete_count": db.query(AthleteProfile).filter(AthleteProfile.video_id == video.id).count(),
                }
                
                # 去重逻辑：如果已存在相同UUID，保留最新的（upload_time最大或db_id最大）
                if video.video_uuid in video_dict:
                    existing = video_dict[video.video_uuid]
                    existing_time = existing.get('analyzed_at') or ''
                    current_time = video_info.get('analyzed_at') or ''
                    
                    # 保留时间更新的，或时间相同时保留ID更大的
                    if current_time > existing_time or (current_time == existing_time and video.id > existing['db_id']):
                        print(f"[DEDUP] Replace duplicate: {video.filename} (old_id={existing['db_id']}, new_id={video.id})")
                        video_dict[video.video_uuid] = video_info
                    else:
                        print(f"[DEDUP] Skip duplicate: {video.filename} (keep_id={existing['db_id']}, skip_id={video.id})")
                else:
                    video_dict[video.video_uuid] = video_info
                    print(f"[INFO] Add video: {video_info['name']} (ID: {video_info['id'][:8]}...)")
        
        # 移除临时的db_id字段
        result = []
        for video_info in video_dict.values():
            del video_info['db_id']
            result.append(video_info)
        
        # 按上传时间倒序排序（最新的在前）
        result.sort(key=lambda x: x.get('analyzed_at') or '', reverse=True)
        
        print(f"[INFO] [OK] Return {len(result)} unique valid videos (raw: {len(videos_with_analysis)})")
        
        return {
            "status": "success",
            "count": len(result),
            "videos": result
        }
    
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed to fetch videos: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "videos": []
        }


# ============================================================================
#  精彩视频生成 API
# ============================================================================

from pydantic import BaseModel

class HighlightVideoRequest(BaseModel):
    """精彩视频生成请求"""
    video_id: str
    target_duration: float = 180.0  # 默认3分钟
    enable_slowmo: bool = True      # 启用慢动作
    enable_zoom: bool = True        # 启用特写
    enable_pip: bool = True         # 启用画中画
    enable_bgm: bool = True         # 启用激昂背景音乐（去除原音频）
    bgm_volume: float = 0.88        # BGM 音量（0~1）

# 精彩视频生成进度存储  { video_id: int(0-100/-1) }
highlight_progress_store = {}
# 心跳时间戳：记录每次进度更新的时间，前端用于判断是否真正卡住
highlight_heartbeat_store = {}   # { video_id: float(timestamp) }
# 错误信息存储
highlight_error_store = {}       # { video_id: str }

@app.post("/api/generate-highlight")
async def generate_highlight_video(
    request: HighlightVideoRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    生成精彩视频
    
    功能：
    1. 自动识别精彩片段（特别是进球时刻）
    2. 应用特效：慢动作、特写、画中画
    3. 生成约3分钟的精华视频
    """
    
    try:
        # 验证视频是否存在
        video = db.query(VideoModel).filter(VideoModel.video_uuid == request.video_id).first()
        
        print(f"[DEBUG] 查询视频ID: {request.video_id}")
        
        if not video:
            raise HTTPException(status_code=404, detail="视频不存在")
        
        print(f"[DEBUG] 找到视频记录，filename={video.filename}")
        
        # 构建视频文件路径
        input_path = os.path.join(UPLOAD_DIR, video.filename)
        
        print(f"[DEBUG] 准备查找文件路径: {input_path}")
        
        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {video.filename}")
        
        # 🔥 任务锁检查：防止同一视频的重复启动
        if request.video_id in highlight_task_locks:
            current_progress = highlight_progress_store.get(request.video_id, 0)
            print(f"[HIGHLIGHT] ⚠️ 任务已在进行中！video_id={request.video_id}, 进度={current_progress}%")
            print(f"[HIGHLIGHT] ⚠️ 拒绝重复启动请求（使用任务锁机制）")
            return {
                "status": "already_running",
                "message": f"精彩视频生成任务已在进行中（进度: {current_progress}%）",
                "video_id": request.video_id,
                "current_progress": current_progress
            }
        
        # 🔥 获取任务锁
        highlight_task_locks.add(request.video_id)
        print(f"[HIGHLIGHT] 🔒 已获取任务锁: {request.video_id}")
        
        # 初始化进度
        highlight_progress_store[request.video_id] = 0
        print(f"[HIGHLIGHT] ✅ 初始化进度为 0%，准备启动任务")
        
        # 生成输出文件名
        output_filename = f"{request.video_id}_highlights.mp4"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        # 后台任务生成精彩视频
        background_tasks.add_task(
            _generate_highlight_task,
            request.video_id,
            input_path,
            output_path,
            request.target_duration,
            request.enable_slowmo,
            request.enable_zoom,
            request.enable_pip,
            request.enable_bgm,
            request.bgm_volume
        )
        
        return {
            "status": "success",
            "message": "精彩视频生成任务已启动",
            "video_id": request.video_id,
            "output_filename": output_filename
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] 启动精彩视频生成失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _generate_highlight_task(
    video_id: str,
    input_path: str,
    output_path: str,
    target_duration: float,
    enable_slowmo: bool,
    enable_zoom: bool,
    enable_pip: bool,
    enable_bgm: bool = True,
    bgm_volume: float = 0.88
):
    """后台任务：生成精彩视频"""
    
    try:
        print(f"\n[HIGHLIGHT] ==================== 任务开始 ====================")
        print(f"[HIGHLIGHT] 视频ID: {video_id}")
        print(f"[HIGHLIGHT] 输入文件: {input_path}")
        print(f"[HIGHLIGHT] 输出文件: {output_path}")
        print(f"[HIGHLIGHT] 目标时长: {target_duration}秒")
        print(f"[HIGHLIGHT] 特效设置: 慢动作={enable_slowmo}, 特写={enable_zoom}, 画中画={enable_pip}, BGM={enable_bgm}(音量{bgm_volume})")
        print(f"[HIGHLIGHT] =====================================================\n")
        
        # 🔥 立即设置进度为 1%，让前端知道任务已启动
        highlight_progress_store[video_id] = 1
        print(f"[HIGHLIGHT] ✅ 进度初始化: 1%")
        
        # 验证输入文件存在
        print(f"[HIGHLIGHT] 检查输入文件是否存在...")
        if not os.path.exists(input_path):
            print(f"[HIGHLIGHT ERROR] ❌ 输入文件不存在: {input_path}")
            highlight_progress_store[video_id] = -1
            return
        print(f"[HIGHLIGHT] ✅ 输入文件存在")
        
        # 获取视频信息
        import cv2
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            print(f"[HIGHLIGHT] 📹 视频信息: {duration:.1f}秒, {fps:.1f} FPS, {total_frames} 帧")
            cap.release()
        
        # 导入精彩视频生成器
        highlight_progress_store[video_id] = 2
        print(f"[HIGHLIGHT] 导入精彩视频生成模块...")
        from highlight_generator import HighlightVideoGenerator
        print(f"[HIGHLIGHT] ✅ 模块导入成功")
        
        # 获取 YOLO 模型
        highlight_progress_store[video_id] = 3
        print(f"[HIGHLIGHT] 加载 YOLO 模型...")
        try:
            # 🔥 修复：直接调用 analysis 模块的函数
            from analysis import get_yolo_model
            yolo_model = get_yolo_model()
            print(f"[HIGHLIGHT] ✅ YOLO 模型加载成功")
        except Exception as e:
            print(f"[HIGHLIGHT ERROR] ❌ YOLO 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            highlight_progress_store[video_id] = -1
            return
        
        # 创建生成器
        highlight_progress_store[video_id] = 5
        print(f"[HIGHLIGHT] 创建精彩视频生成器...")
        generator = HighlightVideoGenerator(yolo_model=yolo_model)
        print(f"[HIGHLIGHT] ✅ 生成器创建成功")
        
        # 进度回调（带心跳时间戳）
        def progress_callback(progress: int):
            old_progress = highlight_progress_store.get(video_id, 0)
            highlight_progress_store[video_id] = progress
            highlight_heartbeat_store[video_id] = __import__('time').time()
            if progress != old_progress:
                print(f"[HIGHLIGHT] 📊 进度更新: {old_progress}% → {progress}%")
        
        # 生成精彩视频
        result = generator.generate_highlight_video(
            input_video=input_path,
            output_video=output_path,
            target_duration=target_duration,
            enable_slowmo=enable_slowmo,
            enable_zoom=enable_zoom,
            enable_pip=enable_pip,
            enable_bgm=enable_bgm,
            bgm_volume=bgm_volume,
            progress_callback=progress_callback
        )
        
        if result['success']:
            print(f"[HIGHLIGHT] ✅ 精彩视频生成成功: {output_path}")
            highlight_progress_store[video_id] = 100
            highlight_heartbeat_store[video_id] = __import__('time').time()
        else:
            err = result.get('error', '未知错误')
            print(f"[HIGHLIGHT] ❌ 精彩视频生成失败: {err}")
            highlight_progress_store[video_id] = -1
            highlight_error_store[video_id] = err
    
    except Exception as e:
        print(f"[HIGHLIGHT] ❌ 生成过程出错: {e}")
        import traceback
        traceback.print_exc()
        highlight_progress_store[video_id] = -1
        highlight_error_store[video_id] = str(e)
    
    finally:
        # 🔥 重要：无论成功还是失败，都要释放任务锁
        if video_id in highlight_task_locks:
            highlight_task_locks.remove(video_id)
            print(f"[HIGHLIGHT] 🔓 已释放任务锁: {video_id}")


@app.get("/api/highlight-progress/{video_id}")
async def get_highlight_progress(video_id: str):
    """
    查询精彩视频生成进度
    额外返回 last_heartbeat（上次进度更新的 Unix 时间戳），前端可用于
    判断任务是否真正卡住（超过 N 秒无心跳则提示用户）。
    """
    import time as _time
    progress  = highlight_progress_store.get(video_id)
    heartbeat = highlight_heartbeat_store.get(video_id)
    error_msg = highlight_error_store.get(video_id, "")

    if progress is None:
        return {
            "status": "not_found",
            "message": "精彩视频生成任务不存在",
            "progress": None
        }
    elif progress == -1:
        # 分析错误类型，给出友好提示
        if "系统找不到指定的文件" in error_msg or "WinError 2" in error_msg:
            friendly = "ffmpeg 路径异常，请重启后端服务后重试"
        elif "TimeoutExpired" in error_msg or "超时" in error_msg:
            friendly = "视频处理超时，建议缩短视频时长或降低分辨率后重试"
        elif "codec" in error_msg.lower() or "decoder" in error_msg.lower():
            friendly = "视频编解码格式不支持，请将视频转换为 H.264/MP4 格式后重试"
        elif "No such file" in error_msg or "不存在" in error_msg:
            friendly = "视频文件已被删除，请重新上传"
        elif error_msg:
            friendly = f"生成失败：{error_msg[:120]}"
        else:
            friendly = "精彩视频生成失败，请重新上传视频后重试"
        return {
            "status": "failed",
            "message": friendly,
            "error_detail": error_msg,
            "progress": -1
        }
    elif progress == 100:
        output_filename = f"{video_id}_highlights.mp4"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            return {
                "status": "completed",
                "message": "精彩视频生成完成",
                "progress": 100,
                "output_file": output_filename,
                "download_url": f"/api/download-highlight/{video_id}",
                "file_size_mb": round(file_size, 2)
            }
        else:
            return {
                "status": "error",
                "message": "精彩视频文件不存在",
                "progress": -1
            }
    else:
        return {
            "status": "processing",
            "message": f"正在生成精彩视频... ({progress}%)",
            "progress": progress,
            "last_heartbeat": heartbeat   # Unix timestamp，前端可做心跳检测
        }


@app.get("/api/download-highlight/{video_id}")
async def download_highlight_video(video_id: str):
    """下载精彩视频（强制下载）"""
    
    output_filename = f"{video_id}_highlights.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="精彩视频文件不存在")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"highlights_{video_id}.mp4",
        headers={"Content-Disposition": f"attachment; filename=highlights_{video_id}.mp4"}
    )


@app.get("/api/stream-highlight/{video_id}")
async def stream_highlight_video(video_id: str, request: Request):
    """精彩视频流式预览（支持 HTTP Range，供浏览器 <video> 标签使用）"""
    output_filename = f"{video_id}_highlights.mp4"
    output_path = os.path.join(UPLOAD_DIR, output_filename)

    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="精彩视频文件不存在")

    file_size = os.path.getsize(output_path)
    range_header = request.headers.get("range")

    if range_header:
        # 解析 Range: bytes=start-end
        try:
            range_val = range_header.strip().replace("bytes=", "")
            parts = range_val.split("-")
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else file_size - 1
        except Exception:
            raise HTTPException(status_code=416, detail="Invalid Range header")

        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def iter_file():
            with open(output_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)
    else:
        # 无 Range 头：返回全文件
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }
        def iter_full():
            with open(output_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    yield data
        return StreamingResponse(iter_full(), status_code=200, headers=headers)


# ============================================================================
#  用户认证模块
# ============================================================================
import hashlib
import secrets
import re as _re

# 管理员会话Token存储：{token: {username, expire_time}}
_admin_tokens: dict = {}

# 访问日志（内存，最多保留500条）
_access_logs: list = []


def _get_user_db() -> Session:
    """获取数据库Session（用于非依赖注入场景）"""
    return SessionLocal()


def _ensure_default_users():
    """确保内置账号（admin/demo）存在于数据库中"""
    db = _get_user_db()
    try:
        defaults = [
            {
                "username": "admin",
                "email": "admin@football.ai",
                "password_hash": hashlib.sha256("Admin@2024".encode()).hexdigest(),
                "role": "admin",
            },
            {
                "username": "demo",
                "email": "demo@football.ai",
                "password_hash": hashlib.sha256("123456".encode()).hexdigest(),
                "role": "user",
            },
        ]
        for d in defaults:
            existing = db.query(UserModel).filter(UserModel.username == d["username"]).first()
            if not existing:
                db.add(UserModel(**d))
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[AUTH] Failed to init default users: {e}")
    finally:
        db.close()


def _lookup_user(db: Session, username_or_email: str) -> Optional[UserModel]:
    """按用户名或邮箱查找用户"""
    user = db.query(UserModel).filter(UserModel.username == username_or_email).first()
    if not user:
        user = db.query(UserModel).filter(UserModel.email == username_or_email).first()
    return user


def _add_access_log(action: str, username: str, detail: str = "", log_type: str = "blue", ip: str = ""):
    """记录访问日志"""
    _access_logs.insert(0, {
        "action": action,
        "user": username,
        "detail": detail,
        "type": log_type,
        "ip": ip,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    # 限制日志条数
    while len(_access_logs) > 500:
        _access_logs.pop()

def _verify_admin_token(request: Request) -> str:
    """验证管理员Token，返回username；验证失败抛出403"""
    token = request.headers.get("X-Admin-Token", "")
    if not token:
        raise HTTPException(status_code=403, detail="缺少管理员Token")
    # 演示token直接通过
    if token.startswith("demo-token-"):
        return "admin"
    session = _admin_tokens.get(token)
    if not session:
        raise HTTPException(status_code=403, detail="Token无效或已过期")
    if datetime.datetime.now().timestamp() > session["expire_time"]:
        _admin_tokens.pop(token, None)
        raise HTTPException(status_code=403, detail="Token已过期，请重新登录")
    return session["username"]




class AuthLoginRequest(BaseModel):
    username: str
    password: str

class AuthRegisterRequest(BaseModel):
    username: str
    email: str
    password: str

@app.post("/api/auth/login")
async def auth_login(req: AuthLoginRequest, request: Request, db: Session = Depends(get_db)):
    """用户登录（支持用户名或邮箱）"""
    user = _lookup_user(db, req.username.strip())
    if not user or not user.is_active:
        _add_access_log("user_login_fail", req.username, "账号不存在或已禁用", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if user.password_hash != pwd_hash:
        _add_access_log("user_login_fail", req.username, "密码错误", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    _add_access_log("user_login", user.username, "用户登录成功", "green",
                    ip=request.client.host if request.client else "")
    return {"username": user.username, "role": user.role, "email": user.email or ""}


@app.post("/api/auth/admin-login")
async def admin_login(req: AuthLoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    管理员专用登录接口
    - 仅允许 role=admin 的账号
    - 成功后返回 session token（有效期2小时）
    """
    user = _lookup_user(db, req.username.strip())
    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()

    if not user or user.password_hash != pwd_hash:
        _add_access_log("admin_login_fail", req.username, "密码错误", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=401, detail="账号或密码错误")

    if user.role != "admin":
        _add_access_log("admin_login_fail", req.username, "非管理员账号尝试登录后台", "red",
                        ip=request.client.host if request.client else "")
        raise HTTPException(status_code=403, detail="无管理员权限")

    # 生成安全Token
    token = secrets.token_urlsafe(32)
    expire = (datetime.datetime.now() + datetime.timedelta(hours=2)).timestamp()
    _admin_tokens[token] = {"username": user.username, "expire_time": expire}

    _add_access_log("admin_login", user.username, "管理员登录后台成功", "gold",
                    ip=request.client.host if request.client else "")

    return {
        "username": user.username,
        "role": user.role,
        "token": token,
        "expire_in": 7200,
    }

@app.post("/api/auth/register")
async def auth_register(req: AuthRegisterRequest, request: Request, db: Session = Depends(get_db)):
    """
    用户注册
    - 用户名唯一性检查
    - 邮箱唯一性检查
    - 邮箱格式验证
    - 用户名长度限制（4-20字符，字母数字下划线）
    - 密码强度要求（至少8位）
    """
    username = req.username.strip()
    email = req.email.strip().lower()
    password = req.password

    # 1. 用户名格式校验
    if len(username) < 4 or len(username) > 20:
        raise HTTPException(status_code=400, detail="用户名长度须在4到20个字符之间")
    if not _re.match(r'^[a-zA-Z0-9_\u4e00-\u9fa5]+$', username):
        raise HTTPException(status_code=400, detail="用户名只能包含字母、数字、下划线或中文")

    # 2. 邮箱格式校验
    if not _re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        raise HTTPException(status_code=400, detail="邮箱格式不正确")

    # 3. 密码强度校验
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="密码至少8位")

    # 4. 用户名唯一性检查
    if db.query(UserModel).filter(UserModel.username == username).first():
        raise HTTPException(status_code=409, detail="用户名已被注册，请换一个")

    # 5. 邮箱唯一性检查
    if db.query(UserModel).filter(UserModel.email == email).first():
        raise HTTPException(status_code=409, detail="该邮箱已被注册，请使用其他邮箱或直接登录")

    # 6. 写入数据库
    new_user = UserModel(
        username=username,
        email=email,
        password_hash=hashlib.sha256(password.encode()).hexdigest(),
        role="user",
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    _add_access_log("register", username, f"新用户注册: {email}", "blue",
                    ip=request.client.host if request.client else "")
    return {"message": "注册成功", "username": username}




# ============================================================================
#  访问日志 & 用户管理 API
# ============================================================================

@app.post("/api/admin/access-log")
async def post_access_log(request: Request):
    """前端写入访问日志"""
    _verify_admin_token(request)
    try:
        body = await request.json()
        _add_access_log(
            action=body.get("action", "unknown"),
            username=body.get("username", "unknown"),
            detail=body.get("detail", ""),
            log_type=body.get("type", "blue"),
            ip=request.client.host if request.client else ""
        )
        return {"ok": True}
    except Exception:
        return {"ok": False}

@app.get("/api/admin/access-logs")
async def get_access_logs(request: Request, limit: int = 100):
    """获取访问日志列表（需管理员Token）"""
    _verify_admin_token(request)
    return {"logs": _access_logs[:limit], "total": len(_access_logs)}

@app.get("/api/admin/users")
async def get_users_list(request: Request, db: Session = Depends(get_db)):
    """获取注册用户列表（需管理员Token）"""
    _verify_admin_token(request)
    users = db.query(UserModel).order_by(UserModel.created_at.asc()).all()
    result = [
        {
            "username": u.username,
            "role": u.role,
            "email": u.email,
            "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else "",
            "is_active": u.is_active,
        }
        for u in users
    ]
    return {"users": result, "total": len(result)}


# ============================================================================
#  数据统计 & 管理 API
# ============================================================================

@app.get("/api/stats/overview")
async def get_stats_overview(db: Session = Depends(get_db)):
    """
    系统概览统计
    返回视频数量、球员数量、精彩视频数量等
    """
    try:
        # 统计视频总数
        total_videos = db.query(VideoModel).count()

        # 统计球员档案总数
        total_athletes = db.query(AthleteProfile).count()

        # 统计已生成精彩视频数（检查文件存在）
        highlight_files = [
            f for f in os.listdir(UPLOAD_DIR)
            if f.endswith("_highlights.mp4")
        ] if os.path.exists(UPLOAD_DIR) else []
        total_highlights = len(highlight_files)

        # 统计上传文件总大小（MB）
        total_size_mb = 0.0
        if os.path.exists(UPLOAD_DIR):
            for f in os.listdir(UPLOAD_DIR):
                fp = os.path.join(UPLOAD_DIR, f)
                if os.path.isfile(fp):
                    total_size_mb += os.path.getsize(fp)
        total_size_mb = round(total_size_mb / (1024 * 1024), 1)

        # 正在进行的生成任务
        active_tasks = len(highlight_task_locks)

        return {
            "total_videos": total_videos,
            "total_athletes": total_athletes,
            "total_highlights": total_highlights,
            "total_size_mb": total_size_mb,
            "active_tasks": active_tasks,
            "registered_users": db.query(UserModel).count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/videos")
async def get_video_stats(db: Session = Depends(get_db)):
    """
    视频分析统计列表
    返回所有视频的分析结果概要
    """
    try:
        # ORM字段名为 upload_time，对外 API 字段名统一用 created_at
        videos = db.query(VideoModel).order_by(VideoModel.upload_time.desc()).limit(50).all()
        result = []
        for v in videos:
            video_id = str(v.id)
            has_highlight = os.path.exists(
                os.path.join(UPLOAD_DIR, f"{video_id}_highlights.mp4")
            )
            athletes = db.query(AthleteProfile).filter(
                AthleteProfile.video_id == video_id
            ).count()
            result.append({
                "video_id": video_id,
                "filename": v.filename,
                "created_at": v.upload_time.isoformat() if v.upload_time else None,
                "athlete_count": athletes,
                "has_highlight": has_highlight,
                "highlight_progress": highlight_progress_store.get(video_id, 0),
            })
        return {"videos": result, "total": len(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats/athletes")
async def get_athlete_stats(db: Session = Depends(get_db)):
    """
    球员统计数据汇总
    返回评分分布、平均能力值等
    """
    try:
        athletes = db.query(AthleteProfile).all()
        if not athletes:
            return {"total": 0, "avg_score": 0, "score_distribution": {}, "top_players": []}

        scores = []
        top_players = []
        for a in athletes:
            score = getattr(a, 'overall_score', None) or 0
            scores.append(score)
            top_players.append({
                "id": str(a.id),
                "video_id": str(a.video_id),
                "overall_score": round(score, 1),
            })

        top_players.sort(key=lambda x: x["overall_score"], reverse=True)

        # 评分段分布
        distribution = {"<60": 0, "60-75": 0, "75-85": 0, "85+": 0}
        for s in scores:
            if s < 60:   distribution["<60"] += 1
            elif s < 75: distribution["60-75"] += 1
            elif s < 85: distribution["75-85"] += 1
            else:        distribution["85+"] += 1

        return {
            "total": len(athletes),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "score_distribution": distribution,
            "top_players": top_players[:10],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/video/{video_id}")
async def admin_delete_video(video_id: str, request: Request, db: Session = Depends(get_db)):
    """管理员删除视频及关联数据（需管理员Token）"""
    admin_user = _verify_admin_token(request)
    try:
        # video_id 为整数主键字符串，转换为 int 确保匹配
        vid_int = int(video_id)
        video = db.query(VideoModel).filter(VideoModel.id == vid_int).first()
        if not video:
            raise HTTPException(status_code=404, detail=f"视频不存在: {video_id}")

        filename = video.filename
        video_uuid = video.video_uuid

        # 删除视频物理文件（原始 + 精彩集锦）
        for suffix in ["", "_highlights"]:
            for name_key in [video_uuid, str(vid_int)]:
                fp = os.path.join(UPLOAD_DIR, f"{name_key}{suffix}.mp4")
                if os.path.exists(fp):
                    os.remove(fp)

        # 删除所有关联数据库记录
        db.query(AthleteProfile).filter(AthleteProfile.video_id == vid_int).delete()
        db.query(AIChatSession).filter(AIChatSession.video_uuid == video_uuid).delete()
        db.query(AIVideoCache).filter(AIVideoCache.video_uuid == video_uuid).delete()
        db.query(VideoModel).filter(VideoModel.id == vid_int).delete()
        db.commit()

        # 清理内存进度缓存
        highlight_progress_store.pop(video_uuid, None)
        highlight_progress_store.pop(str(vid_int), None)
        highlight_task_locks.discard(video_uuid)
        highlight_task_locks.discard(str(vid_int))

        _add_access_log("delete", admin_user, f"删除视频: {filename}", "red",
                        ip=request.client.host if request.client else "")
        return {"message": "删除成功", "video_id": video_id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/reset-all-data")
async def admin_reset_all_data(request: Request, db: Session = Depends(get_db)):
    """管理员：清空所有视频、分析数据、AI缓存（需管理员Token）"""
    admin_user = _verify_admin_token(request)
    try:
        # 1. 删除所有上传文件和导出文件
        deleted_files = 0
        for search_dir in [UPLOAD_DIR]:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for fname in files:
                        fp = os.path.join(root, fname)
                        try:
                            os.remove(fp)
                            deleted_files += 1
                        except Exception:
                            pass

        # 2. 清空数据库所有相关表（按外键依赖顺序）
        ai_chat_del = db.query(AIChatSession).delete()
        ai_cache_del = db.query(AIVideoCache).delete()
        profile_del  = db.query(AthleteProfile).delete()
        video_del    = db.query(VideoModel).delete()
        db.commit()

        # 3. 清空内存缓存
        highlight_progress_store.clear()
        highlight_task_locks.clear()

        _add_access_log("admin_reset", admin_user, "重置所有数据", "red",
                        ip=request.client.host if request.client else "")
        return {
            "message": "重置成功",
            "deleted_files": deleted_files,
            "deleted_rows": {
                "videos": video_del,
                "athlete_profiles": profile_del,
                "ai_chat_sessions": ai_chat_del,
                "ai_video_cache": ai_cache_del,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/system-info")
async def get_system_info(request: Request):
    """系统信息（需管理员Token）"""
    _verify_admin_token(request)
    import platform
    try:
        import psutil, sys as _sys
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        # Windows 使用 C:\ ，其他系统使用 /
        _disk_path = 'C:\\' if _sys.platform.startswith('win') else '/'
        disk = psutil.disk_usage(_disk_path).percent
    except Exception:
        cpu = mem = disk = None

    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "active_highlight_tasks": list(highlight_task_locks),
        "progress_store_size": len(highlight_progress_store),
        "upload_dir": UPLOAD_DIR,
        "cpu_percent": cpu,
        "memory_percent": mem,
        "disk_usage_percent": disk,
    }

    return info


# ============================================================================
#  球衣号码识别 API
# ============================================================================

@app.post("/api/jersey/recognize-video")
async def jersey_recognize_video(
    video_id: str,
    sample_interval: int = 30,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
):
    """
    对指定视频进行球衣号码识别与球队归属分类

    参数：
      video_id:        视频ID
      sample_interval: 每N帧采样1帧，默认30（约1秒1帧）

    返回：
      task_id 任务ID，通过 /api/jersey/result/{task_id} 轮询结果
    """
    # 查找视频文件
    video = db.query(VideoModel).filter(VideoModel.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="视频不存在")

    video_path = os.path.join(UPLOAD_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")

    task_id = f"jersey_{video_id}"

    # 如果已在进行中，直接返回
    if task_id in highlight_task_locks:
        return {"task_id": task_id, "status": "running", "message": "识别任务进行中"}

    highlight_task_locks.add(task_id)
    highlight_progress_store[task_id] = 0

    # 后台异步执行
    async def run_jersey_recognition():
        try:
            from jersey_number_recognition import JerseyNumberRecognizer

            def progress_cb(pct):
                highlight_progress_store[task_id] = pct

            recognizer = JerseyNumberRecognizer(
                yolo_model_path="yolov8n.pt",
                use_ocr=True,
                conf_threshold=0.4,
            )
            results = recognizer.process_video(
                video_path,
                sample_interval=sample_interval,
                progress_callback=progress_cb,
            )

            # 将结果持久化到进度存储（格式化后存储）
            highlight_progress_store[f"{task_id}_result"] = results
            highlight_progress_store[task_id] = 100

        except Exception as e:
            logger.error(f"[JerseyAPI] 识别失败: {e}")
            highlight_progress_store[task_id] = -1
        finally:
            highlight_task_locks.discard(task_id)

    import asyncio
    asyncio.create_task(run_jersey_recognition())

    return {
        "task_id": task_id,
        "status": "started",
        "message": f"号码识别任务已启动，视频ID: {video_id}",
    }


@app.get("/api/jersey/result/{video_id}")
async def jersey_get_result(video_id: str):
    """
    查询球衣号码识别结果

    Returns:
      {
        "status": "running" | "done" | "failed" | "not_found",
        "progress": 0-100,
        "players": [{"player_id": int, "number": int|null, "team": 0|1|-1, "confidence": float}]
      }
    """
    task_id = f"jersey_{video_id}"
    progress = highlight_progress_store.get(task_id)
    result_key = f"{task_id}_result"

    if progress is None:
        return {"status": "not_found", "progress": 0, "players": []}

    if progress == -1:
        return {"status": "failed", "progress": 0, "players": []}

    if progress < 100:
        return {"status": "running", "progress": progress, "players": []}

    # 已完成
    raw_results = highlight_progress_store.get(result_key, {})
    players = []
    for pid, info in raw_results.items():
        team_name = {0: "主队", 1: "客队"}.get(info.get("team", -1), "未知")
        players.append({
            "player_id": pid,
            "number": info.get("number"),
            "team": info.get("team", -1),
            "team_name": team_name,
            "confidence": info.get("confidence", 0.0),
        })

    # 按号码排序（无号码排末尾）
    players.sort(key=lambda x: (x["number"] is None, x["number"] or 999))

    return {
        "status": "done",
        "progress": 100,
        "total_players": len(players),
        "players": players,
    }


@app.get("/api/jersey/progress/{video_id}")
async def jersey_get_progress(video_id: str):
    """获取球衣号码识别进度（0-100）"""
    task_id = f"jersey_{video_id}"
    progress = highlight_progress_store.get(task_id, 0)
    return {"video_id": video_id, "progress": max(0, progress)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)




