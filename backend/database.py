from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, ForeignKey, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from datetime import timezone, timedelta
import os

# ──────────────────────────────────────────────────────────────────────────────
# 数据库连接配置
# 路径固定为 backend/ 目录下，避免因启动目录不同产生多份 DB 文件
# 生产环境可通过环境变量 DATABASE_URL 覆盖（例如 MySQL）
# ──────────────────────────────────────────────────────────────────────────────
_DB_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/ 绝对路径
_DB_FILE = os.path.join(_DB_DIR, "football_demo.db")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{_DB_FILE}")

Base = declarative_base()


class VideoModel(Base):
    """视频记录表（主表）"""
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    video_uuid = Column(String(255), unique=True, index=True)
    filename = Column(String(255))
    # upload_time 是 ORM 字段名，对外 API 统一用 created_at 别名返回
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    # 视频所有者（登录用户名），用于按账号隔离历史记录
    owner = Column(String(64), index=True, nullable=True, default=None)


class AthleteProfile(Base):
    """运动员分析档案表（每个视频对应一条）"""
    __tablename__ = "athlete_profiles"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), index=True)
    overall_score = Column(Float)
    decision_summary = Column(String(1000))
    # 运动员详细分析数据（路线、球位置、多球员数据等）
    detailed_analysis = Column(JSON)


# ── PostureData 表已于 2026-02 移除 ──────────────────────────────────────────
# 原因：该表在整个业务代码中从未被引用（仅在 database.py 中定义），
#       且 init.sql 中对应字段与 ORM 不一致，属于废弃冗余表。
#       数据库中若已存在 posture_data 表不会影响系统运行（SQLAlchemy 不会
#       自动删除不在 metadata 中的表），如需彻底清除可手动执行：
#         DROP TABLE IF EXISTS posture_data;
# ─────────────────────────────────────────────────────────────────────────────


# ── AI 数据持久化表（2026-02 新增） ──────────────────────────────────────────

class AIChatSession(Base):
    """
    AI 对话会话历史表

    将原来纯内存的 DeepSeekAgent.conversation_memory
    持久化到数据库，解决以下问题：
      1. 服务重启后对话历史丢失（幽灵内存数据）
      2. 无法追溯历史对话
      3. 内存无限增长（超过5轮窗口仍保留在内存中）

    字段说明：
      session_id  — 前端生成的随机 session，如 session_xxx
      video_uuid  — 关联视频（可为空，表示通用问答）
      role        — 'user' 或 'assistant'
      content     — 消息正文
      created_at  — 消息写入时间（UTC）
      is_ghost    — 是否为幽灵消息（关联视频已不存在）
    """
    __tablename__ = "ai_chat_sessions"

    id         = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), index=True, nullable=False)
    video_uuid = Column(String(255), index=True, nullable=True)
    role       = Column(String(16), nullable=False)          # 'user' | 'assistant'
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    is_ghost   = Column(Boolean, default=False, index=True)  # 关联视频已删除


class AIVideoCache(Base):
    """
    AI 视频语义知识库缓存表

    将原来纯内存的 DeepSeekAgent.video_knowledge_base
    持久化到数据库，解决以下问题：
      1. 服务重启后知识库丢失，需要重新调用 /ai/analyze-video
      2. 无法校验缓存是否对应已删除视频（幽灵缓存）
      3. 内存占用随视频数量无限增长

    字段说明：
      video_uuid     — 对应视频的 UUID（与 videos.video_uuid 关联）
      semantic_json  — analyze_video_content() 返回的完整语义信息（JSON）
      created_at     — 首次缓存时间
      updated_at     — 最近更新时间
      is_ghost       — 对应视频在 videos 表已不存在
    """
    __tablename__ = "ai_video_cache"

    id            = Column(Integer, primary_key=True, index=True)
    video_uuid    = Column(String(255), unique=True, index=True, nullable=False)
    semantic_json = Column(JSON, nullable=False)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.datetime.utcnow,
                           onupdate=datetime.datetime.utcnow)
    is_ghost      = Column(Boolean, default=False, index=True)

# ─────────────────────────────────────────────────────────────────────────────


class UserModel(Base):
    """
    用户账号持久化表

    替代原来的内存字典 _users_store，解决以下问题：
      1. 服务重启后注册用户数据丢失
      2. 无法实现邮箱唯一性约束
      3. 无法在管理后台持久化查询用户列表

    字段说明：
      username      — 登录用户名（唯一）
      email         — 邮箱地址（唯一，用于邮箱登录）
      password_hash — SHA256 哈希后的密码
      role          — 'admin' | 'user'
      created_at    — 注册时间（UTC）
      is_active     — 是否启用（软删除预留）
    """
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(64), unique=True, index=True, nullable=False)
    email         = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role          = Column(String(16), default="user", nullable=False)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)
    is_active     = Column(Boolean, default=True, nullable=False)

# ─────────────────────────────────────────────────────────────────────────────


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── 时区转换辅助函数 ──────────────────────────────────────────────────────────

def to_beijing_time(utc_datetime):
    """将 UTC datetime 转换为北京时间 (UTC+8)"""
    if utc_datetime is None:
        return None
    if utc_datetime.tzinfo is None:
        utc_datetime = utc_datetime.replace(tzinfo=timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    return utc_datetime.astimezone(beijing_tz)


def init_db():
    """创建所有在 metadata 中注册的表（含新增的 ai_chat_sessions / ai_video_cache / users）"""
    Base.metadata.create_all(bind=engine)

    # 兼容性迁移：为旧版 videos 表补充 owner 列（SQLite 不支持 ALTER TABLE ADD COLUMN IF NOT EXISTS）
    try:
        with engine.connect() as conn:
            # 检查 owner 列是否已存在
            result = conn.execute(
                __import__('sqlalchemy').text("PRAGMA table_info(videos)")
            )
            columns = [row[1] for row in result]
            if "owner" not in columns:
                conn.execute(
                    __import__('sqlalchemy').text(
                        "ALTER TABLE videos ADD COLUMN owner VARCHAR(64) DEFAULT NULL"
                    )
                )
                conn.commit()
                print("[DB_MIGRATE] Added 'owner' column to videos table")
    except Exception as e:
        print(f"[DB_MIGRATE] Migration warning (non-fatal): {e}")


def get_db():
    """FastAPI 依赖注入：提供数据库 Session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
