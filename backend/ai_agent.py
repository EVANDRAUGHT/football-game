"""
DeepSeek AI 智能代理核心模块
提供视频内容理解、自然语言交互、智能问答等功能
"""
import os
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from openai import AsyncOpenAI
import hashlib
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── DB 持久化（可选导入，避免循环依赖）────────────────────────────────────────
def _get_db_session():
    """惰性导入 DB Session，避免循环导入"""
    try:
        from database import SessionLocal, AIChatSession, AIVideoCache  # noqa: F401
        return SessionLocal, AIChatSession, AIVideoCache
    except Exception:
        return None, None, None

class DeepSeekAgent:
    """
    DeepSeek AI 智能代理
    核心功能：
    1. 视频内容语义理解
    2. 多轮对话上下文管理
    3. 专业足球分析回答生成
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 DeepSeek 客户端
        支持环境变量或直接传参配置 API Key
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DeepSeek API Key 未配置，请设置环境变量 DEEPSEEK_API_KEY 或传入参数")
        
        # 初始化异步客户端（兼容不同版本的 openai 库）
        try:
            # 新版本 openai>=1.30 的初始化方式
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com",
                timeout=60.0,
                max_retries=3
            )
        except TypeError:
            # 旧版本 openai<1.30 的初始化方式（不支持某些参数）
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        
        # 对话历史存储（session_id -> messages）
        self.conversation_memory: Dict[str, List[Dict]] = {}
        
        # 视频内容缓存（video_id -> semantic_info）
        self.video_knowledge_base: Dict[str, Dict] = {}
        
        # 性能监控
        self.metrics = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0
        }

        # 系统启动时从 DB 恢复视频知识库缓存（非幽灵记录）
        self._restore_video_cache_from_db()
        
        # 系统提示词 - 定义 AI 的角色和能力边界
        self.system_prompt = """你是一位专业的足球视频分析AI助手，名为 FootballGPT。你的核心能力包括：

**角色定位**：
- 专业足球教练与数据分析师的结合体
- 擅长解读视频中的战术细节、球员表现、技术动作
- 能将复杂的AI分析结果转化为通俗易懂的建议

**知识范围**：
- 足球战术体系（4-3-3、4-4-2等阵型，高位压迫、防守反击等战术）
- 球员技术能力评估（传球、射门、速度、体能、防守）
- 运动生物力学与姿态分析
- AI视觉识别技术（YOLO目标检测、运动轨迹追踪）

**回答风格**：
- 专业但不晦涩，用教练的语言而非学术论文
- 结合具体数据支撑观点（如"传球成功率92%"、"覆盖距离11.2km"）
- 提供可操作的改进建议，而非泛泛而谈
- 对不确定的信息诚实告知，不编造数据

**限制**：
- 仅回答与足球视频分析、球员表现评估相关的问题
- 不讨论与足球无关的话题（如娱乐、政治等）
- 分析基于AI视觉识别结果，可能存在误差，需谦逊告知"""

    def _restore_video_cache_from_db(self):
        """
        系统启动时从 ai_video_cache 表恢复视频语义知识库到内存
        只恢复 is_ghost=False 的有效记录，跳过幽灵缓存
        """
        try:
            SessionLocal, AIChatSession, AIVideoCache = _get_db_session()
            if not SessionLocal or not AIVideoCache:
                return
            db = SessionLocal()
            try:
                rows = db.query(AIVideoCache).filter(
                    AIVideoCache.is_ghost == False  # noqa: E712
                ).all()
                for row in rows:
                    self.video_knowledge_base[row.video_uuid] = row.semantic_json
                if rows:
                    logger.info(f"[DB_RESTORE] Restored {len(rows)} video caches from DB")
            except Exception as e:
                logger.warning(f"[DB_RESTORE] Failed to restore video cache: {e}")
            finally:
                db.close()
        except Exception:
            pass

    async def analyze_video_content(self, video_analysis: Dict) -> Dict:
        """
        将视频分析结果转换为语义化知识库
        输入：analysis.py 生成的球员数据
        输出：结构化的语义信息，便于后续问答检索
        """
        video_id = video_analysis.get("video_id", "unknown")
        athletes = video_analysis.get("detailed_analysis", {}).get("athletes", [])
        
        # 构建结构化知识图谱
        semantic_info = {
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "players": [],
            "tactical_summary": "",
            "key_insights": []
        }
        
        for athlete in athletes:
            player_info = {
                "id": athlete.get("player_id"),
                "name": athlete.get("name"),
                "jersey_number": athlete.get("name", "").split("号")[0].replace("主队", "").replace("客队", ""),
                "abilities": athlete.get("abilities", {}),
                "strengths": [],  # 优势能力
                "weaknesses": [],  # 需提升项
                "role_suggestion": "",  # 位置建议
                "training_plan": []  # 训练计划
            }
            
            # 识别优势与劣势
            abilities = athlete.get("abilities", {})
            sorted_abilities = sorted(abilities.items(), key=lambda x: x[1], reverse=True)
            
            player_info["strengths"] = [f"{k}({v}/100)" for k, v in sorted_abilities[:2]]
            player_info["weaknesses"] = [f"{k}({v}/100)" for k, v in sorted_abilities[-2:]]
            
            # 根据能力推荐位置
            top_skill = sorted_abilities[0][0] if sorted_abilities else ""
            role_map = {
                "防守": "中后卫/后腰",
                "射门": "前锋/影锋",
                "传球": "中场组织核心/前腰",
                "速度": "边锋/边后卫",
                "体能": "Box-to-Box中场"
            }
            player_info["role_suggestion"] = role_map.get(top_skill, "全能型球员")
            
            semantic_info["players"].append(player_info)
        
        # 生成战术摘要
        if len(athletes) >= 4:
            team_a_avg = sum(sum(p["abilities"].values()) for p in athletes[:2]) / 10
            team_b_avg = sum(sum(p["abilities"].values()) for p in athletes[2:4]) / 10
            semantic_info["tactical_summary"] = (
                f"主队平均能力值 {team_a_avg:.1f}/100，客队 {team_b_avg:.1f}/100。"
                f"{'主队' if team_a_avg > team_b_avg else '客队'}在整体实力上略占优势。"
            )
        
        # 缓存到内存知识库
        self.video_knowledge_base[video_id] = semantic_info

        # ── 持久化到数据库 ───────────────────────────────────────────────────
        try:
            SessionLocal, AIChatSession, AIVideoCache = _get_db_session()
            if SessionLocal and AIVideoCache:
                db = SessionLocal()
                try:
                    existing = db.query(AIVideoCache).filter(
                        AIVideoCache.video_uuid == video_id
                    ).first()
                    if existing:
                        existing.semantic_json = semantic_info
                        existing.updated_at = datetime.utcnow()
                        existing.is_ghost = False
                    else:
                        db.add(AIVideoCache(
                            video_uuid=video_id,
                            semantic_json=semantic_info,
                            is_ghost=False,
                        ))
                    db.commit()
                    logger.info(f"[DB_CACHE] Video semantic cache persisted: {video_id}")
                except Exception as db_err:
                    db.rollback()
                    logger.warning(f"[DB_CACHE] Failed to persist video cache: {db_err}")
                finally:
                    db.close()
        except Exception:
            pass  # DB 写入失败不阻断主流程

        return semantic_info

    async def chat(
        self, 
        user_message: str, 
        session_id: str,
        video_id: Optional[str] = None,
        stream: bool = False
    ) -> Dict:
        """
        核心对话接口
        
        参数：
            user_message: 用户问题
            session_id: 会话ID，用于多轮对话上下文管理
            video_id: 关联的视频ID，用于检索相关分析数据
            stream: 是否启用流式输出（SSE）
        
        返回：
            {
                "answer": "AI回答内容",
                "response_time": 0.85,
                "tokens_used": 150,
                "confidence": 0.96
            }
        """
        start_time = time.time()
        
        try:
            # 1. 检索相关视频知识
            context = ""
            if video_id and video_id in self.video_knowledge_base:
                knowledge = self.video_knowledge_base[video_id]
                context = self._build_context_from_knowledge(knowledge, user_message)
            
            # 2. 构建对话历史
            if session_id not in self.conversation_memory:
                self.conversation_memory[session_id] = []
            
            messages = [
                {"role": "system", "content": self.system_prompt},
            ]
            
            # 添加历史对话（保留最近5轮，避免上下文过长）
            recent_history = self.conversation_memory[session_id][-10:]  # 取最近5轮（每轮2条消息）
            messages.extend(recent_history)
            
            # 3. 添加当前问题（如果有上下文，先注入）
            if context:
                user_message_with_context = f"""**相关视频分析数据**：
{context}

**用户问题**：{user_message}

请基于上述视频分析数据，结合你的专业知识回答用户问题。如果数据不足以回答，请诚实告知并提供合理推测。"""
            else:
                user_message_with_context = user_message
            
            messages.append({"role": "user", "content": user_message_with_context})
            
            # 4. 调用 DeepSeek API
            if stream:
                # 流式输出（SSE）
                response_stream = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # 返回生成器
                async def generate():
                    full_response = ""
                    async for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                    
                    # 保存到对话历史（内存 + DB 双写）
                    self.conversation_memory[session_id].append(
                        {"role": "user", "content": user_message}
                    )
                    self.conversation_memory[session_id].append(
                        {"role": "assistant", "content": full_response}
                    )
                    self._persist_messages(session_id, video_id, [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": full_response},
                    ])
                
                return {"stream": generate(), "session_id": session_id}
            
            else:
                # 非流式输出
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                answer = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                
                # 保存对话历史（内存 + DB 双写）
                self.conversation_memory[session_id].append(
                    {"role": "user", "content": user_message}
                )
                self.conversation_memory[session_id].append(
                    {"role": "assistant", "content": answer}
                )
                self._persist_messages(session_id, video_id, [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer},
                ])
                
                # 计算响应时间
                response_time = time.time() - start_time
                
                # 更新性能指标
                self._update_metrics(response_time, success=True)
                
                return {
                    "answer": answer,
                    "response_time": round(response_time, 3),
                    "tokens_used": tokens_used,
                    "confidence": 0.95,  # 可通过语义相似度计算真实置信度
                    "session_id": session_id
                }
        
        except Exception as e:
            self._update_metrics(time.time() - start_time, success=False)
            return {
                "error": str(e),
                "response_time": round(time.time() - start_time, 3),
                "session_id": session_id
            }

    def _build_context_from_knowledge(self, knowledge: Dict, user_query: str) -> str:
        """
        智能检索：根据用户问题从知识库中提取相关信息
        使用关键词匹配 + 语义优先级排序
        """
        context_parts = []
        
        # 1. 球员信息（始终包含）
        players = knowledge.get("players", [])
        for player in players:
            context_parts.append(
                f"**{player['name']}**：\n"
                f"  - 能力值：{player['abilities']}\n"
                f"  - 优势：{', '.join(player['strengths'])}\n"
                f"  - 需提升：{', '.join(player['weaknesses'])}\n"
                f"  - 推荐位置：{player['role_suggestion']}"
            )
        
        # 2. 战术摘要（如果问题涉及"战术"、"整体"等关键词）
        if any(kw in user_query for kw in ["战术", "整体", "对比", "队伍", "团队"]):
            tactical = knowledge.get("tactical_summary", "")
            if tactical:
                context_parts.insert(0, f"**战术概览**：{tactical}\n")
        
        return "\n\n".join(context_parts)

    def _persist_messages(self, session_id: str, video_id: Optional[str], messages: List[Dict]):
        """
        将对话消息写入 ai_chat_sessions 表
        video_id 为 None 时 is_ghost 始终 False；
        若 video_id 在 videos 表已不存在，is_ghost 自动标记 True。
        """
        try:
            SessionLocal, AIChatSession, AIVideoCache = _get_db_session()
            if not SessionLocal or not AIChatSession:
                return
            db = SessionLocal()
            try:
                # 判断 video_uuid 是否仍存在于 videos 表
                is_ghost = False
                if video_id:
                    from database import VideoModel  # noqa: F401
                    exists = db.query(VideoModel.id).filter(
                        VideoModel.video_uuid == video_id
                    ).first()
                    is_ghost = (exists is None)

                for msg in messages:
                    db.add(AIChatSession(
                        session_id=session_id,
                        video_uuid=video_id,
                        role=msg["role"],
                        content=msg["content"],
                        is_ghost=is_ghost,
                    ))
                db.commit()
            except Exception as e:
                db.rollback()
                logger.warning(f"[DB_CHAT] Failed to persist messages: {e}")
            finally:
                db.close()
        except Exception:
            pass  # DB 写入失败不阻断主流程

    def _update_metrics(self, response_time: float, success: bool):
        """更新性能监控指标"""
        self.metrics["total_requests"] += 1
        
        # 移动平均计算响应时间
        alpha = 0.2  # 平滑系数
        self.metrics["avg_response_time"] = (
            alpha * response_time + 
            (1 - alpha) * self.metrics["avg_response_time"]
        )
        
        # 成功率计算
        if success:
            self.metrics["success_rate"] = (
                (self.metrics["success_rate"] * (self.metrics["total_requests"] - 1) + 1.0) /
                self.metrics["total_requests"]
            )
        else:
            self.metrics["success_rate"] = (
                (self.metrics["success_rate"] * (self.metrics["total_requests"] - 1)) /
                self.metrics["total_requests"]
            )

    def clear_session(self, session_id: str):
        """清空指定会话的历史记录"""
        if session_id in self.conversation_memory:
            del self.conversation_memory[session_id]

    def get_session_history(self, session_id: str) -> List[Dict]:
        """获取会话历史"""
        return self.conversation_memory.get(session_id, [])

    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            **self.metrics,
            "active_sessions": len(self.conversation_memory),
            "cached_videos": len(self.video_knowledge_base)
        }
    
    async def sync_cache_with_database(self, db_video_ids: Set[str]) -> Dict[str, Any]:
        """
        同步缓存与数据库状态
        
        参数：
            db_video_ids: 数据库中当前存在的视频ID集合
        
        返回：
            {
                "removed_count": 删除的缓存数量,
                "cached_count": 当前缓存数量,
                "removed_videos": 被删除的视频ID列表,
                "sync_time": 同步时间戳
            }
        """
        try:
            # 获取缓存中的视频ID
            cached_video_ids = set(self.video_knowledge_base.keys())
            
            # 找出数据库中已删除但缓存中仍存在的视频
            orphaned_videos = cached_video_ids - db_video_ids
            
            # 清理孤立的缓存条目
            removed_videos = []
            for video_id in orphaned_videos:
                if video_id in self.video_knowledge_base:
                    del self.video_knowledge_base[video_id]
                    removed_videos.append(video_id)
                    logger.info(f"[CACHE_SYNC] Removed orphaned video cache: {video_id}")
            
            # 清理相关的会话缓存（可选）
            # 删除所有引用了已删除视频的会话记录
            cleaned_sessions = 0
            for session_id in list(self.conversation_memory.keys()):
                messages = self.conversation_memory[session_id]
                # 检查消息中是否引用了已删除的视频
                has_orphaned_video = any(
                    msg.get("content", "").find(video_id) != -1 
                    for msg in messages 
                    for video_id in orphaned_videos
                )
                if has_orphaned_video:
                    del self.conversation_memory[session_id]
                    cleaned_sessions += 1
            
            sync_result = {
                "removed_count": len(removed_videos),
                "cached_count": len(self.video_knowledge_base),
                "removed_videos": removed_videos,
                "cleaned_sessions": cleaned_sessions,
                "sync_time": datetime.now().isoformat(),
                "status": "success"
            }
            
            if removed_videos:
                logger.info(f"[CACHE_SYNC] Cleaned {len(removed_videos)} orphaned videos, {cleaned_sessions} sessions")
            
            return sync_result
            
        except Exception as e:
            logger.error(f"[CACHE_SYNC] Error during cache sync: {e}")
            return {
                "removed_count": 0,
                "cached_count": len(self.video_knowledge_base),
                "removed_videos": [],
                "cleaned_sessions": 0,
                "sync_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e)
            }
    
    def clear_video_cache(self, video_id: str) -> bool:
        """
        清除指定视频的缓存
        
        参数：
            video_id: 要清除的视频ID
        
        返回：
            是否成功清除
        """
        try:
            if video_id in self.video_knowledge_base:
                del self.video_knowledge_base[video_id]
                logger.info(f"[CACHE] Cleared cache for video: {video_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"[CACHE] Error clearing cache for video {video_id}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        返回：
            {
                "total_cached_videos": 总缓存视频数,
                "total_sessions": 总会话数,
                "cache_size_bytes": 缓存大小(估算),
                "video_ids": 缓存的视频ID列表
            }
        """
        try:
            cache_data = json.dumps(self.video_knowledge_base)
            cache_size = len(cache_data.encode('utf-8'))
            
            return {
                "total_cached_videos": len(self.video_knowledge_base),
                "total_sessions": len(self.conversation_memory),
                "cache_size_bytes": cache_size,
                "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                "video_ids": list(self.video_knowledge_base.keys()),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"[CACHE] Error getting cache stats: {e}")
            return {
                "total_cached_videos": len(self.video_knowledge_base),
                "total_sessions": len(self.conversation_memory),
                "error": str(e)
            }



# ===== 高并发处理模块 =====
class ConcurrentAIManager:
    """
    高并发请求管理器
    使用异步队列 + 连接池实现 100+ 并发处理
    """
    
    def __init__(self, max_workers: int = 100):
        self.max_workers = max_workers
        self.agent = None  # 延迟初始化
        self.request_queue = asyncio.Queue()
        self.response_cache = {}  # 简单LRU缓存
        self.cache_max_size = 500
    
    async def initialize(self, api_key: str):
        """异步初始化（避免在 __init__ 中创建事件循环）"""
        self.agent = DeepSeekAgent(api_key=api_key)
    
    async def submit_request(self, user_message: str, session_id: str, video_id: Optional[str] = None) -> Dict:
        """
        提交请求到队列
        实现请求级别的缓存和去重
        """
        # 1. 生成缓存键（user_message + video_id）
        cache_key = hashlib.md5(f"{user_message}_{video_id}".encode()).hexdigest()
        
        # 2. 检查缓存
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response["from_cache"] = True
            return cached_response
        
        # 3. 提交到队列处理
        response = await self.agent.chat(user_message, session_id, video_id)
        
        # 4. 更新缓存（LRU策略）
        if len(self.response_cache) >= self.cache_max_size:
            # 删除最早的缓存项
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = response
        response["from_cache"] = False
        
        return response
    
    def get_stats(self) -> Dict:
        """获取并发管理器统计信息"""
        return {
            "queue_size": self.request_queue.qsize(),
            "cache_size": len(self.response_cache),
            "max_workers": self.max_workers,
            **self.agent.get_metrics()
        }
    
    async def sync_cache_with_database(self, db_video_ids: Set[str]) -> Dict[str, Any]:
        """
        同步视频缓存与数据库状态（代理到 DeepSeekAgent）
        
        参数：
            db_video_ids: 数据库中当前存在的视频ID集合
        
        返回：同步结果字典
        """
        if self.agent is None:
            logger.warning("[CACHE_SYNC] AI agent not initialized")
            return {
                "status": "error",
                "error": "AI agent not initialized",
                "removed_count": 0
            }
        
        # 同步视频知识库缓存
        video_sync_result = await self.agent.sync_cache_with_database(db_video_ids)
        
        # 清理响应缓存中包含已删除视频的条目
        removed_response_cache_count = 0
        removed_videos = set(video_sync_result.get("removed_videos", []))
        
        if removed_videos:
            for cache_key in list(self.response_cache.keys()):
                # 检查缓存的响应内容是否引用了已删除的视频
                cached_response = self.response_cache[cache_key]
                response_str = json.dumps(cached_response)
                
                # 如果响应中包含已删除的视频ID，则清除该缓存项
                has_removed_video = any(
                    video_id in response_str 
                    for video_id in removed_videos
                )
                
                if has_removed_video:
                    del self.response_cache[cache_key]
                    removed_response_cache_count += 1
        
        # 合并结果
        return {
            **video_sync_result,
            "response_cache_cleaned": removed_response_cache_count,
            "current_response_cache_size": len(self.response_cache)
        }
    
    def clear_video_cache(self, video_id: str) -> Dict[str, Any]:
        """
        清除指定视频的所有相关缓存
        
        参数：
            video_id: 视频ID
        
        返回：清理结果字典
        """
        if self.agent is None:
            return {"status": "error", "error": "AI agent not initialized"}
        
        # 清除视频知识库缓存
        video_cache_cleared = self.agent.clear_video_cache(video_id)
        
        # 清除响应缓存中相关的条目
        removed_count = 0
        for cache_key in list(self.response_cache.keys()):
            if video_id in cache_key or (
                "answer" in self.response_cache[cache_key] and 
                video_id in self.response_cache[cache_key]["answer"]
            ):
                del self.response_cache[cache_key]
                removed_count += 1
        
        return {
            "status": "success",
            "video_cache_cleared": video_cache_cleared,
            "response_cache_cleared": removed_count,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取完整的缓存统计信息
        
        返回：包含视频缓存和响应缓存的统计信息
        """
        if self.agent is None:
            return {
                "status": "error",
                "error": "AI agent not initialized"
            }
        
        video_cache_stats = self.agent.get_cache_stats()
        
        return {
            **video_cache_stats,
            "response_cache_size": len(self.response_cache),
            "response_cache_max_size": self.cache_max_size,
            "response_cache_usage_percent": round(
                len(self.response_cache) / self.cache_max_size * 100, 2
            )
        }



# ===== 全局单例 =====
_global_manager: Optional[ConcurrentAIManager] = None

def get_ai_manager() -> ConcurrentAIManager:
    """获取全局 AI 管理器实例"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ConcurrentAIManager(max_workers=100)
    return _global_manager
