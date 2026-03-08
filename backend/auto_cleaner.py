"""
自动数据库清理模块
功能：
1. 启动时清理重复记录
2. 定期后台清理（可选）
3. API触发清理
"""

import asyncio
from collections import defaultdict
from typing import Dict, List, Set
from sqlalchemy.orm import Session
from database import SessionLocal, VideoModel, AthleteProfile
import os

class DatabaseCleaner:
    """数据库自动清理器"""
    
    def __init__(self):
        self.last_clean_time = None
        self.clean_interval = 3600  # 🔑 修改: 延长到60分钟(3600秒)，避免频繁清理
        self.is_cleaning = False
        
    def clean_duplicates(self, db: Session) -> Dict:
        """清理重复的video_uuid记录"""
        if self.is_cleaning:
            return {"status": "skipped", "message": "Cleaning already in progress"}
        
        self.is_cleaning = True
        
        try:
            # 1. 查找重复的video_uuid
            all_videos = db.query(VideoModel).all()
            
            uuid_count = defaultdict(list)
            for video in all_videos:
                uuid_count[video.video_uuid].append(video)
            
            duplicates = {uuid: videos for uuid, videos in uuid_count.items() if len(videos) > 1}
            
            if not duplicates:
                return {
                    "status": "success",
                    "removed_count": 0,
                    "message": "No duplicates found"
                }
            
            # 2. 删除重复记录（保留最新的）
            removed_count = 0
            removed_details = []
            
            for uuid, videos in duplicates.items():
                # 按上传时间排序，保留最新的
                videos.sort(key=lambda v: v.upload_time if v.upload_time else "", reverse=True)
                keep_video = videos[0]
                remove_videos = videos[1:]
                
                for video in remove_videos:
                    # 删除关联的分析记录
                    profiles = db.query(AthleteProfile).filter(
                        AthleteProfile.video_id == video.id
                    ).all()
                    
                    for profile in profiles:
                        db.delete(profile)
                    
                    removed_details.append({
                        "id": video.id,
                        "uuid": video.video_uuid[:8],
                        "filename": video.filename
                    })
                    
                    db.delete(video)
                    removed_count += 1
            
            db.commit()
            
            print(f"[AUTO_CLEAN] Removed {removed_count} duplicate records")
            
            return {
                "status": "success",
                "removed_count": removed_count,
                "removed_details": removed_details,
                "message": f"Cleaned {removed_count} duplicate records"
            }
            
        except Exception as e:
            db.rollback()
            print(f"[AUTO_CLEAN] Error: {e}")
            return {
                "status": "error",
                "message": str(e),
                "removed_count": 0
            }
        finally:
            self.is_cleaning = False
    
    def clean_orphaned_videos(self, db: Session) -> Dict:
        """清理孤立视频（无分析结果）"""
        try:
            # 🔑 关键修复: 导入 progress_store 以检查正在分析的视频
            from analysis import progress_store
            
            orphan_videos = db.query(VideoModel).outerjoin(
                AthleteProfile, VideoModel.id == AthleteProfile.video_id
            ).filter(AthleteProfile.id == None).all()
            
            if not orphan_videos:
                return {
                    "status": "success",
                    "removed_count": 0,
                    "message": "No orphaned videos found"
                }
            
            # 🔑 关键修复: 过滤掉正在分析中的视频
            videos_to_delete = []
            skipped_analyzing = []
            
            for video in orphan_videos:
                # 检查是否正在分析中
                progress = progress_store.get(video.video_uuid, None)
                
                if progress is not None and 0 <= progress < 100:
                    # 正在分析中，跳过删除
                    skipped_analyzing.append(video.video_uuid)
                    print(f"[AUTO_CLEAN] Skipping video {video.video_uuid} (analyzing: {progress}%)")
                else:
                    # 真正的孤立记录（分析失败或已放弃）
                    videos_to_delete.append(video)
            
            if not videos_to_delete:
                return {
                    "status": "success",
                    "removed_count": 0,
                    "message": f"No orphaned videos to clean (skipped {len(skipped_analyzing)} analyzing videos)"
                }
            
            removed_count = len(videos_to_delete)
            for video in videos_to_delete:
                db.delete(video)
            
            db.commit()
            
            print(f"[AUTO_CLEAN] Removed {removed_count} orphaned videos (skipped {len(skipped_analyzing)} analyzing)")
            
            return {
                "status": "success",
                "removed_count": removed_count,
                "skipped_count": len(skipped_analyzing),
                "message": f"Cleaned {removed_count} orphaned videos (skipped {len(skipped_analyzing)} analyzing)"
            }
            
        except Exception as e:
            db.rollback()
            print(f"[AUTO_CLEAN] Error cleaning orphaned videos: {e}")
            return {
                "status": "error",
                "message": str(e),
                "removed_count": 0
            }
    
    def clean_failed_videos(self, db: Session) -> Dict:
        """清理分析失败的视频"""
        try:
            failed_videos = []
            
            all_videos = db.query(VideoModel).join(
                AthleteProfile, VideoModel.id == AthleteProfile.video_id
            ).all()
            
            for video in all_videos:
                profile = db.query(AthleteProfile).filter(
                    AthleteProfile.video_id == video.id
                ).first()
                
                if profile and profile.detailed_analysis:
                    detailed = profile.detailed_analysis
                    if isinstance(detailed, dict) and detailed.get('error'):
                        failed_videos.append((video, profile))
            
            if not failed_videos:
                return {
                    "status": "success",
                    "removed_count": 0,
                    "message": "No failed videos found"
                }
            
            removed_count = len(failed_videos)
            for video, profile in failed_videos:
                db.delete(profile)
                db.delete(video)
            
            db.commit()
            
            print(f"[AUTO_CLEAN] Removed {removed_count} failed videos")
            
            return {
                "status": "success",
                "removed_count": removed_count,
                "message": f"Cleaned {removed_count} failed videos"
            }
            
        except Exception as e:
            db.rollback()
            print(f"[AUTO_CLEAN] Error cleaning failed videos: {e}")
            return {
                "status": "error",
                "message": str(e),
                "removed_count": 0
            }
    
    def full_clean(self, db: Session) -> Dict:
        """执行完整清理"""
        print("[AUTO_CLEAN] Starting full database cleanup...")
        
        results = {
            "duplicates": self.clean_duplicates(db),
            "orphaned": self.clean_orphaned_videos(db),
            "failed": self.clean_failed_videos(db)
        }
        
        total_removed = (
            results["duplicates"].get("removed_count", 0) +
            results["orphaned"].get("removed_count", 0) +
            results["failed"].get("removed_count", 0)
        )
        
        print(f"[AUTO_CLEAN] Full cleanup completed: {total_removed} records removed")
        
        return {
            "status": "success",
            "total_removed": total_removed,
            "details": results
        }

# 全局清理器实例
_cleaner = None

def get_cleaner() -> DatabaseCleaner:
    """获取清理器单例"""
    global _cleaner
    if _cleaner is None:
        _cleaner = DatabaseCleaner()
    return _cleaner

async def auto_clean_on_startup():
    """启动时自动清理"""
    print("[AUTO_CLEAN] Running startup cleanup...")
    db = SessionLocal()
    try:
        cleaner = get_cleaner()
        result = cleaner.full_clean(db)
        print(f"[AUTO_CLEAN] Startup cleanup completed: {result.get('total_removed', 0)} records removed")
    except Exception as e:
        print(f"[AUTO_CLEAN] Startup cleanup failed: {e}")
    finally:
        db.close()

async def periodic_clean_task():
    """后台定期清理任务"""
    cleaner = get_cleaner()
    
    while True:
        try:
            await asyncio.sleep(cleaner.clean_interval)  # 🔑 修改: 从5分钟延长到10分钟
            
            print("[AUTO_CLEAN] Running periodic cleanup...")
            db = SessionLocal()
            try:
                result = cleaner.full_clean(db)
                if result.get('total_removed', 0) > 0:
                    print(f"[AUTO_CLEAN] Periodic cleanup: {result['total_removed']} records removed")
            finally:
                db.close()
                
        except asyncio.CancelledError:
            print("[AUTO_CLEAN] Periodic cleanup task cancelled")
            break
        except Exception as e:
            print(f"[AUTO_CLEAN] Periodic cleanup error: {e}")
