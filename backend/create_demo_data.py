"""
创建演示数据 - 为AI智能助手准备一条完整的视频分析数据
"""

import sys
import os
import io

# 设置标准输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database import VideoModel, AthleteProfile, Base
from datetime import datetime
import uuid

# 数据库文件路径 —— 与 database.py 保持一致，固定在 backend/ 目录
# 历史版本曾错误地指向 '../football_demo.db'（项目根目录），已修正
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'football_demo.db')
DATABASE_URL = f"sqlite:///{DB_PATH}"

def create_demo_video_data():
    """创建一条完整的演示视频数据"""
    
    print("=" * 60)
    print("  创建演示数据 - AI 智能助手专用")
    print("=" * 60)
    print()
    
    # 连接数据库
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(bind=engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # 检查是否已有数据
        existing_videos = session.query(VideoModel).count()
        
        if existing_videos > 0:
            print(f"⚠️  数据库中已有 {existing_videos} 条视频记录")
            response = input("是否清空后重新创建演示数据？(yes/no): ")
            
            if response.lower() not in ['yes', 'y', '是']:
                print("❌ 已取消操作")
                return
            
            # 清空数据
            print("\n正在清空数据库...")
            session.query(AthleteProfile).delete()
            session.query(VideoModel).delete()
            session.commit()
            print("✅ 数据库已清空")
            print()
        
        # 创建演示视频记录
        print("[步骤 1/2] 创建演示视频记录...")
        
        demo_video_uuid = str(uuid.uuid4())
        demo_video = VideoModel(
            video_uuid=demo_video_uuid,
            filename="足球比赛精彩片段-演示.mp4",
            upload_time=datetime.now()
        )
        
        session.add(demo_video)
        session.flush()  # 获取自动生成的 ID
        
        print(f"   ✓ 视频 ID: {demo_video.id}")
        print(f"   ✓ 视频 UUID: {demo_video_uuid}")
        print(f"   ✓ 文件名: {demo_video.filename}")
        print()
        
        # 创建详细的分析数据
        print("[步骤 2/2] 创建球员分析数据...")
        
        # 构建详细的分析结果（4名球员）
        detailed_analysis = {
            "video_duration": 180,  # 3分钟
            "total_frames": 5400,   # 30fps
            "analyzed_frames": 5400,
            "confidence": 0.96,
            "analysis_time": 45.8,
            "athletes": [
                {
                    "player_id": 1,
                    "name": "7号前锋",
                    "jersey_number": 7,
                    "position": "前锋",
                    "abilities": {
                        "shooting": 88,
                        "passing": 75,
                        "dribbling": 82,
                        "defense": 65,
                        "speed": 90,
                        "stamina": 85
                    },
                    "path_accuracy": 0.95,
                    "key_moments": [
                        {"time": 15.2, "action": "射门", "success": True},
                        {"time": 45.8, "action": "突破", "success": True},
                        {"time": 120.5, "action": "助攻", "success": True}
                    ],
                    "performance_summary": "表现出色的前锋球员，射门精准，突破能力强，速度优势明显。建议加强防守意识和体能储备。"
                },
                {
                    "player_id": 2,
                    "name": "10号中场",
                    "jersey_number": 10,
                    "position": "中场",
                    "abilities": {
                        "shooting": 78,
                        "passing": 92,
                        "dribbling": 85,
                        "defense": 72,
                        "speed": 78,
                        "stamina": 88
                    },
                    "path_accuracy": 0.94,
                    "key_moments": [
                        {"time": 30.5, "action": "传球", "success": True},
                        {"time": 75.2, "action": "组织进攻", "success": True},
                        {"time": 150.8, "action": "远射", "success": False}
                    ],
                    "performance_summary": "核心中场球员，传球精准度极高，组织能力出众。建议提升射门能力和速度，进一步完善攻防转换。"
                },
                {
                    "player_id": 3,
                    "name": "5号后卫",
                    "jersey_number": 5,
                    "position": "后卫",
                    "abilities": {
                        "shooting": 60,
                        "passing": 78,
                        "dribbling": 68,
                        "defense": 92,
                        "speed": 75,
                        "stamina": 90
                    },
                    "path_accuracy": 0.93,
                    "key_moments": [
                        {"time": 20.3, "action": "拦截", "success": True},
                        {"time": 60.7, "action": "解围", "success": True},
                        {"time": 135.2, "action": "头球", "success": True}
                    ],
                    "performance_summary": "防守核心球员，拦截和解围能力优秀，体能充沛。建议提升进攻参与度，加强传球精准度。"
                },
                {
                    "player_id": 4,
                    "name": "11号边锋",
                    "jersey_number": 11,
                    "position": "边锋",
                    "abilities": {
                        "shooting": 80,
                        "passing": 80,
                        "dribbling": 90,
                        "defense": 68,
                        "speed": 92,
                        "stamina": 82
                    },
                    "path_accuracy": 0.91,
                    "key_moments": [
                        {"time": 35.8, "action": "过人", "success": True},
                        {"time": 85.3, "action": "传中", "success": True},
                        {"time": 165.7, "action": "反击", "success": True}
                    ],
                    "performance_summary": "速度型边锋，盘带技术娴熟，突破能力强。建议提升体能和防守回追能力，优化传中质量。"
                }
            ],
            "team_statistics": {
                "possession": 58.5,
                "pass_accuracy": 85.2,
                "shots_on_target": 12,
                "total_shots": 18,
                "corners": 7,
                "fouls": 11
            },
            "tactical_analysis": {
                "formation": "4-3-3",
                "style": "快速反击 + 边路进攻",
                "strengths": ["速度优势", "传球精准", "组织能力"],
                "weaknesses": ["防守稳定性", "体能分配", "射门效率"]
            }
        }
        
        # 计算综合评分
        all_abilities = []
        for athlete in detailed_analysis['athletes']:
            all_abilities.extend(athlete['abilities'].values())
        
        overall_score = sum(all_abilities) / len(all_abilities)
        
        # 创建球员分析记录
        athlete_profile = AthleteProfile(
            video_id=demo_video.id,
            overall_score=round(overall_score, 2),
            decision_summary=f"识别到{len(detailed_analysis['athletes'])}名球员，综合评分{round(overall_score, 1)}/100。整体表现优秀，速度和传球是主要优势，防守稳定性需要改进。",
            detailed_analysis=detailed_analysis
        )
        
        session.add(athlete_profile)
        session.commit()
        
        print(f"   ✓ 识别球员数: {len(detailed_analysis['athletes'])}")
        print(f"   ✓ 综合评分: {round(overall_score, 1)}/100")
        print(f"   ✓ 分析置信度: {detailed_analysis['confidence'] * 100}%")
        print()
        
        # 显示球员详情
        print("   球员详情:")
        for athlete in detailed_analysis['athletes']:
            avg_ability = sum(athlete['abilities'].values()) / len(athlete['abilities'])
            print(f"   - {athlete['name']} ({athlete['position']}): 综合 {round(avg_ability, 1)}/100")
        
        print()
        print("=" * 60)
        print("✅ 演示数据创建成功！")
        print("=" * 60)
        print()
        print("📋 演示数据信息:")
        print(f"   视频文件: {demo_video.filename}")
        print(f"   视频 UUID: {demo_video_uuid}")
        print(f"   球员数量: {len(detailed_analysis['athletes'])}")
        print(f"   数据完整度: 100%")
        print()
        print("🚀 现在可以使用 AI 智能助手进行对话了！")
        print()
        print("启动步骤:")
        print("  1. cd backend")
        print("  2. python main.py")
        print("  3. 浏览器访问: http://127.0.0.1:9999/frontend/ai-chat.html")
        print("  4. 在视频选择框中选择演示视频")
        print("  5. 开始提问，例如：")
        print("     - \"分析一下7号球员的表现\"")
        print("     - \"哪个球员的传球能力最强？\"")
        print("     - \"球队的整体战术特点是什么？\"")
        print("     - \"如何提升10号球员的射门能力？\"")
        print()
        
    except Exception as e:
        print(f"❌ 创建演示数据失败: {e}")
        session.rollback()
        import traceback
        traceback.print_exc()
    
    finally:
        session.close()

def verify_demo_data():
    """验证演示数据是否创建成功"""
    engine = create_engine(DATABASE_URL, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        print()
        print("=" * 60)
        print("  验证演示数据")
        print("=" * 60)
        print()
        
        videos = session.query(VideoModel).all()
        
        if not videos:
            print("❌ 数据库为空，未找到演示数据")
            return False
        
        print(f"✓ 视频记录数: {len(videos)}")
        
        for video in videos:
            print(f"\n视频信息:")
            print(f"  - ID: {video.id}")
            print(f"  - UUID: {video.video_uuid}")
            print(f"  - 文件名: {video.filename}")
            print(f"  - 上传时间: {video.upload_time}")
            
            profiles = session.query(AthleteProfile).filter(
                AthleteProfile.video_id == video.id
            ).all()
            
            print(f"  - 分析记录数: {len(profiles)}")
            
            for profile in profiles:
                analysis = profile.detailed_analysis
                if analysis and isinstance(analysis, dict):
                    athletes = analysis.get('athletes', [])
                    print(f"  - 识别球员数: {len(athletes)}")
                    print(f"  - 综合评分: {profile.overall_score}/100")
                    print(f"  - 置信度: {analysis.get('confidence', 0) * 100}%")
        
        print()
        print("=" * 60)
        print("✅ 演示数据验证通过！")
        print("=" * 60)
        return True
        
    finally:
        session.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='演示数据管理工具')
    parser.add_argument('--create', action='store_true', help='创建演示数据')
    parser.add_argument('--verify', action='store_true', help='验证演示数据')
    
    args = parser.parse_args()
    
    if args.create:
        create_demo_video_data()
    elif args.verify:
        verify_demo_data()
    else:
        # 默认：创建演示数据
        create_demo_video_data()
