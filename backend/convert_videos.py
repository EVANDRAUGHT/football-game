"""
将现有的 FMP4 视频转换为浏览器兼容的 H.264 格式
"""
import os
import subprocess
import sys

EXPORT_DIR = os.path.join(os.path.dirname(__file__), "uploads", "exports")

def check_ffmpeg():
    """检查 ffmpeg 是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            print("[OK] ffmpeg 可用")
            return True
    except:
        pass
    
    print("[X] ffmpeg 未安装或不在 PATH 中")
    print("\n请安装 ffmpeg:")
    print("  1. 下载: https://ffmpeg.org/download.html")
    print("  2. 添加到系统 PATH")
    print("  3. 或使用: choco install ffmpeg (需要 Chocolatey)")
    return False

def convert_video(input_path, output_path):
    """使用 ffmpeg 转换视频"""
    print(f"\n转换: {os.path.basename(input_path)}")
    print(f"输出: {os.path.basename(output_path)}")
    
    # ffmpeg 命令：使用 H.264 编码 + AAC 音频
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',      # H.264 视频编码
        '-acodec', 'aac',           # AAC 音频编码
        '-preset', 'fast',          # 编码速度（fast/medium/slow）
        '-crf', '23',               # 质量（18-28，越小质量越好）
        '-movflags', '+faststart',  # 优化网络播放
        '-y',                       # 覆盖已存在文件
        output_path
    ]
    
    try:
        print("执行转换...")
        result = subprocess.run(cmd, 
                              capture_output=True, 
                              text=True,
                              timeout=300)  # 5分钟超时
        
        if result.returncode == 0:
            # 检查输出文件
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / 1024 / 1024
                print(f"[OK] 转换成功！大小: {size_mb:.1f} MB")
                return True
            else:
                print("[X] 转换失败：输出文件不存在")
                return False
        else:
            print(f"[X] 转换失败")
            print(f"错误信息: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[X] 转换超时（超过5分钟）")
        return False
    except Exception as e:
        print(f"[X] 转换出错: {e}")
        return False

def main():
    print("="*70)
    print("视频格式转换工具 (FMP4 → H.264)")
    print("="*70)
    
    if not check_ffmpeg():
        return
    
    if not os.path.exists(EXPORT_DIR):
        print(f"\n[X] 导出目录不存在: {EXPORT_DIR}")
        return
    
    # 查找所有 MP4 文件
    videos = [f for f in os.listdir(EXPORT_DIR) 
              if f.endswith('.mp4') and not f.endswith('_h264.mp4')]
    
    if not videos:
        print("\n[X] 未找到需要转换的视频")
        return
    
    print(f"\n找到 {len(videos)} 个视频文件")
    
    success_count = 0
    for video in videos:
        input_path = os.path.join(EXPORT_DIR, video)
        output_filename = video.replace('.mp4', '_h264.mp4')
        output_path = os.path.join(EXPORT_DIR, output_filename)
        
        if convert_video(input_path, output_path):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"转换完成: {success_count}/{len(videos)} 成功")
    print("="*70)
    
    if success_count > 0:
        print("\n✅ 转换后的视频可以在浏览器中播放")
        print("原始视频已保留，如需删除请手动操作")
        print("\n新视频文件命名格式: *_h264.mp4")

if __name__ == "__main__":
    main()
