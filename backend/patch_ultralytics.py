"""
临时补丁：修复 ultralytics 与 PyTorch 2.8+ 的兼容性
添加安全全局类到 torch 序列化白名单
"""
import sys
import os

# Windows 编码修复
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("  Ultralytics PyTorch 2.8+ Compatibility Patch")
print("=" * 60)
print()

try:
    import torch
    from ultralytics.nn.tasks import DetectionModel
    
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] Adding DetectionModel to safe globals...")
    
    # 添加 ultralytics 类到安全全局列表
    torch.serialization.add_safe_globals([DetectionModel])
    
    print("[OK] Patch applied!")
    print()
    print("Now try loading YOLO:")
    
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("[SUCCESS] YOLO model loaded!")
    print()
    
except Exception as e:
    print(f"[ERROR] Patch failed: {e}")
    print()
    print("Alternative solution:")
    print("  1. Downgrade PyTorch: pip install torch==2.5.1")
    print("  2. Or wait for ultralytics update")
    sys.exit(1)
