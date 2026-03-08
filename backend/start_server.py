"""
启动服务器脚本 - 带编码修复
确保在 Windows 环境下正确处理 UTF-8 编码
"""
import sys
import os

# Windows 编码修复：强制使用 UTF-8
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 60)
print("  Football Analysis AI System")
print("=" * 60)
print()
print("[INFO] Initializing server with UTF-8 encoding...")
print("[INFO] Server will start on http://127.0.0.1:9999")
print()

# 检查 API Key
api_key = os.getenv("DEEPSEEK_API_KEY", "")
if api_key:
    print("[INFO] DEEPSEEK_API_KEY: Configured")
else:
    print("[WARNING] DEEPSEEK_API_KEY: Not configured")
    print("[WARNING] AI chat function will be unavailable")

print()
print("-" * 60)
print()

# 启动主程序
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=9999, reload=False)
