# debug_env.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, dotenv_values

print("=" * 60)
print("🔍 环境变量诊断工具")
print("=" * 60)

# 1. 检查当前工作目录
print(f"\n📁 当前工作目录: {os.getcwd()}")

# 2. 检查项目根目录
project_root = Path(__file__).parent
print(f"📁 项目根目录: {project_root}")

# 3. 检查 .env 文件
env_path = project_root / '.env'
print(f"\n📄 .env 文件路径: {env_path}")
print(f"✅ .env 文件存在: {env_path.exists()}")

if env_path.exists():
    # 4. 显示 .env 文件内容（隐藏部分密钥）
    print("\n📝 .env 文件内容:")
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'API_KEY' in line:
                    key, value = line.split('=', 1)
                    visible_part = value[:10] + '...' + value[-5:] if len(value) > 15 else value
                    print(f"  {key}={visible_part}")
                else:
                    print(f"  {line}")

    # 5. 使用 dotenv_values 读取
    print("\n🔄 dotenv_values 读取结果:")
    env_vars = dotenv_values(env_path)
    for key, value in env_vars.items():
        if 'API_KEY' in key:
            visible = value[:10] + '...' + value[-5:] if value and len(value) > 15 else value
            print(f"  {key}={visible}")
        else:
            print(f"  {key}={value}")

    # 6. 强制加载到环境变量
    print("\n🔄 强制加载到 os.environ:")
    load_dotenv(env_path, override=True)
    for key in ['OPENAI_COMPATIBLE__API_KEY', 'OPENAI_COMPATIBLE__BASE_URL',
                'OPENAI_COMPATIBLE__DEFAULT_MODEL', 'DEFAULT_PROVIDER']:
        value = os.getenv(key)
        if value:
            if 'API_KEY' in key:
                visible = value[:10] + '...' + value[-5:] if len(value) > 15 else value
                print(f"  {key}={visible}")
            else:
                print(f"  {key}={value}")
        else:
            print(f"  {key}=❌ 未找到")

# 7. 尝试导入 settings
print("\n🔄 尝试导入 settings...")
try:
    # 添加 src 到路径
    src_path = project_root / 'src'
    if src_path.exists():
        sys.path.insert(0, str(project_root))

    from src.langchain_v1.config import settings

    print("✅ settings 导入成功!")
    print(f"  DEFAULT_PROVIDER: {settings.DEFAULT_PROVIDER}")
    print(
        f"  openai_compatible.api_key: {settings.openai_compatible.api_key[:10]}...{settings.openai_compatible.api_key[-5:] if settings.openai_compatible.api_key else 'None'}")
    print(f"  openai_compatible.base_url: {settings.openai_compatible.base_url}")
    print(f"  openai_compatible.default_model: {settings.openai_compatible.default_model}")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback

    traceback.print_exc()