"""基础环境配置加载模块"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"


def load_environment():
    """加载环境变量"""
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
        print(f"✅ 已加载环境变量从: {ENV_FILE}")
    else:
        print(f"⚠️ 未找到 .env 文件，使用系统环境变量")
        load_dotenv()  # 尝试从系统环境变量加载


# 配置常量
class Config:
    """应用配置类"""

    # OpenAI 配置
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    # Google 配置
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

    # LangSmith 配置
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "default")

    # 数据库配置
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(ROOT_DIR / "data" / "chroma_db"))

    # API 配置
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

    # 应用配置
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# 初始化加载
load_environment()