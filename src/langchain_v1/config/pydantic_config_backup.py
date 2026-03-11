"""基于 Pydantic Settings 的配置管理"""
from typing import Optional, Literal
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """应用配置模型"""

    # 模型配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # OpenAI 配置
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(0.0, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: Optional[int] = Field(None, alias="OPENAI_MAX_TOKENS")

    # Google 配置
    google_api_key: str = Field("", alias="GOOGLE_API_KEY")
    google_model: str = Field("gemini-2.0-flash", alias="GOOGLE_MODEL")

    # 默认 LLM 提供商
    default_llm_provider: Literal["openai", "google"] = Field(
        "openai",
        alias="DEFAULT_LLM_PROVIDER"
    )

    # LangSmith 配置
    langsmith_tracing: bool = Field(False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field("", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field("default", alias="LANGSMITH_PROJECT")

    # 数据库配置
    chroma_persist_dir: Path = Field(
        Path("./data/chroma_db"),
        alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        "langchain_docs",
        alias="CHROMA_COLLECTION_NAME"
    )

    # API 配置
    api_timeout: int = Field(30, alias="API_TIMEOUT")
    max_retries: int = Field(3, alias="MAX_RETRIES")
    retry_delay: float = Field(1.0, alias="RETRY_DELAY")

    # 应用配置
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # 数据路径
    data_dir: Path = Field(Path("./data"), alias="DATA_DIR")
    upload_dir: Path = Field(Path("./uploads"), alias="UPLOAD_DIR")

    @field_validator("chroma_persist_dir", "data_dir", "upload_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        """确保路径是 Path 对象"""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("chroma_persist_dir", "data_dir", "upload_dir")
    @classmethod
    def create_dirs(cls, v: Path):
        """自动创建目录"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def model_post_init(self, __context):
        """初始化后的处理"""
        # 如果开启了 LangSmith 追踪，设置环境变量
        if self.langsmith_tracing:
            import os
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
            os.environ["LANGSMITH_PROJECT"] = self.langsmith_project


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 导出配置实例
settings = get_settings()