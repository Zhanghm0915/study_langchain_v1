"""基于 Pydantic Settings 的 LangChain 1.2 配置 - 优化版"""
from typing import Optional, Literal
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent.parent  # 这会得到 project_root/
ENV_FILE = ROOT_DIR / '.env'  # project_root/.env

# 调试信息（可以保留，但不会影响运行）
# print(f"📁 项目根目录: {ROOT_DIR}")
# print(f"📄 .env 文件路径: {ENV_FILE}")
# print(f"✅ .env 是否存在: {ENV_FILE.exists()}")

class ModelProvider(str, Enum):
    """支持的模型提供商"""
    OPENAI = "openai"  # 官方 OpenAI
    OPENAI_COMPATIBLE = "openai_compatible"  # OpenAI 兼容接口（阿里云、DeepSeek等）
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    OLLAMA = "ollama"  # 本地模型
    DEEPSEEK = "deepseek"  # DeepSeek（通过 OpenAI 兼容）


class OpenAIConfig(BaseSettings):
    """OpenAI 官方配置"""
    api_key: Optional[str] = Field(None, description="OpenAI API 密钥")
    base_url: Optional[str] = Field(None, description="OpenAI API 基础 URL")
    organization: Optional[str] = Field(None, description="OpenAI 组织 ID")
    default_model: str = Field("gpt-4o-mini", description="默认模型")
    timeout: int = Field(60, description="超时时间")
    max_retries: int = Field(3, description="最大重试次数")


class OpenAICompatibleConfig(BaseSettings):
    """OpenAI 兼容服务配置（阿里云、DeepSeek等）"""
    api_key: Optional[str] = Field(None, description="API 密钥")
    base_url: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1", description="API 基础 URL")
    default_model: str = Field("qwen-plus", description="默认模型")
    timeout: int = Field(120, description="超时时间")
    max_retries: int = Field(0, description="最大重试次数（兼容服务通常不支持重试）")
    provider_name: str = Field("aliyun", description="提供商名称：aliyun/deepseek/etc")


class AnthropicConfig(BaseSettings):
    """Anthropic 配置"""
    api_key: Optional[str] = Field(None, description="Anthropic API 密钥")
    default_model: str = Field("claude-3-5-sonnet-20241022", description="默认模型")
    timeout: int = Field(60, description="超时时间")


class GoogleConfig(BaseSettings):
    """Google 配置"""
    api_key: Optional[str] = Field(None, description="Google API 密钥")
    default_model: str = Field("gemini-1.5-pro", description="默认模型")
    timeout: int = Field(60, description="超时时间")


class MistralConfig(BaseSettings):
    """Mistral 配置"""
    api_key: Optional[str] = Field(None, description="Mistral API 密钥")
    default_model: str = Field("mistral-large-latest", description="默认模型")
    timeout: int = Field(60, description="超时时间")


class GroqConfig(BaseSettings):
    """Groq 配置"""
    api_key: Optional[str] = Field(None, description="Groq API 密钥")
    default_model: str = Field("llama-3.1-70b-versatile", description="默认模型")
    timeout: int = Field(60, description="超时时间")


class OllamaConfig(BaseSettings):
    """Ollama 配置"""
    base_url: str = Field("http://localhost:11434", description="Ollama 基础 URL")
    default_model: str = Field("llama3.2", description="默认模型")
    timeout: int = Field(60, description="超时时间")


class DeepSeekConfig(BaseSettings):
    """DeepSeek 配置（通过 OpenAI 兼容接口）"""
    api_key: Optional[str] = Field(None, description="DeepSeek API 密钥")
    base_url: str = Field("https://api.deepseek.com/v1", description="DeepSeek API 基础 URL")
    default_model: str = Field("deepseek-chat", description="默认模型")
    timeout: int = Field(60, description="超时时间")
    max_retries: int = Field(0, description="最大重试次数")


class Settings(BaseSettings):
    """LangChain 1.2 应用配置 - 优化版"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__"
    )

    # ========== 默认模型配置 ==========
    DEFAULT_PROVIDER: ModelProvider = Field(
        ModelProvider.OPENAI_COMPATIBLE,
        description="默认模型提供商"
    )

    DEFAULT_TEMPERATURE: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="默认温度参数"
    )

    DEFAULT_MAX_TOKENS: Optional[int] = Field(
        4096,
        description="默认最大输出令牌数"
    )

    # ========== 详细配置 ==========
    openai: OpenAIConfig = Field(
        default_factory=OpenAIConfig,
        description="OpenAI 官方配置"
    )

    openai_compatible: OpenAICompatibleConfig = Field(
        default_factory=OpenAICompatibleConfig,
        description="OpenAI 兼容服务配置"
    )

    anthropic: AnthropicConfig = Field(
        default_factory=AnthropicConfig,
        description="Anthropic 配置"
    )

    google: GoogleConfig = Field(
        default_factory=GoogleConfig,
        description="Google 配置"
    )

    mistral: MistralConfig = Field(
        default_factory=MistralConfig,
        description="Mistral 配置"
    )

    groq: GroqConfig = Field(
        default_factory=GroqConfig,
        description="Groq 配置"
    )

    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig,
        description="Ollama 配置"
    )

    deepseek: DeepSeekConfig = Field(
        default_factory=DeepSeekConfig,
        description="DeepSeek 配置"
    )

    # ========== LangSmith 配置 ==========
    LANGCHAIN_TRACING_V2: bool = Field(False, description="启用 LangSmith 追踪")
    LANGCHAIN_API_KEY: Optional[str] = Field(None, description="LangSmith API 密钥")
    LANGCHAIN_PROJECT: str = Field("default", description="LangSmith 项目名称")

    # ========== 应用配置 ==========
    DEBUG: bool = Field(False, description="调试模式")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="日志级别")
    DATA_DIR: Path = Field(Path("./data"), description="数据目录")

    @field_validator("DATA_DIR", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """自动创建必要的目录"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        """验证配置"""
        # 如果使用 OpenAI 兼容服务，检查必要配置
        if self.DEFAULT_PROVIDER == ModelProvider.OPENAI_COMPATIBLE:
            if not self.openai_compatible.api_key:
                raise ValueError(
                    "使用 OpenAI 兼容服务时必须设置 api_key。"
                    "请在 .env 中设置 OPENAI_COMPATIBLE__API_KEY"
                )

        # 设置 LangSmith
        if self.LANGCHAIN_TRACING_V2:
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.LANGCHAIN_API_KEY:
                os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT

        return self


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 导出配置实例
settings = get_settings()