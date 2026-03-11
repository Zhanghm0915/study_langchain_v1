"""基于 Pydantic Settings 的 LangChain 1.2 配置"""
from typing import Optional, Literal, Union, Dict, Any, List
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import Field, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelProvider(str, Enum):
    """支持的模型提供商（LangChain 1.2 官方支持的）"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    GROQ = "groq"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    DEEPSEEK = "deepseek"  # 通过 OpenAI 兼容接口
    TOGETHER = "together"   # 通过 OpenAI 兼容接口
    FIREWORKS = "fireworks" # 通过 OpenAI 兼容接口

class ModelConfig(BaseSettings):
    """单个模型的配置"""
    model_config = ConfigDict(extra="ignore")

    model: str = Field(..., description="模型名称")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)

class ProviderConfig(BaseSettings):
    """单个提供商的配置"""
    model_config = ConfigDict(extra="ignore")

    api_key: Optional[str] = Field(None)
    base_url: Optional[str] = Field(None)
    organization: Optional[str] = Field(None)
    default_model: str = Field(...)
    models: Dict[str, ModelConfig] = Field(default_factory=dict)

class Settings(BaseSettings):
    """LangChain 1.2 应用配置

    使用 Pydantic Settings V2 自动从环境变量加载
    """

    # Pydantic 配置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__"
    )

    # ========== 默认模型配置 ==========
    DEFAULT_PROVIDER: ModelProvider = Field(
        ModelProvider.OPENAI,
        description="默认模型提供商"
    )

    DEFAULT_MODEL: str = Field(
        "gpt-4o-mini",
        description="默认模型名称"
    )

    DEFAULT_TEMPERATURE: float = Field(
        0.7,
        description="默认温度参数"
    )

    DEFAULT_MAX_TOKENS: Optional[int] = Field(
        4096,
        description="默认最大输出令牌数"
    )

    DEFAULT_TIMEOUT: int = Field(
        60,
        description="默认请求超时时间（秒）"
    )

    DEFAULT_MAX_RETRIES: int = Field(
        3,
        description="默认最大重试次数"
    )

    # ========== OpenAI 配置 ==========
    OPENAI_API_KEY: Optional[str] = Field(
        None,
        description="OpenAI API 密钥"
    )

    OPENAI_ORG_ID: Optional[str] = Field(
        None,
        description="OpenAI 组织 ID"
    )

    OPENAI_BASE_URL: Optional[str] = Field(
        None,
        description="OpenAI API 基础 URL（用于代理）"
    )

    OPENAI_MODEL: str = Field(
        "gpt-4o-mini",
        description="OpenAI 默认模型"
    )

    OPENAI_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "gpt-4o": ModelConfig(
                model="gpt-4o",
                temperature=0.7,
                max_tokens=4096
            ),
            "gpt-4o-mini": ModelConfig(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=4096
            ),
            "o1-preview": ModelConfig(
                model="o1-preview",
                temperature=1.0,
                max_tokens=32768
            ),
        },
        description="OpenAI 模型配置"
    )

    # ========== Anthropic (Claude) 配置 ==========
    ANTHROPIC_API_KEY: Optional[str] = Field(
        None,
        description="Anthropic API 密钥"
    )

    ANTHROPIC_MODEL: str = Field(
        "claude-3-5-sonnet-20241022",
        description="Anthropic 默认模型"
    )

    ANTHROPIC_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "claude-3-5-sonnet-20241022": ModelConfig(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=8192
            ),
            "claude-3-haiku-20240307": ModelConfig(
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=4096
            ),
        }
    )

    # ========== Google (Gemini) 配置 ==========
    GOOGLE_API_KEY: Optional[str] = Field(
        None,
        description="Google API 密钥"
    )

    GOOGLE_MODEL: str = Field(
        "gemini-1.5-pro",
        description="Google 默认模型"
    )

    GOOGLE_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "gemini-1.5-pro": ModelConfig(
                model="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=8192
            ),
            "gemini-1.5-flash": ModelConfig(
                model="gemini-1.5-flash",
                temperature=0.7,
                max_tokens=8192
            ),
            "gemini-1.0-pro": ModelConfig(
                model="gemini-1.0-pro",
                temperature=0.7,
                max_tokens=2048
            ),
        }
    )

    # ========== Mistral AI 配置 ==========
    MISTRAL_API_KEY: Optional[str] = Field(
        None,
        description="Mistral API 密钥"
    )

    MISTRAL_MODEL: str = Field(
        "mistral-large-latest",
        description="Mistral 默认模型"
    )

    MISTRAL_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "mistral-large-latest": ModelConfig(
                model="mistral-large-latest",
                temperature=0.7,
                max_tokens=8192
            ),
            "mistral-medium-latest": ModelConfig(
                model="mistral-medium-latest",
                temperature=0.7,
                max_tokens=4096
            ),
            "mistral-small-latest": ModelConfig(
                model="mistral-small-latest",
                temperature=0.7,
                max_tokens=4096
            ),
        }
    )

    # ========== Groq 配置 ==========
    GROQ_API_KEY: Optional[str] = Field(
        None,
        description="Groq API 密钥"
    )

    GROQ_MODEL: str = Field(
        "llama-3.1-70b-versatile",
        description="Groq 默认模型"
    )

    GROQ_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "llama-3.1-70b-versatile": ModelConfig(
                model="llama-3.1-70b-versatile",
                temperature=0.7,
                max_tokens=8192
            ),
            "mixtral-8x7b-32768": ModelConfig(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=32768
            ),
        }
    )

    # ========== Ollama (本地) 配置 ==========
    OLLAMA_BASE_URL: str = Field(
        "http://localhost:11434",
        description="Ollama 基础 URL"
    )

    OLLAMA_MODEL: str = Field(
        "llama3.2",
        description="Ollama 默认模型"
    )

    OLLAMA_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "llama3.2": ModelConfig(
                model="llama3.2",
                temperature=0.7,
                max_tokens=4096
            ),
            "qwen2.5": ModelConfig(
                model="qwen2.5",
                temperature=0.7,
                max_tokens=4096
            ),
            "mistral": ModelConfig(
                model="mistral",
                temperature=0.7,
                max_tokens=4096
            ),
        }
    )

    # ========== DeepSeek 配置 ==========
    DEEPSEEK_API_KEY: Optional[str] = Field(
        None,
        description="DeepSeek API 密钥"
    )

    DEEPSEEK_MODEL: str = Field(
        "deepseek-chat",
        description="DeepSeek 默认模型"
    )

    DEEPSEEK_BASE_URL: str = Field(
        "https://api.deepseek.com/v1",
        description="DeepSeek API 基础 URL"
    )

    DEEPSEEK_MODELS: Dict[str, ModelConfig] = Field(
        default_factory=lambda: {
            "deepseek-chat": ModelConfig(
                model="deepseek-chat",
                temperature=0.7,
                max_tokens=8192
            ),
            "deepseek-coder": ModelConfig(
                model="deepseek-coder",
                temperature=0.7,
                max_tokens=8192
            ),
        }
    )

    # ========== Together AI 配置 ==========
    TOGETHER_API_KEY: Optional[str] = Field(
        None,
        description="Together AI API 密钥"
    )

    TOGETHER_MODEL: str = Field(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        description="Together AI 默认模型"
    )

    TOGETHER_BASE_URL: str = Field(
        "https://api.together.xyz/v1",
        description="Together AI API 基础 URL"
    )

    # ========== Fireworks AI 配置 ==========
    FIREWORKS_API_KEY: Optional[str] = Field(
        None,
        description="Fireworks AI API 密钥"
    )

    FIREWORKS_MODEL: str = Field(
        "accounts/fireworks/models/llama-v3p1-70b-instruct",
        description="Fireworks AI 默认模型"
    )

    FIREWORKS_BASE_URL: str = Field(
        "https://api.fireworks.ai/inference/v1",
        description="Fireworks AI API 基础 URL"
    )

    # ========== LangSmith 配置（LangChain 1.2 官方可观测性）==========
    LANGCHAIN_TRACING_V2: bool = Field(
        False,
        description="启用 LangSmith 追踪"
    )

    LANGCHAIN_API_KEY: Optional[str] = Field(
        None,
        description="LangSmith API 密钥"
    )

    LANGCHAIN_PROJECT: str = Field(
        "default",
        description="LangSmith 项目名称"
    )

    LANGCHAIN_ENDPOINT: str = Field(
        "https://api.smith.langchain.com",
        description="LangSmith API 端点"
    )

    # ========== 应用配置 ==========
    DEBUG: bool = Field(
        False,
        description="调试模式"
    )

    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="日志级别"
    )

    DATA_DIR: Path = Field(
        Path("./data"),
        description="数据目录"
    )

    CACHE_DIR: Path = Field(
        Path("./cache"),
        description="缓存目录"
    )

    # ========== 验证器 ==========
    @field_validator("DATA_DIR", "CACHE_DIR", mode="after")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        """自动创建必要的目录"""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @model_validator(mode="after")
    def validate_default_provider(self) -> "Settings":
        """验证默认提供商的 API 密钥是否存在"""
        provider = self.DEFAULT_PROVIDER

        # 检查对应提供商的 API 密钥
        key_map = {
            ModelProvider.OPENAI: self.OPENAI_API_KEY,
            ModelProvider.ANTHROPIC: self.ANTHROPIC_API_KEY,
            ModelProvider.GOOGLE: self.GOOGLE_API_KEY,
            ModelProvider.MISTRAL: self.MISTRAL_API_KEY,
            ModelProvider.GROQ: self.GROQ_API_KEY,
            ModelProvider.DEEPSEEK: self.DEEPSEEK_API_KEY,
            ModelProvider.TOGETHER: self.TOGETHER_API_KEY,
            ModelProvider.FIREWORKS: self.FIREWORKS_API_KEY,
        }

        if provider in key_map and not key_map[provider] and not self.DEBUG:
            import warnings
            warnings.warn(
                f"默认提供商 {provider} 的 API 密钥未设置，"
                "请检查环境变量配置"
            )

        return self

    def model_post_init(self, __context):
        """初始化后设置环境变量"""
        if self.LANGCHAIN_TRACING_V2:
            import os
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            if self.LANGCHAIN_API_KEY:
                os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()

# 导出配置实例
settings = get_settings()