"""LangChain 1.2 模型工厂 - 优化版"""
from typing import Optional, Union, List, Dict, Any
import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

# LangChain 1.2 官方模型导入
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from .settings import settings, ModelProvider

logger = logging.getLogger(__name__)


class ModelFactory:
    """LangChain 1.2 模型工厂类 - 优化版"""

    # 提供商映射表，便于扩展
    PROVIDER_MAP = {
        ModelProvider.OPENAI: "create_openai",
        ModelProvider.OPENAI_COMPATIBLE: "create_openai_compatible",
        ModelProvider.ANTHROPIC: "create_anthropic",
        ModelProvider.GOOGLE: "create_google",
        ModelProvider.MISTRAL: "create_mistral",
        ModelProvider.GROQ: "create_groq",
        ModelProvider.OLLAMA: "create_ollama",
        ModelProvider.DEEPSEEK: "create_deepseek",
    }

    @classmethod
    def create_openai(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 OpenAI 官方模型实例"""
        model = model_name or settings.openai.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.openai.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.openai.timeout),
            "max_retries": kwargs.get("max_retries", settings.openai.max_retries),
        }

        if settings.openai.organization:
            config["organization"] = settings.openai.organization

        if settings.openai.base_url:
            config["base_url"] = settings.openai.base_url

        # 允许通过 kwargs 覆盖
        config.update(kwargs)

        logger.debug(f"创建 OpenAI 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_openai_compatible(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 OpenAI 兼容服务模型（阿里云、DeepSeek等）"""
        model = model_name or settings.openai_compatible.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.openai_compatible.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "base_url": kwargs.get("base_url", settings.openai_compatible.base_url),
            "timeout": kwargs.get("timeout", settings.openai_compatible.timeout),
            "max_retries": kwargs.get("max_retries", settings.openai_compatible.max_retries),
        }

        # 移除可能的冲突参数
        if "organization" in config:
            del config["organization"]

        config.update(kwargs)

        provider = settings.openai_compatible.provider_name
        logger.debug(f"创建 {provider} 兼容模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_aliyun(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """专门创建阿里云通义千问模型"""
        kwargs["base_url"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return cls.create_openai_compatible(model_name or "qwen-plus", **kwargs)

    @classmethod
    def create_deepseek(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """专门创建 DeepSeek 模型"""
        model = model_name or settings.deepseek.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.deepseek.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "base_url": kwargs.get("base_url", settings.deepseek.base_url),
            "timeout": kwargs.get("timeout", settings.deepseek.timeout),
            "max_retries": kwargs.get("max_retries", settings.deepseek.max_retries),
        }

        config.update(kwargs)

        logger.debug(f"创建 DeepSeek 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_anthropic(cls, model_name: Optional[str] = None, **kwargs) -> ChatAnthropic:
        """创建 Anthropic 模型实例"""
        model = model_name or settings.anthropic.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.anthropic.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.anthropic.timeout),
        }

        config.update(kwargs)

        logger.debug(f"创建 Anthropic 模型: {model}")
        return ChatAnthropic(**config)

    @classmethod
    def create_google(cls, model_name: Optional[str] = None, **kwargs) -> ChatGoogleGenerativeAI:
        """创建 Google Gemini 模型实例"""
        model = model_name or settings.google.default_model

        config = {
            "google_api_key": kwargs.get("api_key", settings.google.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_output_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.google.timeout),
        }

        # 移除多余的 api_key 参数
        if "api_key" in config:
            del config["api_key"]

        config.update(kwargs)

        logger.debug(f"创建 Google 模型: {model}")
        return ChatGoogleGenerativeAI(**config)

    @classmethod
    def create_mistral(cls, model_name: Optional[str] = None, **kwargs) -> ChatMistralAI:
        """创建 Mistral 模型实例"""
        model = model_name or settings.mistral.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.mistral.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.mistral.timeout),
        }

        config.update(kwargs)

        logger.debug(f"创建 Mistral 模型: {model}")
        return ChatMistralAI(**config)

    @classmethod
    def create_groq(cls, model_name: Optional[str] = None, **kwargs) -> ChatGroq:
        """创建 Groq 模型实例"""
        model = model_name or settings.groq.default_model

        config = {
            "api_key": kwargs.get("api_key", settings.groq.api_key),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.groq.timeout),
        }

        config.update(kwargs)

        logger.debug(f"创建 Groq 模型: {model}")
        return ChatGroq(**config)

    @classmethod
    def create_ollama(cls, model_name: Optional[str] = None, **kwargs) -> ChatOllama:
        """创建 Ollama 模型实例"""
        model = model_name or settings.ollama.default_model

        config = {
            "base_url": kwargs.get("base_url", settings.ollama.base_url),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "num_predict": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "timeout": kwargs.get("timeout", settings.ollama.timeout),
        }

        config.update(kwargs)

        logger.debug(f"创建 Ollama 模型: {model}")
        return ChatOllama(**config)

    @classmethod
    def create_default(cls, **kwargs) -> BaseChatModel:
        """根据默认配置创建模型"""
        provider = kwargs.pop("provider", settings.DEFAULT_PROVIDER)

        # 根据提供商获取默认模型名称
        model_name = kwargs.pop("model_name", cls._get_default_model(provider))

        # 合并默认参数
        model_kwargs = {
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            **kwargs
        }

        logger.info(f"使用默认提供商: {provider.value}, 模型: {model_name}")
        return cls.create(provider, model_name=model_name, **model_kwargs)

    @classmethod
    def _get_default_model(cls, provider: ModelProvider) -> str:
        """获取提供商的默认模型"""
        model_map = {
            ModelProvider.OPENAI: settings.openai.default_model,
            ModelProvider.OPENAI_COMPATIBLE: settings.openai_compatible.default_model,
            ModelProvider.ANTHROPIC: settings.anthropic.default_model,
            ModelProvider.GOOGLE: settings.google.default_model,
            ModelProvider.MISTRAL: settings.mistral.default_model,
            ModelProvider.GROQ: settings.groq.default_model,
            ModelProvider.OLLAMA: settings.ollama.default_model,
            ModelProvider.DEEPSEEK: settings.deepseek.default_model,
        }
        return model_map.get(provider, "gpt-4o-mini")

    @classmethod
    def create(cls, provider: Union[str, ModelProvider], **kwargs) -> BaseChatModel:
        """根据提供商创建模型"""
        if isinstance(provider, str):
            provider = ModelProvider(provider.lower())

        if provider not in cls.PROVIDER_MAP:
            raise ValueError(f"不支持的模型提供商: {provider}")

        # 动态调用对应的方法
        method_name = cls.PROVIDER_MAP[provider]
        method = getattr(cls, method_name)

        return method(**kwargs)

    @classmethod
    def create_with_fallback(cls, providers: List[Union[str, ModelProvider]],
                            **kwargs) -> BaseChatModel:
        """尝试按顺序创建模型，失败时回退到下一个"""
        last_error = None

        for provider in providers:
            try:
                logger.info(f"尝试创建 {provider} 模型...")
                return cls.create(provider, **kwargs)
            except Exception as e:
                logger.warning(f"{provider} 模型创建失败: {e}")
                last_error = e
                continue

        raise RuntimeError(f"所有模型创建失败，最后错误: {last_error}")


# ========== 快捷函数 ==========

@lru_cache()
def get_llm(provider: Optional[Union[str, ModelProvider]] = None, **kwargs) -> BaseChatModel:
    """获取 LLM 实例（带缓存）"""
    if provider:
        return ModelFactory.create(provider, **kwargs)
    return ModelFactory.create_default(**kwargs)


def get_openai_llm(**kwargs) -> ChatOpenAI:
    """获取 OpenAI 官方 LLM 实例"""
    return ModelFactory.create_openai(**kwargs)


def get_openai_compatible_llm(**kwargs) -> ChatOpenAI:
    """获取 OpenAI 兼容服务 LLM 实例"""
    return ModelFactory.create_openai_compatible(**kwargs)


def get_aliyun_llm(**kwargs) -> ChatOpenAI:
    """获取阿里云通义千问 LLM 实例"""
    return ModelFactory.create_aliyun(**kwargs)


def get_deepseek_llm(**kwargs) -> ChatOpenAI:
    """获取 DeepSeek LLM 实例"""
    return ModelFactory.create_deepseek(**kwargs)


def get_anthropic_llm(**kwargs) -> ChatAnthropic:
    """获取 Anthropic LLM 实例"""
    return ModelFactory.create_anthropic(**kwargs)


def get_google_llm(**kwargs) -> ChatGoogleGenerativeAI:
    """获取 Google LLM 实例"""
    return ModelFactory.create_google(**kwargs)


def get_mistral_llm(**kwargs) -> ChatMistralAI:
    """获取 Mistral LLM 实例"""
    return ModelFactory.create_mistral(**kwargs)


def get_groq_llm(**kwargs) -> ChatGroq:
    """获取 Groq LLM 实例"""
    return ModelFactory.create_groq(**kwargs)


def get_ollama_llm(**kwargs) -> ChatOllama:
    """获取 Ollama LLM 实例"""
    return ModelFactory.create_ollama(**kwargs)