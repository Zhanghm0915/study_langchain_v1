"""LangChain 1.2 模型工厂"""
from typing import Optional, Dict, Any, Union, List
import logging
from functools import lru_cache

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

# LangChain 1.2 官方支持的模型导入
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_cohere import ChatCohere

from .settings import settings, ModelProvider, ModelConfig

logger = logging.getLogger(__name__)


class ModelFactory:
    """LangChain 1.2 模型工厂类"""

    @classmethod
    def create_openai(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 OpenAI 模型实例（LangChain 1.2）"""
        model = model_name or settings.OPENAI_MODEL

        # 获取模型配置
        model_config = settings.OPENAI_MODELS.get(model, {})

        config = {
            "api_key": kwargs.get("api_key", settings.OPENAI_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens",
                                     getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
            "max_retries": kwargs.get("max_retries", settings.DEFAULT_MAX_RETRIES),
        }

        # 可选参数
        if settings.OPENAI_ORG_ID:
            config["organization"] = settings.OPENAI_ORG_ID

        if settings.OPENAI_BASE_URL:
            config["base_url"] = settings.OPENAI_BASE_URL

        # 添加其他可选参数
        for param in ["top_p", "frequency_penalty", "presence_penalty"]:
            if hasattr(model_config, param) and getattr(model_config, param) is not None:
                config[param] = getattr(model_config, param)

        # 允许通过 kwargs 覆盖
        config.update(kwargs)

        logger.debug(f"创建 OpenAI 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_anthropic(cls, model_name: Optional[str] = None, **kwargs) -> ChatAnthropic:
        """创建 Anthropic 模型实例（LangChain 1.2）"""
        model = model_name or settings.ANTHROPIC_MODEL

        model_config = settings.ANTHROPIC_MODELS.get(model, {})

        config = {
            "api_key": kwargs.get("api_key", settings.ANTHROPIC_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens",
                                     getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
        }

        config.update(kwargs)

        logger.debug(f"创建 Anthropic 模型: {model}")
        return ChatAnthropic(**config)

    @classmethod
    def create_google(cls, model_name: Optional[str] = None, **kwargs) -> ChatGoogleGenerativeAI:
        """创建 Google Gemini 模型实例（LangChain 1.2）"""
        model = model_name or settings.GOOGLE_MODEL

        model_config = settings.GOOGLE_MODELS.get(model, {})

        config = {
            "google_api_key": kwargs.get("api_key", settings.GOOGLE_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_output_tokens": kwargs.get("max_tokens",
                                            getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
        }

        # 移除多余的 api_key 参数
        if "api_key" in config:
            del config["api_key"]

        config.update(kwargs)

        logger.debug(f"创建 Google 模型: {model}")
        return ChatGoogleGenerativeAI(**config)

    @classmethod
    def create_mistral(cls, model_name: Optional[str] = None, **kwargs) -> ChatMistralAI:
        """创建 Mistral 模型实例（LangChain 1.2）"""
        model = model_name or settings.MISTRAL_MODEL

        model_config = settings.MISTRAL_MODELS.get(model, {})

        config = {
            "api_key": kwargs.get("api_key", settings.MISTRAL_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens",
                                     getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
        }

        config.update(kwargs)

        logger.debug(f"创建 Mistral 模型: {model}")
        return ChatMistralAI(**config)

    @classmethod
    def create_groq(cls, model_name: Optional[str] = None, **kwargs) -> ChatGroq:
        """创建 Groq 模型实例（LangChain 1.2）"""
        model = model_name or settings.GROQ_MODEL

        model_config = settings.GROQ_MODELS.get(model, {})

        config = {
            "api_key": kwargs.get("api_key", settings.GROQ_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens",
                                     getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
        }

        config.update(kwargs)

        logger.debug(f"创建 Groq 模型: {model}")
        return ChatGroq(**config)

    @classmethod
    def create_ollama(cls, model_name: Optional[str] = None, **kwargs) -> ChatOllama:
        """创建 Ollama 模型实例（LangChain 1.2）"""
        model = model_name or settings.OLLAMA_MODEL

        model_config = settings.OLLAMA_MODELS.get(model, {})

        config = {
            "base_url": kwargs.get("base_url", settings.OLLAMA_BASE_URL),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "num_predict": kwargs.get("max_tokens",
                                      getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
        }

        config.update(kwargs)

        logger.debug(f"创建 Ollama 模型: {model}")
        return ChatOllama(**config)

    @classmethod
    def create_deepseek(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 DeepSeek 模型实例（通过 OpenAI 兼容接口）"""
        model = model_name or settings.DEEPSEEK_MODEL

        model_config = settings.DEEPSEEK_MODELS.get(model, {})

        config = {
            "api_key": kwargs.get("api_key", settings.DEEPSEEK_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature",
                                      getattr(model_config, "temperature", settings.DEFAULT_TEMPERATURE)),
            "max_tokens": kwargs.get("max_tokens",
                                     getattr(model_config, "max_tokens", settings.DEFAULT_MAX_TOKENS)),
            "base_url": kwargs.get("base_url", settings.DEEPSEEK_BASE_URL),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
            "max_retries": kwargs.get("max_retries", settings.DEFAULT_MAX_RETRIES),
        }

        config.update(kwargs)

        logger.debug(f"创建 DeepSeek 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_together(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 Together AI 模型实例（通过 OpenAI 兼容接口）"""
        model = model_name or settings.TOGETHER_MODEL

        config = {
            "api_key": kwargs.get("api_key", settings.TOGETHER_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "base_url": kwargs.get("base_url", settings.TOGETHER_BASE_URL),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
            "max_retries": kwargs.get("max_retries", settings.DEFAULT_MAX_RETRIES),
        }

        config.update(kwargs)

        logger.debug(f"创建 Together AI 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_fireworks(cls, model_name: Optional[str] = None, **kwargs) -> ChatOpenAI:
        """创建 Fireworks AI 模型实例（通过 OpenAI 兼容接口）"""
        model = model_name or settings.FIREWORKS_MODEL

        config = {
            "api_key": kwargs.get("api_key", settings.FIREWORKS_API_KEY),
            "model": model,
            "temperature": kwargs.get("temperature", settings.DEFAULT_TEMPERATURE),
            "max_tokens": kwargs.get("max_tokens", settings.DEFAULT_MAX_TOKENS),
            "base_url": kwargs.get("base_url", settings.FIREWORKS_BASE_URL),
            "timeout": kwargs.get("timeout", settings.DEFAULT_TIMEOUT),
            "max_retries": kwargs.get("max_retries", settings.DEFAULT_MAX_RETRIES),
        }

        config.update(kwargs)

        logger.debug(f"创建 Fireworks AI 模型: {model}")
        return ChatOpenAI(**config)

    @classmethod
    def create_default(cls, **kwargs) -> BaseChatModel:
        """根据默认配置创建模型"""
        provider = kwargs.pop("provider", settings.DEFAULT_PROVIDER)

        # 合并默认参数
        model_kwargs = {
            "temperature": settings.DEFAULT_TEMPERATURE,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "timeout": settings.DEFAULT_TIMEOUT,
            "max_retries": settings.DEFAULT_MAX_RETRIES,
            **kwargs
        }

        if "model_name" not in model_kwargs and "model" not in model_kwargs:
            model_kwargs["model_name"] = settings.DEFAULT_MODEL

        logger.info(f"使用默认提供商: {provider}")
        return cls.create(provider, **model_kwargs)

    @classmethod
    def create(cls, provider: Union[str, ModelProvider], **kwargs) -> BaseChatModel:
        """根据提供商创建模型"""
        if isinstance(provider, str):
            provider = ModelProvider(provider.lower())

        # 提供商到创建方法的映射
        creators = {
            ModelProvider.OPENAI: cls.create_openai,
            ModelProvider.ANTHROPIC: cls.create_anthropic,
            ModelProvider.GOOGLE: cls.create_google,
            ModelProvider.MISTRAL: cls.create_mistral,
            ModelProvider.GROQ: cls.create_groq,
            ModelProvider.OLLAMA: cls.create_ollama,
            ModelProvider.COHERE: lambda **kw: ChatCohere(**kw),
            ModelProvider.DEEPSEEK: cls.create_deepseek,
            ModelProvider.TOGETHER: cls.create_together,
            ModelProvider.FIREWORKS: cls.create_fireworks,
        }

        if provider not in creators:
            raise ValueError(f"不支持的模型提供商: {provider}")

        return creators[provider](**kwargs)

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


# 快捷函数
@lru_cache()
def get_llm(provider: Optional[str] = None, **kwargs) -> BaseChatModel:
    """获取 LLM 实例（带缓存）"""
    if provider:
        return ModelFactory.create(provider, **kwargs)
    return ModelFactory.create_default(**kwargs)


def get_openai_llm(**kwargs) -> ChatOpenAI:
    """获取 OpenAI LLM 实例"""
    return ModelFactory.create_openai(**kwargs)


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