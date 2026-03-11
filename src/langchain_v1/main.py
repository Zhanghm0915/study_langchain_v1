#!/usr/bin/env python
"""LangChain 1.2 主程序示例"""
import asyncio
import logging
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.langchain_v1.config import settings, get_llm, ModelFactory, ModelProvider

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demonstrate_default_model():
    """演示默认模型使用（LangChain 1.2 风格）"""
    logger.info("=== 演示 1: 默认模型 ===")

    # 使用默认配置创建模型
    llm = get_llm()

    # LangChain 1.2 支持直接 await
    response = await llm.ainvoke("你好，请用一句话介绍 LangChain 1.2")
    logger.info(f"默认模型响应: {response.content}")


async def demonstrate_different_providers():
    """演示不同提供商"""
    logger.info("\n=== 演示 2: 不同模型提供商 ===")

    providers = [
        ("OpenAI", ModelProvider.OPENAI),
        ("Anthropic", ModelProvider.ANTHROPIC),
        ("Google", ModelProvider.GOOGLE),
        ("Mistral", ModelProvider.MISTRAL),
        ("Groq", ModelProvider.GROQ),
        ("DeepSeek", ModelProvider.DEEPSEEK),
    ]

    for name, provider in providers:
        try:
            llm = ModelFactory.create(provider)
            response = await llm.ainvoke("Hello, who are you?")
            logger.info(f"{name}: {response.content[:50]}...")
        except Exception as e:
            logger.warning(f"{name} 不可用: {e}")


async def demonstrate_streaming():
    """演示流式输出（LangChain 1.2 新特性）"""
    logger.info("\n=== 演示 3: 流式输出 ===")

    llm = get_openai_llm()

    logger.info("流式响应:")
    async for chunk in llm.astream("请写一个关于 AI 的短诗"):
        print(chunk.content, end="", flush=True)
    print()


async def demonstrate_batch_processing():
    """演示批量处理"""
    logger.info("\n=== 演示 4: 批量处理 ===")

    llm = get_llm()

    messages = [
        "你好",
        "How are you?",
        "Bonjour",
    ]

    # LangChain 1.2 支持批量处理
    responses = await llm.abatch([msg for msg in messages])

    for msg, resp in zip(messages, responses):
        logger.info(f"输入: {msg}")
        logger.info(f"输出: {resp.content[:30]}...")


async def demonstrate_with_config():
    """演示带配置的调用"""
    logger.info("\n=== 演示 5: 带配置的调用 ===")

    llm = get_llm()

    # LangChain 1.2 支持运行时配置
    response = await llm.ainvoke(
        "讲一个笑话",
        config={
            "tags": ["joke", "chinese"],
            "metadata": {"user_id": "123"},
        }
    )

    logger.info(f"带标签的响应: {response.content}")


async def main():
    """主函数"""
    logger.info(f"🚀 LangChain 1.2 应用启动")
    logger.info(f"📊 当前配置:")
    logger.info(f"  - 默认提供商: {settings.DEFAULT_PROVIDER}")
    logger.info(f"  - 默认模型: {settings.DEFAULT_MODEL}")
    logger.info(f"  - 默认温度: {settings.DEFAULT_TEMPERATURE}")
    logger.info(f"  - LangSmith 追踪: {settings.LANGCHAIN_TRACING_V2}")

    # 运行演示
    await demonstrate_default_model()
    await demonstrate_different_providers()
    await demonstrate_streaming()
    await demonstrate_batch_processing()
    await demonstrate_with_config()

    logger.info("\n✅ 演示完成")


if __name__ == "__main__":
    asyncio.run(main())