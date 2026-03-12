#!/usr/bin/env python
"""LangChain 1.2 完整示例主程序"""
import asyncio
import logging
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.langchain_v1.config import (
    settings,
    get_llm,
    get_openai_llm,
    get_openai_compatible_llm,
    get_aliyun_llm,
    get_deepseek_llm,
    get_anthropic_llm,
    get_google_llm,
    get_mistral_llm,
    get_groq_llm,
    get_ollama_llm,
    ModelProvider,
    ModelFactory
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TestScenario(Enum):
    """测试场景枚举"""
    ALL = "all"
    DEFAULT = "default"
    ALIYUN = "aliyun"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    STREAMING = "streaming"
    BATCH = "batch"
    FALLBACK = "fallback"
    TOOLS = "tools"


def print_banner(text: str):
    """打印横幅"""
    logger.info("=" * 60)
    logger.info(f"  {text}")
    logger.info("=" * 60)


def print_config():
    """打印当前配置"""
    logger.info("📊 系统配置:")
    logger.info(f"  - 默认提供商: {settings.DEFAULT_PROVIDER.value}")
    logger.info(f"  - 默认温度: {settings.DEFAULT_TEMPERATURE}")
    logger.info(f"  - 默认最大令牌: {settings.DEFAULT_MAX_TOKENS}")
    logger.info(f"  - 调试模式: {settings.DEBUG}")
    logger.info(f"  - 日志级别: {settings.LOG_LEVEL}")
    logger.info(f"  - 数据目录: {settings.DATA_DIR}")

    # 打印各提供商配置状态
    logger.info("\n🔑 提供商配置状态:")

    # OpenAI
    if settings.openai.api_key:
        logger.info(f"  ✅ OpenAI: {settings.openai.default_model}")
    else:
        logger.info(f"  ❌ OpenAI: 未配置")

    # OpenAI兼容（阿里云）
    if settings.openai_compatible.api_key:
        logger.info(f"  ✅ {settings.openai_compatible.provider_name}: {settings.openai_compatible.default_model}")
        logger.info(f"     - API Key: {settings.openai_compatible.api_key[:10]}...{settings.openai_compatible.api_key[-5:]}")
        logger.info(f"     - Base URL: {settings.openai_compatible.base_url}")
    else:
        logger.info(f"  ❌ {settings.openai_compatible.provider_name}: 未配置")

    # Anthropic
    if settings.anthropic.api_key:
        logger.info(f"  ✅ Anthropic: {settings.anthropic.default_model}")
    else:
        logger.info(f"  ❌ Anthropic: 未配置")

    # Google
    if settings.google.api_key:
        logger.info(f"  ✅ Google: {settings.google.default_model}")
    else:
        logger.info(f"  ❌ Google: 未配置")

    # DeepSeek
    if settings.deepseek.api_key:
        logger.info(f"  ✅ DeepSeek: {settings.deepseek.default_model}")
    else:
        logger.info(f"  ❌ DeepSeek: 未配置")

    # Ollama
    logger.info(f"  ✅ Ollama: {settings.ollama.default_model} (本地)")

    logger.info("")


async def test_default_model():
    """测试默认模型"""
    print_banner("测试 1: 默认模型")

    try:
        llm = get_llm()
        logger.info(f"✅ 创建模型: {type(llm).__name__}")

        response = await llm.ainvoke("你好！请用一句话介绍你自己。")
        logger.info(f"🤖 响应: {response.content}")

        # 打印token使用情况（如果有）
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            logger.info(
                f"📊 Token使用: 输入={usage.get('input_tokens', 'N/A')}, 输出={usage.get('output_tokens', 'N/A')}")

        return True
    except Exception as e:
        logger.error(f"❌ 默认模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_aliyun():
    """测试阿里云通义千问"""
    print_banner("测试 2: 阿里云通义千问")

    if not settings.openai_compatible.api_key:
        logger.warning("⚠️ 阿里云API密钥未配置，跳过测试")
        return False

    try:
        # 打印配置信息
        logger.info(f"📝 阿里云配置:")
        logger.info(f"  - API Key: {settings.openai_compatible.api_key[:10]}...{settings.openai_compatible.api_key[-5:]}")
        logger.info(f"  - Base URL: {settings.openai_compatible.base_url}")
        logger.info(f"  - Model: {settings.openai_compatible.default_model}")

        # 方法1: 使用专门的阿里云函数
        # llm1 = get_aliyun_llm()
        # logger.info(f"✅ 方法1 - 创建阿里云模型: {type(llm1).__name__}")
        #
        # response1 = await llm1.ainvoke("你好！请问你是谁？")
        # logger.info(f"🤖 方法1响应: {response1.content[:100]}...")

        # 方法2: 使用通用兼容接口
        llm2 = get_openai_compatible_llm()
        logger.info(f"✅ 方法2 - 创建兼容模型: {type(llm2).__name__}")

        response2 = await llm2.ainvoke("langchain1+的架构是什么(当前版本1.2)？")
        logger.info(f"🤖 方法2响应: {response2.content[:100]}...")

        # 测试不同模型
        # models_to_test = ["qwen-turbo", "qwen-plus", "qwen-max"]
        # for model in models_to_test:
        #     try:
        #         logger.info(f"🔄 测试模型: {model}")
        #         llm = get_aliyun_llm(model_name=model)
        #         response = await llm.ainvoke("你好")
        #         logger.info(f"  ✅ 模型 {model} 可用: {response.content[:30]}...")
        #         break  # 找到一个可用的就行
        #     except Exception as e:
        #         logger.info(f"  ❌ 模型 {model} 不可用: {e}")
        #         continue

        return True
    except Exception as e:
        logger.error(f"❌ 阿里云测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_deepseek():
    """测试 DeepSeek"""
    print_banner("测试 3: DeepSeek")

    if not settings.deepseek.api_key:
        logger.warning("⚠️ DeepSeek API密钥未配置，跳过测试")
        return False

    try:
        llm = get_deepseek_llm()
        logger.info(f"✅ 创建DeepSeek模型: {type(llm).__name__}")

        response = await llm.ainvoke("你好！请简单介绍DeepSeek")
        logger.info(f"🤖 响应: {response.content[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ DeepSeek测试失败: {e}")
        return False


async def test_openai():
    """测试OpenAI官方"""
    print_banner("测试 4: OpenAI官方")

    if not settings.openai.api_key:
        logger.warning("⚠️ OpenAI API密钥未配置，跳过测试")
        return False

    try:
        llm = get_openai_llm()
        logger.info(f"✅ 创建OpenAI模型: {type(llm).__name__}")

        response = await llm.ainvoke("Hello! Please introduce yourself briefly.")
        logger.info(f"🤖 响应: {response.content[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ OpenAI测试失败: {e}")
        return False


async def test_anthropic():
    """测试Anthropic Claude"""
    print_banner("测试 5: Anthropic Claude")

    if not settings.anthropic.api_key:
        logger.warning("⚠️ Anthropic API密钥未配置，跳过测试")
        return False

    try:
        llm = get_anthropic_llm()
        logger.info(f"✅ 创建Anthropic模型: {type(llm).__name__}")

        response = await llm.ainvoke("Hello! Please introduce yourself briefly.")
        logger.info(f"🤖 响应: {response.content[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Anthropic测试失败: {e}")
        return False


async def test_google():
    """测试Google Gemini"""
    print_banner("测试 6: Google Gemini")

    if not settings.google.api_key:
        logger.warning("⚠️ Google API密钥未配置，跳过测试")
        return False

    try:
        llm = get_google_llm()
        logger.info(f"✅ 创建Google模型: {type(llm).__name__}")

        response = await llm.ainvoke("Hello! Please introduce yourself briefly.")
        logger.info(f"🤖 响应: {response.content[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Google测试失败: {e}")
        return False


async def test_ollama():
    """测试Ollama本地模型"""
    print_banner("测试 7: Ollama本地模型")

    try:
        llm = get_ollama_llm()
        logger.info(f"✅ 创建Ollama模型: {type(llm).__name__}")

        response = await llm.ainvoke("Hello! Please introduce yourself.")
        logger.info(f"🤖 响应: {response.content[:100]}...")

        return True
    except Exception as e:
        logger.error(f"❌ Ollama测试失败: {e}")
        logger.info("💡 提示: 确保Ollama服务已启动且模型已下载")
        return False


async def test_streaming():
    """测试流式输出"""
    print_banner("测试 8: 流式输出")

    try:
        llm = get_llm()
        logger.info("📝 流式响应:")

        async for chunk in llm.astream("讲一个简短的笑话"):
            print(chunk.content, end="", flush=True)
        print("\n")

        logger.info("✅ 流式输出完成")
        return True
    except Exception as e:
        logger.error(f"❌ 流式输出测试失败: {e}")
        return False


async def test_batch():
    """测试批量处理"""
    print_banner("测试 9: 批量处理")

    try:
        llm = get_llm()

        messages = [
            "你好",
            "How are you?",
            "Bonjour",
            "Ciao",
            "こんにちは"
        ]

        logger.info(f"📦 批量处理 {len(messages)} 条消息...")
        start_time = time.time()

        responses = await llm.abatch([msg for msg in messages])

        elapsed = time.time() - start_time
        logger.info(f"⏱️ 耗时: {elapsed:.2f}秒")

        for i, (msg, resp) in enumerate(zip(messages, responses)):
            logger.info(f"  [{i + 1}] '{msg}' -> {resp.content[:30]}...")

        return True
    except Exception as e:
        logger.error(f"❌ 批量处理测试失败: {e}")
        return False


async def test_fallback():
    """测试模型回退机制"""
    print_banner("测试 10: 模型回退机制")

    providers = [
        ModelProvider.ANTHROPIC,
        ModelProvider.OPENAI,
        ModelProvider.GOOGLE,
        ModelProvider.OPENAI_COMPATIBLE
    ]

    logger.info(f"🔄 尝试按顺序创建模型: {[p.value for p in providers]}")

    try:
        llm = ModelFactory.create_with_fallback(
            providers=providers,
            temperature=0.5
        )

        logger.info(f"✅ 成功创建模型: {type(llm).__name__}")

        response = await llm.ainvoke("Hello")
        logger.info(f"🤖 响应: {response.content[:50]}...")

        return True
    except Exception as e:
        logger.error(f"❌ 回退机制测试失败: {e}")
        return False


async def test_tools():
    """测试工具调用"""
    print_banner("测试 11: 工具调用")

    try:
        from langchain_core.tools import tool

        @tool
        def get_weather(city: str) -> str:
            """获取指定城市的天气"""
            return f"{city} 的天气是晴朗，22度。"

        @tool
        def calculate(expression: str) -> str:
            """计算数学表达式"""
            try:
                result = eval(expression)
                return f"计算结果: {result}"
            except:
                return "无法计算该表达式"

        tools = [get_weather, calculate]

        llm = get_llm().bind_tools(tools)
        logger.info(f"✅ 创建支持工具的模型")

        # 测试天气工具
        response1 = await llm.ainvoke("北京今天天气怎么样？")
        logger.info(f"🌤️ 天气查询: {response1.content}")

        # 测试计算工具
        response2 = await llm.ainvoke("计算 15 * 7 + 3 等于多少？")
        logger.info(f"🧮 计算: {response2.content}")

        return True
    except Exception as e:
        logger.error(f"❌ 工具调用测试失败: {e}")
        return False


async def run_selected_tests(scenario: TestScenario):
    """运行选定的测试场景"""
    test_functions = {
        TestScenario.DEFAULT: [test_default_model],
        TestScenario.ALIYUN: [test_aliyun],
        TestScenario.DEEPSEEK: [test_deepseek],
        TestScenario.OPENAI: [test_openai],
        TestScenario.ANTHROPIC: [test_anthropic],
        TestScenario.GOOGLE: [test_google],
        TestScenario.OLLAMA: [test_ollama],
        TestScenario.STREAMING: [test_streaming],
        TestScenario.BATCH: [test_batch],
        TestScenario.FALLBACK: [test_fallback],
        TestScenario.TOOLS: [test_tools],
        TestScenario.ALL: [
            # test_default_model,
            test_aliyun,
            # test_deepseek,
            # test_openai,
            # test_anthropic,
            # test_google,
            # test_ollama,
            # test_streaming,
            # test_batch,
            # test_fallback,
            # test_tools
        ]
    }

    tests_to_run = test_functions.get(scenario, [test_default_model])

    results = {}
    for test_func in tests_to_run:
        try:
            logger.info(f"\n🔄 开始执行: {test_func.__name__}")
            success = await test_func()
            results[test_func.__name__] = "✅ 通过" if success else "❌ 失败"
        except Exception as e:
            logger.error(f"执行 {test_func.__name__} 时发生未处理异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_func.__name__] = "💥 异常"

    # 打印测试汇总
    print_banner("测试结果汇总")
    for test_name, result in results.items():
        logger.info(f"  {result} - {test_name}")

    return results


def parse_arguments():
    """解析命令行参数"""
    import argparse

    parser = argparse.ArgumentParser(description="LangChain 1.2 测试程序")
    parser.add_argument(
        "--test", "-t",
        type=str,
        choices=[s.value for s in TestScenario],
        default="aliyun",
        help="要运行的测试场景 (默认: aliyun)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="指定模型名称"
    )

    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_arguments()

    # 如果指定了详细模式，调整日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 打印启动横幅
    print_banner(f"🚀 LangChain 1.2 测试程序启动")
    logger.info(f"测试场景: {args.test}")
    if args.model:
        logger.info(f"指定模型: {args.model}")

    # 打印系统配置
    print_config()

    # 如果指定了模型，可以在这里处理
    if args.model and args.test == "aliyun":
        # 临时修改配置中的模型名
        settings.openai_compatible.default_model = args.model
        logger.info(f"📝 已临时将阿里云模型改为: {args.model}")

    # 运行选定的测试
    scenario = TestScenario(args.test)
    results = await run_selected_tests(scenario)

    # 计算成功率
    success_count = sum(1 for r in results.values() if "✅" in r)
    total_count = len(results)

    print_banner(f"测试完成: {success_count}/{total_count} 通过")

    # 如果所有测试都失败，给出提示
    if success_count == 0:
        logger.warning("")
        logger.warning("💡 提示:")
        logger.warning("  1. 检查 .env 文件中的 API Key 是否正确")
        logger.warning("  2. 确认网络连接是否正常")
        logger.warning("  3. 检查各服务商账号余额/配额")
        logger.warning("  4. 尝试使用基础模型: qwen-turbo 或 qwen-plus")
        logger.warning("  5. 运行 python test_simple.py 进行基础连接测试")


if __name__ == "__main__":
    asyncio.run(main())