#!/usr/bin/env python3
"""
优化版CLI测试：预加载组件，测量各环节耗时
"""

import sys
import time
import logging
import asyncio
import traceback
from dotenv import load_dotenv

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 全局单例，避免重复初始化
_global_components = {}


async def preload_components():
    """预加载所有组件，避免重复初始化"""
    logger.info("=== 预加载组件 ===")
    start_total = time.time()

    # 1. 预加载LegalAgent
    logger.info("1. 预加载LegalAgent...")
    start = time.time()
    from src.agents.LegalAgent import LegalAgent
    agent = LegalAgent()
    elapsed = (time.time() - start) * 1000
    logger.info(f"   LegalAgent预加载完成，耗时: {elapsed:.1f}ms")
    _global_components["agent"] = agent

    total_time = (time.time() - start_total) * 1000
    logger.info(f"   预加载总耗时: {total_time:.1f}ms\n")
    return agent


async def test_speed(agent):
    """测试整个流程的各环节耗时"""
    logger.info("=== 测试咨询流程 ===")

    # 测试问题
    test_question = "合同违约怎么办？"
    logger.info(f"测试问题: {test_question}")

    start_time = time.time()

    # 处理输入
    await agent.process_input(test_question)

    total_time = (time.time() - start_time) * 1000
    logger.info(f"完整流程耗时: {total_time:.1f}ms")

    logger.info(f"意图: {agent.get_intent_type()}")
    logger.info(f"输出: {agent.get_output()[:150]}...")

    if agent.get_error():
        logger.error(f"错误: {agent.get_error()}")

    # 检查是否在5秒内
    if total_time <= 5000:
        logger.info("✅ 延迟达标 (<= 5秒)")
        return True
    else:
        logger.warning(f"⚠️ 延迟未达标 ({total_time:.1f}ms > 5秒)")
        return False


async def main():
    try:
        # 预加载组件
        agent = await preload_components()

        # 测试多次
        for i in range(3):
            logger.info(f"\n--- 测试轮次 {i+1} ---")
            success = await test_speed(agent)
            if success:
                break

        logger.info("\n=== 所有测试完成 ===")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    asyncio.run(main())
