#!/usr/bin/env python3
"""
优化版CLI测试：测量各环节耗时并输出详细报告
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


async def test_speed():
    """测试整个流程的各环节耗时"""
    logger.info("=== 法律AI咨询系统优化版测试 ===\n")

    try:
        logger.info("1. 正在导入组件...")
        from src.agents.LegalAgent import LegalAgent

        logger.info("2. 初始化法律Agent...")
        agent = LegalAgent()

        # 测试问题
        test_question = "合同违约怎么办？"
        logger.info(f"3. 测试问题: {test_question}")

        start_time = time.time()

        # 处理输入
        await agent.process_input(test_question)

        total_time = (time.time() - start_time) * 1000
        logger.info(f"4. 完整流程耗时: {total_time:.1f}ms")

        logger.info(f"意图: {agent.get_intent_type()}")
        logger.info(f"输出: {agent.get_output()[:100]}...")

        if agent.get_error():
            logger.error(f"错误: {agent.get_error()}")

        # 检查是否在5秒内
        if total_time <= 5000:
            logger.info("✅ 延迟达标 (<= 5秒)")
        else:
            logger.warning(f"⚠️ 延迟未达标 ({total_time:.1f}ms > 5秒)")

    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        logger.error(traceback.format_exc())
        return False

    logger.info("\n=== 测试完成 ===")
    return True


async def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        await test_speed()
    else:
        logger.info("用法: python cli_test_speed.py test")


if __name__ == "__main__":
    asyncio.run(main())
