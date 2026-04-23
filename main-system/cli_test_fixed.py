#!/usr/bin/env python3
"""
固定输入的CLI测试脚本，用于测试系统功能和延迟
"""

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


async def test_query(agent, question, expected_intent=None):
    """测试单个查询"""
    start_total = time.time()
    await agent.process_input(question)
    total_time = (time.time() - start_total) * 1000

    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"延迟: {total_time:.1f}ms ({total_time/1000:.2f}秒)")
    print("-"*60)
    print(f"意图: {agent.get_intent_type()}")
    print(f"回答: {agent.get_output()[:150]}...")
    print(f"{'='*60}")

    if total_time > 5000:
        logger.warning(f"⚠️ 延迟未达标: {total_time:.1f}ms > 5000ms")
    else:
        logger.info("✅ 延迟达标")

    return total_time <= 5000


async def main():
    try:
        logger.info("=== 初始化法律AI咨询系统 ===")
        from src.agents.LegalAgent import LegalAgent
        agent = LegalAgent()

        # 测试问题列表
        test_questions = [
            "合同违约怎么办？",
            "什么是侵权责任？",
            "离婚财产如何分割？",
            "如何申请专利？",
            "你好"
        ]

        # 逐个测试
        all_passed = True
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- 测试问题 {i} ---")
            success = await test_query(agent, question)
            if not success:
                all_passed = False
            await asyncio.sleep(0.1)  # 避免过快调用

        logger.info("\n" + "-"*60)
        if all_passed:
            logger.info("✅ 所有测试通过，延迟均在5秒以内！")
        else:
            logger.warning("⚠️ 部分测试延迟未达标")

        # 打印系统信息
        logger.info(f"\n系统信息:")
        logger.info(f"  意图识别器已加载")
        logger.info(f"  RAG系统已初始化")
        logger.info(f"  法律条文数据库已准备")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
