#!/usr/bin/env python3
"""
交互式法律AI咨询终端 - 本地运行，方便测试
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


# 全局单例
_global_agent = None


async def init_agent():
    """初始化LegalAgent"""
    global _global_agent
    if _global_agent is not None:
        return _global_agent

    print("\n" + "="*60)
    print("  正在初始化法律AI咨询系统...")
    print("="*60)

    start_time = time.time()
    from src.agents.LegalAgent import LegalAgent
    _global_agent = LegalAgent()
    elapsed = (time.time() - start_time) * 1000

    print(f"  系统初始化完成，耗时: {elapsed:.1f}ms")
    print("="*60)
    print("\n  提示：输入 'quit' 或 'exit' 退出")
    print("        输入 'clear' 清空对话历史")
    print("="*60 + "\n")

    return _global_agent


async def handle_query(agent, question):
    """处理用户问题"""
    start_total = time.time()

    print(f"\n正在思考您的问题...")
    await agent.process_input(question)

    total_time = (time.time() - start_total) * 1000

    # 输出结果
    print(f"\n" + "-"*60)
    print(f"延迟统计: {total_time:.1f}ms ({total_time/1000:.2f}秒)")
    print("-"*60)
    print(f"\n雅子的回答：\n{agent.get_output()}")
    print("-"*60 + "\n")

    # 检查是否达标
    if total_time <= 5000:
        print("延迟达标 (<= 5秒)")
    else:
        print(f"延迟未达标 ({total_time:.1f}ms > 5秒)")
    print()


async def main():
    try:
        # 初始化agent
        agent = await init_agent()

        # 交互式循环
        while True:
            try:
                # 读取输入
                question = input("您的问题: ").strip()

                # 处理退出命令
                if question.lower() in ["quit", "exit", "退出"]:
                    print("\n再见！感谢使用雅子法律小助手！")
                    break

                # 处理清空命令
                if question.lower() in ["clear", "清空"]:
                    agent.clear_memory()
                    print("\n对话历史已清空\n")
                    continue

                # 跳过空输入
                if not question:
                    continue

                # 处理问题
                await handle_query(agent, question)

            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                logger.error(f"处理问题时出错: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
