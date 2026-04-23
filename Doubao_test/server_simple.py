#!/usr/bin/env python3
"""
豆包 LLM 对话终端 - 简化版（无 LangChain，支持流式输出）
直接管理对话历史，避免 LangChain 额外开销
"""

import sys
import time
import logging
from typing import List, Dict, Any
from volcenginesdkarkruntime import Ark

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 配置直接写在这里 ====================
ARK_API_KEY = "f775bf4f-cfc1-452f-9346-2f5221b0ca51"
DOUBAO_MODEL_ID = "doubao-seed-2-0-mini-260215"
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
# ===========================================================


class ConversationManager:
    """对话管理类 - 直接管理对话历史，避免 LangChain 开销"""

    def __init__(self):
        self.client = Ark(
            base_url=DOUBAO_BASE_URL,
            api_key=ARK_API_KEY
        )
        self.history: List[Dict[str, str]] = []
        self.latencies: List[float] = []

        # 系统提示
        self.system_prompt = "你是一个专业的AI助手。请详细回答用户的问题，内容长度控制在500字左右，确保信息全面且有条理。"
        self.history.append({
            "role": "system",
            "content": self.system_prompt
        })

    def send_message(self, user_input: str) -> tuple[str, float]:
        start_time = time.time()
        try:
            # 添加用户消息到历史
            self.history.append({
                "role": "user",
                "content": user_input
            })

            full_response = ""
            print("\nAI: ", end="", flush=True)

            # 调用 API - 使用正确的 chat completions 接口
            response = self.client.chat.completions.create(
                model=DOUBAO_MODEL_ID,
                messages=self.history,
                stream=True
            )

            for event in response:
                # 正确的事件结构：event.choices[0].delta.content
                text = ""
                if hasattr(event, 'choices') and event.choices:
                    choice = event.choices[0]
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            text = delta.content

                if text:
                    print(text, end="", flush=True)
                    full_response += text

            # 添加AI回复到历史
            self.history.append({
                "role": "assistant",
                "content": full_response
            })

            elapsed_time = (time.time() - start_time) * 1000
            self.latencies.append(elapsed_time)

            return full_response.strip(), elapsed_time
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            logger.error(f"调用失败: {e}")
            return f"错误: {e}", elapsed_time

    def get_stats(self) -> dict:
        if not self.latencies:
            return {
                "total_messages": 0,
                "avg_latency": 0,
                "min_latency": 0,
                "max_latency": 0
            }
        return {
            "total_messages": len(self.latencies),
            "avg_latency": sum(self.latencies) / len(self.latencies),
            "min_latency": min(self.latencies),
            "max_latency": max(self.latencies)
        }

    def clear_history(self):
        self.history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        self.latencies = []

    def get_conversation_length(self) -> int:
        return len(self.history) - 1


def print_welcome():
    print("\n" + "="*60)
    print("        豆包 LLM 对话终端 (无 LangChain 简化版 + 流式输出)")
    print("="*60)
    print("命令:")
    print("  /stats   - 显示延迟统计")
    print("  /clear   - 清空对话历史")
    print("  /quit    - 退出程序")
    print("="*60 + "\n")


def main():
    try:
        conv = ConversationManager()
        logger.info("豆包 LLM 初始化成功")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        return

    print_welcome()

    while True:
        try:
            user_input = input("你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit', '退出']:
                print("\n再见！")
                break

            if user_input.lower() in ['/stats', 'stats']:
                stats = conv.get_stats()
                print("\n" + "-"*60)
                print("延迟统计:")
                print(f"  消息总数: {stats['total_messages']}")
                print(f"  平均延迟: {stats['avg_latency']:.1f}ms")
                print(f"  最小延迟: {stats['min_latency']:.1f}ms")
                print(f"  最大延迟: {stats['max_latency']:.1f}ms")
                print(f"  对话历史: {conv.get_conversation_length()}")
                print("-"*60 + "\n")
                continue

            if user_input.lower() in ['/clear', 'clear']:
                conv.clear_history()
                print("\n对话历史已清空\n")
                continue

            response, latency = conv.send_message(user_input)
            print(f"\n[延迟: {latency:.1f}ms] [对话历史: {conv.get_conversation_length()}]\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            print("\n\n再见！")
            break
        except Exception as e:
            logger.error(f"出错: {e}")


if __name__ == "__main__":
    main()
