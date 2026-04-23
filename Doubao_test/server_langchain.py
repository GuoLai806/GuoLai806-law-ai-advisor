import sys
import time
import logging
from typing import List, Dict, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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


class DoubaoLLM(LLM):
    """豆包 LLM 模型封装类 - LangChain 兼容版本 + 流式输出"""

    model_id: str = DOUBAO_MODEL_ID
    api_key: str = ARK_API_KEY
    base_url: str = DOUBAO_BASE_URL
    timeout: int = 60

    @property
    def _llm_type(self) -> str:
        return "doubao"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "base_url": self.base_url,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if not self.api_key:
            raise Exception("ARK_API_KEY 未配置")

        client = Ark(
            base_url=self.base_url,
            api_key=self.api_key
        )

        full_response = ""
        print("\nAI: ", end="", flush=True)

        try:
            # 使用正确的 chat completions 接口
            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for event in response:
                text = ""
                # 正确的事件结构：event.choices[0].delta.content
                if hasattr(event, 'choices') and event.choices:
                    choice = event.choices[0]
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            text = delta.content

                if text:
                    print(text, end="", flush=True)
                    full_response += text
                    if run_manager:
                        run_manager.on_llm_new_token(text)

            return full_response.strip()
        except Exception as e:
            raise Exception(f"豆包模型调用失败: {e}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        return self._call(prompt, stop, run_manager, **kwargs)


class ConversationManager:
    """对话管理类 - 处理对话记忆和统计"""

    def __init__(self):
        self.llm = DoubaoLLM()
        self.memory = ConversationBufferMemory(return_messages=True)

        # 创建prompt模板
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""你是一个专业的AI助手。请详细回答用户的问题，内容长度控制在500字左右，确保信息全面且有条理。

对话历史：
{history}

用户的问题：
{input}

AI回答：
"""
        )

        # 创建对话链
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )

        self.latencies: List[float] = []

    def send_message(self, user_input: str) -> tuple[str, float]:
        start_time = time.time()
        try:
            response = self.chain.predict(input=user_input)
            elapsed_time = (time.time() - start_time) * 1000
            self.latencies.append(elapsed_time)
            return response, elapsed_time
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            return f"\n错误: {e}", elapsed_time

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
        self.memory.clear()
        self.latencies = []


def print_welcome():
    print("\n" + "="*60)
    print("        豆包 LLM 对话终端 (LangChain 版本 + 流式输出)")
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
                print("-"*60 + "\n")
                continue

            if user_input.lower() in ['/clear', 'clear']:
                conv.clear_history()
                print("\n对话历史已清空\n")
                continue

            response, latency = conv.send_message(user_input)
            print(f"\n[延迟: {latency:.1f}ms]\n")

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
