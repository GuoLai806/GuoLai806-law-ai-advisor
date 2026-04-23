"""
豆包 LLM 模型封装 - LangChain 兼容版本

使用火山引擎 Ark 平台的豆包模型 API。
"""

import os
import requests
from typing import List, Dict, Any, Optional, Iterator
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field


class DoubaoLLM(LLM):
    """豆包 LLM 模型封装类 - LangChain 兼容版本"""

    model_id: str = Field(
        default="doubao-seed-2-0-mini-260215",
        description="豆包模型 ID"
    )
    api_key: str = Field(
        default="",
        description="豆包 API 密钥"
    )
    base_url: str = Field(
        default="https://ark.cn-beijing.volces.com/api/v3",
        description="API 基础 URL"
    )
    timeout: int = Field(
        default=120,
        description="请求超时时间（秒）"
    )

    def __init__(
        self,
        model_id: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
        **kwargs
    ):
        """
        初始化豆包 LLM 模型

        Args:
            model_id: 模型 ID，默认值为豆包种子模型
            api_key: API 密钥，默认为环境变量 ARK_API_KEY
            base_url: API 基础 URL，默认火山引擎 Ark 平台
            timeout: 请求超时时间（秒），默认30秒
        """
        values = {}
        if model_id is not None:
            values["model_id"] = model_id
        else:
            values["model_id"] = os.getenv("DOUBAO_MODEL_ID", "doubao-seed-2-0-mini-260215")

        if api_key is not None:
            values["api_key"] = api_key
        else:
            values["api_key"] = os.getenv("ARK_API_KEY", "")

        if base_url is not None:
            values["base_url"] = base_url
        else:
            values["base_url"] = os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")

        if timeout is not None:
            values["timeout"] = timeout
        else:
            values["timeout"] = 60  # 增加到60秒

        super().__init__(**values, **kwargs)

    @property
    def _llm_type(self) -> str:
        """返回 LLM 类型"""
        return "doubao"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回用于识别该 LLM 的参数"""
        return {
            "model_id": self.model_id,
            "base_url": self.base_url,
        }

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        格式化消息格式

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "内容"}, ...]

        Returns:
            格式化后的消息列表
        """
        formatted_messages = []
        for msg in messages:
            content = []

            if isinstance(msg["content"], str):
                content.append({
                    "type": "input_text",
                    "text": msg["content"]
                })
            elif isinstance(msg["content"], list):
                content.extend(msg["content"])
            else:
                raise ValueError("消息内容格式不支持")

            formatted_messages.append({
                "role": msg["role"],
                "content": content
            })

        return formatted_messages

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        调用豆包模型生成回复

        Args:
            prompt: 提示词
            stop: 停止词列表
            run_manager: 回调管理器

        Returns:
            模型回复文本

        Raises:
            Exception: 调用失败时抛出异常
        """
        if not self.api_key:
            raise Exception("ARK_API_KEY 未配置")

        messages = [{"role": "user", "content": prompt}]
        formatted_messages = self._format_messages(messages)

        payload = {
            "model": self.model_id,
            "input": formatted_messages
        }

        try:
            response = requests.post(
                f"{self.base_url}/responses",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            if "output" in result:
                # 查找type为"message"的项目
                for output_item in result["output"]:
                    if output_item.get("type") == "message" and "content" in output_item:
                        content_list = output_item["content"]
                        for content_item in content_list:
                            if content_item.get("type") == "output_text":
                                return content_item["text"].strip()

            raise Exception("响应格式无效")

        except Exception as e:
            raise Exception(f"豆包模型调用失败: {e}")

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        """
        异步调用豆包模型

        Args:
            prompt: 提示词
            stop: 停止词列表
            run_manager: 回调管理器

        Returns:
            模型回复文本
        """
        return self._call(prompt, stop, run_manager, **kwargs)

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        """
        调用豆包模型生成回复（保持向后兼容）

        Args:
            messages: 对话消息列表

        Returns:
            模型回复文本

        Raises:
            Exception: 调用失败时抛出异常
        """
        if not self.api_key:
            raise Exception("ARK_API_KEY 未配置")

        formatted_messages = self._format_messages(messages)

        payload = {
            "model": self.model_id,
            "input": formatted_messages
        }

        try:
            response = requests.post(
                f"{self.base_url}/responses",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            if "output" in result:
                # 查找type为"message"的项目
                for output_item in result["output"]:
                    if output_item.get("type") == "message" and "content" in output_item:
                        content_list = output_item["content"]
                        for content_item in content_list:
                            if content_item.get("type") == "output_text":
                                return content_item["text"].strip()

            raise Exception("响应格式无效")

        except Exception as e:
            raise Exception(f"豆包模型调用失败: {e}")


# 便捷函数
def create_doubao_llm(
    model_id: str = None,
    api_key: str = None
) -> DoubaoLLM:
    """
    创建豆包 LLM 实例

    Args:
        model_id: 模型 ID
        api_key: API 密钥

    Returns:
        DoubaoLLM 实例
    """
    return DoubaoLLM(model_id, api_key)
