from ..agents.LegalAgent import LegalAgent
from ..context.AgentContext import AgentState
from typing import Optional


class useLegalAgent:
    """法律 Agent Hook"""

    def __init__(self):
        self.agent = LegalAgent()

    def get_agent(self):
        """获取 Agent 实例"""
        return self.agent

    def get_state(self):
        """获取状态"""
        return self.agent.get_state()

    async def send_message(self, input_message: str) -> AgentState:
        """发送消息并获取最新状态"""
        await self.agent.process_input(input_message)
        return self.agent.get_state()

    def reset(self):
        """重置状态"""
        from ..context.AgentContext import AgentActionType
        self.agent.dispatch({
            "type": AgentActionType.RESET_STATE.value
        })

    def get_execution_status(self):
        """获取执行状态"""
        return self.agent.get_execution_status()

    def get_output(self):
        """获取输出结果"""
        return self.agent.get_output()

    def get_error(self):
        """获取错误信息"""
        return self.agent.get_error()

    def get_intent_type(self):
        """获取识别到的意图类型"""
        return self.agent.get_intent_type()
