from typing import Dict, Any
import time
import logging
from .context.AgentContext import AgentState, AgentActionType, IntentType, ExecutionStatus
from .context.AgentReducer import AgentReducer
from .components.IntentClassifier import IntentClassifier
from .components.RetrievalModule import RetrievalModule
from .components.ConsultationModule import ConsultationModule
from .components.DocumentGeneration import DocumentGeneration
from .components.FallbackHandler import FallbackHandler

logger = logging.getLogger(__name__)


class LegalAgent:
    """主法律 Agent 组件"""

    def __init__(self):
        self.state = AgentState()
        self.reducer = AgentReducer()
        self.intent_classifier = IntentClassifier()
        self.retrieval_module = RetrievalModule()
        self.consultation_module = ConsultationModule()
        self.document_generator = DocumentGeneration()
        self.fallback_handler = FallbackHandler()

    def dispatch(self, action: Dict[str, Any]):
        """状态更新方法"""
        self.state = self.reducer.reduce(self.state, action)

    async def process_input(self, input_message: str):
        """处理用户输入的主方法"""
        total_start = time.time()

        # 1. 直接设置输入，保留对话历史
        self.dispatch({
            "type": AgentActionType.SET_INPUT.value,
            "payload": input_message
        })

        # 2. 意图识别
        intent_start = time.time()
        intent_result = await self.intent_classifier.recognize(input_message)
        intent_time = (time.time() - intent_start) * 1000
        logger.info(f"⏱️ [意图识别] 耗时: {intent_time:.1f}ms")
        self.dispatch({
            "type": AgentActionType.SET_INTENT.value,
            "payload": {
                "intent_type": intent_result["intent_type"],
                "intent_confidence": intent_result["intent_confidence"],
                "document_type": intent_result.get("document_type")
            }
        })

        # 3. 根据意图执行相应流程（条件渲染）
        handler_start = time.time()
        if self.state.intent_type == IntentType.LEGAL_CONSULTATION:
            await self._handle_consultation()
        elif self.state.intent_type == IntentType.DOCUMENT_GENERATION:
            await self._handle_document_generation()
        else:
            await self._handle_fallback()
        handler_time = (time.time() - handler_start) * 1000

        total_time = (time.time() - total_start) * 1000
        logger.info(f"📊 [处理输入] 总耗时: {total_time:.1f}ms | 意图识别: {intent_time:.1f}ms | 业务处理: {handler_time:.1f}ms")

    async def _handle_consultation(self):
        """处理法律咨询流程"""
        try:
            # 暂时禁用RAG检索，只使用LLM回答
            # retrieval_result = await self.retrieval_module.retrieve(self.state.input_message)
            # self.dispatch({
            #     "type": AgentActionType.SET_CONTEXT.value,
            #     "payload": retrieval_result
            # })

            # 生成咨询回复 - 不传递检索结果
            consultation_result = await self.consultation_module.consult(
                self.state.input_message,
                None  # 不传递检索结果
            )

            # 保存完整结果（包括引用）
            self._consultation_result = consultation_result

            # 只设置响应文本
            self.dispatch({
                "type": AgentActionType.SET_OUTPUT.value,
                "payload": consultation_result["response"]
            })

        except Exception as e:
            self.dispatch({
                "type": AgentActionType.SET_ERROR.value,
                "payload": str(e)
            })

    async def _handle_document_generation(self):
        """处理法律文书生成流程"""
        try:
            # 生成法律文书
            document_result = await self.document_generator.generate(
                self.state.document_type,
                self.state.input_message
            )
            self.dispatch({
                "type": AgentActionType.SET_OUTPUT.value,
                "payload": document_result
            })

        except Exception as e:
            self.dispatch({
                "type": AgentActionType.SET_ERROR.value,
                "payload": str(e)
            })

    async def _handle_fallback(self):
        """处理未识别意图的情况"""
        fallback_response = await self.fallback_handler.handle(
            self.state.input_message
        )
        self.dispatch({
            "type": AgentActionType.SET_OUTPUT.value,
            "payload": fallback_response
        })

    def get_state(self):
        """获取当前状态的只读视图"""
        return self.state

    def get_execution_status(self):
        """获取执行状态"""
        return self.state.execution_status

    def get_output(self):
        """获取输出结果"""
        return self.state.output_message

    def get_error(self):
        """获取错误信息"""
        return self.state.error

    def get_intent_type(self):
        """获取识别到的意图类型"""
        return self.state.intent_type

    def get_references(self):
        """获取引用的法律条文"""
        return getattr(self, '_consultation_result', {}).get('references', [])

    def clear_memory(self):
        """清空对话记忆"""
        if hasattr(self, 'consultation_module'):
            self.consultation_module.clear_memory()
        # 重置状态
        from .context.AgentContext import AgentState
        self.state = AgentState()
