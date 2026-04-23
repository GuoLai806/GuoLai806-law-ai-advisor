from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Dict, Optional
import datetime


class ExecutionStatus(Enum):
    """执行状态枚举"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    ERROR = "error"
    DONE = "done"


class IntentType(Enum):
    """意图类型枚举"""
    LEGAL_CONSULTATION = "legal_consultation"
    DOCUMENT_GENERATION = "document_generation"
    FALLBACK = "fallback"


@dataclass
class AgentState:
    """Agent 的完整状态"""
    # 输入输出状态
    input_message: str = ""
    output_message: str = ""

    # 意图识别状态
    intent_type: IntentType = IntentType.FALLBACK
    intent_confidence: float = 0.0
    document_type: Optional[str] = None

    # RAG 检索状态
    retrieved_context: str = ""
    retrieved_sections: List[str] = field(default_factory=list)

    # 文档生成状态
    document_fields: Dict[str, str] = field(default_factory=dict)
    current_field_index: int = 0
    generated_document: Optional[str] = None

    # 执行状态
    execution_status: ExecutionStatus = ExecutionStatus.IDLE
    error: Optional[str] = None

    # 会话信息
    session_id: str = ""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    # 元数据
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())


class AgentActionType(Enum):
    """Agent 动作类型"""
    RESET_STATE = "reset_state"
    SET_INPUT = "set_input"
    SET_INTENT = "set_intent"
    SET_CONTEXT = "set_context"
    SET_OUTPUT = "set_output"
    SET_ERROR = "set_error"
    SET_EXECUTION_STATUS = "set_execution_status"
    UPDATE_DOCUMENT_FIELDS = "update_document_fields"
    SET_DOCUMENT_TYPE = "set_document_type"
    ADD_CONVERSATION_HISTORY = "add_conversation_history"
