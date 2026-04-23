from .AgentContext import AgentState, AgentActionType, ExecutionStatus


class AgentReducer:
    """Agent 状态管理 Reducer"""

    @staticmethod
    def reduce(state: AgentState, action: dict) -> AgentState:
        """状态更新逻辑"""
        match action["type"]:
            case AgentActionType.RESET_STATE.value:
                return AgentState()

            case AgentActionType.SET_INPUT.value:
                new_state = AgentState()
                # 复制除了需要更新的字段外的所有字段
                for key, value in state.__dict__.items():
                    if key not in ['input_message', 'execution_status', 'conversation_history']:
                        setattr(new_state, key, value)
                # 更新特定字段
                new_state.input_message = action["payload"]
                new_state.execution_status = ExecutionStatus.ANALYZING
                new_state.conversation_history = state.conversation_history + [
                    {
                        "role": "user",
                        "content": action["payload"]
                    }
                ]
                return new_state

            case AgentActionType.SET_INTENT.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key not in ['intent_type', 'intent_confidence', 'document_type', 'execution_status']:
                        setattr(new_state, key, value)
                new_state.intent_type = action["payload"]["intent_type"]
                new_state.intent_confidence = action["payload"]["intent_confidence"]
                new_state.document_type = action["payload"].get("document_type")
                new_state.execution_status = ExecutionStatus.RETRIEVING
                return new_state

            case AgentActionType.SET_CONTEXT.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key not in ['retrieved_context', 'retrieved_sections', 'execution_status']:
                        setattr(new_state, key, value)
                new_state.retrieved_context = action["payload"]["context"]
                new_state.retrieved_sections = action["payload"]["sections"]
                new_state.execution_status = ExecutionStatus.GENERATING
                return new_state

            case AgentActionType.SET_OUTPUT.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key not in ['output_message', 'execution_status', 'conversation_history']:
                        setattr(new_state, key, value)
                new_state.output_message = action["payload"]
                new_state.execution_status = ExecutionStatus.DONE
                new_state.conversation_history = state.conversation_history + [
                    {
                        "role": "assistant",
                        "content": action["payload"]
                    }
                ]
                return new_state

            case AgentActionType.SET_ERROR.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key not in ['error', 'execution_status']:
                        setattr(new_state, key, value)
                new_state.error = action["payload"]
                new_state.execution_status = ExecutionStatus.ERROR
                return new_state

            case AgentActionType.UPDATE_DOCUMENT_FIELDS.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key not in ['document_fields', 'current_field_index']:
                        setattr(new_state, key, value)
                new_fields = {**state.document_fields, **action["payload"]}
                new_state.document_fields = new_fields
                new_state.current_field_index = state.current_field_index + 1
                return new_state

            case AgentActionType.ADD_CONVERSATION_HISTORY.value:
                new_state = AgentState()
                for key, value in state.__dict__.items():
                    if key != 'conversation_history':
                        setattr(new_state, key, value)
                new_state.conversation_history = state.conversation_history + [
                    action["payload"]
                ]
                return new_state

            case _:
                return state
