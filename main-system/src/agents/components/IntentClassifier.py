import re
from typing import Dict, Optional, Any
from ..context.AgentContext import IntentType


class IntentClassifier:
    """意图识别组件"""

    INTENT_KEYWORDS = {
        IntentType.LEGAL_CONSULTATION: [
            r"法律.*咨询", r"法律.*问题", r"如何.*法律", r"法律.*规定",
            r"违法", r"违规", r"侵权", r"赔偿", r"合同.*纠纷",
            r"怎么办", r"怎么处理", r"如何解决", r"有什么办法",
            r"民法典.*第.*条", r"法律.*条文", r"法律.*条款", r"第.*条",
            r"什么是.*", r"介绍.*一下", r"解释.*一下"
        ],
        IntentType.DOCUMENT_GENERATION: [
            r"文书.*生成", r"合同.*起草", r"起诉状.*写", r"律师函.*开",
            r"协议.*模板", r"法律.*文件", r"帮我写.*合同", r"帮我写.*协议",
            r"写一份.*合同", r"写一份.*协议", r"写一份.*起诉状", r"写一份.*律师函"
        ]
    }

    async def recognize(self, input_message: str) -> Dict[str, Any]:
        """识别用户意图"""
        for intent_type, keywords in self.INTENT_KEYWORDS.items():
            for pattern in keywords:
                if re.search(pattern, input_message, re.IGNORECASE):
                    return {
                        "intent_type": intent_type,
                        "intent_confidence": 0.95,
                        "document_type": await self._detect_document_type(input_message)
                    }

        return {
            "intent_type": IntentType.FALLBACK,
            "intent_confidence": 0.5,
            "document_type": None
        }

    async def _detect_document_type(self, input_message: str) -> Optional[str]:
        """检测需要生成的文书类型"""
        document_patterns = {
            "contract": [r"合同.*起草", r"合同.*模板", r"协议.*写", r"协议.*模板"],
            "complaint": [r"起诉状.*写", r"起诉书.*拟", r"提起.*诉讼"],
            "legal_letter": [r"律师函.*开", r"律师信.*写", r"函件.*准备"]
        }

        for doc_type, patterns in document_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_message, re.IGNORECASE):
                    return doc_type

        return None
