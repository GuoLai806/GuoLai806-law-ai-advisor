#!/usr/bin/env python
"""
标准化数据接口定义
解决模块间通信问题，统一数据格式
"""
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """标准化的检索结果"""
    chunks: List[str]
    sections: List[int]
    context: str
    scores: Optional[List[Dict[str, float]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievalResult':
        """从字典创建对象，带容错处理"""
        chunks = data.get("chunks", [])
        sections = data.get("sections", [])
        context = data.get("context", "")
        scores = data.get("scores")

        # 确保chunks和sections长度一致
        if len(chunks) != len(sections):
            logger.warning(f"chunks({len(chunks)})和sections({len(sections)})长度不一致")
            # 补齐长度
            min_len = min(len(chunks), len(sections))
            chunks = chunks[:min_len]
            sections = sections[:min_len]

        return cls(chunks=chunks, sections=sections, context=context, scores=scores)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @property
    def is_valid(self) -> bool:
        """检查结果是否有效"""
        return len(self.chunks) > 0 and len(self.chunks) == len(self.sections)

    @property
    def count(self) -> int:
        """返回结果数量"""
        return len(self.chunks)

    def get_reference(self, index: int) -> Optional[Dict[str, Any]]:
        """获取指定索引的引用"""
        if 0 <= index < len(self.chunks):
            return {
                "title": f"民法典条文 {index + 1}",
                "content": self.chunks[index],
                "page": self.sections[index]
            }
        return None

    def get_all_references(self) -> List[Dict[str, Any]]:
        """获取所有引用"""
        return [self.get_reference(i) for i in range(len(self.chunks)) if self.get_reference(i)]


@dataclass
class ConsultationInput:
    """咨询输入数据"""
    query: str
    retrieval_result: Optional[RetrievalResult] = None

    @classmethod
    def create(cls, query: str, retrieval_data: Any = None) -> 'ConsultationInput':
        """创建咨询输入，自动处理不同格式的retrieval_data"""
        retrieval_result = None
        if retrieval_data:
            if isinstance(retrieval_data, RetrievalResult):
                retrieval_result = retrieval_data
            elif isinstance(retrieval_data, dict):
                retrieval_result = RetrievalResult.from_dict(retrieval_data)
            else:
                # 兼容旧版本的字符串格式
                retrieval_result = RetrievalResult(
                    chunks=[],
                    sections=[],
                    context=str(retrieval_data)
                )
        return cls(query=query, retrieval_result=retrieval_result)


@dataclass
class ConsultationOutput:
    """咨询输出数据"""
    response: str
    references: List[Dict[str, Any]]
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def success(cls, response: str, references: List[Dict[str, Any]] = None) -> 'ConsultationOutput':
        """创建成功的输出"""
        return cls(
            response=response,
            references=references or [],
            success=True
        )

    @classmethod
    def failure(cls, error: str) -> 'ConsultationOutput':
        """创建失败的输出"""
        return cls(
            response=f"抱歉，处理失败: {error}",
            references=[],
            success=False,
            error=error
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为API输出格式"""
        return {
            "response": self.response,
            "references": self.references,
            "success": self.success
        }
