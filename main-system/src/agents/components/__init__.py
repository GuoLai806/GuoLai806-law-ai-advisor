"""
RAG组件模块

提供法律AI系统的核心组件：
- OptimizedRAGSystem：优化后的RAG系统（支持混合检索和RLHF）
- BM25Retriever：BM25关键词检索器
- HybridRAG：BM25+向量混合检索器
- PDFProcessor：PDF文档处理
- VectorDatabase：ChromaDB向量存储
- HybridRAG：混合检索

作者：GuoLai
版本：2.0.0
"""

from .OptimizedRAGSystem import (
    OptimizedRAGSystem,
    RAGConfig
)

from .BM25Retriever import (
    BM25Retriever
)

from .PDFProcessor import (
    PDFProcessor,
    DocumentChunk
)

from .VectorDatabase import (
    VectorDatabase
)

from .HybridRAG import (
    HybridRAG,
    RetrievalResult
)

from .IntentClassifier import IntentClassifier
from .RetrievalModule import RetrievalModule
from .ConsultationModule import ConsultationModule
from .DocumentGeneration import DocumentGeneration
from .FallbackHandler import FallbackHandler

__all__ = [
    # 主系统
    "OptimizedRAGSystem",
    "RAGConfig",

    # 高级检索
    "BM25Retriever",
    "HybridRAG",
    "RetrievalResult",

    # 文档处理
    "PDFProcessor",
    "DocumentChunk",
    "VectorDatabase",

    # 业务组件
    "IntentClassifier",
    "RetrievalModule",
    "ConsultationModule",
    "DocumentGeneration",
    "FallbackHandler",
]

__version__ = "2.0.0"
