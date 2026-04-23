"""
RAG 检索组件 - 使用优化后的 RAG 系统

这是 React 风格架构中的 RAG 组件，负责：
1. 检索相关法律条文
2. 格式化检索结果
3. 提供检索统计信息

作者：GuoLai
版本：2.0.0
"""

from typing import Dict, Any, List, Optional
import os

from .OptimizedRAGSystem import OptimizedRAGSystem, RAGConfig


class RetrievalModule:
    """
    RAG 检索组件

    使用 OptimizedRAGSystem 进行检索，提供简洁的接口
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化检索组件

        Args:
            config: RAG 配置对象，为 None 时使用默认配置
        """
        # 创建 RAG 系统
        self.rag_system = OptimizedRAGSystem(config)

        # 延迟初始化标记
        self._initialized = False

    async def initialize(self) -> bool:
        """
        初始化 RAG 系统

        Returns:
            bool: 初始化是否成功
        """
        if self._initialized:
            return True

        success = self.rag_system.initialize()
        if success:
            self._initialized = True

        return success

    async def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        检索相关法律条文

        Args:
            query: 用户查询
            k: 返回的结果数量

        Returns:
            Dict[str, Any]: 检索结果
                - context: 格式化的上下文文本
                - sections: 相关页码列表
                - chunks: 完整的文档块列表
                - scores: 相似度分数列表
        """
        # 确保已初始化
        if not self._initialized:
            success = await self.initialize()
            if not success:
                return {
                    "context": "RAG系统未初始化，无法提供法律条文支持，请检查是否已添加民法典PDF文件",
                    "sections": [],
                    "chunks": [],
                    "scores": []
                }

        # 执行混合检索
        try:
            result = self.rag_system.retrieve(query, k)
            return result
        except Exception as e:
            return {
                "context": f"检索法律条文时出错: {e}",
                "sections": [],
                "chunks": [],
                "scores": []
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取 RAG 系统统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return self.rag_system.get_statistics()

    def get_rag_system(self) -> OptimizedRAGSystem:
        """
        获取底层 RAG 系统实例（用于高级操作）

        Returns:
            OptimizedRAGSystem: RAG 系统实例
        """
        return self.rag_system


# 为了保持向后兼容，保留旧的接口
# 但建议使用上面的新接口
class LegacyRetrievalModule:
    """
    旧版检索组件（保留向后兼容）

    新代码请使用 RetrievalModule
    """

    def __init__(self):
        from ..models.BailianEmbeddings import BailianEmbeddings
        from langchain_community.vectorstores import Chroma

        try:
            bailian_api_key = os.getenv("BAILIAN_API_KEY", "")
            self.vector_db = Chroma(
                persist_directory="./chroma_civil_code",
                embedding_function=BailianEmbeddings(bailian_api_key)
            )
        except Exception as e:
            print(f"向量数据库初始化失败: {e}")
            self.vector_db = None

    async def retrieve(self, query: str, k: int = 3) -> Dict[str, Any]:
        """旧版检索方法"""
        if not self.vector_db:
            return {
                "context": "向量数据库未初始化，无法提供法律条文支持",
                "sections": []
            }

        try:
            results = self.vector_db.similarity_search(query, k=k)

            context_parts = []
            sections = []
            for i, doc in enumerate(results, 1):
                page_num = doc.metadata.get("page", "未知")
                context_parts.append(f"法律条文{i}: {doc.page_content[:200]}... (第{page_num}页)")
                sections.append(doc.metadata.get("page", "未知"))

            return {
                "context": "\n\n".join(context_parts),
                "sections": sections
            }

        except Exception as e:
            return {
                "context": f"检索法律条文时出错: {e}",
                "sections": []
            }

    def get_vector_db(self):
        """获取向量数据库"""
        return self.vector_db
