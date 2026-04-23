"""
混合检索器 - BM25 + 向量检索

作者：GuoLai
版本：1.0.0
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document

from .BM25Retriever import BM25Retriever


class HybridRetriever:
    """
    混合检索器 - 结合BM25关键词匹配和向量语义检索

    特性：
    1. 双路并行检索
    2. 结果去重
    3. 分数归一化
    4. 加权融合（BM25:0.6, 向量:0.4）
    5. 可配置权重
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever = None,
        vector_store=None,
        bm25_weight: float = 0.6,
        vector_weight: float = 0.4,
    ):
        """
        初始化混合检索器

        Args:
            bm25_retriever: BM25检索器实例
            vector_store: Chroma向量存储实例
            bm25_weight: BM25权重（默认0.6）
            vector_weight: 向量权重（默认0.4）
        """
        self.bm25_retriever = bm25_retriever
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # 验证权重总和
        total_weight = self.bm25_weight + self.vector_weight
        if total_weight != 1.0:
            print(f"[WARN] 权重总和({total_weight})不为1，将自动归一化")
            self.bm25_weight = self.bm25_weight / total_weight
            self.vector_weight = self.vector_weight / total_weight

    def set_weights(self, bm25_weight: float, vector_weight: float):
        """
        动态设置权重

        Args:
            bm25_weight: BM25权重
            vector_weight: 向量权重
        """
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        total_weight = self.bm25_weight + self.vector_weight
        if total_weight != 1.0:
            self.bm25_weight = self.bm25_weight / total_weight
            self.vector_weight = self.vector_weight / total_weight

    def _normalize_scores(
        self,
        doc_score_pairs: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """
        归一化分数到[0,1]范围

        Args:
            doc_score_pairs: [(doc, score), ...]列表

        Returns:
            归一化后的[(doc, normalized_score), ...]列表
        """
        if not doc_score_pairs:
            return []

        scores = [score for _, score in doc_score_pairs]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # 所有分数相同，归一化为0.5
            return [(doc, 0.5) for doc, _ in doc_score_pairs]

        return [
            (doc, (score - min_score) / (max_score - min_score))
            for doc, score in doc_score_pairs
        ]

    def _get_document_id(self, doc: Document) -> str:
        """
        获取文档唯一标识

        优先使用metadata中的chunk_id，否则使用文档内容的hash
        """
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id is not None:
            return str(chunk_id)
        # 回退到内容hash
        return str(hash(doc.page_content))

    def hybrid_retrieve(
        self,
        query: str,
        k: int = 10,
        bm25_k: int = 20,
        vector_k: int = 20,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        """
        混合检索核心方法

        流程：
        1. BM25检索获取Top-bm25_k
        2. 向量检索获取Top-vector_k
        3. 分别归一化分数
        4. 结果去重和融合
        5. 加权计算最终分数
        6. 排序返回Top-k

        Args:
            query: 查询文本
            k: 最终返回结果数量
            bm25_k: BM25检索候选数量
            vector_k: 向量检索候选数量
            score_threshold: 最终分数阈值

        Returns:
            [(Document, final_score), ...]列表，按分数降序排列
        """
        # 1. BM25检索
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, bm25_k)

        # 2. 向量检索
        vector_results = []
        if self.vector_store:
            try:
                vector_results_with_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=vector_k
                )
                # Chroma返回的是(Document, distance)，distance越小越好，转换为相似度
                vector_results = [
                    (doc, 1.0 - distance)
                    for doc, distance in vector_results_with_scores
                ]
            except Exception as e:
                print(f"[WARN] 向量检索失败: {e}")

        # 3. 归一化分数
        bm25_normalized = self._normalize_scores(bm25_results)
        vector_normalized = self._normalize_scores(vector_results)

        # 4. 构建文档ID到分数的映射
        doc_map = {}

        # 处理BM25结果
        for doc, score in bm25_normalized:
            doc_id = self._get_document_id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc": doc,
                    "bm25_score": score,
                    "vector_score": 0.0
                }
            else:
                doc_map[doc_id]["bm25_score"] = max(
                    doc_map[doc_id]["bm25_score"],
                    score
                )

        # 处理向量结果
        for doc, score in vector_normalized:
            doc_id = self._get_document_id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc": doc,
                    "bm25_score": 0.0,
                    "vector_score": score
                }
            else:
                doc_map[doc_id]["vector_score"] = max(
                    doc_map[doc_id]["vector_score"],
                    score
                )

        # 5. 计算最终分数
        final_results = []
        for doc_id, data in doc_map.items():
            final_score = (
                self.bm25_weight * data["bm25_score"] +
                self.vector_weight * data["vector_score"]
            )

            if final_score >= score_threshold:
                final_results.append((data["doc"], final_score))

        # 6. 排序返回Top-k
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:k]

    def hybrid_retrieve_with_debug(
        self,
        query: str,
        k: int = 10,
        bm25_k: int = 20,
        vector_k: int = 20,
    ) -> Dict[str, Any]:
        """
        调试模式的混合检索，返回详细信息

        Returns:
            包含BM25结果、向量结果、融合结果的详细信息
        """
        # 获取BM25结果
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.retrieve(query, bm25_k)

        # 获取向量结果
        vector_results = []
        if self.vector_store:
            try:
                vector_results_with_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=vector_k
                )
                vector_results = [
                    (doc, 1.0 - distance)
                    for doc, distance in vector_results_with_scores
                ]
            except Exception as e:
                print(f"[WARN] 向量检索失败: {e}")

        # 获取混合结果
        hybrid_results = self.hybrid_retrieve(query, k, bm25_k, vector_k)

        return {
            "query": query,
            "weights": {
                "bm25": self.bm25_weight,
                "vector": self.vector_weight
            },
            "bm25_results": [
                {
                    "content": doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content,
                    "score": score,
                    "page": doc.metadata.get("page", "unknown")
                }
                for doc, score in bm25_results[:5]
            ],
            "vector_results": [
                {
                    "content": doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content,
                    "score": score,
                    "page": doc.metadata.get("page", "unknown")
                }
                for doc, score in vector_results[:5]
            ],
            "hybrid_results": [
                {
                    "content": doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content,
                    "final_score": score,
                    "page": doc.metadata.get("page", "unknown")
                }
                for doc, score in hybrid_results
            ]
        }


# 便捷函数
def create_hybrid_retriever(
    documents: List[Document],
    vector_store,
    bm25_weight: float = 0.6,
    vector_weight: float = 0.4,
) -> HybridRetriever:
    """
    便捷创建混合检索器

    Args:
        documents: 文档列表
        vector_store: Chroma向量存储实例
        bm25_weight: BM25权重
        vector_weight: 向量权重

    Returns:
        HybridRetriever实例
    """
    bm25_retriever = BM25Retriever(documents)
    return HybridRetriever(bm25_retriever, vector_store, bm25_weight, vector_weight)
