"""
混合RAG检索器（0.4 Embedding + 0.6 BM25）

功能：
1. 混合检索（向量 + BM25）
2. 结果归一化与合并
3. RLHF线性重排序
"""
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """检索结果"""
    text: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    doc_id: str = ""


class HybridRAG:
    """混合RAG检索器"""

    # 权重配置
    VECTOR_WEIGHT = 0.4
    BM25_WEIGHT = 0.6

    # 重排序特征权重
    RERANK_FEATURES = {
        "bm25": 0.65,
        "vector": 0.30,
        "position": 0.05
    }

    def __init__(
        self,
        vector_db,
        bm25_retriever,
        vector_weight: float = None,
        bm25_weight: float = None
    ):
        """
        初始化混合RAG检索器

        Args:
            vector_db: VectorDatabase实例
            bm25_retriever: BM25Retriever实例
            vector_weight: 向量权重
            bm25_weight: BM25权重
        """
        self.vector_db = vector_db
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight or self.VECTOR_WEIGHT
        self.bm25_weight = bm25_weight or self.BM25_WEIGHT

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int = 10,
        enable_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        混合检索主入口

        Args:
            query: 查询文本
            top_k: 返回结果数量
            enable_rerank: 是否启用重排序

        Returns:
            检索结果字典
        """
        try:
            logger.info(f"开始混合检索: {query[:50]}...")

            # 1. 并行检索（这里模拟，实际可以使用asyncio）
            vector_results = self._query_vector_db(query, top_k * 2)
            bm25_results = self._query_bm25(query, top_k * 2)

            # 2. 结果合并与去重
            combined_results = self._merge_and_normalize(
                vector_results,
                bm25_results
            )

            # 3. 重排序
            if enable_rerank:
                reranked_results = self._linear_rerank(query, combined_results)
            else:
                reranked_results = combined_results

            # 4. 排序并取top_k
            final_results = sorted(
                reranked_results,
                key=lambda x: x.rerank_score if enable_rerank else x.hybrid_score,
                reverse=True
            )[:top_k]

            # 5. 格式化输出
            output = self._format_output(final_results)

            logger.info(f"混合检索完成，返回 {len(output['chunks'])} 个结果")

            return output

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise

    def _query_vector_db(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """查询向量数据库"""
        try:
            results = self.vector_db.query(query, top_k=top_k)

            formatted_results = []
            for res in results:
                formatted_results.append({
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "score": res["score"],
                    "id": res["id"]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"向量数据库查询失败: {e}")
            return []

    def _query_bm25(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """查询BM25"""
        try:
            results = self.bm25_retriever.retrieve(query, k=top_k)

            formatted_results = []
            for res in results:
                formatted_results.append({
                    "text": res["text"],
                    "metadata": res["metadata"],
                    "score": res["score"],
                    "id": res["id"]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"BM25查询失败: {e}")
            return []

    def _merge_and_normalize(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]]
    ) -> List[RetrievalResult]:
        """
        合并和归一化结果

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果

        Returns:
            合并后的结果列表
        """
        result_map = {}

        # 归一化向量得分
        vector_scores = [r["score"] for r in vector_results]
        if vector_scores:
            vec_min, vec_max = min(vector_scores), max(vector_scores)
            vec_range = vec_max - vec_min if vec_max > vec_min else 1

        # 添加向量结果
        for res in vector_results:
            doc_id = res["id"]

            if vector_scores:
                normalized_score = (res["score"] - vec_min) / vec_range
            else:
                normalized_score = 0

            result = RetrievalResult(
                text=res["text"],
                metadata=res["metadata"],
                vector_score=normalized_score,
                doc_id=doc_id
            )
            result_map[doc_id] = result

        # 归一化BM25得分
        bm25_scores = [r["score"] for r in bm25_results]
        if bm25_scores:
            bm25_min, bm25_max = min(bm25_scores), max(bm25_scores)
            bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1

        # 合并BM25结果
        for res in bm25_results:
            doc_id = res["id"]

            if bm25_scores:
                normalized_score = (res["score"] - bm25_min) / bm25_range
            else:
                normalized_score = 0

            if doc_id in result_map:
                result_map[doc_id].bm25_score = normalized_score
            else:
                result = RetrievalResult(
                    text=res["text"],
                    metadata=res["metadata"],
                    bm25_score=normalized_score,
                    doc_id=doc_id
                )
                result_map[doc_id] = result

        # 计算混合得分
        results_list = list(result_map.values())
        for result in results_list:
            result.hybrid_score = (
                self.vector_weight * result.vector_score +
                self.bm25_weight * result.bm25_score
            )

        return results_list

    def _linear_rerank(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        线性重排序（RLHF第一阶段）

        Args:
            query: 查询文本
            results: 检索结果列表

        Returns:
            重排序后的结果列表
        """
        try:
            # 为每个结果计算重排序得分
            for result in results:
                # 提取关键词
                keywords = self.bm25_retriever.extract_keywords(query, top_k=5)
                query_words = [word for word, _ in keywords]

                # 计算查询词匹配
                text_lower = result.text.lower()
                keyword_match = sum(
                    1 for word in query_words if word in text_lower
                ) / max(len(query_words), 1)

                # 位置特征（页码越小得分越高）
                page_number = result.metadata.get("page_number", 100)
                position_score = max(0, 1 - (page_number / 200))

                # 线性组合重排序
                result.rerank_score = (
                    self.RERANK_FEATURES["bm25"] * result.bm25_score +
                    self.RERANK_FEATURES["vector"] * result.vector_score +
                    self.RERANK_FEATURES["position"] * position_score +
                    0.1 * keyword_match  # 额外的关键词匹配奖励
                )

            # 归一化重排序得分
            rerank_scores = [r.rerank_score for r in results]
            if rerank_scores:
                r_min, r_max = min(rerank_scores), max(rerank_scores)
                r_range = r_max - r_min if r_max > r_min else 1

                for result in results:
                    result.rerank_score = (result.rerank_score - r_min) / r_range

            return results

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 重排序失败时返回原始结果
            return results

    def _format_output(
        self,
        results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        格式化输出

        Args:
            results: 结果列表

        Returns:
            格式化输出字典
        """
        chunks = []
        sections = []
        scores = []

        for result in results:
            chunks.append(result.text)
            sections.append(result.metadata.get("page_number", 0))
            scores.append({
                "hybrid": result.hybrid_score,
                "rerank": result.rerank_score,
                "vector": result.vector_score,
                "bm25": result.bm25_score
            })

        # 构建上下文文本
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = sections[i-1]
            context_parts.append(f"法律条文{i} (第{page}页):\n{chunk}\n")

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "chunks": chunks,
            "sections": sections,
            "scores": scores,
            "result_count": len(results)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        vector_stats = self.vector_db.get_statistics()
        bm25_stats = self.bm25_retriever.get_statistics()

        return {
            "vector_database": vector_stats,
            "bm25_retriever": bm25_stats,
            "weights": {
                "vector": self.vector_weight,
                "bm25": self.bm25_weight
            },
            "rerank_features": self.RERANK_FEATURES
        }
