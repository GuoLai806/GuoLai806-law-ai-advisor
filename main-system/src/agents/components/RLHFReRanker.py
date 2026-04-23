"""
RLHF重排序器 - 线性强化学习重排序

作者：GuoLai
版本：1.0.0
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from langchain.schema import Document


@dataclass
class FeedbackItem:
    """反馈数据项"""
    query: str
    doc_content: str
    doc_metadata: Dict[str, Any]
    original_rank: int
    original_score: float
    is_positive: bool  # True表示用户认为好，False表示不好
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass
class FeatureWeights:
    """特征权重"""
    # 检索分数特征
    bm25_score_weight: float = 0.3
    vector_score_weight: float = 0.3

    # 文档特征
    doc_length_weight: float = 0.1
    keyword_match_weight: float = 0.2
    page_recency_weight: float = 0.05
    chunk_position_weight: float = 0.05


class RLHFReRanker:
    """
    线性RLHF重排序器

    核心思想：
    1. 基于用户反馈数据调整特征权重
    2. 使用线性模型计算最终重排序分数
    3. 支持特征工程和权重优化
    4. 持久化反馈数据和权重

    特征包括：
    - BM25检索分数
    - 向量检索分数
    - 文档长度
    - 关键词匹配度
    - 页码新鲜度
    - 文档块位置
    """

    def __init__(
        self,
        data_dir: str = "./rlhf_data",
        learning_rate: float = 0.01,
    ):
        """
        初始化RLHF重排序器

        Args:
            data_dir: 反馈数据存储目录
            learning_rate: 学习率
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_file = self.data_dir / "feedback.json"
        self.weights_file = self.data_dir / "weights.json"

        self.learning_rate = learning_rate
        self.feedbacks: List[FeedbackItem] = []
        self.weights = FeatureWeights()

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载反馈数据和权重"""
        # 加载权重
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r', encoding='utf-8') as f:
                    weights_dict = json.load(f)
                    for key, value in weights_dict.items():
                        if hasattr(self.weights, key):
                            setattr(self.weights, key, value)
                print(f"[OK] 已加载权重: {self.weights}")
            except Exception as e:
                print(f"[WARN] 加载权重失败: {e}")

        # 加载反馈
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_list = json.load(f)
                    self.feedbacks = [FeedbackItem(**item) for item in feedback_list]
                print(f"[OK] 已加载 {len(self.feedbacks)} 条反馈数据")
            except Exception as e:
                print(f"[WARN] 加载反馈失败: {e}")

    def _save_data(self):
        """保存反馈数据和权重"""
        # 保存权重
        try:
            weights_dict = {
                k: v for k, v in self.weights.__dict__.items()
                if not k.startswith('_')
            }
            with open(self.weights_file, 'w', encoding='utf-8') as f:
                json.dump(weights_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 保存权重失败: {e}")

        # 保存反馈（只保留最近1000条）
        try:
            feedback_list = []
            for fb in self.feedbacks[-1000:]:
                feedback_list.append({
                    k: v for k, v in fb.__dict__.items()
                    if not k.startswith('_')
                })
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 保存反馈失败: {e}")

    def _extract_features(
        self,
        query: str,
        doc: Document,
        bm25_score: float = 0.0,
        vector_score: float = 0.0,
        index: int = 0,
        total_docs: int = 10,
    ) -> Dict[str, float]:
        """
        提取文档特征

        Args:
            query: 查询文本
            doc: 文档
            bm25_score: BM25分数
            vector_score: 向量分数
            index: 文档索引
            total_docs: 总文档数

        Returns:
            特征字典
        """
        features = {}

        # 1. 检索分数特征
        features['bm25_score'] = bm25_score
        features['vector_score'] = vector_score

        # 2. 文档长度特征（归一化）
        doc_length = len(doc.page_content)
        features['doc_length'] = min(doc_length / 2000, 1.0)

        # 3. 关键词匹配特征
        query_keywords = set(query.replace('的', '').replace('了', '').split())
        doc_text = doc.page_content
        match_count = sum(1 for kw in query_keywords if kw in doc_text)
        features['keyword_match'] = min(match_count / max(len(query_keywords), 1), 1.0)

        # 4. 页码新鲜度（页码越小权重越高）
        page = doc.metadata.get('page', 1000)
        features['page_recency'] = max(0, 1.0 - page / 1000)

        # 5. 文档块位置（越靠前越好）
        chunk_id = doc.metadata.get('chunk_id', total_docs)
        features['chunk_position'] = max(0, 1.0 - chunk_id / max(total_docs, 1))

        return features

    def _compute_final_score(
        self,
        features: Dict[str, float],
    ) -> float:
        """
        计算最终重排序分数 - 线性模型

        Args:
            features: 特征字典

        Returns:
            最终分数
        """
        score = (
            self.weights.bm25_score_weight * features.get('bm25_score', 0) +
            self.weights.vector_score_weight * features.get('vector_score', 0) +
            self.weights.doc_length_weight * features.get('doc_length', 0) +
            self.weights.keyword_match_weight * features.get('keyword_match', 0) +
            self.weights.page_recency_weight * features.get('page_recency', 0) +
            self.weights.chunk_position_weight * features.get('chunk_position', 0)
        )
        return score

    def _update_weights_from_feedback(self, feedback: FeedbackItem):
        """
        根据单条反馈更新权重（梯度下降）

        Args:
            feedback: 反馈数据
        """
        # 这里简化处理：
        # 正反馈：增加相关特征的权重
        # 负反馈：减少相关特征的权重

        # 这是一个简化的实现，实际RLHF会更复杂
        if feedback.is_positive:
            # 正反馈：增加关键词匹配和向量分数的权重
            self.weights.keyword_match_weight += self.learning_rate * 0.1
            self.weights.vector_score_weight += self.learning_rate * 0.1
        else:
            # 负反馈：减少BM25的权重
            self.weights.bm25_score_weight -= self.learning_rate * 0.1

        # 归一化权重，确保总和为1
        self._normalize_weights()

    def _normalize_weights(self):
        """归一化权重，确保总和为1"""
        weights = [
            self.weights.bm25_score_weight,
            self.weights.vector_score_weight,
            self.weights.doc_length_weight,
            self.weights.keyword_match_weight,
            self.weights.page_recency_weight,
            self.weights.chunk_position_weight,
        ]

        # 确保权重不为负
        weights = [max(0.01, w) for w in weights]
        total = sum(weights)

        # 归一化
        self.weights.bm25_score_weight = weights[0] / total
        self.weights.vector_score_weight = weights[1] / total
        self.weights.doc_length_weight = weights[2] / total
        self.weights.keyword_match_weight = weights[3] / total
        self.weights.page_recency_weight = weights[4] / total
        self.weights.chunk_position_weight = weights[5] / total

    def add_feedback(
        self,
        query: str,
        doc: Document,
        original_rank: int,
        original_score: float,
        is_positive: bool,
    ):
        """
        添加用户反馈

        Args:
            query: 查询文本
            doc: 文档
            original_rank: 原始排名
            original_score: 原始分数
            is_positive: 是否为正反馈
        """
        feedback = FeedbackItem(
            query=query,
            doc_content=doc.page_content,
            doc_metadata=doc.metadata,
            original_rank=original_rank,
            original_score=original_score,
            is_positive=is_positive,
        )

        self.feedbacks.append(feedback)

        # 更新权重
        self._update_weights_from_feedback(feedback)

        # 保存数据
        self._save_data()

        print(f"[OK] 反馈已记录: {'[LIKE]' if is_positive else '[DISLIKE]'} '{query[:30]}...'")

    def rerank(
        self,
        query: str,
        doc_score_pairs: List[Tuple[Document, float]],
        bm25_scores: Optional[List[float]] = None,
        vector_scores: Optional[List[float]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        重排序核心方法

        Args:
            query: 查询文本
            doc_score_pairs: [(doc, original_score), ...]列表
            bm25_scores: 可选的BM25分数列表
            vector_scores: 可选的向量分数列表

        Returns:
            重排序后的[(doc, final_score), ...]列表
        """
        if not doc_score_pairs:
            return []

        total_docs = len(doc_score_pairs)
        reranked_with_features = []

        for i, (doc, original_score) in enumerate(doc_score_pairs):
            # 获取BM25和向量分数
            bm25_score = bm25_scores[i] if (bm25_scores and i < len(bm25_scores)) else 0.0
            vector_score = vector_scores[i] if (vector_scores and i < len(vector_scores)) else original_score

            # 提取特征
            features = self._extract_features(
                query=query,
                doc=doc,
                bm25_score=bm25_score,
                vector_score=vector_score,
                index=i,
                total_docs=total_docs,
            )

            # 计算最终分数
            final_score = self._compute_final_score(features)

            reranked_with_features.append({
                'doc': doc,
                'original_score': original_score,
                'final_score': final_score,
                'features': features,
                'original_rank': i,
            })

        # 按最终分数排序
        reranked_with_features.sort(
            key=lambda x: x['final_score'],
            reverse=True
        )

        # 返回结果
        return [
            (item['doc'], item['final_score'])
            for item in reranked_with_features
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_feedbacks": len(self.feedbacks),
            "positive_feedbacks": sum(1 for fb in self.feedbacks if fb.is_positive),
            "negative_feedbacks": sum(1 for fb in self.feedbacks if not fb.is_positive),
            "current_weights": {
                k: v for k, v in self.weights.__dict__.items()
                if not k.startswith('_')
            },
        }

    def reset_weights(self):
        """重置权重为默认值"""
        self.weights = FeatureWeights()
        self._save_data()
        print("[OK] 权重已重置")


# 便捷函数
def create_rlhf_reranker(data_dir: str = "./rlhf_data") -> RLHFReRanker:
    """
    便捷创建RLHF重排序器

    Args:
        data_dir: 数据存储目录

    Returns:
        RLHFReRanker实例
    """
    return RLHFReRanker(data_dir)
