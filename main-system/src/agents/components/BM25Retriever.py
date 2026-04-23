"""
BM25检索器 - 关键词匹配检索

作者：GuoLai
版本：2.0.0
"""
import os
import re
import json
import logging
import jieba
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25关键词检索器

    功能：
    1. 基于BM25算法的关键词匹配
    2. 支持中文分词
    3. 支持停用词过滤
    4. 提供Top-K检索结果
    5. 支持持久化（模拟）
    """

    DEFAULT_INDEX_DIR = "./bm25_civil_code"

    def __init__(self, index_dir: str = None):
        """
        初始化BM25检索器

        Args:
            index_dir: 索引存储目录
        """
        self.index_dir = index_dir or self.DEFAULT_INDEX_DIR
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._initialized = False
        self.stop_words = self._load_default_stop_words()

    def _load_default_stop_words(self) -> set:
        """加载默认停用词"""
        return {
            "的", "了", "在", "是", "我", "有", "和", "就",
            "不", "人", "都", "一", "一个", "上", "也", "很",
            "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "那", "有什么", "怎么办",
            "如何", "怎么", "什么", "为什么", "哪个", "哪些",
            "请问", "麻烦", "那个", "这个", "呢", "吗", "吧",
            "啊", "呀", "噢", "哦", "啦", "嘛", "咯"
        }

    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本

        步骤：
        1. 去除特殊字符
        2. 统一大小写
        3. 去除多余空格
        """
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词

        步骤：
        1. 预处理
        2. jieba分词
        3. 过滤停用词
        """
        text = self._preprocess_text(text)
        tokens = jieba.lcut(text)
        tokens = [token for token in tokens
                 if token.strip() and token not in self.stop_words]
        return tokens

    def _save_documents(self):
        """保存文档到磁盘"""
        try:
            docs_file = os.path.join(self.index_dir, "documents.json")
            with open(docs_file, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)
            logger.debug(f"文档已保存到: {docs_file}")
        except Exception as e:
            logger.error(f"保存文档失败: {e}")

    def _load_documents(self) -> bool:
        """从磁盘加载文档"""
        try:
            docs_file = os.path.join(self.index_dir, "documents.json")
            if os.path.exists(docs_file):
                with open(docs_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

                # 重新分词并构建索引
                self.tokenized_docs = []
                for doc in self.documents:
                    tokens = self._tokenize(doc.get("text", ""))
                    self.tokenized_docs.append(tokens)

                if self.tokenized_docs:
                    self.bm25 = BM25Okapi(self.tokenized_docs)

                logger.info(f"已加载 {len(self.documents)} 个文档")
                return True
            return False
        except Exception as e:
            logger.error(f"加载文档失败: {e}")
            return False

    def initialize(self, recreate: bool = False) -> bool:
        """
        初始化BM25检索器

        Args:
            recreate: 是否重新创建

        Returns:
            是否成功
        """
        try:
            logger.info(f"初始化BM25检索器，索引目录: {self.index_dir}")

            # 确保目录存在
            os.makedirs(self.index_dir, exist_ok=True)

            # 检查是否有现有索引
            marker_file = os.path.join(self.index_dir, "bm25_index.marker")

            if os.path.exists(marker_file) and not recreate:
                logger.info("正在加载现有BM25索引...")
                if self._load_documents():
                    self._initialized = True
                    logger.info("BM25索引加载成功")
                    return True
                else:
                    logger.warning("索引加载失败，将重新创建")

            # 清空现有数据
            self.documents = []
            self.tokenized_docs = []
            self.bm25 = None

            self._initialized = True
            logger.info("BM25检索器初始化成功（新索引）")
            return True

        except Exception as e:
            logger.error(f"BM25检索器初始化失败: {e}")
            return False

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        添加文档到BM25索引

        Args:
            chunks: 文档块列表，格式: [{"id": str, "text": str, "metadata": dict}]
        """
        if not self._initialized:
            raise Exception("BM25检索器未初始化")

        try:
            logger.info(f"准备添加 {len(chunks)} 个文档到BM25索引")

            for chunk in chunks:
                self.documents.append(chunk)
                tokens = self._tokenize(chunk.get("text", ""))
                self.tokenized_docs.append(tokens)

            # 重建BM25索引
            if self.tokenized_docs:
                self.bm25 = BM25Okapi(self.tokenized_docs)

            # 保存文档和索引
            self._save_documents()

            # 创建标记文件
            marker_file = os.path.join(self.index_dir, "bm25_index.marker")
            with open(marker_file, "w", encoding="utf-8") as f:
                f.write(f"Indexed {len(self.documents)} documents\n")

            logger.info(f"成功添加 {len(chunks)} 个文档到BM25索引")

        except Exception as e:
            logger.error(f"添加文档到BM25索引失败: {e}")
            raise

    def clear(self):
        """清空索引"""
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None

    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        BM25检索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            结果列表: [{"id": str, "text": str, "metadata": dict, "score": float}]
        """
        if not self.bm25 or not self.documents:
            return []

        try:
            # 查询分词
            query_tokens = self._tokenize(query)

            if not query_tokens:
                return []

            # 计算BM25分数
            scores = self.bm25.get_scores(query_tokens)

            # 获取Top-K结果
            doc_score_pairs = list(zip(self.documents, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # 格式化结果
            results = []
            for doc, score in doc_score_pairs[:k]:
                results.append({
                    "id": doc.get("id", ""),
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(score)
                })

            return results

        except Exception as e:
            logger.error(f"BM25检索失败: {e}")
            return []

    def extract_keywords(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        提取关键词（用于重排序）

        Args:
            text: 文本
            top_k: 返回数量

        Returns:
            [(word, score), ...]
        """
        tokens = self._tokenize(text)
        if not tokens:
            return []

        # 简单的词频统计
        from collections import Counter
        word_counts = Counter(tokens)
        total = sum(word_counts.values())

        results = []
        for word, count in word_counts.most_common(top_k):
            results.append((word, count / total if total > 0 else 0))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            "document_count": len(self.documents),
            "index_dir": self.index_dir,
            "initialized": self._initialized
        }

    def search_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        简单的关键词搜索（用于调试）

        Args:
            keyword: 关键词
            top_k: 返回数量

        Returns:
            搜索结果
        """
        matches = []
        for doc in self.documents:
            text = doc.get("text", "")
            if keyword in text:
                matches.append({
                    "id": doc.get("id", ""),
                    "text": text,
                    "metadata": doc.get("metadata", {}),
                    "score": text.count(keyword)  # 简单的词频得分
                })

        # 排序并返回
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]


# 便捷函数
def create_bm25_retriever(index_dir: str = None) -> BM25Retriever:
    """
    便捷创建BM25检索器

    Args:
        index_dir: 索引目录

    Returns:
        BM25Retriever实例
    """
    return BM25Retriever(index_dir=index_dir)
