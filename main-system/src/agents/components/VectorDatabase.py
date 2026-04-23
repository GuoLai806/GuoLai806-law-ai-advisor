"""
向量数据库模块（ChromaDB）

功能：
1. 向量存储与查询
2. 相似度计算
3. 持久化存储
"""
import os
import logging
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.config import Settings
from src.agents.models.BailianEmbeddings import BailianEmbeddings

logger = logging.getLogger(__name__)


class VectorDatabase:
    """ChromaDB向量数据库封装"""

    DEFAULT_PERSIST_DIR = "./chroma_civil_code"
    COLLECTION_NAME = "civil_code_collection"
    DEFAULT_EMBEDDING_DIM = 1536
    DEFAULT_METADATA = {
        "source": "民法典",
        "language": "Chinese"
    }

    def __init__(self, persist_dir: str = None, api_key: str = None):
        """
        初始化向量数据库

        Args:
            persist_dir: 持久化目录
            api_key: API密钥，None时从环境变量读取
        """
        self.persist_dir = persist_dir or self.DEFAULT_PERSIST_DIR
        if api_key is None:
            api_key = os.getenv("BAILIAN_API_KEY", "")
        self.embedding_model = BailianEmbeddings(api_key)
        self.client = None
        self.collection = None

    def initialize(self, recreate: bool = False) -> bool:
        """
        初始化向量数据库

        Args:
            recreate: 是否重新创建

        Returns:
            是否成功
        """
        try:
            logger.info(f"初始化向量数据库，持久化目录: {self.persist_dir}")

            # 初始化客户端
            self.client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    persist_directory=self.persist_dir
                )
            )

            # 检查集合是否存在
            existing_collections = [col.name for col in self.client.list_collections()]

            if self.COLLECTION_NAME in existing_collections:
                if recreate:
                    logger.warning("集合已存在，正在重建...")
                    self.client.delete_collection(self.COLLECTION_NAME)
                    self._create_collection()
                else:
                    logger.info("使用现有集合")
                    self.collection = self.client.get_collection(self.COLLECTION_NAME)
            else:
                self._create_collection()

            logger.info("向量数据库初始化成功")
            return True

        except Exception as e:
            logger.error(f"向量数据库初始化失败: {e}")
            return False

    def _create_collection(self):
        """创建集合"""
        try:
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata=self.DEFAULT_METADATA
            )
            logger.info("集合创建成功")
        except Exception as e:
            logger.error(f"集合创建失败: {e}")
            raise

    def add_documents(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[str]:
        """
        向数据库添加文档块

        Args:
            chunks: 文档块列表，格式: {"id": str, "text": str, "metadata": dict}

        Returns:
            成功添加的文档ID列表
        """
        if not self.collection:
            raise Exception("数据库未初始化")

        try:
            logger.info(f"准备添加 {len(chunks)} 个文档块")

            # 提取数据
            ids = []
            texts = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                doc_id = str(i)
                ids.append(doc_id)
                texts.append(chunk.get("text", ""))
                metadatas.append(chunk.get("metadata", {}))

            # 生成嵌入向量
            logger.info("正在生成嵌入向量...")
            embeddings = self.embedding_model.embed_documents(texts)

            # 添加到数据库
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"成功添加 {len(ids)} 个文档块")
            return ids

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        metadata_filter: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        查询向量数据库

        Args:
            query_text: 查询文本
            top_k: 返回数量
            metadata_filter: 元数据过滤

        Returns:
            查询结果: [{"text": str, "metadata": dict, "score": float, "id": str}]
        """
        if not self.collection:
            raise Exception("数据库未初始化")

        try:
            logger.info(f"查询向量数据库: {query_text[:50]}...")

            # 生成查询向量
            query_embedding = self.embedding_model.embed_query(query_text)

            # 查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter
            )

            # 格式化结果
            formatted_results = []
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # 转换为相似度
                })

            logger.info(f"查询完成，返回 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"查询失败: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息

        Returns:
            统计信息
        """
        if not self.collection:
            return {"document_count": 0}

        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "persist_dir": self.persist_dir,
                "collection_name": self.COLLECTION_NAME
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {"document_count": 0}

    def reset(self) -> bool:
        """
        重置数据库（危险操作）

        Returns:
            是否成功
        """
        try:
            logger.warning("正在重置向量数据库")

            if self.COLLECTION_NAME in [col.name for col in self.client.list_collections()]:
                self.client.delete_collection(self.COLLECTION_NAME)
            self._create_collection()

            logger.warning("向量数据库重置成功")
            return True

        except Exception as e:
            logger.error(f"重置失败: {e}")
            return False

    def search_by_keyword(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        简单的关键词搜索（用于调试）

        Args:
            keyword: 关键词
            top_k: 返回数量

        Returns:
            搜索结果
        """
        if not self.collection:
            raise Exception("数据库未初始化")

        try:
            # 查询所有文档
            all_docs = self.collection.get(include=["documents", "metadatas"])

            # 筛选包含关键词的文档
            matches = []
            for i, doc in enumerate(all_docs["documents"]):
                if keyword in doc:
                    matches.append({
                        "id": all_docs["ids"][i],
                        "text": doc,
                        "metadata": all_docs["metadatas"][i],
                        "score": doc.count(keyword)  # 简单的词频得分
                    })

            # 排序并返回
            matches.sort(key=lambda x: x["score"], reverse=True)
            return matches[:top_k]

        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            raise
