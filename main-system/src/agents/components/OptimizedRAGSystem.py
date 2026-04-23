"""
民法典 RAG 系统（0.4 Embedding + 0.6 BM25 + RLHF 线性重排序）

功能：
1. 民法典PDF文档处理与分块
2. ChromaDB向量存储
3. Whoosh BM25检索
4. 混合检索（0.4向量 + 0.6 BM25）
5. RLHF线性重排序
"""
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .PDFProcessor import PDFProcessor, DocumentChunk
from .VectorDatabase import VectorDatabase
from .BM25Retriever import BM25Retriever
from .HybridRAG import HybridRAG, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG 配置类"""
    # PDF 配置
    pdf_path: str = "中华人民共和国民法典 - 中华人民共和国最高人民法院.pdf"
    persist_dir: str = "./chroma_civil_code"
    index_dir: str = "./bm25_civil_code"

    # 分块配置
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # 检索配置
    default_k: int = 3

    # 混合检索配置
    vector_weight: float = 0.4
    bm25_weight: float = 0.6

    # RLHF重排序配置
    use_rerank: bool = True
    rerank_features = {
        "bm25": 0.65,
        "vector": 0.30,
        "position": 0.05
    }

    # 启用特征
    enable_vector: bool = True
    enable_bm25: bool = True
    enable_rerank: bool = True


class OptimizedRAGSystem:
    """优化后的 RAG 系统"""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化 RAG 系统

        Args:
            config: RAG 配置对象，为 None 时使用默认配置
        """
        self.config = config or RAGConfig()

        # 组件初始化
        self.pdf_processor = PDFProcessor(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        self.vector_db = VectorDatabase(persist_dir=self.config.persist_dir)
        self.bm25_retriever = BM25Retriever(index_dir=self.config.index_dir)
        self.hybrid_rag = None
        self.documents = []

        self._initialized = False

    def initialize(self, force_rebuild: bool = False) -> bool:
        """
        初始化 RAG 系统，包括创建或加载索引

        Args:
            force_rebuild: 是否强制重建索引

        Returns:
            初始化是否成功
        """
        if self._initialized and not force_rebuild:
            logger.info("RAG 系统已初始化，跳过")
            return True

        logger.info("正在初始化 RAG 系统...")

        try:
            # 1. 检查是否需要重建索引
            needs_rebuilding = force_rebuild or self._needs_rebuilding()

            if needs_rebuilding:
                logger.info("需要重建索引")
                self._rebuild_index()
            else:
                logger.info("加载现有索引")
                self._load_index()

            self._initialized = True
            logger.info("RAG 系统初始化成功")
            return True

        except Exception as e:
            logger.error(f"RAG 系统初始化失败: {e}")
            self._initialized = False
            return False

    def _needs_rebuilding(self) -> bool:
        """检查是否需要重建索引"""
        # 检查向量数据库（只检查目录是否存在即可）
        vector_exists = os.path.exists(self.config.persist_dir) and os.listdir(self.config.persist_dir)
        # 检查 BM25 索引（只检查目录是否存在即可）
        bm25_exists = os.path.exists(self.config.index_dir) and os.listdir(self.config.index_dir)

        if not vector_exists or not bm25_exists:
            logger.warning("未找到完整索引，需要重建")
            return True

        logger.info("现有索引已存在，不需要重建")
        return False

    def _rebuild_index(self):
        """重建索引（处理 PDF）"""
        logger.info("开始重建索引...")

        # 1. 处理 PDF 分块
        logger.info("处理民法典 PDF 文件...")
        chunks = self.pdf_processor.process_pdf(self.config.pdf_path)

        # 2. 准备数据格式
        self.documents = []
        for chunk in chunks:
            self.documents.append({
                "id": str(chunk.chunk_index),
                "text": chunk.content,
                "metadata": chunk.metadata
            })

        # 3. 初始化向量数据库
        logger.info("初始化向量数据库...")
        self.vector_db.initialize(recreate=True)
        self.vector_db.add_documents(self.documents)

        # 4. 初始化 BM25 检索器
        logger.info("初始化 BM25 检索器...")
        self.bm25_retriever.initialize(recreate=True)
        self.bm25_retriever.add_documents(self.documents)

        # 5. 初始化混合检索器
        self.hybrid_rag = HybridRAG(
            vector_db=self.vector_db,
            bm25_retriever=self.bm25_retriever,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight
        )

        logger.info(f"索引重建完成，共 {len(chunks)} 个文档块")

    def _load_index(self):
        """加载现有索引"""
        logger.info("加载向量数据库...")
        self.vector_db.initialize(recreate=False)

        logger.info("加载 BM25 索引...")
        self.bm25_retriever.initialize(recreate=False)

        self.hybrid_rag = HybridRAG(
            vector_db=self.vector_db,
            bm25_retriever=self.bm25_retriever,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight
        )

    def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """
        检索相关法律条文

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            检索结果:
            {
                "context": 格式化的上下文文本,
                "chunks": 文档块列表,
                "sections": 页码列表,
                "scores": 得分信息
            }
        """
        import time
        start_time = time.time()

        if not self._initialized:
            logger.warning("RAG 系统未初始化，自动初始化")
            if not self.initialize():
                return {
                    "context": "RAG 系统初始化失败，无法提供法律条文支持",
                    "chunks": [],
                    "sections": [],
                    "scores": []
                }

        if not self.hybrid_rag:
            logger.error("混合检索器未初始化")
            return {
                "context": "检索系统未初始化",
                "chunks": [],
                "sections": [],
                "scores": []
            }

        try:
            k = k or self.config.default_k

            retrieve_start = time.time()
            result = self.hybrid_rag.hybrid_retrieve(
                query,
                top_k=k,
                enable_rerank=self.config.enable_rerank
            )
            retrieve_time = (time.time() - retrieve_start) * 1000
            total_time = (time.time() - start_time) * 1000

            chunk_count = len(result.get("chunks", []))
            logger.info(f"⏱️ [RAG检索] 耗时: {retrieve_time:.1f}ms | 找到 {chunk_count} 个结果 | 总耗时: {total_time:.1f}ms")

            return result

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {
                "context": f"检索法律条文时出错: {e}",
                "chunks": [],
                "sections": [],
                "scores": []
            }

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        vector_stats = self.vector_db.get_statistics()
        bm25_stats = self.bm25_retriever.get_statistics()

        return {
            "initialized": self._initialized,
            "document_count": vector_stats.get("document_count", 0),
            "vector_stats": vector_stats,
            "bm25_stats": bm25_stats,
            "retrieval_weights": {
                "vector": self.config.vector_weight,
                "bm25": self.config.bm25_weight
            },
            "rerank_features": self.config.rerank_features if self.config.enable_rerank else None
        }

    def reset_rerank_features(self, features: Dict[str, float]):
        """
        重置重排序特征权重

        Args:
            features: 特征权重字典
        """
        if self.hybrid_rag:
            self.hybrid_rag.RERANK_FEATURES = features
            logger.info("重排序特征权重已更新")

    def rebuild_index(self):
        """重建索引"""
        return self.initialize(force_rebuild=True)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 测试代码
    logger.info("=" * 50)
    logger.info("民法典 RAG 系统测试")
    logger.info("=" * 50)

    # 创建 RAG 系统
    rag = OptimizedRAGSystem()

    # 初始化
    logger.info("正在初始化 RAG 系统...")
    success = rag.initialize()

    if success:
        logger.info("✅ RAG 系统初始化成功！")
        stats = rag.get_statistics()
        logger.info(f"📊 统计信息: {stats}")

        # 测试检索
        logger.info("\n🔍 测试检索:")
        test_queries = [
            "合同违约怎么办？",
            "什么是合同纠纷？",
            "民法典第一条内容是什么？",
            "知识产权保护"
        ]

        for query in test_queries:
            logger.info(f"\n查询: {query}")
            result = rag.retrieve(query)

            if result["chunks"]:
                logger.info(f"找到 {len(result['chunks'])} 个相关结果")
                logger.info("\n相关法律条文:")
                for i, (chunk, score) in enumerate(
                    zip(result["chunks"], [s["hybrid"] for s in result["scores"]]), 1
                ):
                    logger.info(f"\n#{i} (得分: {score:.3f}):")
                    logger.info(f"{chunk[:200]}...")
            else:
                logger.warning("未找到相关内容")
    else:
        logger.error("❌ RAG 系统初始化失败")
