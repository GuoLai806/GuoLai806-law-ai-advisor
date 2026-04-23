"""
民法典PDF处理模块

功能：
1. PDF解析与OCR
2. 智能语义分块
3. 元数据提取
"""
import os
import re
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """文档块"""
    content: str
    page_number: int
    chunk_index: int
    metadata: Dict[str, Any]


class PDFProcessor:
    """PDF处理器"""

    # 中文标点符号
    CHINESE_PUNCTUATION = r'[。！？；\n]+'
    # 法律条款标题模式
    LAW_TITLE_PATTERNS = [
        r'第[零一二三四五六七八九十百千万0-9]+条',
        r'第[零一二三四五六七八九十百千万0-9]+章',
        r'第[零一二三四五六七八九十百千万0-9]+节',
        r'第[零一二三四五六七八九十百千万0-9]+编',
    ]

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        初始化PDF处理器

        Args:
            chunk_size: 每个块的字符数
            overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """
        处理PDF文件并返回分块结果

        Args:
            pdf_path: PDF文件路径

        Returns:
            分块列表
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

        logger.info(f"开始处理PDF文件: {pdf_path}")

        try:
            # 打开PDF
            doc = fitz.open(pdf_path)
            logger.info(f"PDF包含 {len(doc)} 页")

            all_chunks = []
            chunk_index = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()

                # 提取页码信息
                page_info = {
                    "page_number": page_num + 1,  # 从1开始
                    "source_file": os.path.basename(pdf_path)
                }

                # 处理页面文本
                page_chunks = self._process_page_text(text, page_info, chunk_index)
                all_chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

            doc.close()
            logger.info(f"PDF处理完成，共生成 {len(all_chunks)} 个块")

            return all_chunks

        except Exception as e:
            logger.error(f"PDF处理失败: {e}")
            raise

    def _process_page_text(
        self,
        text: str,
        page_info: Dict[str, Any],
        start_chunk_index: int
    ) -> List[DocumentChunk]:
        """
        处理单个页面的文本并分块

        Args:
            text: 页面文本
            page_info: 页面元数据
            start_chunk_index: 起始块索引

        Returns:
            分块列表
        """
        # 清理文本
        text = self._clean_text(text)

        if not text.strip():
            return []

        # 语义分割
        segments = self._semantic_segmentation(text)

        # 分块
        chunks = []
        current_chunk = ""
        chunk_index = start_chunk_index

        for segment in segments:
            # 如果当前块加上新段超出chunk_size，就保存当前块
            if len(current_chunk) + len(segment) > self.chunk_size and current_chunk:
                chunks.append(DocumentChunk(
                    content=current_chunk.strip(),
                    page_number=page_info["page_number"],
                    chunk_index=chunk_index,
                    metadata=page_info.copy()
                ))
                chunk_index += 1

                # 应用重叠
                if self.overlap > 0:
                    current_chunk = current_chunk[-self.overlap:] + segment
                else:
                    current_chunk = segment
            else:
                current_chunk += segment

        # 保存最后一个块
        if current_chunk.strip():
            chunks.append(DocumentChunk(
                content=current_chunk.strip(),
                page_number=page_info["page_number"],
                chunk_index=chunk_index,
                metadata=page_info.copy()
            ))

        return chunks

    def _clean_text(self, text: str) -> str:
        """
        清理文本

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        # 移除页眉页脚模式（可以根据实际PDF调整）
        text = re.sub(r'中华人民共和国民法典.*?最高人民法院', '', text)
        # 移除页码（简单模式）
        text = re.sub(r'第\s*\d+\s*页', '', text)
        # 整理换行
        text = text.strip()
        return text

    def _semantic_segmentation(self, text: str) -> List[str]:
        """
        语义分割

        Args:
            text: 文本

        Returns:
            分割后的片段列表
        """
        segments = []
        current_segment = ""
        words = list(text)

        for i, char in enumerate(words):
            current_segment += char

            # 检查是否是句子结束
            if char in '。！？；':
                # 检查后面的内容是否是新的法律标题
                next_text = ''.join(words[i+1:i+20]) if i+1 < len(words) else ''

                # 如果是标题或足够长，就分割
                is_title = any(re.search(pattern, next_text) for pattern in self.LAW_TITLE_PATTERNS)
                if is_title or len(current_segment) >= 200:
                    segments.append(current_segment)
                    current_segment = ""

        # 添加最后一段
        if current_segment.strip():
            segments.append(current_segment)

        return segments

    def get_chunk_for_query(
        self,
        chunks: List[DocumentChunk],
        query: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        简单的关键词匹配（用于早期测试）

        Args:
            chunks: 分块列表
            query: 查询
            top_k: 返回数量

        Returns:
            匹配的分块
        """
        # 简单的关键词匹配
        query_keywords = self._extract_keywords(query)

        # 评分
        scored_chunks = []
        for chunk in chunks:
            score = 0
            for keyword in query_keywords:
                if keyword in chunk.content:
                    score += chunk.content.count(keyword)
            if score > 0:
                scored_chunks.append((score, chunk))

        # 排序并返回top_k
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def _extract_keywords(self, text: str) -> List[str]:
        """
        提取关键词（简单实现）

        Args:
            text: 文本

        Returns:
            关键词列表
        """
        # 简单实现：提取大于等于2个字符的词
        words = re.findall(r'[\w\u4e00-\u9fff]+', text)
        keywords = [word for word in words if len(word) >= 2]
        return keywords
