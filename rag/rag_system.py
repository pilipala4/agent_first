import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
import markdown
import re
from typing import List, Dict, Any, Optional
import argparse
import sys



class DocumentRAG:
    def __init__(self, db_path="./chroma_db"):
        """初始化 ChromaDB 客户端，使用指定路径"""
        self.db_path = db_path
        print(f"Initializing database at: {os.path.abspath(db_path)}")

        # 创建目录如果不存在
        os.makedirs(db_path, exist_ok=True)

        # 初始化客户端
        self.client = chromadb.PersistentClient(path=db_path)

        # 使用默认嵌入函数（仅做向量转换，不涉及大模型推理）
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        # 创建集合
        self.collection_name = "documents"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Database initialized. Current collection count: {self.collection.count()}")
        print("Note: This system only performs vector similarity search, no LLM involved in retrieval.")

    def read_pdf(self, file_path: str) -> str:
        """读取 PDF 文件内容"""
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def read_markdown(self, file_path: str) -> str:
        """读取 Markdown 文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 转换 Markdown 为纯文本
        html = markdown.markdown(content)
        # 移除 HTML 标签，获取纯文本
        clean_text = re.sub('<[^<]+?>', '', html)
        return clean_text

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """智能分块：确保标题（含'引言'）独立成块"""
        # 中英文常见标题关键词（按学术论文结构排序）
        title_keywords = [
            '摘要', 'Abstract', '引言', 'Introduction', '绪论', '背景', 'Background',
            '相关工作', 'Related Work', '方法', 'Method', 'Methodology', '实验', 'Experiment',
            '结果', 'Result', '讨论', 'Discussion', '结论', 'Conclusion', '参考文献', 'References',
            '致谢', 'Acknowledgement', '附录', 'Appendix'
        ]

        # 构建健壮的标题匹配正则（匹配：换行+可选编号+标题词+标点+换行）
        pattern = r'(\n\s*(?:' \
                  r'(?:第[零一二三四五六七八九十百千]+[章节]|Chapter\s*\d+|[0-9]+\.[0-9]*|[\dIVX]+)\s*[:：]?\s*)?' \
                  r'(?:' + '|'.join(re.escape(kw) for kw in title_keywords) + r')' \
                                                                              r'\s*[:：]?\s*\n)'

        sections = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)

        # 无标题匹配 → 全文按基础逻辑分割
        if len(sections) <= 1:
            return self._split_text_with_overlap(text, chunk_size, overlap)

        chunks = []
        # sections[0]: 标题前的前置内容（如封面文字）
        if sections[0].strip():
            chunks.extend(self._split_text_with_overlap(sections[0], chunk_size, overlap))

        # 交替处理：[标题, 内容, 标题, 内容...]
        for i in range(1, len(sections), 2):
            title_part = sections[i].strip() if i < len(sections) else ""
            content_part = sections[i + 1] if i + 1 < len(sections) else ""

            # 标题块独立保留（关键！确保"引言"等完整存在）
            if title_part:
                chunks.append(title_part)  # 不strip换行，保留原始格式特征

            # 内容块安全分割
            if content_part.strip():
                chunks.extend(self._split_text_with_overlap(content_part, chunk_size, overlap))

        return chunks



    def _split_text_with_overlap(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """安全分割长文本块（保留语义边界）"""
        if not text.strip():
            return []
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # 优先在句子结尾/换行处分割
            if end < text_len:
                # 尝试找最近的句号、换行或空格
                candidates = [
                    text.rfind('.', start, end),
                    text.rfind('。', start, end),
                    text.rfind('\n', start, end),
                    text.rfind(' ', start, end)
                ]
                valid_pos = [p for p in candidates if p != -1 and p > start]
                if valid_pos:
                    end = max(valid_pos) + 1  # 包含标点

            chunk = text[start:end].strip()
            if chunk:  # 跳过空块
                chunks.append(chunk)

            # 计算下个起点（避免负索引）
            next_start = end - overlap
            if next_start <= start:  # 防止死循环
                next_start = end
            start = next_start

        return chunks



    def add_document(self, file_path: str, doc_type: str = None, metadata: Optional[Dict] = None, chunk_size: int = 500, overlap: int = 50):
        """添加文档到向量数据库，支持自定义元数据"""
        if not metadata:
            metadata = {}

        if not doc_type:
            if file_path.lower().endswith('.pdf'):
                doc_type = 'pdf'
            elif file_path.lower().endswith('.md'):
                doc_type = 'markdown'
            else:
                raise ValueError("Unsupported file type. Only PDF and MD files are supported.")

        # 读取文档内容
        if doc_type == 'pdf':
            content = self.read_pdf(file_path)
        elif doc_type == 'markdown':
            content = self.read_markdown(file_path)
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

        # 分块处理
        chunks = self.chunk_text(text=content, chunk_size=chunk_size, overlap=overlap)

        

        # 添加到集合中
        documents = []
        metadatas = []
        ids = []

        # 为每个chunk添加文档标识
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            # 合并默认元数据和自定义元数据
            chunk_metadata = {
                "source": file_path,
                "chunk_id": i,
                "doc_type": doc_type,
                "doc_id": doc_id
            }
            chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
            ids.append(f"{doc_id}_chunk_{i}")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Added {len(chunks)} chunks from {file_path} to collection '{self.collection_name}'")
        print(f"Database location: {os.path.abspath(self.db_path)}")
        self.print_db_size()

    def print_db_size(self):
        """打印数据库占用的空间大小"""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(self.db_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        size_mb = total_size / (1024 * 1024)
        print(f"Current database size: {size_mb:.2f} MB")

    def query(self, question: str, n_results: int = 3, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """查询文档 - 仅执行向量相似度搜索，不涉及LLM

        参数:
        question: 查询问题
        n_results: 返回结果数量
        filter_metadata: 元数据过滤条件，例如 {"doc_id": "基于改进YOLOv5s"}
        """
        print(f"Performing vector similarity search for: '{question}'")
        print("(This only finds semantically similar text chunks, no LLM summarization occurs here)")

        # 如果有元数据过滤条件，使用ChromaDB的where参数
        if filter_metadata:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                where=filter_metadata
            )
        else:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )

        # 整理结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })

        return formatted_results