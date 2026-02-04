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
        """将文本分割成块，优化关键部分识别"""
        chunks = []
        start = 0

        # 优化分块逻辑：确保章节标题能成为独立块
        # 首先按章节标题分割
        sections = re.split(r'(\d+\s*[\u4e00-\u9fa5]+)', text)

        if len(sections) > 1:
            # 重新组织文本，确保章节标题单独成块
            for i in range(1, len(sections), 2):
                section_title = sections[i].strip()
                section_content = sections[i + 1].strip() if i + 1 < len(sections) else ""

                # 确保章节标题单独成块
                if section_title:
                    chunks.append(section_title)

                # 分割章节内容
                if section_content:
                    section_chunks = self._split_text_with_overlap(section_content, chunk_size, overlap)
                    chunks.extend(section_chunks)
        else:
            # 如果没有章节标题，使用默认分块
            while start < len(text):
                end = start + chunk_size
                # 确保不在单词中间切分
                if end < len(text):
                    # 找到下一个空格位置
                    next_space = text.find(' ', end)
                    if next_space != -1 and next_space - end < 50:
                        # 如果距离太远则忽略
                        end = next_space
                    chunk = text[start:end]
                    chunks.append(chunk)
                    # 计算下一个开始位置，考虑重叠
                    start = end - overlap
                else:
                    chunks.append(text[start:])
                    start = len(text)

        return chunks

    def _split_text_with_overlap(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """辅助函数：按指定大小和重叠分割文本"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break
            # 找到下一个空格位置
            next_space = text.find(' ', end)
            if next_space != -1 and next_space - end < 50:
                end = next_space
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def add_document(self, file_path: str, doc_type: str = None, metadata: Optional[Dict] = None):
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
        chunks = self.chunk_text(content)

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