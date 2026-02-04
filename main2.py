import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from rag.rag_system import DocumentRAG
from llm_call import LLMClient
from llm_call import llm_call
from llm_call import DEFAULT_MODEL, DEFAULT_BASE_URL
from typing import List, Dict, Any, Optional
import argparse
import sys



class RAGSystem:
    def __init__(self, api_key: str = None, db_path="./chroma_db"):
        self.rag = DocumentRAG(db_path)
        self.llm_client = LLMClient(api_key)

    def answer_question(self, question: str, n_results: int = 3) -> Dict[str, Any]:
        """回答问题：检索相关文档并使用LLM生成答案"""
        print(f"\n正在处理问题: {question}")

        # 1. 检索相关文档
        retrieved_docs = self.rag.query(question, n_results=n_results)

        if not retrieved_docs:
            return {
                "success": False,
                "answer": "抱歉，未能从文档中找到相关信息。",
                "retrieved_docs": [],
                "question": question
            }

        # 2. 构建上下文
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source = doc['metadata']['source']
            distance = doc['distance']
            content = doc['document'][:1000]  # 限制长度避免超出token限制
            context_parts.append(f"文档{i + 1} ({source}, 相似度得分: {1 / (1 + distance):.3f}):\n{content}\n---")

        context = "\n".join(context_parts)

        # 3. 构建提示词
        system_prompt = """你是一个基于文档内容回答问题的助手。请仔细阅读提供的文档片段，并根据这些内容回答用户的问题。
        - 如果问题的答案可以从文档中找到，请提供准确的回答。
        - 如果问题与文档内容无关，请明确说明无法从文档中找到相关信息。
        - 回答应简洁明了，重点突出。
        - 不要编造信息，仅基于文档内容回答。"""

        user_prompt = f"""请基于以下文档内容回答问题：

文档内容：
{context}

问题：
{question}

请提供准确、简洁的答案："""

        # 4. 调用LLM生成答案
        result = llm_call(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=1000
        )

        if result['success']:
            return {
                "success": True,
                "answer": result['data'],
                "retrieved_docs": retrieved_docs,
                "question": question
            }
        else:
            return {
                "success": False,
                "answer": f"生成答案时出错: {result['error_message']}",
                "retrieved_docs": retrieved_docs,
                "question": question
            }


def main():
    parser = argparse.ArgumentParser(description="本地文档RAG问答助手V1.0")
    parser.add_argument("--add", default="./基於改進YOLOv5s的無人機圖像識別_李傑.pdf", help="添加PDF或MD文件到数据库")
    parser.add_argument("--query",default="这篇论文的主要内容是什么？", help="询问关于文档的问题")
    parser.add_argument("--chunk-size", type=int, default=1000, help="分块大小")
    parser.add_argument("--overlap", type=int, default=100, help="分块重叠长度")
    parser.add_argument("--db-path", default="./chroma_db", help="数据库存储路径")
    parser.add_argument("--n-results", type=int, default=3, help="返回检索结果的数量")
    parser.add_argument("--list-docs", action="store_true", help="列出数据库中的所有文档")

    args = parser.parse_args()

    # 如果没有参数，显示帮助信息
    if not any([args.add, args.query, args.list_docs]):
        print("本地文档RAG问答助手V1.0")
        print("=" * 60)
        print("功能:")
        print("  - 向量相似性搜索: 基于语义匹配查找相关文档")
        print("  - LLM答案生成: 基于检索到的文档内容生成答案")
        print("  - 准确率优化: 分块大小512，Top3检索，提高回答准确性")
        print("")
        print("使用方法:")
        print("  添加文档: python main2.py --add <pdf_or_md_file>")
        print("  问答查询: python main2.py --query \"你的问题\"")
        print("  查看文档: python main2.py --list-docs")
        print("  组合使用: python main2.py --add <file> --query \"问题\"")
        print("=" * 60)
        parser.print_help()
        return

    # 初始化RAG系统
    rag_system = RAGSystem(db_path=args.db_path)

    # 添加文档
    if args.add:
        file_path = args.add
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            sys.exit(1)

        try:
            # 自定义分块大小和重叠长度
            rag_system.rag.add_document(file_path, chunk_size=args.chunk_size, overlap=args.overlap)
            print(f"成功添加 {file_path} 到数据库")
        except Exception as e:
            print(f"添加文档时出错: {e}")
            sys.exit(1)

    # 列出文档
    if args.list_docs:
        count = rag_system.rag.collection.count()
        print(f"数据库中有 {count} 个文档块")
        # 获取所有唯一文档ID
        all_docs = rag_system.rag.collection.get(limit=count)
        unique_sources = set()
        for meta in all_docs['metadatas']:
            unique_sources.add(meta['source'])

        print("文档列表:")
        for i, source in enumerate(unique_sources, 1):
            print(f"  {i}. {source}")

    # 问答查询
    if args.query:
        if rag_system.rag.collection.count() == 0:
            print("数据库中没有文档。请先添加文档。")
            sys.exit(1)

        result = rag_system.answer_question(args.query, n_results=args.n_results)

        print("\n" + "=" * 60)
        print(f"问题: {result['question']}")
        print("=" * 60)

        if result['success']:
            print(f"答案: {result['answer']}")
        else:
            print(f"错误: {result['answer']}")

        print("\n检索到的相关文档:")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            source = doc['metadata']['source']
            similarity_score = 1 / (1 + doc['distance'])
            print(f"\n{i}. 来源: {source}")
            print(f"   相似度: {similarity_score:.3f}")
            print(f"   内容预览: {doc['document'][:200]}...")
            print("-" * 40)


if __name__ == "__main__":
    main()
