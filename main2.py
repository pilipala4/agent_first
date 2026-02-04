import os
import chromadb
from chromadb.utils import embedding_functions
from PyPDF2 import PdfReader
from rag.rag_system import DocumentRAG
import argparse
import sys



def main():
    parser = argparse.ArgumentParser(description="Pure Vector Search RAG System (No LLM)")
    parser.add_argument("--add", help="Path to PDF or MD file to add to database")
    parser.add_argument("--query", help="Question to ask about the documents")
    parser.add_argument("--db-path", default="./chroma_db", help="Path to store the database")
    parser.add_argument("--n-results", type=int, default=2, help="Number of results to return")

    args = parser.parse_args()

    # 检查是否有参数传入，如果没有则显示帮助信息
    if not args.add and not args.query:
        print("纯向量搜索 RAG 系统")
        print("=" * 50)
        print("此系统仅执行向量相似性搜索。")
        print("它不会使用任何大语言模型进行以下操作：")
        print("  - 查询理解")
        print("  - 答案生成")
        print("  - 文本摘要")
        print("  - 响应格式化")
        print("")
        print("系统只是将您的查询转换为向量，并基于语义含义")
        print("找到最相似的文本块。")
        print("=" * 50)
        print("\n使用方法:")
        print("  添加文档: python script.py --add <pdf_file_path> --db-path <database_path>")
        print("  查询文档: python script.py --query <your_question> --db-path <database_path>")
        print("  同时添加和查询: python script.py --add <pdf_file_path> --query <your_question> --db-path <database_path>")
        parser.print_help()
        return

    # 初始化 RAG 系统
    rag_system = DocumentRAG(db_path=args.db_path)

    if args.add:
        file_path = args.add
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            sys.exit(1)

        try:
            rag_system.add_document(file_path)
            print(f"Successfully added {file_path} to the database")
        except Exception as e:
            print(f"Error adding document: {e}")
            sys.exit(1)

    if args.query:
        if rag_system.collection.count() == 0:
            print("No documents in the database. Please add documents first.")
            sys.exit(1)

        results = rag_system.query(args.query, n_results=args.n_results)

        print("\n" + "=" * 60)
        print(f"Query: {args.query}")
        print("=" * 60)
        print("Results (vector similarity matches, no LLM processing):")

        for i, result in enumerate(results):
            print(f"\nResult {i + 1}:")
            print(f"Source: {result['metadata']['source']}")
            print(f"Chunk ID: {result['metadata']['chunk_id']}")
            print(f"Similarity Score: {1 / (1 + result['distance']):.4f}")  # 转换为相似度分数
            print(f"Content: {result['document'][:500]}...")
            print("-" * 40)

if __name__ == "__main__":
     main()
