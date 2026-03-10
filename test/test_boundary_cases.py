
import sys
import os

# 将上级目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from rag.rag_system import DocumentRAG
from main_first import StructuredAgent, InputValidator
from main2 import RAGSystem

def test_tool_no_return():
    agent = StructuredAgent()
    result = agent.process_with_tools("调用一个不存在的工具")
    assert not result["success"]
    assert "暂无法获取信息" in result["error_message"]

def test_long_input():
    validator = InputValidator()
    long_input = "a" * 1001  # 超过最大长度1000
    is_valid, msg, _ = validator.validate_and_clean(long_input)
    assert not is_valid
    assert "输入内容太长" in msg

def test_invalid_characters():
    validator = InputValidator()
    invalid_input = "<script>alert('xss')</script>"
    is_valid, msg, _ = validator.validate_and_clean(invalid_input)
    assert not is_valid
    assert "输入包含非法字符" in msg

def test_no_relevant_documents():
    Document_system = DocumentRAG()
    Document_system.clear_database()
    # 动态创建无关文档
    test_file_path = "./无关文档.md"
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("# 无关文档\n\n这是一个用于测试的无关文档。")
    Document_system.add_document(test_file_path, doc_type="markdown")
    os.remove(test_file_path)  # 清理测试文件

    rag_system = RAGSystem()
    result = rag_system.answer_question("这是一个完全无关的问题")
    assert not result["success"]
    assert "未能从文档中找到相关信息" in result["answer"]

def test_rag_long_question():
    rag_system = RAGSystem()
    long_question = "a" * 2001  # 超过最大长度2000
    result = rag_system.answer_question(long_question)
    assert not result["success"]
    assert "问题过长" in result["answer"]

def test_empty_database():
    Document_system = DocumentRAG()
    # 清空数据库
    Document_system.clear_database()
    assert Document_system.collection.count() == 0
    rag_system = RAGSystem()
    result = rag_system.answer_question("任何问题")
    assert not result["success"]
    assert "数据库中没有文档" in result["answer"]

if __name__ == "__main__":
    pytest.main()
