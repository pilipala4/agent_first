import sys
import os
import pytest

# 将上级目录添加到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_first import ConversationAgent
from main4 import SmartOfficeAssistant
from tools.weekly_report_generator import WeeklyReportGenerator
from tools.office_tools import OfficeToolsManager
from rag.rag_system import DocumentRAG


@pytest.fixture(scope="function")
def conversation_agent():
    """为每个测试函数提供一个新的 ConversationAgent 实例"""
    agent = ConversationAgent()
    yield agent
    # 清理操作
    agent.clear_history()


@pytest.fixture(scope="function")
def smart_assistant():
    """为每个测试函数提供一个新的 SmartOfficeAssistant 实例"""
    assistant = SmartOfficeAssistant()
    yield assistant
    # 清理操作
    assistant.clear_history()


@pytest.fixture(scope="function")
def report_generator():
    """为每个测试函数提供一个新的周报生成器实例"""
    generator = WeeklyReportGenerator()
    yield generator


@pytest.fixture(scope="function")
def office_tools():
    """为每个测试函数提供一个新的办公工具管理器实例"""
    tools = OfficeToolsManager()
    yield tools


@pytest.fixture(scope="function")
def rag_system():
    """为每个测试函数提供一个新的 RAG 系统实例"""
    rag = DocumentRAG(db_path="./test_chroma_db")
    yield rag
    # 清理测试数据库
    try:
        import shutil
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
    except Exception:
        pass


@pytest.fixture(scope="session")
def sample_inputs():
    """提供样本输入数据"""
    return {
        "valid": "这是一个有效的测试输入",
        "empty": "",
        "long": "a" * 500,
        "with_special_chars": "!@#$%^&*()" * 10,
        "weekly_report_simple": "本周完成了项目开发",
        "weekly_report_detailed": """
        工作内容：
        1. 完成智能办公助手开发
        2. 集成 RAG 文档问答
        3. 实现周报自动生成

        成果：系统性能提升 50%

        下周计划：
        1. 优化用户体验
        2. 编写技术文档
        """,
        "todo_add": "添加待办：下午 3 点开会",
        "calendar_query": "查询今天的日程",
        "search_query": "搜索人工智能新闻"
    }


@pytest.fixture(scope="session")
def test_config():
    """测试配置"""
    return {
        "max_response_time": 10.0,  # 最大响应时间（秒）
        "min_pass_rate": 95.0,  # 最低通过率要求（%）
        "min_coverage": 80.0,  # 最低覆盖率要求（%）
        "max_conversation_history": 20,  # 最大对话历史条数
        "max_input_length": 1000,  # 最大输入长度
    }


def pytest_configure(config):
    """pytest 配置钩子"""
    # 添加自定义标记
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "boundary: marks tests as boundary tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
