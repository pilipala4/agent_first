# conftest.py
import pytest
from structured_prompt_dialogV1_0 import ConversationAgent


@pytest.fixture(scope="function")
def conversation_agent():
    """为每个测试函数提供一个新的ConversationAgent实例"""
    agent = ConversationAgent()
    yield agent
    # 清理操作
    agent.clear_history()


@pytest.fixture(scope="session")
def sample_inputs():
    """提供样本输入数据"""
    return {
        "valid": "这是一个有效的测试输入",
        "empty": "",
        "long": "a" * 500,
        "with_special_chars": "!@#$%^&*()" * 10
    }
