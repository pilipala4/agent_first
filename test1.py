# test_conversation_agent.py
import pytest
import json
from unittest.mock import Mock, patch
from structured_prompt_dialogV1_0 import ConversationAgent, InputValidator, StructuredAgent, LLMClient
import os


class TestConversationAgent:
    """对话助手功能测试类"""

    def setup_method(self):
        """测试前初始化"""
        self.agent = ConversationAgent()

    def test_normal_input(self):
        """测试正常输入功能"""
        user_input = "你好，今天天气怎么样？"

        # 模拟LLM响应
        mock_response = {
            "success": True,
            "data": '{"generated_copy": "你好！今天的天气很好"}',
            "parsed_data": {"generated_copy": "你好！今天的天气很好"}
        }

        with patch.object(self.agent.agent, 'chat_completion', return_value=mock_response):
            result = self.agent.chat(user_input)

            assert result["success"] is True
            assert result["data"] is not None
            assert len(self.agent.conversation_history) >= 2  # 用户输入 + 助手回复

    def test_empty_input_validation(self):
        """测试空输入验证"""
        empty_inputs = ["", "   ", "\t", "\n", None]

        for empty_input in empty_inputs:
            if empty_input is not None:  # None的情况单独测试
                result = self.agent.chat(empty_input)
                assert result["success"] is False
                assert result["error_type"] == "InputValidationError"

    def test_long_input_validation(self):
        """测试超长输入验证"""
        long_input = "a" * 1001  # 超过最大长度1000

        result = self.agent.chat(long_input)
        assert result["success"] is False
        assert result["error_type"] == "InputValidationError"
        assert "太长" in result["error_message"]

    def test_special_characters_input(self):
        """测试包含特殊字符的输入验证"""
        special_input = "!@#$%^&*()" * 100  # 特殊字符占比过高

        result = self.agent.chat(special_input)
        assert result["success"] is False
        assert result["error_type"] == "InputValidationError"
        assert "特殊字符" in result["error_message"]

    def test_clear_history_function(self):
        """测试清空对话历史功能"""
        # 添加一些历史记录
        self.agent.add_to_history("user", "测试输入")
        self.agent.add_to_history("assistant", "测试回复")

        assert len(self.agent.conversation_history) == 2

        # 清空历史
        self.agent.clear_history()

        assert len(self.agent.conversation_history) == 0

    def test_get_conversation_context(self):
        """测试获取对话上下文功能"""
        self.agent.add_to_history("user", "测试输入")
        context = self.agent.get_conversation_context()

        assert len(context) == 1
        assert context[0]["role"] == "user"
        assert context[0]["content"] == "测试输入"

    def test_api_exception_handling(self):
        """测试API异常处理"""
        user_input = "测试异常"

        # 模拟API调用异常
        mock_response = {
            "success": False,
            "error_type": "APIError",
            "error_message": "API调用失败",
            "data": None
        }

        with patch.object(self.agent.agent, 'chat_completion', return_value=mock_response):
            result = self.agent.chat(user_input)

            assert result["success"] is False
            assert result["error_type"] == "APIError"
            assert "API调用失败" in result["error_message"]

    def test_input_validator_edge_cases(self):
        """测试输入验证器边界情况"""
        validator = InputValidator()

        # 测试HTML标签字符
        html_input = "<script>alert('test')</script>"
        is_valid, msg = validator.validate(html_input)
        assert is_valid is False

        # 测试过多空行
        multiline_input = "line1\n\n\n\n\nline2"
        is_valid, msg = validator.validate(multiline_input)
        assert is_valid is False
        assert "空行" in msg


class TestInputValidator:
    """输入验证器测试类"""

    def setup_method(self):
        self.validator = InputValidator()

    @pytest.mark.parametrize("valid_input", [
        "这是一个有效的输入",
        "Hello World!",
        "123456",
        "Mixed 中英文 text 内容"
    ])
    def test_valid_inputs(self, valid_input):
        """测试各种有效输入"""
        is_valid, msg = self.validator.validate(valid_input)
        assert is_valid is True

    @pytest.mark.parametrize("invalid_input", [
        "",
        "   ",
        "\t\t\t",
        "\n\n\n"
    ])
    def test_empty_inputs(self, invalid_input):
        """测试各种空输入"""
        is_valid, msg = self.validator.validate(invalid_input)
        assert is_valid is False
        assert "有效内容" in msg or "空" in msg

    def test_length_validation(self):
        """测试长度验证"""
        # 测试过短输入
        short_input = "a"
        is_valid, msg = self.validator.validate(short_input)
        # 根据代码，最短长度是1，所以这个应该通过
        # 实际上根据代码，min_length=1，所以"a"是有效的
        # 修正测试
        is_valid, msg = self.validator.validate("")
        assert is_valid is False

        # 测试超长输入
        long_input = "x" * 1001
        is_valid, msg = self.validator.validate(long_input)
        assert is_valid is False
        assert "太长" in msg


class TestStructuredAgent:
    """结构化代理测试类"""

    def test_create_math_prompt(self):
        """测试数学问题提示词生成"""
        agent = StructuredAgent()
        problem = "2+2等于多少？"

        prompt = agent.create_math_prompt(problem)

        assert problem in prompt
        assert "思维链" in prompt
        assert "JSON格式" in prompt
        assert "analysis" in prompt
        assert "steps" in prompt

    def test_create_copywriting_prompt(self):
        """测试文案生成提示词生成"""
        agent = StructuredAgent()
        requirements = "写一个产品介绍"

        prompt = agent.create_copywriting_prompt(requirements)

        assert requirements in prompt
        assert "思维链" in prompt
        assert "JSON格式" in prompt
        assert "target_audience" in prompt
        assert "generated_copy" in prompt


class TestLLMClientIntegration:
    """LLM客户端集成测试类"""

    @patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fake-api-key"})
    def test_llm_client_initialization_with_env_var(self):
        """测试使用环境变量初始化LLM客户端"""
        client = LLMClient()
        assert client.client.api_key == "fake-api-key"

    def test_llm_client_initialization_with_provided_key(self):
        """测试使用提供密钥初始化LLM客户端"""
        client = LLMClient(api_key="provided-api-key")
        assert client.client.api_key == "provided-api-key"


# conftest.py - 配置文件（可选）
@pytest.fixture(scope="session")
def test_agent():
    """创建一个测试用的对话代理实例"""
    return ConversationAgent()


# 运行测试的命令说明
def run_tests_instructions():
    """
    运行测试的命令：

    1. 基本运行：
       pytest test_conversation_agent.py -v

    2. 生成覆盖率报告：
       pytest test_conversation_agent.py --cov=. --cov-report=html

    3. 生成JUnit XML报告：
       pytest test_conversation_agent.py --junitxml=report.xml

    4. 运行特定测试类：
       pytest test_conversation_agent.py::TestConversationAgent::test_normal_input

    5. 详细输出：
       pytest test_conversation_agent.py -v -s
    """
    pass
