import re
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
from dotenv import load_dotenv
from functools import wraps

from openai._exceptions import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIError
)

# 尝试导入超时异常，兼容不同版本
try:
    from openai._exceptions import APITimeoutError as Timeout
except ImportError:
    try:
        from openai import APITimeoutError as Timeout
    except ImportError:
        class Timeout(Exception):
            pass

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_calls.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置常量
DEFAULT_MODEL = "qwen3-32b"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def handle_llm_exceptions(func):
    """装饰器：统一处理 LLM API 异常"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            error_msg = f"认证错误: {str(e)}"
            logger.error(error_msg)
            return _create_error_result("AuthenticationError", error_msg)
        except RateLimitError as e:
            error_msg = f"速率限制错误: {str(e)}"
            logger.warning(error_msg)
            return _create_error_result("RateLimitError", error_msg)
        except APIConnectionError as e:
            error_msg = f"网络连接错误: {str(e)}"
            logger.error(error_msg)
            return _create_error_result("APIConnectionError", error_msg)
        except APIError as e:
            error_msg = f"API 错误: {str(e)}"
            logger.error(error_msg)
            return _create_error_result("APIError", error_msg)
        except Timeout as e:
            error_msg = f"请求超时: {str(e)}"
            logger.error(error_msg)
            return _create_error_result("Timeout", error_msg)
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(error_msg)
            return _create_error_result("UnknownError", error_msg)

    return wrapper


def _create_error_result(error_type: str, error_message: str) -> Dict[str, Any]:
    """创建错误结果的辅助函数"""
    return {
        "success": False,
        "error_type": error_type,
        "error_message": error_message,
        "data": None
    }


def _get_usage_info(usage) -> Dict[str, Any]:
    """获取使用情况信息，兼容不同版本"""
    if hasattr(usage, 'model_dump'):
        return usage.model_dump()
    elif hasattr(usage, 'dict'):
        return usage.dict()
    else:
        return {}


def _log_api_call_start(model: str, messages_count: int):
    """记录 API 调用开始"""
    logger.info(f"开始调用 LLM API: model={model}, messages_count={messages_count}")


def _log_api_call_success(model: str, tokens_used: int):
    """记录 API 调用成功"""
    logger.info(f"LLM API 调用成功: model={model}, tokens_used={tokens_used}")


class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = DEFAULT_BASE_URL):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥，默认从环境变量获取
            base_url: API 基础 URL
        """
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @handle_llm_exceptions
    def call_llm(
            self,
            messages: List[Dict],
            model: str = DEFAULT_MODEL,
            temperature: float = 0.7,
            max_tokens: int = 2000,
            response_format: Optional[Dict] = None,
            retry_times: int = 3,
            retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        调用 LLM API 并处理异常，支持重试机制

        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            response_format: 响应格式
            retry_times: 重试次数
            retry_delay: 重试延迟（秒）

        Returns:
            包含响应结果或错误信息的字典
        """
        import time

        for attempt in range(retry_times):
            try:
                # 记录调用开始
                _log_api_call_start(model, len(messages))

                # 构建请求参数
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "extra_body": {"enable_thinking": False}
                }

                if response_format:
                    params["response_format"] = response_format

                # 执行 API 调用
                completion = self.client.chat.completions.create(**params)

                # 解析响应
                usage_info = _get_usage_info(completion.usage)
                result = {
                    "success": True,
                    "data": completion.choices[0].message.content,
                    "model": completion.model,
                    "usage": usage_info,
                    "request_id": getattr(completion, 'id', None)
                }

                tokens_used = result['usage'].get('total_tokens', 0)
                _log_api_call_success(model, tokens_used)

                return result

            except (RateLimitError, APIConnectionError, Timeout) as e:
                # 对于可重试的错误进行重试
                if attempt < retry_times - 1:
                    logger.warning(f"第 {attempt + 1} 次尝试失败，{retry_delay} 秒后重试: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                else:
                    # 重试次数用完，抛出异常让装饰器处理
                    raise e
            except Exception as e:
                # 其他异常直接抛出
                raise e


# 便捷调用函数
def llm_call(
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str = DEFAULT_MODEL,
        retry_times: int = 3,
        **kwargs
) -> Dict[str, Any]:
    """
    便捷的 LLM 调用函数

    Args:
        prompt: 用户输入提示
        system_prompt: 系统提示
        model: 模型名称
        retry_times: 重试次数
        **kwargs: 其他参数

    Returns:
        API 调用结果
    """
    # 从环境变量获取 API 密钥
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        logger.error("未找到 DASHSCOPE_API_KEY 环境变量")
        return _create_error_result("ConfigurationError", "未找到 DASHSCOPE_API_KEY 环境变量")

    client = LLMClient(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    return client.call_llm(messages, model=model, retry_times=retry_times, **kwargs)


# 重构的 StructuredAgent 类
class StructuredAgent:
    def __init__(self, api_key: str = None):
        self.llm_client = LLMClient(api_key)

    def create_math_prompt(self, problem: str) -> str:
        """创建数学解题的结构化提示词"""
        return f"""
请使用思维链（Chain of Thought）方法解决以下数学问题，并以JSON格式返回结构化结果：

问题：{problem}

要求步骤：
1. 首先分析问题的关键信息
2. 列出解题思路和步骤
3. 逐步推导计算过程
4. 得出最终答案
5. 验证答案的合理性

请严格按照以下JSON格式返回：
{{
  "problem": "...",
  "analysis": "...",
  "steps": [
    {{
      "step_number": 1,
      "description": "...",
      "calculation": "..."
    }}
  ],
  "final_answer": "...",
  "verification": "..."
}}
"""

    def create_copywriting_prompt(self, requirements: str) -> str:
        """创建文案生成的结构化提示词"""
        return f"""
请使用思维链（Chain of Thought）方法生成满足以下要求的文案，并以JSON格式返回结构化结果：

需求：{requirements}

要求步骤：
1. 分析文案目标受众和目的
2. 确定文案风格和语调
3. 构思核心信息和要点
4. 组织文案结构
5. 撰写文案内容

请严格按照以下JSON格式返回：
{{
  "requirement": "...",
  "target_audience": "...",
  "style_tone": "...",
  "key_points": ["...", "..."],
  "structure_plan": "...",
  "generated_copy": "...",
  "call_to_action": "..."
}}
"""

    def chat_completion(self, prompt: str, model: str = DEFAULT_MODEL, retry_times: int = 3) -> Dict[str, Any]:
        """通用聊天完成方法"""
        result = self.llm_client.call_llm(
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的AI助手，擅长使用思维链方法进行逻辑推理和内容创作。请严格按照JSON格式返回结构化输出。"
                },
                {"role": "user", "content": prompt}
            ],
            model=model,
            response_format={"type": "json_object"},
            retry_times=retry_times
        )

        if result["success"]:
            try:
                parsed_content = json.loads(result["data"])
                result["parsed_data"] = parsed_content
            except json.JSONDecodeError:
                logger.error("JSON 解析失败")
                result["parsed_data"] = None

        return result


class ConversationAgent:
    def __init__(self, api_key: str = None):
        self.agent = StructuredAgent(api_key=api_key)
        # 存储对话历史
        self.conversation_history: List[Dict[str, str]] = []
        # 用户输入验证规则
        self.input_validator = InputValidator()

    def validate_input(self, user_input: str) -> tuple[bool, str]:
        """验证用户输入的合法性"""
        return self.input_validator.validate(user_input)

    def add_to_history(self, role: str, content: str):
        """添加消息到对话历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": self._get_timestamp()
        })

    def get_conversation_context(self) -> List[Dict[str, str]]:
        """获取当前对话上下文"""
        return self.conversation_history.copy()

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()

    def chat(self, user_input: str) -> Dict[str, Any]:
        # 验证输入
        is_valid, validation_msg = self.validate_input(user_input)
        if not is_valid:
            return {
                "success": False,
                "error_type": "InputValidationError",
                "error_message": validation_msg,
                "data": None
            }

        # 添加用户输入到历史记录
        self.add_to_history("user", user_input)

        try:
            # 构建完整的消息历史
            messages = self._build_messages()

            # 调用 LLM
            response = self.agent.chat_completion(
                prompt=self._format_current_prompt(user_input),
                retry_times=2
            )

            if response["success"]:
                # 安全地获取助手回复
                parsed_data = response.get("parsed_data")
                if isinstance(parsed_data, dict):
                    assistant_reply = parsed_data.get("generated_copy", response["data"])
                elif isinstance(parsed_data, (int, float)):  # 如果是数字类型
                    assistant_reply = str(parsed_data)
                else:
                    assistant_reply = response.get("data", "")

                self.add_to_history("assistant", assistant_reply)
                return response
            else:
                # 发生错误时仍记录到历史
                error_msg = f"助手暂时无法回应: {response['error_message']}"
                self.add_to_history("assistant", error_msg)
                return response  # 确保返回response

        except Exception as e:
            error_msg = f"对话处理出错: {str(e)}"
            self.add_to_history("assistant", error_msg)
            logger.error(error_msg)
            return {
                "success": False,
                "error_type": "ConversationError",
                "error_message": error_msg,
                "data": None
            }

    def _build_messages(self) -> List[Dict[str, str]]:
        """构建完整的消息列表"""
        # 使用最近的几轮对话作为上下文
        recent_history = self.conversation_history[-6:]  # 最近3轮对话（用户+助手）

        messages = [{
            "role": "system",
            "content": "你是一个专业的AI助手，能够进行多轮对话。请参考之前的对话历史来回答问题。"
        }]

        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages

    def _format_current_prompt(self, user_input: str) -> str:
        """格式化当前用户的输入"""
        if len(self.conversation_history) <= 2:  # 只有当前这次输入
            return user_input
        else:
            # 提供对话上下文
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-4:]  # 最近2轮
            ])
            return f"之前的对话:\n{context}\n\n现在的问题: {user_input}"

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class InputValidator:
    """用户输入验证器"""

    def __init__(self):
        # 定义非法字符模式
        self.invalid_patterns = [
            r'[<>{}[\]\\]',  # HTML/XML标签字符
            r'(\n\s*){3,}',  # 过多的空行
        ]
        # 定义最大输入长度
        self.max_length = 1000
        # 定义最小有效长度
        self.min_length = 1

    def validate(self, user_input: str) -> tuple[bool, str]:
        """验证用户输入"""
        if user_input is None:
            return False, "输入不能为空"

        # 去除首尾空白
        cleaned_input = user_input.strip()

        # 检查是否为空
        if not cleaned_input:
            return False, "请输入有效内容"

        # 检查长度
        if len(cleaned_input) < self.min_length:
            return False, "输入内容太短，请输入更多内容"

        if len(cleaned_input) > self.max_length:
            return False, f"输入内容太长，最多允许{self.max_length}个字符"

        # 检查非法字符
        for pattern in self.invalid_patterns:
            if re.search(pattern, cleaned_input):
                return False, "输入包含非法字符，请重新输入"

        # 检查是否包含过多特殊字符
        special_char_ratio = sum(1 for c in cleaned_input if not c.isalnum() and not c.isspace()) / len(cleaned_input)
        if special_char_ratio > 0.7:  # 特殊字符占比超过70%
            return False, "输入包含过多特殊字符，请使用正常文字"

        return True, "输入有效"


def interactive_chat():
    """交互式聊天界面"""
    print("=== AI 对话助手 ===")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 'clear' 清空对话历史")
    print("输入 'history' 查看对话历史")
    print("-" * 30)

    # 初始化对话助手
    agent = ConversationAgent()

    while True:
        try:
            user_input = input("\n您: ").strip()

            # 处理特殊命令
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            elif user_input.lower() == 'clear':
                agent.clear_history()
                print("🤖 助手: 对话历史已清空")
                continue
            elif user_input.lower() == 'history':
                history = agent.get_conversation_context()
                if history:
                    print("🤖 对话历史:")
                    for i, msg in enumerate(history, 1):
                        print(f"  {i}. [{msg['role']}] {msg['content']}")
                else:
                    print("🤖 助手: 当前没有对话历史")
                continue
            elif not user_input:
                print("🤖 助手: 请输入有效内容")
                continue

            # 处理用户输入
            result = agent.chat(user_input)

            if result["success"]:
                response_data = result.get("parsed_data") or result.get("data")
                if isinstance(response_data, dict):
                    # 如果是结构化数据，提取主要内容
                    content = response_data.get("generated_copy") or response_data.get("final_answer") or str(
                        response_data)
                else:
                    content = response_data

                print(f"🤖 助手: {content}")
            else:
                print(f"🤖 助手: {result['error_message']}")

        except KeyboardInterrupt:
            print("\n\n对话被中断，再见！")
            break
        except Exception as e:
            print(f"🤖 助手: 发生错误 - {str(e)}")


if __name__ == '__main__':
    # 运行交互式聊天
    interactive_chat()
