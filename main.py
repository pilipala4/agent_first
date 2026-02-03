import re
import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from openai import OpenAI
from dotenv import load_dotenv
from functools import wraps
from logger import logger
from llm_call import LLMClient
from llm_call import DEFAULT_MODEL, DEFAULT_BASE_URL
from tools.tool_encapsulation import ToolManager, determine_tool_usage


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


# 重构的 StructuredAgent 类
class StructuredAgent:
    def __init__(self, api_key: str = None):
        self.llm_client = LLMClient(api_key)
        self.tool_manager = ToolManager(api_key)  # 添加工具管理器

    def process_with_tools(self, prompt: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """
        处理包含工具调用的请求
        """
        # 首先判断是否需要使用工具
        tool_decision = determine_tool_usage(prompt, self.tool_manager)

        if tool_decision["use_tool"]:
            # 需要使用工具
            tool_name = tool_decision["tool_name"]
            arguments = tool_decision["arguments"]

            # 调用相应工具
            tool_func = self.tool_manager.tools[tool_name]
            tool_result = tool_func(**arguments)

            # 将工具结果整合到提示中
            if tool_result["success"]:
                enhanced_prompt = f"""
    原始问题：{prompt}

    工具调用结果：
    {json.dumps(tool_result, ensure_ascii=False, indent=2)}

    请基于以上信息回答原始问题。
    """
            else:
                enhanced_prompt = f"""
    原始问题：{prompt}

    工具调用失败：{tool_result.get('error', '未知错误')}

    请尝试其他方式回答问题或告知用户工具调用失败。
    """

            # 使用增强后的提示调用LLM
            return self.chat_completion(enhanced_prompt, model=model)
        else:
            # 不需要工具，直接处理原问题
            return self.chat_completion(prompt, model=model)

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

    def handle_history_query(self, user_input: str) -> bool:
        """判断是否为历史查询请求"""
        history_keywords = [
            "前面问了什么", "之前的问题", "历史记录",
            "对话历史", " earlier questions", "previous questions"
        ]
        return any(keyword in user_input.lower() for keyword in history_keywords)

    def get_previous_questions_summary(self) -> str:
        """获取之前问题的摘要"""
        user_messages = [
            msg for msg in self.conversation_history
            if msg["role"] == "user" and msg["content"] != "我这轮对话中前面问了什么问题"
        ]

        if len(user_messages) <= 1:  # 只有当前这个问题
            return "这是我们第一轮对话，您还没有问过其他问题。"

        previous_questions = [msg["content"] for msg in user_messages[:-1]]
        return f"您之前问过的问题包括：{'；'.join(previous_questions)}"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()

    def chat(self, user_input: str) -> Dict[str, Any]:
        # 验证输入
        is_valid, validation_msg, cleaned_input = self.input_validator.validate_and_clean(user_input)
        #is_valid, validation_msg = self.validate_input(user_input)
        if not is_valid:
            return {
                "success": False,
                "error_type": "InputValidationError",
                "error_message": validation_msg,
                "data": None
            }
        # 特殊处理：历史查询
        if self.handle_history_query(cleaned_input):
            summary = self.get_previous_questions_summary()
            self.add_to_history("assistant", summary)
            return {
                "success": True,
                "data": summary,
                "parsed_data": {"summary": summary}
            }

        # 添加用户输入到历史记录
        self.add_to_history("user", cleaned_input)
        #self.add_to_history("user", user_input)

        try:
            '''
            # 构建完整的消息历史
            messages = self._build_messages()

            # 调用 LLM
            response = self.agent.chat_completion(
                prompt=self._format_current_prompt(cleaned_input),
                retry_times=2
            )
            '''
            # 使用工具增强处理
            response = self.agent.process_with_tools(cleaned_input)

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
                "content": f"[{msg['timestamp']}] {msg['content']}"
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

    def validate_and_clean(self, user_input: str) -> tuple[bool, str, str]:
        """验证并清理用户输入"""
        if user_input is None:
            return False, "输入不能为空", ""

        # 清理代理字符
        try:
            cleaned_input = user_input.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            cleaned_input = user_input

        # 原有的验证逻辑
        stripped_input = cleaned_input.strip()

        if not stripped_input:
            return False, "请输入有效内容", ""

        if len(stripped_input) < self.min_length:
            return False, "输入内容太短，请输入更多内容", ""

        if len(stripped_input) > self.max_length:
            return False, f"输入内容太长，最多允许{self.max_length}个字符", ""

        for pattern in self.invalid_patterns:
            if re.search(pattern, stripped_input):
                return False, "输入包含非法字符，请重新输入", ""

        special_char_ratio = sum(1 for c in stripped_input if not c.isalnum() and not c.isspace()) / len(stripped_input)
        if special_char_ratio > 0.7:
            return False, "输入包含过多特殊字符，请使用正常文字", ""

        return True, "输入有效", stripped_input


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
