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


class StructuredAgent:
    def __init__(self, api_key: str = None):
        self.llm_client = LLMClient(api_key)
        self.tool_manager = ToolManager(api_key)

    def process_with_tools(self, prompt: str, context_history: List[Dict] = None, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
        """
        处理包含工具调用的请求，支持对话上下文
        """
        tool_decision = determine_tool_usage(prompt, self.tool_manager)

        if tool_decision["use_tool"]:
            tool_name = tool_decision["tool_name"]
            arguments = tool_decision["arguments"]

            tool_func = self.tool_manager.tools[tool_name]
            tool_result = tool_func(**arguments)

            # 构建上下文字符串（如果存在）
            context_str = ""
            if context_history:
                context_str = f"根据以下对话历史背景：\n{json.dumps(context_history, ensure_ascii=False, indent=2)}"

            if tool_result["success"]:
                enhanced_prompt = f"""
原始问题：{prompt}

工具调用结果：
{json.dumps(tool_result, ensure_ascii=False, indent=2)}

{context_str}

请基于以上信息回答原始问题。
""".strip()
            else:
                enhanced_prompt = f"""
原始问题：{prompt}

工具调用失败：{tool_result.get('error', '未知错误')}

{context_str}

请尝试其他方式回答问题或告知用户工具调用失败。
""".strip()

            return self.chat_completion(enhanced_prompt, model=model)
        else:
            # 不使用工具：将上下文融入提示
            if context_history:
                context_str = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in context_history[-6:]  # 最近2轮对话
                ])
                full_prompt = f"之前的对话:\n{context_str}\n\n当前问题: {prompt}"
            else:
                full_prompt = prompt

            return self.chat_completion(full_prompt, model=model)

    def create_math_prompt(self, problem: str) -> str:
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


class InputValidator:
    def __init__(self):
        self.invalid_patterns = [
            r'[<>{}[\]\\]',
            r'(\n\s*){3,}',
        ]
        self.max_length = 1000
        self.min_length = 1

    def validate_and_clean(self, user_input: str) -> tuple[bool, str, str]:
        if user_input is None:
            return False, "输入不能为空", ""

        try:
            cleaned_input = user_input.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            cleaned_input = user_input

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


class ConversationAgent:
    def __init__(self, api_key: str = None):
        self.agent = StructuredAgent(api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.input_validator = InputValidator()

    def add_to_history(self, role: str, content: str):
        from datetime import datetime
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    def get_conversation_context(self) -> List[Dict[str, str]]:
        return self.conversation_history.copy()

    def clear_history(self):
        self.conversation_history.clear()

    def chat(self, user_input: str) -> Dict[str, Any]:
        is_valid, validation_msg, cleaned_input = self.input_validator.validate_and_clean(user_input)
        if not is_valid:
            return {
                "success": False,
                "error_type": "InputValidationError",
                "error_message": validation_msg,
                "data": None
            }

        self.add_to_history("user", cleaned_input)

        try:
            # ✅ 关键修复：获取上下文并传入
            context_history = self.get_conversation_context()

            # ✅ 只调用一次，且传入上下文
            response = self.agent.process_with_tools(
                prompt=cleaned_input,
                context_history=context_history  # 👈 传入上下文！
            )

            if response["success"]:
                parsed_data = response.get("parsed_data")
                if isinstance(parsed_data, dict):
                    assistant_reply = parsed_data.get("generated_copy") or parsed_data.get("final_answer") or str(parsed_data)
                elif isinstance(parsed_data, (int, float)):
                    assistant_reply = str(parsed_data)
                else:
                    assistant_reply = response.get("data", "")

                self.add_to_history("assistant", assistant_reply)
                return response
            else:
                error_msg = f"助手暂时无法回应: {response['error_message']}"
                self.add_to_history("assistant", error_msg)
                return response

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


def interactive_chat():
    print("=== AI 对话助手 ===")
    print("输入 'quit' 或 'exit' 退出对话")
    print("输入 'clear' 清空对话历史")
    print("输入 'history' 查看对话历史")
    print("-" * 30)

    agent = ConversationAgent()

    while True:
        try:
            user_input = input("\n您: ").strip()

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

            result = agent.chat(user_input)

            if result["success"]:
                response_data = result.get("parsed_data") or result.get("data")
                if isinstance(response_data, dict):
                    content = response_data.get("generated_copy") or response_data.get("final_answer") or str(response_data)
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
    interactive_chat()