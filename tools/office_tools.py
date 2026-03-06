# -*- coding: utf-8 -*-
"""
办公专属工具模块
提供实时搜索、日历管理、待办清单等办公功能
"""
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pickle
from logger import logger
from tools.tool_encapsulation import ToolManager
from llm_call import llm_call, DEFAULT_MODEL


class OfficeToolsManager:
    """
    办公工具管理器，整合所有办公相关工具
    支持：实时搜索、日历查询、待办管理、工具调用重试/降级
    """

    def __init__(self, api_key: str = None, storage_path: str = "./office_tools_data.pkl"):
        """
        初始化办公工具管理器

        Args:
            api_key: API 密钥
            storage_path: 数据持久化路径
        """
        self.api_key = api_key
        self.storage_path = storage_path
        self.tool_manager = ToolManager(api_key=api_key, storage_path=storage_path)

        # 工具调用统计
        self.tool_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_count': 0
        }

        # 工具描述（用于 LLM 识别）
        self.tools_description = self._build_tools_description()

    def _build_tools_description(self) -> str:
        """构建工具描述的文本版本，用于提示词"""
        return """可用办公工具：
1. search - 实时搜索行业资讯、新闻、天气等信息
2. query_calendar - 查询指定日期的日历安排和会议
3. add_todo - 添加待办事项到清单
4. list_todos - 查看待办事项清单
5. complete_todo - 标记待办事项为已完成"""

    def execute_tool_with_retry(self, tool_name: str, arguments: Dict[str, Any],
                                max_retries: int = 2) -> Dict[str, Any]:
        """
        执行工具调用，支持重试机制

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            max_retries: 最大重试次数

        Returns:
            工具执行结果
        """
        self.tool_stats['total_calls'] += 1

        for attempt in range(max_retries + 1):
            try:
                logger.info(f"执行工具：{tool_name}, 尝试次数：{attempt + 1}")

                if tool_name not in self.tool_manager.tools:
                    return {
                        "success": False,
                        "error": f"未知工具：{tool_name}",
                        "tool_name": tool_name
                    }

                tool_func = self.tool_manager.tools[tool_name]
                result = tool_func(**arguments)

                if result.get('success'):
                    self.tool_stats['successful_calls'] += 1
                    logger.info(f"工具执行成功：{tool_name}")
                    return result
                else:
                    if attempt < max_retries:
                        self.tool_stats['retry_count'] += 1
                        logger.warning(f"工具执行失败，准备重试：{tool_name}, 错误：{result.get('error')}")
                        continue
                    else:
                        self.tool_stats['failed_calls'] += 1
                        logger.error(f"工具执行失败且重试用完：{tool_name}")
                        return self._get_degradation_result(tool_name, arguments)

            except Exception as e:
                logger.error(f"工具调用异常：{tool_name}, 错误：{str(e)}")
                if attempt < max_retries:
                    self.tool_stats['retry_count'] += 1
                    continue
                else:
                    self.tool_stats['failed_calls'] += 1
                    return {
                        "success": False,
                        "error": f"工具调用异常：{str(e)}",
                        "tool_name": tool_name,
                        "degraded": True
                    }

        return self._get_degradation_result(tool_name, arguments)

    def _get_degradation_result(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取降级结果，当工具调用失败时提供友好的错误提示

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            降级结果
        """
        degradation_messages = {
            "search": f"抱歉，搜索服务暂时不可用。您可以稍后重试搜索：{arguments.get('query', '')}",
            "query_calendar": "抱歉，日历查询服务暂时不可用。请稍后重试。",
            "add_todo": "抱歉，待办添加服务暂时不可用。建议您手动记录此事项。",
            "list_todos": "抱歉，待办清单查询暂时不可用。请稍后重试。",
            "complete_todo": "抱歉，待办完成标记服务暂时不可用。请稍后重试。"
        }

        message = degradation_messages.get(tool_name, "抱歉，该服务暂时不可用，请稍后重试。")

        return {
            "success": False,
            "error": message,
            "tool_name": tool_name,
            "degraded": True
        }

    def parse_user_intent(self, user_input: str) -> Dict[str, Any]:
        """
        使用 LLM 解析用户意图，确定需要调用的工具和参数

        Args:
            user_input: 用户输入

        Returns:
            解析结果，包含工具名称和参数
        """
        intent_prompt = f"""
    你是一个办公助手意图识别专家。请分析用户输入，判断需要调用哪个办公工具。

    {self.tools_description}

    请按照以下 JSON 格式返回结果：
    {{
        "tool_name": "工具名称（如果没有匹配到工具则返回 null）",
        "arguments": {{
            "参数名": "参数值"
        }},
        "confidence": "置信度 (0-1)",
        "reason": "判断理由"
    }}

    用户输入：{user_input}

    注意：
    1. 如果涉及查询实时信息、新闻、天气等，使用 search 工具
    2. 如果涉及查询日程、会议、安排，使用 query_calendar 工具
    3. 如果涉及添加待办事项，使用 add_todo 工具，参数必须是：task (必填), priority (可选), due_date (可选)
    4. 如果涉及查看待办清单，使用 list_todos 工具
    5. 如果涉及标记待办完成，使用 complete_todo 工具
    6. 如果没有匹配到合适的工具，tool_name 返回 null

    重要：参数名称必须严格按照工具定义，不要使用 todo_item 等错误的参数名。
    """

        try:
            response = llm_call(
                prompt=intent_prompt,
                system_prompt="你是一个精准的办公助手意图识别系统，只返回 JSON 格式结果，参数名必须正确",
                model=DEFAULT_MODEL,
                retry_times=2
            )

            if isinstance(response, dict) and response.get('success'):
                result_text = response.get('data', '{}')

                try:
                    result_json = json.loads(result_text)
                    return self._validate_and_fix_arguments(result_json)
                except json.JSONDecodeError:
                    logger.warning(f"LLM 返回非 JSON 格式：{result_text}")
                    return self._fallback_intent_parsing(user_input)
            else:
                logger.error("意图识别失败")
                return self._fallback_intent_parsing(user_input)

        except Exception as e:
            logger.error(f"意图识别异常：{e}")
            return self._fallback_intent_parsing(user_input)

    def _validate_and_fix_arguments(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证和修正工具参数

        Args:
            result: LLM 返回的意图识别结果

        Returns:
            修正后的结果
        """
        tool_name = result.get('tool_name')
        arguments = result.get('arguments', {})

        if not tool_name or not arguments:
            return result

        # 修正 add_todo 的参数
        if tool_name == 'add_todo':
            if 'todo_item' in arguments:
                arguments['task'] = arguments.pop('todo_item')
            if 'item' in arguments:
                arguments['task'] = arguments.pop('item')

        # 修正 query_calendar 的参数
        elif tool_name == 'query_calendar':
            if 'date' not in arguments:
                arguments['date'] = 'today'

        result['arguments'] = arguments
        return result

    def _fallback_intent_parsing(self, user_input: str) -> Dict[str, Any]:
        """
        回退的意图识别方法，使用简单的规则匹配
        """
        from tools.tool_encapsulation import determine_tool_usage

        tool_decision = determine_tool_usage(user_input, self.tool_manager)

        if tool_decision.get('use_tool'):
            return {
                "tool_name": tool_decision['tool_name'],
                "arguments": tool_decision['arguments'],
                "confidence": 0.7,
                "reason": "基于规则匹配"
            }
        else:
            return {
                "tool_name": None,
                "arguments": {},
                "confidence": 0.5,
                "reason": "未匹配到工具"
            }

    def execute_office_tool(self, user_input: str) -> Dict[str, Any]:
        """
        一站式执行办公工具：解析意图 -> 调用工具 -> 返回结果

        Args:
            user_input: 用户输入

        Returns:
            执行结果
        """
        logger.info(f"收到办公工具请求：{user_input}")

        # 1. 解析意图
        intent_result = self.parse_user_intent(user_input)
        tool_name = intent_result.get('tool_name')

        if not tool_name:
            logger.info("未匹配到工具，返回通用响应")
            return {
                "success": True,
                "result": "我无法理解您的需求，请用更明确的方式描述，比如'查询今天的日程'或'添加一个待办事项'",
                "tool_used": None,
                "intent_confidence": intent_result.get('confidence', 0)
            }

        arguments = intent_result.get('arguments', {})

        # 2. 执行工具（带重试）
        tool_result = self.execute_tool_with_retry(tool_name, arguments, max_retries=2)

        # 3. 整合结果
        if tool_result.get('success'):
            return {
                "success": True,
                "result": tool_result.get('result'),
                "tool_used": tool_name,
                "intent_confidence": intent_result.get('confidence'),
                "intent_reason": intent_result.get('reason')
            }
        else:
            return {
                "success": False,
                "error": tool_result.get('error'),
                "tool_used": tool_name,
                "degraded": tool_result.get('degraded', False),
                "intent_confidence": intent_result.get('confidence')
            }

    def get_tool_statistics(self) -> Dict[str, Any]:
        """获取工具调用统计信息"""
        total = self.tool_stats['total_calls']
        success = self.tool_stats['successful_calls']
        success_rate = (success / total * 100) if total > 0 else 0

        return {
            "total_calls": total,
            "successful_calls": success,
            "failed_calls": self.tool_stats['failed_calls'],
            "retry_count": self.tool_stats['retry_count'],
            "success_rate": f"{success_rate:.2f}%"
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.tool_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'retry_count': 0
        }



