import requests
import subprocess
import re
from typing import Dict, Any, List
from llm_call import DEFAULT_MODEL
import json
from serpapi import SerpApiClient
import os
from dotenv import load_dotenv

load_dotenv()


class ToolManager:
    """
    工具管理器，封装百度搜索和代码运行工具
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.tools = {
            "search": self.search,
            "execute_code": self.execute_code
        }

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        返回可用工具的描述，用于LLM识别
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "使用搜索引擎获取实时信息，适用于查询最新新闻、天气、股票、事实信息等问题",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词或问题"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "返回结果数量，默认为3",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "执行Python代码，适用于数学计算、数据分析、文本处理等任务",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要执行的Python代码"
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:  # 添加 self 参数
        """
        一个基于SerpApi的实战网页搜索引擎工具。
        它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
        """
        print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                return {"success": False, "error": "SERPAPI_API_KEY 未配置"}

            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "gl": "cn",  # 国家代码
                "hl": "zh-cn",  # 语言代码
                "num": num_results  # 添加结果数量参数
            }

            client = SerpApiClient(params)
            results = client.get_dict()

            # 智能解析：优先寻找最直接的答案
            if "answer_box_list" in results:
                return {"success": True, "result": "\n".join(results["answer_box_list"])}
            if "answer_box" in results and "answer" in results["answer_box"]:
                return {"success": True, "result": results["answer_box"]["answer"]}
            if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                return {"success": True, "result": results["knowledge_graph"]["description"]}
            if "organic_results" in results and results["organic_results"]:
                # 如果没有直接答案，则返回前几个有机结果的摘要
                snippets = [
                    f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(results["organic_results"][:num_results])
                ]
                return {"success": True, "result": "\n\n".join(snippets)}

            return {"success": True, "result": f"对不起，没有找到关于 '{query}' 的信息。"}

        except Exception as e:
            return {"success": False, "error": f"搜索时发生错误: {e}", "query": query}

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        执行Python代码工具
        """
        try:
            # 限制代码执行的安全性
            # 检查是否有危险操作
            dangerous_patterns = [
                r'import\s+os',
                r'import\s+sys',
                r'exec\s*\(',
                r'eval\s*\(',
                r'open\s*\(',
                r'requests\s*\('
            ]

            for pattern in dangerous_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    return {
                        "success": False,
                        "error": "检测到潜在危险操作，禁止执行",
                        "code": code
                    }

            # 安全检查通过，执行代码
            exec_globals = {}
            exec(code, exec_globals)

            # 获取执行结果
            result = exec_globals.get('result', '代码执行完成，但未返回结果')

            return {
                "success": True,
                "output": str(result),
                "code": code
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": code
            }


def determine_tool_usage(user_input: str, tool_manager: ToolManager) -> Dict[str, Any]:
    """
    根据用户输入判断是否需要调用工具以及调用哪个工具
    """
    # 关键词匹配策略
    search_keywords = [
        '今天', '明天', '天气', '新闻', '股票', '实时', '最新',
        '查询', '搜索', '了解', '有什么', '怎么样', '如何'
    ]

    code_keywords = [
        '计算', '算一下', '数学', '加减乘除', '统计', '求和',
        '平均值', '编程', '代码', '算法', '公式'
    ]

    input_lower = user_input.lower()

    # 检查是否需要搜索
    for keyword in search_keywords:
        if keyword in input_lower:
            # 如果是询问天气等特定问题，准备搜索查询
            if '天气' in input_lower:
                location = extract_location(input_lower)
                search_query = f"天气预报 {location}" if location else "天气预报"
            else:
                search_query = user_input

            return {
                "use_tool": True,
                "tool_name": "search",
                "arguments": {
                    "query": search_query,
                    "num_results": 3
                }
            }

    # 检查是否需要执行代码
    for keyword in code_keywords:
        if keyword in input_lower:
            # 提取数学表达式或代码片段
            math_expr = extract_math_expression(input_lower)
            if math_expr:
                code_to_execute = f"result = {math_expr}"

                return {
                    "use_tool": True,
                    "tool_name": "execute_code",
                    "arguments": {
                        "code": code_to_execute
                    }
                }

    # 不需要使用工具
    return {
        "use_tool": False,
        "tool_name": None,
        "arguments": {}
    }


def extract_location(text: str) -> str:
    """
    从文本中提取地点信息
    """
    # 简单的位置提取逻辑
    location_patterns = [
        r'([北京|上海|广州|深圳|杭州|南京|武汉|成都|西安|重庆]\s*天气)',
        r'(.*?市)',
        r'(.*?省)',
        r'(.*?县)'
    ]

    for pattern in location_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace('天气', '').strip()

    return ""


def extract_math_expression(text: str) -> str:
    """
    从文本中提取数学表达式
    """
    # 匹配常见的数学运算
    math_pattern = r'([\d\+\-\*\/\.\(\)\s]+)'
    match = re.search(math_pattern, text)
    if match:
        expr = match.group(1).strip()
        # 确保表达式安全
        if re.match(r'^[\d\+\-\*\/\.\(\)\s]+$', expr):
            return expr

    return ""
