import requests
import subprocess
import re
from typing import Dict, Any, List
from llm_call import DEFAULT_MODEL
import json
from serpapi import SerpApiClient
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pickle
import hashlib

load_dotenv()


class ToolManager:
    """
    工具管理器，封装百度搜索、代码运行、日历查询和待办清单工具
    """

    def __init__(self, api_key: str = None, storage_path: str = "./office_tools_data.pkl"):
        self.api_key = api_key
        self.storage_path = storage_path
        self.tools = {
            "search": self.search,
            "execute_code": self.execute_code,
            "query_calendar": self.query_calendar,
            "add_todo": self.add_todo,
            "list_todos": self.list_todos,
            "complete_todo": self.complete_todo
        }
        self._load_data()

    def _load_data(self):
        """加载持久化数据"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.calendar_events = data.get('calendar', {})
                    self.todos = data.get('todos', [])
            else:
                self.calendar_events = {}
                self.todos = []
        except Exception as e:
            print(f"加载数据失败：{e}")
            self.calendar_events = {}
            self.todos = []

    def _save_data(self):
        """保存持久化数据"""
        try:
            data = {
                'calendar': self.calendar_events,
                'todos': self.todos
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存数据失败：{e}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        返回可用工具的描述，用于 LLM 识别
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
                                "description": "返回结果数量，默认为 3",
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
                    "description": "执行 Python 代码，适用于数学计算、数据分析、文本处理等任务",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "要执行的 Python 代码"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_calendar",
                    "description": "查询指定日期的日历安排、会议和事件",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "查询的日期，格式为 YYYY-MM-DD，或使用'today'、'tomorrow'"
                            }
                        },
                        "required": ["date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_todo",
                    "description": "添加待办事项到清单",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "待办事项内容"
                            },
                            "priority": {
                                "type": "string",
                                "description": "优先级：high, medium, low",
                                "enum": ["high", "medium", "low"],
                                "default": "medium"
                            },
                            "due_date": {
                                "type": "string",
                                "description": "截止日期，格式为 YYYY-MM-DD"
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_todos",
                    "description": "列出所有待办事项，可按状态和优先级筛选",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {
                                "type": "string",
                                "description": "筛选状态：pending, completed, all",
                                "enum": ["pending", "completed", "all"],
                                "default": "pending"
                            },
                            "priority": {
                                "type": "string",
                                "description": "筛选优先级：high, medium, low, all",
                                "enum": ["high", "medium", "low", "all"],
                                "default": "all"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_todo",
                    "description": "标记待办事项为已完成",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "integer",
                                "description": "待办事项的 ID"
                            }
                        },
                        "required": ["task_id"]
                    }
                }
            }
        ]

    def search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """
        一个基于 SerpApi 的实战网页搜索引擎工具。
        它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
        """
        print(f"🔍 正在执行 [SerpApi] 网页搜索：{query}")
        try:
            api_key = os.getenv("SERPAPI_API_KEY")
            if not api_key:
                return {"success": False, "error": "SERPAPI_API_KEY 未配置"}

            params = {
                "engine": "google",
                "q": query,
                "api_key": api_key,
                "gl": "cn",
                "hl": "zh-cn",
                "num": num_results
            }

            client = SerpApiClient(params)
            results = client.get_dict()

            if "answer_box_list" in results:
                return {"success": True, "result": "\n".join(results["answer_box_list"])}
            if "answer_box" in results and "answer" in results["answer_box"]:
                return {"success": True, "result": results["answer_box"]["answer"]}
            if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
                return {"success": True, "result": results["knowledge_graph"]["description"]}
            if "organic_results" in results and results["organic_results"]:
                snippets = [
                    f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                    for i, res in enumerate(results["organic_results"][:num_results])
                ]
                return {"success": True, "result": "\n\n".join(snippets)}

            return {"success": True, "result": f"对不起，没有找到关于 '{query}' 的信息。"}

        except Exception as e:
            return {"success": False, "error": f"搜索时发生错误：{e}", "query": query}

    def execute_code(self, code: str) -> Dict[str, Any]:
        """
        执行 Python 代码工具
        """
        try:
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

            exec_globals = {}
            exec(code, exec_globals)

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

    def query_calendar(self, date: str) -> Dict[str, Any]:
        """
        查询指定日期的日历安排
        """
        print(f"📅 正在查询日历：{date}")
        try:
            if date.lower() == 'today':
                date = datetime.now().strftime('%Y-%m-%d')
            elif date.lower() == 'tomorrow':
                date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            events = self.calendar_events.get(date, [])

            if not events:
                return {
                    "success": True,
                    "result": f"{date} 暂无日程安排"
                }

            result_lines = [f"📅 {date} 的日程安排："]
            for event in events:
                time_str = event.get('time', '全天')
                title = event.get('title', '无标题')
                desc = event.get('description', '')
                result_lines.append(f"  • {time_str} - {title}")
                if desc:
                    result_lines.append(f"    {desc}")

            return {
                "success": True,
                "result": "\n".join(result_lines)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"查询日历失败：{e}",
                "date": date
            }

    def add_calendar_event(self, date: str, title: str, time: str = None, description: str = None) -> Dict[str, Any]:
        """
        添加日历事件
        """
        try:
            if date.lower() == 'today':
                date = datetime.now().strftime('%Y-%m-%d')
            elif date.lower() == 'tomorrow':
                date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

            if date not in self.calendar_events:
                self.calendar_events[date] = []

            event = {
                'title': title,
                'time': time,
                'description': description
            }

            self.calendar_events[date].append(event)
            self._save_data()

            return {
                "success": True,
                "result": f"✅ 已添加日程：{date} - {title}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"添加日程失败：{e}"
            }

    def add_todo(self, task: str = None, todo_item: str = None, item: str = None,
                 priority: str = "medium", due_date: str = None) -> Dict[str, Any]:
        """
        添加待办事项
        """
        # 兼容多种参数名
        actual_task = task or todo_item or item or "未命名任务"

        print(f"✓ 正在添加待办：{actual_task}")
        try:
            todo_id = len(self.todos) + 1

            todo = {
                'id': todo_id,
                'task': actual_task,
                'priority': priority,
                'due_date': due_date,
                'status': 'pending',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'completed_at': None
            }

            self.todos.append(todo)
            self._save_data()

            priority_map = {'high': '高', 'medium': '中', 'low': '低'}
            return {
                "success": True,
                "result": f"✅ 已添加待办事项 #{todo_id}: {actual_task} (优先级：{priority_map[priority]})"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"添加待办失败：{e}"
            }

    def list_todos(self, status: str = "pending", priority: str = "all") -> Dict[str, Any]:
        """
        列出待办事项
        """
        print(f"📋 正在列出自办事项：status={status}, priority={priority}")
        try:
            filtered_todos = self.todos

            if status != "all":
                filtered_todos = [t for t in filtered_todos if t['status'] == status]

            if priority != "all":
                filtered_todos = [t for t in filtered_todos if t['priority'] == priority]

            if not filtered_todos:
                status_map = {'pending': '待处理', 'completed': '已完成', 'all': '所有'}
                return {
                    "success": True,
                    "result": f"暂无{status_map[status]}的待办事项"
                }

            result_lines = ["📋 待办事项清单："]
            priority_map = {'high': '🔴 高', 'medium': '🟡 中', 'low': '🟢 低'}

            for todo in filtered_todos:
                status_icon = '✓' if todo['status'] == 'completed' else '○'
                line = f"{status_icon} #{todo['id']} [{priority_map[todo['priority']]}] {todo['task']}"
                if todo['due_date']:
                    line += f" (截止：{todo['due_date']})"
                result_lines.append(line)

            return {
                "success": True,
                "result": "\n".join(result_lines)
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"列出待办失败：{e}"
            }

    def complete_todo(self, task_id: int) -> Dict[str, Any]:
        """
        完成待办事项
        """
        print(f"✓ 正在标记待办 #{task_id} 为已完成")
        try:
            for todo in self.todos:
                if todo['id'] == task_id:
                    todo['status'] = 'completed'
                    todo['completed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self._save_data()

                    return {
                        "success": True,
                        "result": f"✅ 已标记待办 #{task_id} 为已完成：{todo['task']}"
                    }

            return {
                "success": False,
                "error": f"未找到 ID 为 {task_id} 的待办事项"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"完成待办失败：{e}"
            }


def determine_tool_usage(user_input: str, tool_manager: ToolManager) -> Dict[str, Any]:
    """
    根据用户输入判断是否需要调用工具以及调用哪个工具
    """
    search_keywords = [
        '今天', '明天', '天气', '新闻', '股票', '实时', '最新',
        '查询', '搜索', '了解', '有什么', '怎么样', '如何', '资讯', '行业'
    ]

    calendar_keywords = ['日历', '日程', '会议', '安排', '预约']

    todo_keywords = ['待办', '代办', '清单', '任务', '计划', '要做', '记得']

    code_keywords = [
        '计算', '算一下', '数学', '加减乘除', '统计', '求和',
        '平均值', '编程', '代码', '算法', '公式'
    ]

    input_lower = user_input.lower()

    for keyword in calendar_keywords:
        if keyword in input_lower:
            if '查询' in input_lower or '看' in input_lower:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2}|today|tomorrow|今天 | 明天)', input_lower)
                date = date_match.group(1) if date_match else 'today'

                return {
                    "use_tool": True,
                    "tool_name": "query_calendar",
                    "arguments": {"date": date}
                }
            elif '添加' in input_lower or '增加' in input_lower:
                return {
                    "use_tool": True,
                    "tool_name": "add_calendar_event",
                    "arguments": {"date": "today", "title": "新日程"}
                }

    for keyword in todo_keywords:
        if keyword in input_lower:
            if '添加' in input_lower or '增加' in input_lower:
                task_match = re.search(r'(?:添加 | 增加)(.*?)(?:优先级 | 截止|$)', input_lower)
                task = task_match.group(1).strip() if task_match else "新待办事项"

                priority = "medium"
                if '高' in input_lower or '紧急' in input_lower:
                    priority = "high"
                elif '低' in input_lower:
                    priority = "low"

                return {
                    "use_tool": True,
                    "tool_name": "add_todo",
                    "arguments": {"task": task, "priority": priority}
                }
            elif '查看' in input_lower or '列表' in input_lower or '有' in input_lower:
                return {
                    "use_tool": True,
                    "tool_name": "list_todos",
                    "arguments": {"status": "pending", "priority": "all"}
                }
            elif '完成' in input_lower or '做完' in input_lower:
                id_match = re.search(r'#?(\d+)', input_lower)
                task_id = int(id_match.group(1)) if id_match else 1

                return {
                    "use_tool": True,
                    "tool_name": "complete_todo",
                    "arguments": {"task_id": task_id}
                }

    for keyword in search_keywords:
        if keyword in input_lower:
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

    for keyword in code_keywords:
        if keyword in input_lower:
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

    return {
        "use_tool": False,
        "tool_name": None,
        "arguments": {}
    }


def extract_location(text: str) -> str:
    """
    从文本中提取地点信息
    """
    location_patterns = [
        r'([北京 | 上海 | 广州 | 深圳 | 杭州 | 南京 | 武汉 | 成都 | 西安 | 重庆]\s*天气)',
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
    math_pattern = r'([\d\+\-\*\/\.\(\)\s]+)'
    match = re.search(math_pattern, text)
    if match:
        expr = match.group(1).strip()
        if re.match(r'^[\d\+\-\*\/\.\(\)\s]+$', expr):
            return expr

    return ""

