# -*- coding: utf-8 -*-
"""
办公助手 LangGraph 工作流实现
基于 llm_call.py 脚本构建，实现文档查询 RAG、周报生成、办公工具等功能
"""
import os
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
import logger

from rag.rag_system import DocumentRAG

# 从 llm_call.py 导入 LLMClient
from llm_call import LLMClient, llm_call
from llm_call import DEFAULT_MODEL, DEFAULT_BASE_URL

# 导入办公工具模块
from tools.office_tools import OfficeToolsManager
from tools.weekly_report_generator import WeeklyReportGenerator

class OfficeAssistantState(BaseModel):
    """工作流状态模型"""
    user_input: str = Field(description="用户输入的原始问题")
    action_type: str = Field(default="其他", description="分析出的操作类型")
    rag_result: Optional[str] = Field(default=None, description="RAG 检索结果")
    report_result: Optional[str] = Field(default=None, description="周报生成结果")
    tool_result: Optional[str] = Field(default=None, description="通用工具执行结果")
    office_tool_result: Optional[str] = Field(default=None, description="办公工具执行结果")
    report_file_paths: Optional[List[str]] = Field(default=None, description="周报导出文件路径")
    final_response: Optional[str] = Field(default=None, description="最终返回给用户的响应")


class OfficeAssistant:
    """办公助手工作流实现类"""

    def __init__(self, api_key: str = None, db_path: str = "./chroma_db"):
        """
        初始化办公助手

        Args:
            api_key: 阿里云 API 密钥，如果未提供则从环境变量获取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        # 初始化 LLM 客户端
        self.llm_client = LLMClient(api_key=self.api_key)

        # 初始化 RAG 引擎
        self.rag_engine = DocumentRAG(db_path=db_path)

        # 初始化办公工具管理器
        self.office_tools_manager = OfficeToolsManager(api_key=self.api_key)

        # 初始化周报生成器
        self.report_generator = WeeklyReportGenerator(api_key=self.api_key)

        # 构建 LangGraph 工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(OfficeAssistantState)

        # 添加节点
        workflow.add_node("input", self._input_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("rag_query", self._rag_query_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("office_tools", self._office_tools_node)
        workflow.add_node("tool_execution", self._tool_execution_node)
        workflow.add_node("final_response", self._final_response_node)

        # 设置入口节点
        workflow.set_entry_point("input")

        # 添加边
        workflow.add_edge("input", "decision")
        workflow.add_conditional_edges(
            "decision",
            self._route_decision,
            {
                "rag_query": "rag_query",
                "report_generation": "report_generation",
                "office_tools": "office_tools",
                "tool_execution": "tool_execution",
                "other": "tool_execution"
            }
        )
        workflow.add_edge("office_tools", "final_response")
        workflow.add_edge("rag_query", "final_response")
        workflow.add_edge("report_generation", "final_response")
        workflow.add_edge("tool_execution", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    def _input_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """输入节点：接收用户输入"""
        logger.info(f"接收用户输入：{state.user_input}")
        return state

    def _decision_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """决策节点：分析用户意图，确定操作类型"""
        decision_prompt = f"""
        请分析以下用户输入，并确定需要执行的操作类型。操作类型包括：
        - rag_query: 文档查询（涉及文档、资料、信息检索，特别是技术文档、论文等）
        - report_generation: 周报生成（涉及报告、总结、工作汇报）
        - office_tools: 办公工具（涉及日程查询、待办事项、日历管理、实时搜索等）
        - tool_execution: 其他操作（通用工具调用）
        - other: 无法确定的操作类型

        用户输入：{state.user_input}

        请只返回操作类型，不要包含其他内容。
        """

        # 使用 LLM 进行决策
        response = llm_call(
            prompt=decision_prompt,
            system_prompt="你是一个办公助手决策系统，负责分析用户需求并分类",
            model=DEFAULT_MODEL,
            retry_times=2
        )

        # 兼容处理：如果返回的是字典
        if isinstance(response, dict):
            is_success = response.get('success', False)
            result_data = response.get('data', '')
            error_msg = response.get('error_message', '未知错误')
        else:
            # 如果仍然是对象（以防万一）
            is_success = getattr(response, 'success', False)
            result_data = getattr(response, 'data', '')
            error_msg = getattr(response, 'error_message', '未知错误')

        if is_success:
            action_type = str(result_data).strip().lower()

            # 优化决策结果 - 优先匹配办公工具关键词
            if "日程" in state.user_input or "会议" in state.user_input or "安排" in state.user_input or \
                    "待办" in state.user_input or "日历" in state.user_input or "任务" in state.user_input or \
                    "搜索" in state.user_input or "资讯" in state.user_input or "新闻" in state.user_input or \
                    "天气" in state.user_input:
                action_type = "office_tools"
            elif "文档" in state.user_input or "资料" in state.user_input or "查询" in state.user_input or \
                    "模型" in state.user_input or "论文" in state.user_input or "技术" in state.user_input or \
                    "HIC-YOLOv5" in state.user_input or "YOLOv5" in state.user_input:
                action_type = "rag_query"
            elif "周报" in state.user_input or "报告" in state.user_input or "总结" in state.user_input:
                action_type = "report_generation"
            else:
                action_type = "other"

            logger.info(f"决策结果：{action_type}")
            return {"action_type": action_type}
        else:
            logger.error(f"决策失败：{error_msg}")
            return {"action_type": "other"}

    def _route_decision(self, state: OfficeAssistantState) -> str:
        """路由决策：根据决策结果确定下一个节点"""
        action_type = state.action_type
        logger.debug(f"路由决策：{action_type}")

        if action_type == "rag_query":
            return "rag_query"
        elif action_type == "report_generation":
            return "report_generation"
        elif action_type == "office_tools":
            return "office_tools"
        elif action_type == "tool_execution":
            return "tool_execution"
        else:
            return "other"

    def _report_generation_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """周报生成节点：生成结构化周报"""
        logger.info(f"进入周报生成节点：{state.user_input}")

        # 使用周报生成器生成结构化周报
        result = self.report_generator.generate_and_export(
            user_input=state.user_input,
            template="professional",
            export_formats=['markdown']
        )

        if result.get('success'):
            report_content = result.get('report_content', '')
            file_paths = [f['path'] for f in result.get('exported_files', [])]

            logger.info(f"周报生成成功，已导出到：{file_paths}")

            response_text = f"{report_content}\n\n📁 周报已导出到：{', '.join(file_paths)}"

            return {
                "report_result": response_text,
                "report_file_paths": file_paths,
                "action_type": "report_generation"
            }
        else:
            error_msg = result.get('error', '周报生成失败')
            logger.error(f"周报生成失败：{error_msg}")
            return {
                "report_result": f"周报生成失败：{error_msg}",
                "action_type": "report_generation"
            }

    def _rag_query_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """RAG 节点：处理文档查询"""
        rag_prompt = f"""
        你是一个文档问答助手，请基于以下检索到的文档内容回答用户问题：

        文档内容摘要:
        {self._get_document_context(state.user_input)}

        用户问题：{state.user_input}

        请用简洁明了的语言回答问题，不要提及文档内容。
        """

        response = llm_call(
            prompt=rag_prompt,
            system_prompt="你是一个专业的文档问答助手",
            model=DEFAULT_MODEL,
            retry_times=2
        )

        if isinstance(response, dict):
            is_success = response.get('success', False)
            result_data = response.get('data', '')
            error_msg = response.get('error_message', 'Unknown error')
        else:
            is_success = getattr(response, 'success', False)
            result_data = getattr(response, 'data', '')
            error_msg = getattr(response, 'error_message', 'Unknown error')

        if is_success:
            logger.info("RAG 查询成功")
            return {"rag_result": result_data}
        else:
            logger.error(f"RAG 查询失败：{error_msg}")
            return {"rag_result": f"文档查询失败：{error_msg}"}

    def _get_document_context(self, query: str) -> str:
        """调用 RAG 引擎进行真实文档检索"""
        try:
            results = self.rag_engine.query(question=query, n_results=3)

            if not results:
                return "未找到相关文档内容。"

            context_parts = []
            for i, res in enumerate(results):
                source = res['metadata'].get('source', '未知来源')
                doc_content = res['document']
                context_parts.append(f"[来源：{os.path.basename(source)}]\n{doc_content}")

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"RAG 检索异常：{e}")
            return f"检索系统暂时不可用：{str(e)}"

    def upload_document(self, file_path: str):
        """对外提供的文档上传方法"""
        if not os.path.exists(file_path):
            return f"❌ 文件不存在：{file_path}"
        try:
            self.rag_engine.add_document(file_path)
            return f"✅ 文档 {os.path.basename(file_path)} 已成功索引"
        except Exception as e:
            return f"❌ 上传失败：{str(e)}"

    def _report_generation_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """周报生成节点：生成周报"""
        report_prompt = f"""
        请生成一份专业的周报，基于以下信息：

        本周工作主题：{state.user_input}

        周报要求：
        1. 本周工作总结（3-4 点）
        2. 下周工作计划（3-4 点）
        3. 遇到的问题及解决方案（2-3 点）
        4. 重要事项提醒

        请用简洁专业的语言，使用正式的商务语气。
        """

        response = llm_call(
            prompt=report_prompt,
            system_prompt="你是一个专业的周报生成助手",
            model=DEFAULT_MODEL,
            retry_times=2
        )
        if isinstance(response, dict):
            is_success = response.get('success', False)
            result_data = response.get('data', '')
            error_msg = response.get('error_message', 'Unknown error')
        else:
            is_success = getattr(response, 'success', False)
            result_data = getattr(response, 'data', '')
            error_msg = getattr(response, 'error_message', 'Unknown error')

        if is_success:
            logger.info("周报生成成功")
            return {"report_result": result_data}
        else:
            logger.error(f"周报生成失败：{error_msg}")
            return {"report_result": f"周报生成失败：{error_msg}"}

    def _office_tools_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """办公工具节点：处理日程查询、待办管理、实时搜索等办公需求"""
        logger.info(f"进入办公工具节点：{state.user_input}")

        result = self.office_tools_manager.execute_office_tool(state.user_input)

        if result.get('success'):
            logger.info(f"办公工具执行成功：{result.get('tool_used')}")
            return {
                "office_tool_result": result.get('result'),
                "action_type": "office_tools"
            }
        else:
            logger.error(f"办公工具执行失败：{result.get('error')}")
            if result.get('degraded'):
                return {
                    "office_tool_result": result.get('error'),
                    "action_type": "office_tools"
                }
            else:
                return {
                    "office_tool_result": f"抱歉，处理您的请求时出现问题：{result.get('error')}",
                    "action_type": "office_tools"
                }

    def _tool_execution_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """工具执行节点：处理其他操作"""
        tool_result = f"已处理您的请求：{state.user_input}"
        logger.info(f"工具执行：{tool_result}")
        return {"tool_result": tool_result}

    def _final_response_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """最终响应节点：整合结果并生成最终回复"""
        if state.office_tool_result:
            final_response = state.office_tool_result
        elif state.rag_result:
            final_response = f"文档查询结果：{state.rag_result}"
        elif state.report_result:
            final_response = f"周报生成结果：{state.report_result}"
        elif state.tool_result:
            final_response = f"操作执行结果：{state.tool_result}"
        else:
            final_response = "未找到相关操作结果，请重试。"

        logger.info(f"生成最终响应：{final_response}")
        return {"final_response": final_response}

    def run(self, user_input: str) -> str:
        """运行工作流并返回最终响应"""
        initial_state = {
            "user_input": user_input,
            "action_type": "其他"
        }

        result = self.workflow.invoke(initial_state)

        return result.get("final_response", "未生成有效回复")

    def generate_weekly_report(self, user_input: str,
                               template: str = "professional",
                               export_formats: List[str] = None) -> Dict[str, Any]:
        """
        直接调用周报生成器（不经过工作流）

        Args:
            user_input: 用户输入
            template: 模板类型（professional/simple/detailed）
            export_formats: 导出格式列表

        Returns:
            生成结果
        """
        if export_formats is None:
            export_formats = ['markdown']

        return self.report_generator.generate_and_export(
            user_input=user_input,
            template=template,
            export_formats=export_formats
        )

    def get_tool_statistics(self) -> Dict[str, Any]:
        """获取办公工具调用统计信息"""
        return self.office_tools_manager.get_tool_statistics()


# 测试示例
if __name__ == "__main__":
    print("=" * 60)
    print("办公助手 LangGraph 工作流测试 - 整合办公工具模块")
    print("=" * 60)

    try:
        assistant = OfficeAssistant()
        print("✅ 办公助手初始化成功")
        print("📦 已加载功能：")
        print("   • RAG 文档问答")
        print("   • 结构化周报生成（支持 Markdown/Word 导出）")
        print("   • 周报自动生成")
        print("   • 实时搜索（行业资讯/新闻/天气）")
        print("   • 日历查询与管理")
        print("   • 待办清单管理")
        print("=" * 60)
    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        exit(1)

    test_files = [
        "2309.16393v2.pdf",
    ]

    print("\n--- 阶段 1: 文档上传与索引 ---")
    for file in test_files:
        if os.path.exists(file):
            print(assistant.upload_document(file))
        else:
            print(f"⚠️ 跳过不存在的文件：{file}")

    print("\n--- 阶段 2: RAG 文档问答测试 ---")
    rag_queries = [
        "HIC-YOLOv5 模型的最新进展是什么？",
        "HIC-YOLOv5 相比原始 YOLOv5 有哪些改进？"
    ]

    for query in rag_queries:
        print(f"\n👤 用户：{query}")
        response = assistant.run(query)
        print(f"🤖 助手：{response}")
        print("-" * 60)

    print("\n--- 阶段 3: 办公工具功能测试 ---")
    office_queries = [
        ("搜索今天的人工智能行业新闻", "实时搜索"),
        ("查询今天的日程安排", "日历查询"),
        ("添加一个待办事项：下午 3 点参加项目会议", "待办添加"),
        ("查看我的待办清单", "待办列表"),
        ("明天广州天气怎么样", "天气搜索")
    ]

    for query, desc in office_queries:
        print(f"\n👤 用户 ({desc}): {query}")
        response = assistant.run(query)
        print(f"🤖 助手：{response}")
        print("-" * 60)

    print("\n--- 阶段 4: 周报生成测试 ---")

    # 测试 1：简单输入
    print("\n【测试 4.1】简单输入")
    report_query_1 = "本周完成了 AI 项目开发，下周准备优化性能"
    print(f"👤 用户：{report_query_1}")
    response = assistant.run(report_query_1)
    print(f"🤖 助手：{response}")
    print("-" * 60)

    # 测试 2：详细输入
    print("\n【测试 4.2】详细输入（使用周报生成器 API）")
    report_query_2 = """
    工作内容：
    1. 完成办公助手系统开发
    2. 集成实时搜索工具
    3. 实现日历和待办功能

    成果：系统响应速度提升 50%，用户满意度 95%

    下周计划：
    1. 添加更多办公工具
    2. 优化用户体验
    3. 编写技术文档

    问题：API 调用超时，已通过重试机制解决
    """
    print(f"👤 用户：{report_query_2[:100]}...")

    result = assistant.generate_weekly_report(
        user_input=report_query_2,
        template="professional",
        export_formats=['markdown', 'word']
    )

    if result['success']:
        print("✅ 周报生成成功")
        print(f"\n📄 内容预览：\n{result['report_content'][:300]}...\n")
        for file_info in result.get('exported_files', []):
            print(f"📁 导出文件：{file_info['format']} -> {file_info['path']}")
    else:
        print(f"❌ 生成失败：{result.get('error')}")

    print("-" * 60)

    print("\n--- 阶段 5: 工具调用统计 ---")
    stats = assistant.get_tool_statistics()
    print("📊 工具调用统计:")
    print(f"   总调用次数：{stats.get('total_calls', 0)}")
    print(f"   成功次数：{stats.get('successful_calls', 0)}")
    print(f"   失败次数：{stats.get('failed_calls', 0)}")
    print(f"   重试次数：{stats.get('retry_count', 0)}")
    print(f"   成功率：{stats.get('success_rate', '0%')}")
    print("=" * 60)

    print("\n✅ 所有测试完成！")
