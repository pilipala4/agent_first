#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
办公助手LangGraph工作流实现
基于llm_call.py脚本构建，实现文档查询RAG、周报生成等核心功能
"""
import os
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage
#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
import logger

from rag.rag_system import DocumentRAG

# 从llm_call.py导入LLMClient
from llm_call import LLMClient, llm_call
from llm_call import DEFAULT_MODEL, DEFAULT_BASE_URL



class OfficeAssistantState(BaseModel):
    """工作流状态模型"""
    user_input: str = Field(description="用户输入的原始问题")
    action_type: str = Field(default="其他", description="分析出的操作类型")
    rag_result: Optional[str] = Field(default=None, description="RAG检索结果")
    report_result: Optional[str] = Field(default=None, description="周报生成结果")
    tool_result: Optional[str] = Field(default=None, description="通用工具执行结果")
    final_response: Optional[str] = Field(default=None, description="最终返回给用户的响应")


class OfficeAssistant:
    """办公助手工作流实现类"""

    def __init__(self, api_key: str = None, db_path: str = "./office_docs_db"):
        """
        初始化办公助手

        Args:
            api_key: 阿里云API密钥，如果未提供则从环境变量获取
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置DASHSCOPE_API_KEY环境变量")

        # 初始化LLM客户端
        self.llm_client = LLMClient(api_key=self.api_key)

        # 初始化 RAG 引擎 (新增)
        self.rag_engine = DocumentRAG(db_path=db_path)

        # 构建LangGraph工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(OfficeAssistantState)

        # 添加节点
        workflow.add_node("input", self._input_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("rag_query", self._rag_query_node)
        workflow.add_node("report_generation", self._report_generation_node)
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
                "tool_execution": "tool_execution",
                "other": "tool_execution"
            }
        )
        workflow.add_edge("rag_query", "final_response")
        workflow.add_edge("report_generation", "final_response")
        workflow.add_edge("tool_execution", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    def _input_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """输入节点：接收用户输入"""
        logger.info(f"接收用户输入: {state.user_input}")
        return state

    def _decision_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """决策节点：分析用户意图，确定操作类型"""
        decision_prompt = f"""
        请分析以下用户输入，并确定需要执行的操作类型。操作类型包括：
        - rag_query: 文档查询（涉及文档、资料、信息检索）
        - report_generation: 周报生成（涉及报告、总结、工作汇报）
        - tool_execution: 其他操作（日程安排、邮件处理等）
        - other: 无法确定的操作类型

        用户输入: {state.user_input}

        请只返回操作类型，不要包含其他内容。
        """

        # 使用LLM进行决策
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

            # 优化决策结果 (保持原有逻辑)
            if "文档" in state.user_input or "资料" in state.user_input or "查询" in state.user_input:
                action_type = "rag_query"
            elif "周报" in state.user_input or "报告" in state.user_input or "总结" in state.user_input:
                action_type = "report_generation"
            elif "日程" in state.user_input or "会议" in state.user_input or "安排" in state.user_input:
                action_type = "tool_execution"
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
        logger.debug(f"路由决策: {action_type}")

        if action_type in ["rag_query", "report_generation", "tool_execution"]:
            return action_type
        else:
            return "other"

    def _rag_query_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """RAG节点：处理文档查询"""
        rag_prompt = f"""
        你是一个文档问答助手，请基于以下检索到的文档内容回答用户问题：

        文档内容摘要:
        {self._get_document_context(state.user_input)}

        用户问题: {state.user_input}

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
        """模拟文档检索，返回文档摘要"""
        # 实际应用中这里应连接到RAG系统
        # 模拟文档内容
        document_samples = [
            "项目A的最新进展：已完成需求分析，正在开发核心功能模块",
            "公司政策文档：员工请假需提前3天申请，需部门经理批准",
            "市场调研报告：Q3季度智能手机市场增长15%，主要增长区域为东南亚",
            "项目B技术方案：采用微服务架构，使用Spring Boot和Docker",
            "会议纪要：2023-10-05会议讨论了新产品的UI设计，确定了主要功能点"
        ]

        # 简单匹配文档
        matches = [doc for doc in document_samples if query.lower() in doc.lower()]

        if matches:
            return "相关文档摘要: " + ", ".join(matches[:2])
        else:
            return "未找到相关文档内容。请尝试更具体的查询。"

    def _report_generation_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """周报生成节点：生成周报"""
        report_prompt = f"""
        请生成一份专业的周报，基于以下信息：

        本周工作主题: {state.user_input}

        周报要求：
        1. 本周工作总结（3-4点）
        2. 下周工作计划（3-4点）
        3. 遇到的问题及解决方案（2-3点）
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
            logger.error(f"周报生成失败: {error_msg}")
            return {"report_result": f"周报生成失败: {error_msg}"}

    def _tool_execution_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """工具执行节点：处理其他操作"""
        # 实际应用中这里应调用具体工具API
        tool_result = f"已处理您的请求: {state.user_input}"

        logger.info(f"工具执行: {tool_result}")
        return {"tool_result": tool_result}

    def _final_response_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """最终响应节点：整合结果并生成最终回复"""
        # 根据操作类型选择结果
        if state.rag_result:
            final_response = f"文档查询结果: {state.rag_result}"
        elif state.report_result:
            final_response = f"周报生成结果: {state.report_result}"
        elif state.tool_result:
            final_response = f"操作执行结果: {state.tool_result}"
        else:
            final_response = "未找到相关操作结果，请重试。"

        logger.info(f"生成最终响应: {final_response}")
        return {"final_response": final_response}

    def run(self, user_input: str) -> str:
        """运行工作流并返回最终响应"""
        # 初始化状态时可以直接传字典，LangGraph 会自动处理
        initial_state = {
            "user_input": user_input,
            "action_type": "其他"
        }

        # invoke 返回的是最终状态的字典
        result = self.workflow.invoke(initial_state)

        # 通过字典键访问 final_response
        return result.get("final_response", "未生成有效回复")


# 测试示例
if __name__ == "__main__":
    print("=" * 50)
    print("办公助手LangGraph工作流测试")
    print("=" * 50)

    # 初始化办公助手
    try:
        assistant = OfficeAssistant()
        print("✅ 办公助手初始化成功")
    except ValueError as e:
        print(f"❌ 初始化失败: {e}")
        exit(1)

    # 测试用例
    test_queries = [
        "查询项目A的最新进展",
        "帮我写一份本周工作周报",
        "安排下周三的团队会议",
        "今天天气怎么样",
        "如何设置公司邮箱的自动回复"
    ]

    for query in test_queries:
        print(f"\n--- 用户输入: {query} ---")
        response = assistant.run(query)
        print(f"助手回复: {response}")
        print("-" * 50)

    print("\n测试完成！")