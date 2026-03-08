# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0
一站式办公解决方案：文档问答 + 周报生成 + 工具调用
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 导入本地模块
from logger import logger
from llm_call import LLMClient, llm_call, DEFAULT_MODEL
from rag.rag_system import DocumentRAG
from tools.office_tools import OfficeToolsManager
from tools.weekly_report_generator import WeeklyReportGenerator
from tools.tool_encapsulation import ToolManager


class OfficeAssistantState(BaseModel):
    """工作流状态模型"""
    user_input: str = Field(description="用户输入的原始问题")
    action_type: str = Field(default="其他", description="分析出的操作类型")
    rag_result: Optional[str] = Field(default=None, description="RAG 检索结果")
    report_result: Optional[str] = Field(default=None, description="周报生成结果")
    office_tool_result: Optional[str] = Field(default=None, description="办公工具执行结果")
    report_file_paths: Optional[List[str]] = Field(default=None, description="周报导出文件路径")
    final_response: Optional[str] = Field(default=None, description="最终返回给用户的响应")
    conversation_history: List[Dict] = Field(default_factory=list, description="对话历史")


class SmartOfficeAssistant:
    """智能办公助手 - 统一入口"""

    def __init__(self, api_key: str = None, db_path: str = "./chroma_db"):
        """
        初始化智能办公助手

        Args:
            api_key: 阿里云 API 密钥
            db_path: RAG 数据库路径
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        # 初始化各模块
        self.llm_client = LLMClient(api_key=self.api_key)
        self.rag_engine = DocumentRAG(db_path=db_path)
        self.office_tools = OfficeToolsManager(api_key=self.api_key)
        self.report_generator = WeeklyReportGenerator(api_key=self.api_key)
        self.tool_manager = ToolManager(api_key=self.api_key)

        # 构建工作流
        self.workflow = self._build_workflow()

        # 对话历史
        self.conversation_history = []

    def _build_workflow(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        workflow = StateGraph(OfficeAssistantState)

        # 添加节点
        workflow.add_node("input", self._input_node)
        workflow.add_node("decision", self._decision_node)
        workflow.add_node("rag_query", self._rag_query_node)
        workflow.add_node("report_generation", self._report_generation_node)
        workflow.add_node("office_tools", self._office_tools_node)
        workflow.add_node("final_response", self._final_response_node)

        # 设置入口
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
                "other": "final_response"
            }
        )
        workflow.add_edge("rag_query", "final_response")
        workflow.add_edge("report_generation", "final_response")
        workflow.add_edge("office_tools", "final_response")
        workflow.add_edge("final_response", END)

        return workflow.compile()

    def _input_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """输入节点"""
        logger.info(f"接收用户输入：{state.user_input}")
        return state

    def _decision_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """决策节点：分析用户意图"""
        decision_prompt = f"""
请分析用户输入，确定需要执行的操作类型：

可用操作：
- rag_query: 文档查询（涉及文档、资料、论文、技术文档等内容检索）
- report_generation: 周报生成（涉及报告、总结、工作汇报等）
- office_tools: 办公工具（涉及日程、会议、待办、搜索、天气、新闻等）
- other: 其他或闲聊

用户输入：{state.user_input}

请只返回操作类型名称，不要包含其他内容。
"""

        try:
            response = llm_call(
                prompt=decision_prompt,
                system_prompt="你是办公助手意图识别系统，只返回操作类型名称",
                model=DEFAULT_MODEL,
                retry_times=2
            )

            if isinstance(response, dict) and response.get('success'):
                action_type = str(response.get('data', '')).strip().lower()

                # 关键词优化匹配
                input_lower = state.user_input.lower()
                if any(kw in input_lower for kw in
                       ['日程', '会议', '待办', '日历', '任务', '搜索', '资讯', '新闻', '天气']):
                    action_type = "office_tools"
                elif any(kw in input_lower for kw in
                         ['文档', '资料', '论文', '查询', '模型', '技术', 'HIC-YOLOv5', 'YOLOv5']):
                    action_type = "rag_query"
                elif any(kw in input_lower for kw in ['周报', '报告', '总结', '汇报']):
                    action_type = "report_generation"
                else:
                    action_type = "other"

                logger.info(f"决策结果：{action_type}")
                return {"action_type": action_type}
            else:
                logger.error(f"决策失败：{response.get('error_message')}")
                return {"action_type": "other"}

        except Exception as e:
            logger.error(f"决策异常：{e}")
            return {"action_type": "other"}

    def _route_decision(self, state: OfficeAssistantState) -> str:
        """路由决策"""
        action_type = state.action_type
        logger.debug(f"路由决策：{action_type}")

        if action_type == "rag_query":
            return "rag_query"
        elif action_type == "report_generation":
            return "report_generation"
        elif action_type == "office_tools":
            return "office_tools"
        else:
            return "other"

    def _rag_query_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """RAG 文档查询节点"""
        logger.info(f"进入 RAG 查询节点：{state.user_input}")

        try:
            # 检查数据库是否有文档
            if self.rag_engine.collection.count() == 0:
                return {
                    "rag_result": "❌ 数据库中没有文档，请先添加文档到系统。",
                    "action_type": "rag_query"
                }

            # 检索相关文档
            results = self.rag_engine.query(question=state.user_input, n_results=3)

            if not results:
                return {
                    "rag_result": "抱歉，未能从文档中找到相关信息。",
                    "action_type": "rag_query"
                }

            # 构建上下文
            context_parts = []
            for i, res in enumerate(results, 1):
                source = os.path.basename(res['metadata'].get('source', '未知来源'))
                doc_content = res['document'][:1000]  # 限制长度
                context_parts.append(f"[来源：{source}]\n{doc_content}")

            context = "\n\n---\n\n".join(context_parts)

            # 调用 LLM 生成答案
            rag_prompt = f"""
你是一个专业的文档问答助手。请基于以下检索到的文档内容回答用户问题：

文档内容：
{context}

用户问题：{state.user_input}

请用简洁明了的语言回答问题，不要提及文档来源。
"""

            response = llm_call(
                prompt=rag_prompt,
                system_prompt="你是专业的文档问答助手",
                model=DEFAULT_MODEL,
                retry_times=2
            )

            if isinstance(response, dict) and response.get('success'):
                answer = response.get('data', '无法生成答案')
                logger.info("RAG 查询成功")
                return {
                    "rag_result": answer,
                    "action_type": "rag_query"
                }
            else:
                error_msg = response.get('error_message', '未知错误') if isinstance(response, dict) else '未知错误'
                logger.error(f"RAG 查询失败：{error_msg}")
                return {
                    "rag_result": f"文档查询失败：{error_msg}",
                    "action_type": "rag_query"
                }

        except Exception as e:
            logger.error(f"RAG 查询异常：{e}")
            return {
                "rag_result": f"文档查询异常：{str(e)}",
                "action_type": "rag_query"
            }

    def _report_generation_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """周报生成节点"""
        logger.info(f"进入周报生成节点：{state.user_input}")

        try:
            # 使用周报生成器
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

        except Exception as e:
            logger.error(f"周报生成异常：{e}")
            return {
                "report_result": f"周报生成异常：{str(e)}",
                "action_type": "report_generation"
            }

    def _office_tools_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """办公工具节点"""
        logger.info(f"进入办公工具节点：{state.user_input}")

        try:
            result = self.office_tools.execute_office_tool(state.user_input)

            if result.get('success'):
                logger.info(f"办公工具执行成功：{result.get('tool_used')}")
                return {
                    "office_tool_result": result.get('result'),
                    "action_type": "office_tools"
                }
            else:
                error_msg = result.get('error', '办公工具执行失败')
                logger.error(f"办公工具执行失败：{error_msg}")
                return {
                    "office_tool_result": f"办公工具执行失败：{error_msg}",
                    "action_type": "office_tools"
                }

        except Exception as e:
            logger.error(f"办公工具执行异常：{e}")
            return {
                "office_tool_result": f"办公工具执行异常：{str(e)}",
                "action_type": "office_tools"
            }

    def _final_response_node(self, state: OfficeAssistantState) -> Dict[str, Any]:
        """最终响应节点"""
        if state.office_tool_result:
            final_response = state.office_tool_result
        elif state.rag_result:
            final_response = state.rag_result
        elif state.report_result:
            final_response = state.report_result
        else:
            # 其他情况：使用通用对话
            general_prompt = f"""
你是一个友好的办公助手。请回复用户的问题：

用户输入：{state.user_input}

请用友好、专业的语气回复。
"""
            try:
                response = llm_call(
                    prompt=general_prompt,
                    system_prompt="你是友好的办公助手",
                    model=DEFAULT_MODEL,
                    retry_times=2
                )
                final_response = response.get('data', '我无法理解您的问题，请换个方式描述。') if isinstance(response,
                                                                                                          dict) else '我无法理解您的问题。'
            except:
                final_response = "我无法理解您的问题，请换个方式描述。"

        logger.info(f"生成最终响应")
        return {"final_response": final_response}

    def run(self, user_input: str) -> str:
        """运行工作流并返回响应"""
        initial_state = {
            "user_input": user_input,
            "action_type": "其他",
            "conversation_history": self.conversation_history
        }

        result = self.workflow.invoke(initial_state)

        # 更新对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": result.get("final_response", "")})

        # 限制历史记录长度
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return result.get("final_response", "未生成有效回复")

    def upload_document(self, file_path: str) -> str:
        """上传文档到 RAG 系统"""
        if not os.path.exists(file_path):
            return f"❌ 文件不存在：{file_path}"

        try:
            self.rag_engine.add_document(file_path)
            return f"✅ 文档 {os.path.basename(file_path)} 已成功索引"
        except Exception as e:
            return f"❌ 上传失败：{str(e)}"

    def generate_weekly_report(self, user_input: str,
                               template: str = "professional",
                               export_formats: List[str] = None) -> Dict[str, Any]:
        """直接调用周报生成器"""
        if export_formats is None:
            export_formats = ['markdown']

        return self.report_generator.generate_and_export(
            user_input=user_input,
            template=template,
            export_formats=export_formats
        )

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "conversation_turns": len(self.conversation_history) // 2,
            "rag_docs_count": self.rag_engine.collection.count(),
            "office_tools_stats": self.office_tools.get_tool_statistics()
        }


def print_welcome():
    """打印欢迎界面"""
    print("\n" + "=" * 70)
    print("🤖 智能办公助手 V1.0")
    print("=" * 70)
    print("功能列表:")
    print("  📚 文档问答 - 上传 PDF/Word/Markdown文档，进行智能问答")
    print("  📝 周报生成 - 自动生成结构化周报，支持 Markdown/Word导出")
    print("  🔧 办公工具 - 实时搜索、日历管理、待办清单")
    print("=" * 70)
    print("使用说明:")
    print("  • 上传文档：upload <文件路径>")
    print("  • 对话交流：直接输入问题即可")
    print("  • 生成周报：report <工作内容>")
    print("  • 查看帮助：help 或 h")
    print("  • 清空历史：clear 或 cls")
    print("  • 退出系统：quit 或 exit 或 q")
    print("=" * 70 + "\n")


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 70)
    print("📖 使用帮助")
    print("=" * 70)
    print("\n【文档问答】")
    print("  1. 上传文档：upload ./path/to/file.pdf")
    print("  2. 提问：HIC-YOLOv5 模型有什么特点？")
    print("\n【周报生成】")
    print("  方式 1: report 本周完成了项目开发，下周继续优化")
    print("  方式 2: 帮我写一份周报，本周...")
    print("\n【办公工具】")
    print("  • 搜索：今天的人工智能新闻")
    print("  • 日历：查询今天的日程安排")
    print("  • 待办：添加一个待办事项：下午 3 点开会")
    print("  • 天气：明天广州天气怎么样")
    print("\n【其他命令】")
    print("  • help/h - 查看帮助")
    print("  • clear/cls - 清空对话历史")
    print("  • quit/exit/q - 退出系统")
    print("=" * 70 + "\n")


def interactive_mode():
    """交互式命令行模式"""
    print_welcome()

    try:
        assistant = SmartOfficeAssistant()
        print("✅ 系统初始化成功\n")
    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        print("请确保设置了 DASHSCOPE_API_KEY 环境变量")
        return

    while True:
        try:
            user_input = input("👤 您：").strip()

            if not user_input:
                continue

            # 处理特殊命令
            cmd_lower = user_input.lower()

            if cmd_lower in ['quit', 'exit', 'q', '退出']:
                print("\n👋 再见！祝您工作顺利！")
                break

            elif cmd_lower in ['help', 'h', '帮助']:
                print_help()
                continue

            elif cmd_lower in ['clear', 'cls', '清空']:
                assistant.clear_history()
                print("✅ 对话历史已清空")
                continue

            elif cmd_lower.startswith('upload '):
                file_path = user_input[7:].strip()
                # 去除引号
                if file_path.startswith('"') and file_path.endswith('"'):
                    file_path = file_path[1:-1]
                elif file_path.startswith("'") and file_path.endswith("'"):
                    file_path = file_path[1:-1]
                result = assistant.upload_document(file_path)
                print(result)
                continue

            elif cmd_lower.startswith('report '):
                report_input = user_input[7:].strip()
                print("\n⏳ 正在生成周报...\n")
                result = assistant.generate_weekly_report(
                    user_input=report_input,
                    template="professional",
                    export_formats=['markdown']
                )
                if result.get('success'):
                    print("✅ 周报生成成功！\n")
                    print(result['report_content'])
                    for file_info in result.get('exported_files', []):
                        print(f"\n📁 已导出：{file_info['path']}")
                else:
                    print(f"❌ 周报生成失败：{result.get('error')}")
                continue

            # 正常工作流处理
            print("\n⏳ 思考中...\n")
            response = assistant.run(user_input)
            print(f"🤖 助手：{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 对话被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 发生错误：{str(e)}\n")
            logger.error(f"交互式模式异常：{e}")


def main():
    """主函数"""
    # 检查是否需要上传文档
    if len(sys.argv) > 1:
        if sys.argv[1] == '--upload' and len(sys.argv) > 2:
            # 命令行上传文档
            file_path = sys.argv[2]
            try:
                assistant = SmartOfficeAssistant()
                result = assistant.upload_document(file_path)
                print(result)
                return
            except Exception as e:
                print(f"❌ 上传失败：{e}")
                return

    # 默认进入交互模式
    interactive_mode()


if __name__ == '__main__':
    main()
