# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0 - 综合功能测试套件
覆盖所有核心功能：文档问答、周报生成、办公工具、工作流
"""
import os
import sys
import pytest
from pathlib import Path
from datetime import datetime
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main4 import SmartOfficeAssistant
from tools.weekly_report_generator import WeeklyReportGenerator
from tools.office_tools import OfficeToolsManager
from rag.rag_system import DocumentRAG


class TestSmartOfficeAssistantInit:
    """智能办公助手初始化测试"""

    def test_initialization_success(self):
        """测试正常初始化"""
        assistant = SmartOfficeAssistant()
        assert assistant is not None
        assert assistant.api_key is not None
        assert assistant.llm_client is not None
        assert assistant.rag_engine is not None
        assert assistant.office_tools is not None
        assert assistant.report_generator is not None

    def test_conversation_history_empty(self):
        """测试初始对话历史为空"""
        assistant = SmartOfficeAssistant()
        assert len(assistant.conversation_history) == 0

    def test_workflow_built(self):
        """测试工作流已构建"""
        assistant = SmartOfficeAssistant()
        assert assistant.workflow is not None


class TestDocumentUploadAndQuery:
    """文档上传与问答测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    @pytest.fixture
    def test_md_file(self, tmp_path):
        """创建临时测试文件"""
        md_file = tmp_path / "test_doc.md"
        md_file.write_text("""
# 测试文档

## 简介
这是一个用于测试的 Markdown文档。

## 主要内容
HIC-YOLOv5 是一种改进的目标检测模型，具有以下特点：
1. 引入了注意力机制
2. 优化了特征金字塔结构
3. 提升了检测精度和速度

## 实验结果
在公开数据集上 mAP 达到 95.6%。
""", encoding='utf-8')
        return str(md_file)

    def test_upload_markdown_document(self, assistant, test_md_file):
        """测试上传 Markdown文档"""
        result = assistant.upload_document(test_md_file)
        assert "✅" in result or "成功" in result

    def test_upload_nonexistent_file(self, assistant):
        """测试上传不存在的文件"""
        result = assistant.upload_document("./nonexistent_file.pdf")
        assert "❌" in result or "不存在" in result

    def test_query_after_upload(self, assistant, test_md_file):
        """测试上传文档后查询"""
        # 先上传文档
        assistant.upload_document(test_md_file)

        # 等待索引建立
        time.sleep(1)

        # 查询相关内容
        response = assistant.run("HIC-YOLOv5 有什么特点？")
        assert response is not None
        assert len(response) > 10

    def test_query_empty_database(self, assistant):
        """测试空数据库查询"""
        # 确保数据库为空（新建的 chroma_db）
        response = assistant.run("测试问题")
        assert response is not None
        # 应该提示数据库中没有文档或进入其他处理流程


class TestWeeklyReportGeneration:
    """周报生成测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    @pytest.fixture
    def report_generator(self):
        """创建周报生成器实例"""
        return WeeklyReportGenerator()

    def test_simple_weekly_report(self, assistant):
        """测试简单周报生成"""
        simple_input = "本周完成了项目开发"
        result = assistant.generate_weekly_report(
            user_input=simple_input,
            template="simple",
            export_formats=['markdown']
        )

        assert result['success'] is True
        assert 'report_content' in result
        assert len(result['report_content']) > 50
        assert 'exported_files' in result
        assert len(result['exported_files']) > 0

    def test_professional_weekly_report(self, assistant):
        """测试专业版周报生成"""
        detailed_input = """
        工作内容：
        1. 完成智能办公助手开发
        2. 集成 RAG 文档问答
        3. 实现周报自动生成

        成果：系统性能提升 50%

        下周计划：
        1. 优化用户体验
        2. 编写技术文档
        """

        result = assistant.generate_weekly_report(
            user_input=detailed_input,
            template="professional",
            export_formats=['markdown', 'word']
        )

        assert result['success'] is True
        assert 'report_content' in result
        # 检查是否包含关键部分
        content = result['report_content']
        assert any(kw in content for kw in ['工作总结', '工作成果', '工作计划', '本周', '下周'])

        # 检查导出文件
        assert len(result['exported_files']) >= 2
        formats = [f['format'] for f in result['exported_files']]
        assert 'markdown' in formats
        assert 'word' in formats

    def test_detailed_weekly_report(self, assistant):
        """测试详细版周报生成"""
        input_text = "负责项目管理，协调 3 个团队协作"

        result = assistant.generate_weekly_report(
            user_input=input_text,
            template="detailed",
            export_formats=['markdown']
        )

        assert result['success'] is True
        assert len(result['report_content']) > 200  # 详细版应该更长

    def test_weekly_report_via_workflow(self, assistant):
        """测试通过工作流生成周报"""
        query = "帮我写一份周报，本周完成了系统开发"
        response = assistant.run(query)

        assert response is not None
        assert len(response) > 50
        # 应该包含周报相关内容
        assert any(kw in response.lower() for kw in ['周报', '总结', '计划', '工作'])

    def test_weekly_report_export_paths(self, assistant):
        """测试周报导出路径正确性"""
        result = assistant.generate_weekly_report(
            user_input="测试周报导出",
            template="simple",
            export_formats=['markdown']
        )

        assert result['success'] is True
        for file_info in result['exported_files']:
            assert 'path' in file_info
            assert os.path.exists(file_info['path'])
            assert file_info['path'].endswith('.md')


class TestOfficeTools:
    """办公工具测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    def test_add_todo_item(self, assistant):
        """测试添加待办事项"""
        query = "添加一个待办事项：下午 3 点参加项目会议"
        response = assistant.run(query)

        assert response is not None
        assert len(response) > 10
        # 应该包含成功添加的提示
        assert any(kw in response for kw in ['✅', '已添加', '成功', '待办'])

    def test_list_todos(self, assistant):
        """测试查看待办清单"""
        # 先添加一些待办
        assistant.run("添加一个待办：完成任务 A")
        assistant.run("添加一个待办：完成任务 B")

        # 查看清单
        response = assistant.run("查看我的待办清单")
        assert response is not None
        assert len(response) > 10

    def test_calendar_query(self, assistant):
        """测试日历查询"""
        query = "查询今天的日程安排"
        response = assistant.run(query)

        assert response is not None
        # 应该返回日历信息（即使为空）
        assert isinstance(response, str)
        assert len(response) > 0

    def test_search_news(self, assistant):
        """测试搜索新闻"""
        query = "搜索今天的人工智能新闻"
        response = assistant.run(query)

        assert response is not None
        # 搜索可能成功或失败（取决于 API 配置）
        assert isinstance(response, str)
        assert len(response) > 0

    def test_weather_query(self, assistant):
        """测试天气查询"""
        query = "明天广州天气怎么样"
        response = assistant.run(query)

        assert response is not None
        assert isinstance(response, str)

    def test_complete_todo(self, assistant):
        """测试完成待办标记"""
        # 先添加待办
        add_response = assistant.run("添加一个待办：测试任务")
        assert "✅" in add_response or "成功" in add_response

        # 标记完成
        complete_response = assistant.run("标记第 1 个待办为已完成")
        assert complete_response is not None


class TestWorkflowIntegration:
    """工作流整合测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    def test_rag_query_routing(self, assistant):
        """测试 RAG 查询路由"""
        queries = [
            "HIC-YOLOv5 模型的特点",
            "这篇论文的主要内容",
            "文档中提到了什么技术"
        ]

        for query in queries:
            response = assistant.run(query)
            assert response is not None
            assert len(response) > 0

    def test_report_generation_routing(self, assistant):
        """测试周报生成路由"""
        queries = [
            "帮我写周报",
            "生成一份周报",
            "本周工作总结"
        ]

        for query in queries:
            response = assistant.run(query)
            assert response is not None
            # 应该触发周报生成或相关回复
            assert len(response) > 10

    def test_office_tool_routing(self, assistant):
        """测试办公工具路由"""
        queries = [
            "添加待办：开会",
            "查询日程",
            "搜索新闻"
        ]

        for query in queries:
            response = assistant.run(query)
            assert response is not None
            assert len(response) > 0

    def test_general_chat_routing(self, assistant):
        """测试通用聊天路由"""
        queries = [
            "你好",
            "你是谁",
            "你能做什么"
        ]

        for query in queries:
            response = assistant.run(query)
            assert response is not None
            assert len(response) > 0

    def test_conversation_history_maintained(self, assistant):
        """测试对话历史维护"""
        # 进行多轮对话
        assistant.run("你好")
        assistant.run("你能做什么")
        assistant.run("介绍一下你自己")

        # 检查历史记录
        stats = assistant.get_statistics()
        assert stats['conversation_turns'] >= 3
        assert len(assistant.conversation_history) >= 6  # 每轮对话包含用户和助手


class TestStatisticsAndMonitoring:
    """统计与监控测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    def test_get_statistics(self, assistant):
        """测试获取统计信息"""
        stats = assistant.get_statistics()

        assert 'conversation_turns' in stats
        assert 'rag_docs_count' in stats
        assert 'office_tools_stats' in stats

        assert isinstance(stats['conversation_turns'], int)
        assert isinstance(stats['rag_docs_count'], int)

    def test_clear_history(self, assistant):
        """测试清空对话历史"""
        # 添加一些对话
        assistant.run("测试 1")
        assistant.run("测试 2")

        # 清空历史
        assistant.clear_history()

        # 验证已清空
        stats = assistant.get_statistics()
        assert stats['conversation_turns'] == 0
        assert len(assistant.conversation_history) == 0


class TestPerformanceAndStability:
    """性能与稳定性测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    def test_response_time_basic(self, assistant):
        """测试基本响应时间"""
        start_time = time.time()
        response = assistant.run("你好")
        elapsed = time.time() - start_time

        # 响应时间应该在合理范围内（10 秒内）
        assert elapsed < 10
        assert response is not None

    def test_concurrent_requests_simulation(self, assistant):
        """模拟并发请求（顺序执行）"""
        queries = ["测试 1", "测试 2", "测试 3", "测试 4", "测试 5"]

        start_time = time.time()
        for query in queries:
            response = assistant.run(query)
            assert response is not None

        total_time = time.time() - start_time
        # 5 个请求总时间应该在合理范围内（60 秒内）
        assert total_time < 60

    def test_memory_leak_check(self, assistant):
        """检查内存泄漏（简化版）"""
        initial_history_size = len(assistant.conversation_history)

        # 执行多次对话
        for i in range(10):
            assistant.run(f"测试对话 {i}")

        # 历史记录应该被限制在一定范围内
        assert len(assistant.conversation_history) <= 20  # 最多 20 条

    def test_error_recovery(self, assistant):
        """测试错误恢复能力"""
        # 发送可能导致错误的输入
        invalid_queries = [
            "",  # 空输入（可能被过滤）
            "   ",  # 空格
            "<script>alert('xss')</script>",  # 非法字符
        ]

        for query in invalid_queries:
            try:
                response = assistant.run(query)
                # 应该有某种响应
                assert response is not None
            except Exception as e:
                # 如果抛出异常，应该是可处理的
                pytest.fail(f"未处理的异常：{e}")


class TestEdgeCasesAndExceptions:
    """边界情况与异常测试"""

    @pytest.fixture
    def assistant(self):
        """创建助手实例"""
        return SmartOfficeAssistant()

    def test_very_long_input(self, assistant):
        """测试超长输入"""
        long_input = "测试内容 " * 1000
        response = assistant.run(long_input)
        assert response is not None

    def test_special_characters_input(self, assistant):
        """测试特殊字符输入"""
        special_input = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        response = assistant.run(special_input)
        assert response is not None

    def test_mixed_language_input(self, assistant):
        """测试混合语言输入"""
        mixed_input = "Hello 你好こんにちは안녕하세요"
        response = assistant.run(mixed_input)
        assert response is not None

    def test_code_injection_attempt(self, assistant):
        """测试代码注入尝试"""
        injection_input = "'); DROP TABLE users; --"
        response = assistant.run(injection_input)
        assert response is not None
        # 系统应该安全处理，不执行恶意代码

    def test_html_tags_input(self, assistant):
        """测试 HTML 标签输入"""
        html_input = "<div>测试</div><script>alert('xss')</script>"
        response = assistant.run(html_input)
        assert response is not None

    def test_repeated_same_input(self, assistant):
        """测试重复相同输入"""
        query = "测试重复输入"
        responses = []

        for _ in range(3):
            response = assistant.run(query)
            responses.append(response)

        # 所有响应都应该有效
        for resp in responses:
            assert resp is not None
            assert len(resp) > 0

    def test_rapid_sequential_requests(self, assistant):
        """测试快速连续请求"""
        queries = [f"快速测试{i}" for i in range(5)]

        for query in queries:
            response = assistant.run(query)
            assert response is not None


def run_all_tests():
    """运行所有测试的辅助函数"""
    print("\n" + "=" * 70)
    print("智能办公助手 V1.0 - 综合测试套件")
    print("=" * 70)

    # 使用 pytest 运行
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-q"
    ])


if __name__ == "__main__":
    run_all_tests()
