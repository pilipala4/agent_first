# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0 - 高级边界测试
覆盖 10+ 边界场景，确保系统健壮性
"""
import os
import sys
import pytest
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main4 import SmartOfficeAssistant
from tools.weekly_report_generator import WeeklyReportGenerator
from tools.office_tools import OfficeToolsManager
from rag.rag_system import DocumentRAG


class TestInputBoundaryCases:
    """输入边界测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_minimum_length_input(self, assistant):
        """测试最小长度输入（1 个字符）"""
        response = assistant.run("好")
        assert response is not None
        assert len(response) > 0

    def test_maximum_length_input(self, assistant):
        """测试接近最大长度的输入"""
        # 生成接近但不超过限制的输入
        long_input = "测试 " * 300  # 约 900 字符
        response = assistant.run(long_input)
        assert response is not None

    def test_unicode_characters(self, assistant):
        """测试 Unicode 字符"""
        unicode_input = "😀😁😂🤣😃 表情符号测试 🎉🎊🎈"
        response = assistant.run(unicode_input)
        assert response is not None

    def test_numeric_only_input(self, assistant):
        """测试纯数字输入"""
        numeric_input = "1234567890"
        response = assistant.run(numeric_input)
        assert response is not None

    def test_whitespace_variations(self, assistant):
        """测试各种空白字符"""
        whitespace_inputs = [
            "  前后有空格  ",
            "\t制表符\t",
            "\n换行符\n",
            "  \t\n  混合空白  \n\t  "
        ]

        for ws_input in whitespace_inputs:
            response = assistant.run(ws_input)
            assert response is not None

    def test_sql_injection_patterns(self, assistant):
        """测试 SQL 注入模式"""
        sql_patterns = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "SELECT * FROM users WHERE 1=1",
            "UNION SELECT password FROM users"
        ]

        for pattern in sql_patterns:
            response = assistant.run(pattern)
            assert response is not None
            # 系统应该安全处理，不执行 SQL

    def test_xss_attack_patterns(self, assistant):
        """测试 XSS 攻击模式"""
        xss_patterns = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]

        for pattern in xss_patterns:
            response = assistant.run(pattern)
            assert response is not None
            # 系统应该安全处理，不执行脚本


class TestDocumentBoundaryCases:
    """文档处理边界测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    @pytest.fixture
    def empty_md_file(self, tmp_path):
        """创建空 Markdown 文件"""
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding='utf-8')
        return str(md_file)

    @pytest.fixture
    def large_md_file(self, tmp_path):
        """创建大 Markdown 文件"""
        md_file = tmp_path / "large.md"
        # 写入大量内容
        large_content = "# 大文档\n\n" + "这是一段测试内容。\n" * 10000
        md_file.write_text(large_content, encoding='utf-8')
        return str(md_file)

    def test_upload_empty_document(self, assistant, empty_md_file):
        """测试上传空文档"""
        result = assistant.upload_document(empty_md_file)
        # 应该能处理空文档（可能警告但不应崩溃）
        assert result is not None

    def test_upload_large_document(self, assistant, large_md_file):
        """测试上传大文档"""
        result = assistant.upload_document(large_md_file)
        # 应该能处理大文档
        assert result is not None

    def test_query_very_short(self, assistant, empty_md_file):
        """测试极短查询"""
        assistant.upload_document(empty_md_file)
        time.sleep(1)

        response = assistant.run("？")
        assert response is not None

    def test_upload_invalid_path(self, assistant):
        """测试上传无效路径"""
        invalid_paths = [
            "",
            None,
            "/nonexistent/path/file.pdf",
            "C:\\Windows\\System32\\config\\SAM"  # Windows 系统文件
        ]

        for path in invalid_paths:
            if path is not None:
                result = assistant.upload_document(path)
                assert "❌" in result or "不" in result


class TestWeeklyReportBoundaryCases:
    """周报生成边界测试"""

    @pytest.fixture
    def report_generator(self):
        return WeeklyReportGenerator()

    def test_empty_input(self, report_generator):
        """测试空输入生成周报"""
        result = report_generator.generate_and_export(
            user_input="",
            template="simple"
        )
        # 应该能处理空输入（可能降级处理）
        assert result is not None

    def test_minimal_input(self, report_generator):
        """测试最小输入（几个字）"""
        result = report_generator.generate_and_export(
            user_input="工作",
            template="simple"
        )
        assert result is not None

    def test_very_detailed_input(self, report_generator):
        """测试非常详细的输入"""
        detailed_input = "工作内容：" + "\n".join([f"任务{i}" for i in range(50)])
        detailed_input += "\n成果：" + "\n".join([f"成果{i}" for i in range(20)])

        result = report_generator.generate_and_export(
            user_input=detailed_input,
            template="detailed"
        )
        assert result is not None
        assert result['success'] is True

    def test_special_characters_in_input(self, report_generator):
        """测试输入包含特殊字符"""
        special_input = "工作@#$%^&*() 内容！@#￥%……&*（）"
        result = report_generator.generate_and_export(
            user_input=special_input,
            template="simple"
        )
        assert result is not None

    def test_multiple_templates_sequential(self, report_generator):
        """测试连续使用多个模板"""
        base_input = "本周完成工作"

        templates = ["simple", "professional", "detailed"]
        results = []

        for template in templates:
            result = report_generator.generate_and_export(
                user_input=base_input,
                template=template
            )
            results.append(result)
            assert result is not None

    def test_export_format_combinations(self, report_generator):
        """测试各种导出格式组合"""
        format_combinations = [
            ['markdown'],
            ['word'],
            ['markdown', 'word']
        ]

        for formats in format_combinations:
            result = report_generator.generate_and_export(
                user_input="测试导出",
                template="simple",
                export_formats=formats
            )
            assert result is not None
            if result['success']:
                assert len(result['exported_files']) == len(formats)


class TestOfficeToolsBoundaryCases:
    """办公工具边界测试"""

    @pytest.fixture
    def office_tools(self):
        return OfficeToolsManager()

    def test_add_todo_empty_task(self, office_tools):
        """测试添加空任务"""
        result = office_tools.execute_office_tool("添加一个待办")
        assert result is not None

    def test_add_todo_very_long_task(self, office_tools):
        """测试添加超长任务"""
        long_task = "任务 " * 100
        result = office_tools.execute_office_tool(f"添加待办：{long_task}")
        assert result is not None

    def test_query_nonexistent_date(self, office_tools):
        """测试查询不存在的日期"""
        result = office_tools.execute_office_tool("查询 2099 年 1 月 1 日的日程")
        assert result is not None

    def test_complete_nonexistent_todo(self, office_tools):
        """测试完成不存在的待办"""
        result = office_tools.execute_office_tool("标记第 9999 个待办为已完成")
        assert result is not None

    def test_search_empty_query(self, office_tools):
        """测试空搜索"""
        result = office_tools.execute_office_tool("搜索")
        assert result is not None

    def test_rapid_todo_operations(self, office_tools):
        """测试快速待办操作"""
        # 连续添加多个待办
        for i in range(10):
            result = office_tools.execute_office_tool(f"添加待办：任务{i}")
            assert result is not None

        # 查看清单
        result = office_tools.execute_office_tool("查看待办清单")
        assert result is not None


class TestConcurrencyAndTiming:
    """并发与时序边界测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_rapid_fire_questions(self, assistant):
        """测试快速连续提问"""
        questions = [f"问题{i}" for i in range(20)]

        responses = []
        for q in questions:
            response = assistant.run(q)
            responses.append(response)
            assert response is not None

    def test_conversation_history_limit(self, assistant):
        """测试对话历史限制"""
        # 进行超过限制的对话
        for i in range(30):
            assistant.run(f"对话{i}")

        # 检查历史记录是否被限制
        assert len(assistant.conversation_history) <= 20

    def test_state_persistence(self, assistant):
        """测试状态持久性"""
        # 添加待办
        assistant.run("添加待办：测试持久化")

        # 立即查询
        response1 = assistant.run("查看待办清单")
        assert response1 is not None

        # 等待一小段时间
        time.sleep(2)

        # 再次查询
        response2 = assistant.run("查看待办清单")
        assert response2 is not None

    def test_workflow_timeout_handling(self, assistant):
        """测试工作流超时处理"""
        # 发送可能导致超时的复杂查询
        complex_query = "请详细分析一下这个复杂的问题：" + "？" * 100
        response = assistant.run(complex_query)
        assert response is not None


class TestResourceManagement:
    """资源管理边界测试"""

    @pytest.fixture
    def rag_system(self):
        return DocumentRAG()

    def test_database_size_monitoring(self, rag_system):
        """测试数据库大小监控"""
        initial_count = rag_system.collection.count()

        # 添加文档
        test_md = "./test_resource.md"
        with open(test_md, "w", encoding='utf-8') as f:
            f.write("# 测试\n\n内容")

        rag_system.add_document(test_md)
        os.remove(test_md)

        new_count = rag_system.collection.count()
        assert new_count > initial_count

    def test_clear_database(self, rag_system):
        """测试清空数据库"""
        # 先添加一些文档
        test_md = "./test_clear.md"
        with open(test_md, "w", encoding='utf-8') as f:
            f.write("# 测试\n\n内容")

        rag_system.add_document(test_md)
        os.remove(test_md)

        # 清空数据库
        rag_system.clear_database()

        # 验证已清空
        count = rag_system.collection.count()
        assert count == 0


def run_boundary_tests():
    """运行边界测试"""
    print("\n" + "=" * 70)
    print("智能办公助手 V1.0 - 边界测试套件")
    print("=" * 70)

    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-q"
    ])


if __name__ == "__main__":
    run_boundary_tests()
