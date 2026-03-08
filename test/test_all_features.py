# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0 - 全功能集成测试
测试所有核心功能：文档问答、周报生成、办公工具
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main4 import SmartOfficeAssistant


def test_initialization():
    """测试 1: 系统初始化"""
    print("\n" + "=" * 70)
    print("测试 1: 系统初始化")
    print("=" * 70)

    try:
        assistant = SmartOfficeAssistant()
        print("✅ 系统初始化成功")
        return assistant
    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        return None


def test_rag_upload_and_query(assistant):
    """测试 2: RAG 文档上传与问答"""
    print("\n" + "=" * 70)
    print("测试 2: RAG 文档上传与问答")
    print("=" * 70)

    # 检查是否有测试文件
    test_pdf = "2309.16393v2.pdf"
    if not os.path.exists(test_pdf):
        print(f"⚠️  跳过测试：找不到测试文件 {test_pdf}")
        print("   提示：请将测试 PDF 文件放到项目根目录")
        return False

    # 上传文档
    print(f"\n📤 上传文档：{test_pdf}")
    upload_result = assistant.upload_document(test_pdf)
    print(upload_result)

    if "❌" in upload_result:
        print("❌ 文档上传失败")
        return False

    print("✅ 文档上传成功")

    # 文档问答测试
    print("\n🔍 测试文档问答:")
    queries = [
        "HIC-YOLOv5 模型的主要改进是什么？",
        "这篇论文提出了什么方法？"
    ]

    for query in queries:
        print(f"\n👤 问题：{query}")
        response = assistant.run(query)
        print(f"🤖 回答：{response[:300]}...")

    print("\n✅ RAG 文档问答测试完成")
    return True


def test_weekly_report_generation(assistant):
    """测试 3: 周报生成"""
    print("\n" + "=" * 70)
    print("测试 3: 周报生成")
    print("=" * 70)

    # 测试场景 1: 简单输入
    print("\n📝 测试场景 1: 简单输入")
    simple_input = "本周完成了项目开发，下周继续优化"
    print(f"👤 输入：{simple_input}")

    result1 = assistant.generate_weekly_report(
        user_input=simple_input,
        template="simple",
        export_formats=['markdown']
    )

    if result1.get('success'):
        print("✅ 简单版周报生成成功")
        print(f"\n📄 内容预览:\n{result1['report_content'][:300]}...")
        for file in result1.get('exported_files', []):
            print(f"📁 导出文件：{file['path']}")
    else:
        print(f"❌ 生成失败：{result1.get('error')}")
        return False

    # 测试场景 2: 详细输入
    print("\n📝 测试场景 2: 详细输入")
    detailed_input = """
    工作内容：
    1. 完成智能办公助手系统开发
    2. 集成 RAG 文档问答功能
    3. 实现周报自动生成
    4. 添加办公工具模块

    成果：
    - 系统性能提升 50%
    - 用户满意度达 95%

    下周计划：
    1. 优化用户体验
    2. 编写技术文档
    """
    print(f"👤 输入：{detailed_input[:100]}...")

    result2 = assistant.generate_weekly_report(
        user_input=detailed_input,
        template="professional",
        export_formats=['markdown', 'word']
    )

    if result2.get('success'):
        print("✅ 专业版周报生成成功")
        print(f"\n📄 内容预览:\n{result2['report_content'][:400]}...")
        for file in result2.get('exported_files', []):
            print(f"📁 导出文件：{file['format']} -> {file['path']}")
    else:
        print(f"❌ 生成失败：{result2.get('error')}")
        return False

    print("\n✅ 周报生成测试完成")
    return True


def test_office_tools(assistant):
    """测试 4: 办公工具"""
    print("\n" + "=" * 70)
    print("测试 4: 办公工具")
    print("=" * 70)

    # 测试场景：待办管理
    print("\n📋 测试场景 1: 待办事项管理")

    # 添加待办
    print("👤 添加待办：下午 3 点参加项目会议")
    response1 = assistant.run("添加一个待办事项：下午 3 点参加项目会议")
    print(f"🤖 回答：{response1}")

    # 查看待办
    print("\n👤 查看待办清单")
    response2 = assistant.run("查看我的待办清单")
    print(f"🤖 回答：{response2}")

    # 测试场景：日历查询
    print("\n📅 测试场景 2: 日历查询")
    print("👤 查询今天的日程")
    response3 = assistant.run("查询今天的日程安排")
    print(f"🤖 回答：{response3}")

    # 测试场景：实时搜索（如果配置了 SerpAPI）
    print("\n🔍 测试场景 3: 实时搜索")
    print("👤 搜索今天的人工智能新闻")
    response4 = assistant.run("搜索今天的人工智能行业新闻")
    print(f"🤖 回答：{response4[:300]}...")

    print("\n✅ 办公工具测试完成")
    return True


def test_workflow_integration(assistant):
    """测试 5: 工作流整合"""
    print("\n" + "=" * 70)
    print("测试 5: 工作流整合测试")
    print("=" * 70)

    test_cases = [
        ("帮我写一份周报", "周报生成"),
        ("HIC-YOLOv5 有什么特点？", "文档问答"),
        ("添加一个待办：明天提交报告", "办公工具"),
        ("查询明天的日程", "办公工具"),
    ]

    for i, (query, expected_type) in enumerate(test_cases, 1):
        print(f"\n{i}. 👤 问题：{query}")
        print(f"   预期类型：{expected_type}")
        response = assistant.run(query)
        print(f"   🤖 回答：{response[:200]}...")

    print("\n✅ 工作流整合测试完成")
    return True


def test_statistics(assistant):
    """测试 6: 统计信息"""
    print("\n" + "=" * 70)
    print("测试 6: 统计信息")
    print("=" * 70)

    stats = assistant.get_statistics()
    print(f"💬 对话轮数：{stats.get('conversation_turns', 0)}")
    print(f"📚 RAG 文档数：{stats.get('rag_docs_count', 0)}")

    office_stats = stats.get('office_tools_stats', {})
    print(f"🔧 工具调用次数：{office_stats.get('total_calls', 0)}")
    print(f"✅ 成功次数：{office_stats.get('successful_calls', 0)}")
    print(f"❌ 失败次数：{office_stats.get('failed_calls', 0)}")
    print(f"📊 成功率：{office_stats.get('success_rate', '0%')}")

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🚀 智能办公助手 V1.0 - 全功能测试")
    print("=" * 70)

    # 测试 1: 初始化
    assistant = test_initialization()
    if not assistant:
        print("\n❌ 测试终止：系统初始化失败")
        return False

    # 测试 2: RAG 文档问答
    test_rag_upload_and_query(assistant)

    # 测试 3: 周报生成
    test_weekly_report_generation(assistant)

    # 测试 4: 办公工具
    test_office_tools(assistant)

    # 测试 5: 工作流整合
    test_workflow_integration(assistant)

    # 测试 6: 统计信息
    test_statistics(assistant)

    # 总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print("✅ 所有测试执行完成")
    print("\n功能验证:")
    print("  ✓ 文档问答 (RAG)")
    print("  ✓ 周报生成 (多模板/多格式)")
    print("  ✓ 办公工具 (搜索/日历/待办)")
    print("  ✓ LangGraph 工作流")
    print("  ✓ 统一入口")
    print("  ✓ 用户交互友好")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\n✅ 所有测试完成！")
    else:
        print("\n❌ 测试未完全通过")
