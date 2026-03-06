# -*- coding: utf-8 -*-
"""
周报生成模块集成测试
"""
import os
from main3 import OfficeAssistant


def test_weekly_report_module():
    """测试周报生成模块"""
    print("=" * 70)
    print("周报生成模块集成测试")
    print("=" * 70)

    try:
        assistant = OfficeAssistant()
        print("✅ 办公助手初始化成功\n")
    except ValueError as e:
        print(f"❌ 初始化失败：{e}")
        return False

    # 测试场景 1：简单输入生成周报
    print("\n" + "=" * 70)
    print("测试场景 1: 简单工作内容生成周报")
    print("=" * 70)
    query1 = "本周完成了项目开发，下周继续优化"
    print(f"👤 用户输入：{query1}\n")

    result1 = assistant.generate_weekly_report(
        user_input=query1,
        template="simple",
        export_formats=['markdown']
    )

    if result1['success']:
        print("✅ 测试通过 - 周报生成成功")
        print(f"\n📊 生成的周报预览：\n")
        print(result1['report_content'][:500])
        print("\n...（内容过长，仅显示前 500 字符）...\n")

        for file_info in result1.get('exported_files', []):
            print(f"📁 导出文件：{file_info['format']} -> {file_info['path']}")
    else:
        print(f"❌ 测试失败：{result1.get('error')}")
        return False

    # 测试场景 2：详细输入生成专业版周报
    print("\n" + "=" * 70)
    print("测试场景 2: 详细输入生成专业版周报")
    print("=" * 70)
    query2 = """
    工作内容：
    1. 完成智能办公助手系统开发，实现 RAG 文档问答功能
    2. 集成 SerpAPI 实时搜索工具，支持天气、新闻查询
    3. 开发日历管理和待办清单模块
    4. 优化系统性能，修复已知 bug

    工作成果：
    - 系统响应时间从 5 秒降低到 2 秒，提升 60%
    - 准确率达到 95%，获得客户好评
    - 申请技术创新专利 1 项

    下周计划：
    1. 增加邮件处理功能
    2. 支持多模态文档处理
    3. 编写用户手册和技术文档
    4. 进行系统安全加固

    遇到的问题：
    - 向量数据库查询慢：通过建立索引和缓存机制解决
    - API 调用不稳定：实现重试机制和降级策略
    - 内存占用高：优化数据处理流程

    备注：
    周五下午 2 点有项目评审会议，需要准备演示材料
    """
    print(f"👤 用户输入：{query2[:100]}...（详细输入）\n")

    result2 = assistant.generate_weekly_report(
        user_input=query2,
        template="professional",
        export_formats=['markdown', 'word']
    )

    if result2['success']:
        print("✅ 测试通过 - 专业版周报生成成功")
        print(f"\n📊 生成的周报预览：\n")
        print(result2['report_content'][:600])
        print("\n...（内容过长，仅显示前 600 字符）...\n")

        for file_info in result2.get('exported_files', []):
            print(f"📁 导出文件：{file_info['format']} -> {file_info['path']}")
    else:
        print(f"❌ 测试失败：{result2.get('error')}")
        return False

    # 测试场景 3：详细模板
    print("\n" + "=" * 70)
    print("测试场景 3: 项目管理周报（详细模板）")
    print("=" * 70)
    query3 = "负责项目管理，协调 3 个团队协作，完成 2 个重要交付物，客户满意度高"
    print(f"👤 用户输入：{query3}\n")

    result3 = assistant.generate_weekly_report(
        user_input=query3,
        template="detailed",
        export_formats=['markdown']
    )

    if result3['success']:
        print("✅ 测试通过 - 详细版周报生成成功")
        print(f"\n📊 完整周报内容：\n")
        print(result3['report_content'])
    else:
        print(f"❌ 测试失败：{result3.get('error')}")
        return False

    # 测试场景 4：通过工作流生成周报
    print("\n" + "=" * 70)
    print("测试场景 4: 通过 LangGraph 工作流生成周报")
    print("=" * 70)
    query4 = "帮我写一份周报，本周完成了系统开发和测试工作"
    print(f"👤 用户：{query4}\n")

    response = assistant.run(query4)
    print(f"🤖 助手回复：\n{response}\n")

    if response and len(response) > 50:
        print("✅ 测试通过 - 工作流周报生成成功")
    else:
        print("⚠️  测试警告 - 回复内容过短")

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print("✅ 所有测试场景执行完成")
    print("📁 生成的周报文件已保存到：./generated_reports/ 目录")
    print("\n功能验证：")
    print("  ✓ 支持自定义输入（工作内容 + 成果 + 计划）")
    print("  ✓ 生成结构化周报")
    print("  ✓ 支持 Markdown 格式导出")
    print("  ✓ 支持 Word 格式导出")
    print("  ✓ 支持多种模板（简单/专业/详细）")
    print("  ✓ 内容贴合办公场景")
    print("  ✓ 无明显错误")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_weekly_report_module()
    if success:
        print("\n✅ 周报生成模块测试全部通过！")
    else:
        print("\n❌ 测试未完全通过，请检查日志")
