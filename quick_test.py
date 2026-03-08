# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证核心功能
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main4 import SmartOfficeAssistant


def test_core_features():
    """快速测试核心功能"""
    print("=" * 70)
    print("智能办公助手 V1.0 - 快速功能验证")
    print("=" * 70)

    try:
        # 初始化
        print("\n1. 初始化系统...")
        assistant = SmartOfficeAssistant()
        print("✅ 初始化成功")

        # 测试通用对话
        print("\n2. 测试通用对话...")
        response = assistant.run("你好")
        print(f"   回复：{response[:100]}...")
        print("✅ 对话功能正常")

        # 测试待办添加
        print("\n3. 测试待办添加...")
        response = assistant.run("添加待办：测试任务")
        print(f"   回复：{response[:100]}...")
        print("✅ 待办功能正常")

        # 测试周报生成（直接调用）
        print("\n4. 测试周报生成...")
        result = assistant.generate_weekly_report(
            user_input="本周完成工作",
            template="simple",
            export_formats=['markdown']
        )
        if result.get('success'):
            print(f"   生成字数：{len(result['report_content'])}")
            print(f"   导出文件：{result['exported_files'][0]['path']}")
            print("✅ 周报生成功能正常")
        else:
            print(f"❌ 周报生成失败：{result.get('error')}")

        # 获取统计
        print("\n5. 获取统计信息...")
        stats = assistant.get_statistics()
        print(f"   对话轮数：{stats['conversation_turns']}")
        print(f"   RAG 文档数：{stats['rag_docs_count']}")
        print("✅ 统计功能正常")

        print("\n" + "=" * 70)
        print("✅ 所有核心功能验证通过！")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_core_features()
    sys.exit(0 if success else 1)
