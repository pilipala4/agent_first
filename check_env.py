# -*- coding: utf-8 -*-
"""
环境检查脚本 - 验证智能办公助手的运行环境
"""
import sys
import os
from pathlib import Path


def print_check(title: str, status: bool, message: str = ""):
    """打印检查结果"""
    icon = "✅" if status else "❌"
    print(f"{icon} {title}: {message}")
    return status


def check_python_version():
    """检查 Python 版本"""
    version = sys.version_info
    required = (3, 8)
    status = version >= required
    print_check(
        "Python 版本",
        status,
        f"{version.major}.{version.minor}.{version.micro} (要求：{'.'.join(map(str, required))}+)"
    )
    return status


def check_env_variables():
    """检查环境变量"""
    dashscope_key = os.getenv('DASHSCOPE_API_KEY')
    serpapi_key = os.getenv('SERPAPI_API_KEY')

    status1 = print_check(
        "DASHSCOPE_API_KEY",
        dashscope_key is not None,
        "已设置" if dashscope_key else "未设置（必需）"
    )

    status2 = print_check(
        "SERPAPI_API_KEY",
        serpapi_key is not None,
        "已设置" if serpapi_key else "未设置（可选，用于实时搜索）"
    )

    return status1 and status2


def check_dotenv_file():
    """检查.env 文件"""
    dotenv_path = Path("./.env")
    exists = dotenv_path.exists()

    if exists:
        content = dotenv_path.read_text(encoding='utf-8')
        has_dashscope = 'DASHSCOPE_API_KEY' in content
        has_serpapi = 'SERPAPI_API_KEY' in content

        print_check(".env 文件", True, "存在")
        print_check("  - DASHSCOPE_API_KEY", has_dashscope, "已配置" if has_dashscope else "未配置")
        print_check("  - SERPAPI_API_KEY", has_serpapi, "已配置" if has_serpapi else "未配置（可选）")
        return has_dashscope
    else:
        print_check(".env 文件", False, "不存在，请创建该文件")
        return False


def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'openai': 'OpenAI SDK',
        'langgraph': 'LangGraph 工作流',
        'chromadb': '向量数据库',
        'PyPDF2': 'PDF 处理',
        'docx': 'Word 处理 (python-docx)',
        'markdown': 'Markdown 处理',
        'pydantic': '数据验证',
        'requests': 'HTTP 请求',
        'google_search_results': 'SerpAPI 搜索',
    }

    all_ok = True
    print("\n依赖包检查:")

    for package, description in required_packages.items():
        try:
            __import__(package)
            print_check(f"  {description} ({package})", True, "✓")
        except ImportError:
            print_check(f"  {description} ({package})", False, "缺失")
            all_ok = False

    return all_ok


def check_directories():
    """检查必要目录"""
    dirs = {
        './rag': 'RAG 模块',
        './tools': '工具模块',
        './test': '测试模块',
    }

    print("\n目录检查:")
    all_ok = True

    for dir_path, description in dirs.items():
        exists = Path(dir_path).exists()
        print_check(description, exists, dir_path if exists else "缺失")
        if not exists:
            all_ok = False

    return all_ok


def check_main_files():
    """检查主要文件"""
    files = {
        './main.py': '主入口文件',
        './llm_call.py': 'LLM 调用模块',
        './logger.py': '日志模块',
        './requirements.txt': '依赖列表',
        './rag/rag_system.py': 'RAG 系统',
        './tools/office_tools.py': '办公工具',
        './tools/weekly_report_generator.py': '周报生成器',
    }

    print("\n文件检查:")
    all_ok = True

    for file_path, description in files.items():
        exists = Path(file_path).exists()
        print_check(description, exists, file_path if exists else "缺失")
        if not exists:
            all_ok = False

    return all_ok


def check_test_files():
    """检查测试文件"""
    test_files = {
        './test/test_all_features.py': '全功能测试',
        './test/test_weekly_report.py': '周报测试',
    }

    print("\n测试文件检查:")

    for file_path, description in test_files.items():
        exists = Path(file_path).exists()
        print_check(description, exists, file_path if exists else "缺失")

    return True


def show_summary(checks: list):
    """显示总结"""
    print("\n" + "=" * 70)
    print("检查总结")
    print("=" * 70)

    all_passed = all(checks)

    if all_passed:
        print("✅ 所有检查通过！系统已准备就绪。\n")
        print("下一步:")
        print("  1. 确保已配置 API 密钥")
        print("  2. 运行：python main.py")
    else:
        print("❌ 部分检查未通过，请根据提示修复。\n")
        print("修复建议:")
        print("  1. 安装缺失的依赖：pip install -r requirements.txt")
        print("  2. 创建.env 文件并配置 API 密钥")
        print("  3. 确保所有必要文件存在")

    print("=" * 70)

    return all_passed


def main():
    """主函数"""
    print("=" * 70)
    print("智能办公助手 V1.0 - 环境检查")
    print("=" * 70)
    print()

    checks = []

    # 执行检查
    checks.append(check_python_version())
    checks.append(check_env_variables())
    checks.append(check_dotenv_file())
    checks.append(check_dependencies())
    checks.append(check_directories())
    checks.append(check_main_files())
    checks.append(check_test_files())

    # 显示总结
    show_summary(checks)


if __name__ == "__main__":
    main()
