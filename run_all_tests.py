# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0 - 一键运行所有测试
自动执行完整测试套件并生成报告
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime


def print_banner(text):
    """打印横幅"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70 + "\n")


def check_environment():
    """检查测试环境"""
    print_banner("步骤 1: 环境检查")

    # 检查 Python 版本
    import sys
    print(f"Python 版本：{sys.version}")

    # 检查必要依赖
    required_packages = ['pytest', 'pytest_cov']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            print(f"   请运行：pip install {package}")
            return False

    # 检查环境变量
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("⚠️  警告：DASHSCOPE_API_KEY 未设置")
        print("   某些测试可能需要 API 密钥")
    else:
        print("✅ DASHSCOPE_API_KEY 已配置")

    print()
    return True


def run_comprehensive_tests():
    """运行综合功能测试"""
    print_banner("步骤 2: 运行综合功能测试")

    cmd = [
        "pytest",
        "test/test_comprehensive.py",
        "-v",
        "--tb=short",
        "-s"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_boundary_tests():
    """运行边界测试"""
    print_banner("步骤 3: 运行边界测试")

    cmd = [
        "pytest",
        "test/test_edge_cases_advanced.py",
        "-v",
        "--tb=short",
        "-s"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def run_performance_tests():
    """运行性能测试"""
    print_banner("步骤 4: 运行性能测试")

    cmd = [
        "pytest",
        "test/test_performance.py",
        "-v",
        "--tb=short",
        "-s"
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def generate_report():
    """生成测试报告"""
    print_banner("步骤 5: 生成测试报告")

    from test.generate_test_report import main as generate_report_main
    generate_report_main()


def main():
    """主函数"""
    print_banner("智能办公助手 V1.0 - 自动化测试套件")

    start_time = datetime.now()

    # 步骤 1: 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，无法继续测试")
        return 1

    # 步骤 2-4: 运行各类测试
    test_results = {
        '综合功能测试': run_comprehensive_tests(),
        '边界测试': run_boundary_tests(),
        '性能测试': run_performance_tests(),
    }

    # 步骤 5: 生成报告
    generate_report()

    # 总结
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_banner("测试执行总结")

    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {test_name}")

    print(f"\n总耗时：{duration:.2f}秒")
    print(f"通过率：{passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n⚠️  部分测试失败，请查看报告")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
