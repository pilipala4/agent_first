# -*- coding: utf-8 -*-
"""
智能办公助手 V1.0 - 性能测试
测试系统在各种负载下的性能表现
"""
import os
import sys
import pytest
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main4 import SmartOfficeAssistant


class TestResponseTime:
    """响应时间性能测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_average_response_time_simple(self, assistant):
        """测试简单查询的平均响应时间"""
        query = "你好"
        times = []

        # 执行 10 次测量
        for i in range(10):
            start = time.time()
            assistant.run(query)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        print(f"\n简单查询平均响应时间：{avg_time:.2f}秒 (标准差：{std_dev:.2f})")

        # 平均响应时间应小于 5 秒
        assert avg_time < 5.0

    def test_average_response_time_complex(self, assistant):
        """测试复杂查询的平均响应时间"""
        complex_query = "请详细解释 HIC-YOLOv5 模型的技术原理和创新点"
        times = []

        for i in range(5):
            start = time.time()
            assistant.run(complex_query)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = statistics.mean(times)

        print(f"\n复杂查询平均响应时间：{avg_time:.2f}秒")

        # 复杂查询平均响应时间应小于 15 秒
        assert avg_time < 15.0

    def test_response_time_percentile(self, assistant):
        """测试响应时间百分位数"""
        query = "测试响应时间"
        times = []

        for i in range(20):
            start = time.time()
            assistant.run(query)
            elapsed = time.time() - start
            times.append(elapsed)

        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]

        print(f"\nP95 响应时间：{p95:.2f}秒")
        print(f"P99 响应时间：{p99:.2f}秒")

        # P95 应小于 8 秒
        assert p95 < 8.0


class TestThroughput:
    """吞吐量性能测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_requests_per_minute(self, assistant):
        """测试每分钟请求数"""
        query = "性能测试"
        start_time = time.time()
        request_count = 0

        # 运行 1 分钟
        while time.time() - start_time < 60:
            assistant.run(query)
            request_count += 1

        rpm = request_count  # 1 分钟的请求数
        print(f"\n吞吐量：{rpm} 请求/分钟")

        # 至少应该能处理 10 个请求/分钟
        assert rpm >= 10

    def test_concurrent_users_simulation(self, assistant):
        """模拟多用户并发"""
        queries = [f"用户{i}的查询" for i in range(5)]

        def make_request(q):
            return assistant.run(q)

        start_time = time.time()

        # 使用线程池模拟并发
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, q) for q in queries]
            results = [f.result() for f in futures]

        elapsed = time.time() - start_time

        # 所有请求应在 30 秒内完成
        assert elapsed < 30
        assert all(r is not None for r in results)

        print(f"\n并发测试耗时：{elapsed:.2f}秒")


class TestMemoryUsage:
    """内存使用性能测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_memory_growth_over_time(self, assistant):
        """测试内存增长"""
        import tracemalloc
        tracemalloc.start()

        # 执行多次操作
        for i in range(20):
            assistant.run(f"测试对话{i}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        print(f"\n峰值内存使用：{peak_mb:.2f} MB")

        # 峰值内存应小于 500MB
        assert peak_mb < 500

    def test_memory_after_large_input(self, assistant):
        """测试大输入后的内存使用"""
        import tracemalloc
        tracemalloc.start()

        # 发送大输入
        large_input = "测试内容 " * 1000
        assistant.run(large_input)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        print(f"\n大输入后峰值内存：{peak_mb:.2f} MB")

        assert peak_mb < 500


class TestStability:
    """稳定性测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_long_running_stability(self, assistant):
        """测试长时间运行稳定性"""
        error_count = 0
        total_requests = 50

        for i in range(total_requests):
            try:
                assistant.run(f"稳定性测试{i}")
            except Exception as e:
                error_count += 1
                print(f"请求{i}失败：{e}")

        success_rate = (total_requests - error_count) / total_requests * 100
        print(f"\n成功率：{success_rate:.2f}%")

        # 成功率应大于 95%
        assert success_rate >= 95.0

    def test_error_rate_under_load(self, assistant):
        """测试负载下的错误率"""
        error_count = 0
        total_requests = 30

        queries = [
            "正常查询",
            "测试",
            "你好",
            "",  # 空查询
            "   ",  # 空格
            "<script>alert('xss')</script>",  # 恶意输入
        ]

        for i in range(total_requests):
            query = queries[i % len(queries)]
            try:
                response = assistant.run(query)
                if response is None:
                    error_count += 1
            except Exception as e:
                error_count += 1

        error_rate = error_count / total_requests * 100
        print(f"\n错误率：{error_rate:.2f}%")

        # 错误率应小于 5%
        assert error_rate < 5.0


class TestScalability:
    """可扩展性测试"""

    @pytest.fixture
    def assistant(self):
        return SmartOfficeAssistant()

    def test_document_count_impact(self, assistant):
        """测试文档数量对性能的影响"""
        import tempfile
        import shutil

        # 创建临时目录存放测试文档
        temp_dir = tempfile.mkdtemp()

        try:
            # 测试不同文档数量下的查询性能
            doc_counts = [1, 5, 10]

            for count in doc_counts:
                # 添加文档
                for i in range(count):
                    test_file = os.path.join(temp_dir, f"test_{i}.md")
                    with open(test_file, "w", encoding='utf-8') as f:
                        f.write(f"# 测试文档{i}\n\n这是第{i}个测试文档的内容。")
                    assistant.upload_document(test_file)

                # 测量查询时间
                start = time.time()
                assistant.run("测试查询")
                elapsed = time.time() - start

                print(f"\n{count}个文档时查询时间：{elapsed:.2f}秒")

                # 清理以便下次测试
                assistant.rag_engine.clear_database()

        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir)


def run_performance_tests():
    """运行性能测试"""
    print("\n" + "=" * 70)
    print("智能办公助手 V1.0 - 性能测试套件")
    print("=" * 70)

    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-q",
        "-s"
    ])


if __name__ == "__main__":
    run_performance_tests()
