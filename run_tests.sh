#!/bin/bash
# run_tests.sh

echo "开始运行对话助手测试..."


# 设置测试用的API密钥环境变量（测试时使用模拟值）
export DASHSCOPE_API_KEY=fake-test-key

# 运行测试并生成报告
pytest test1.py -v --cov=. --cov-report=html --junitxml=test-results.xml

echo "测试完成，查看以下输出："
echo "- 控制台输出显示测试结果"
echo "- htmlcov/ 目录下有覆盖率报告"
echo "- test-results.xml 包含XML格式的测试结果"
