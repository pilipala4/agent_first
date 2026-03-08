# 智能办公助手 V1.1

> 一站式办公解决方案：文档问答 + 周报生成 + 工具调用 + 记忆管理

## 🎯 核心功能

### 1. 文档问答 (RAG)
- 支持 PDF、Word、Markdown 文档上传
- 基于向量数据库的智能检索
- 结合 LLM 生成准确答案

### 2. 周报生成
- 自定义输入工作内容
- 三种模板：简单/专业/详细
- 支持 Markdown 和 Word 格式导出
- 自动生成结构化报告

### 3. 办公工具
- **实时搜索**: 行业资讯、新闻、天气
- **日历管理**: 日程查询、会议安排
- **待办清单**: 任务添加、查看、标记完成

### 4. 记忆管理
查看历史：查看历史问答记录
搜索历史：通过关键词搜索历史记录
清空历史：清空历史记录clear 或 cls"

## 🚀 快速开始

### 环境准备

1. **安装依赖**
pip install -r requirements.txt
2. **配置 API 密钥**
DASHSCOPE_API_KEY=your_dashscope_api_key SERPAPI_API_KEY=your_serpapi_api_key # 可选，用于实时搜索
在项目根目录创建 `.env` 文件：
### 启动系统
python main.py
## 💡 使用指南

### 1. 文档问答

**上传文档：**
upload ./path/to/document.pdf
**提问：**
HIC-YOLOv5 模型有什么特点？ 这篇论文的主要内容是什么？
### 2. 周报生成

**方式一：快捷命令**
report 本周完成了项目开发，下周继续优
**方式二：自然语言**
帮我写一份周报，本周完成了系统开发和测试
**示例输入：**
### 3. 办公工具

**搜索资讯：**
今天的人工智能新闻 搜索最新的行业动态
**日历管理：**
查询今天的日程安排 明天有什么会议
**待办事项：**
添加一个待办：下午 3 点参加项目会议 查看我的待办清单 标记第 1 个待办为已完成
**天气查询：**
明天广州天气怎么样？ 北京周末会下雨吗？
### 4. 系统命令
help / h - 查看帮助 history 或 hist 查看历史 search 关键词 搜索历史 clear / cls - 清空对话历史 quit / exit - 退出系统
## 📁 项目结构
agent_first/ 
├── main.py # 主入口（智能办公助手） 
├── rag/ # RAG 文档问答模块 │ └── rag_system.py 
├── tools/ # 工具模块 │ ├── office_tools.py # 办公工具管理器 │ ├── weekly_report_generator.py # 周报生成器 │ └── tool_encapsulation.py # 工具封装 
├── test/ # 测试目录 │ ├── test_all_features.py # 全功能测试 │ └── test_weekly_report.py # 周报测试 
├── generated_reports/ # 生成的周报文件 ├── chroma_db/ # RAG 向量数据库 ├── llm_call.py # LLM 调用封装 └── logger.py # 日志模块
## 🧪 运行测试

**全功能测试：**
python run_all_tests.py
**核心功能测试：**
python quick_test.py
---

**版本**: V1.1  
**更新日期**: 2026-03-08