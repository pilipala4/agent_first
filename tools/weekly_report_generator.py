# -*- coding: utf-8 -*-
"""
结构化周报生成模块
支持用户输入"工作内容 + 成果 + 计划"，生成多格式周报 (markdown/word)
实现"自定义输入→生成结构化周报→导出"功能
"""
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from llm_call import llm_call, DEFAULT_MODEL
from logger import logger


class WeeklyReportInput(BaseModel):
    """周报输入数据模型"""
    work_content: str = Field(description="本周工作内容")
    achievements: str = Field(default="", description="工作成果")
    next_plan: str = Field(default="", description="下周计划")
    issues: str = Field(default="", description="遇到的问题及解决方案")
    remarks: str = Field(default="", description="备注/重要事项")


class WeeklyReportGenerator:
    """结构化周报生成器"""

    def __init__(self, api_key: str = None):
        """
        初始化周报生成器

        Args:
            api_key: API 密钥
        """
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY')
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")

        self.output_dir = "./generated_reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def parse_user_input(self, user_input: str) -> WeeklyReportInput:
        """
        解析用户输入，提取结构化信息

        Args:
            user_input: 用户输入的原始文本

        Returns:
            结构化的输入对象
        """
        parse_prompt = f"""
你是一个专业的周报信息提取助手。请从用户输入中提取以下信息，并以 JSON 格式返回：

用户输入：{user_input}

请提取以下字段（如果某项信息不存在，请留空）：
{{
    "work_content": "本周完成的主要工作内容",
    "achievements": "取得的工作成果和亮点",
    "next_plan": "下周工作计划",
    "issues": "遇到的问题及解决方案",
    "remarks": "备注或重要事项提醒"
}}

注意：
1. 只返回 JSON 格式，不要包含其他文字
2. 每个字段都应该简洁明了
3. 如果没有相关信息，该字段为空字符串
"""

        try:
            response = llm_call(
                prompt=parse_prompt,
                system_prompt="你是一个精准的信息提取助手，只返回 JSON 格式结果",
                model=DEFAULT_MODEL,
                retry_times=2
            )

            if isinstance(response, dict) and response.get('success'):
                result_text = response.get('data', '{}')

                try:
                    # 尝试解析 JSON
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result_json = json.loads(json_match.group())
                    else:
                        result_json = json.loads(result_text)

                    return WeeklyReportInput(
                        work_content=result_json.get('work_content', ''),
                        achievements=result_json.get('achievements', ''),
                        next_plan=result_json.get('next_plan', ''),
                        issues=result_json.get('issues', ''),
                        remarks=result_json.get('remarks', '')
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析失败：{e}")
                    # 降级处理：将全部输入作为工作内容
                    return WeeklyReportInput(work_content=user_input)
            else:
                logger.error("意图识别失败")
                return WeeklyReportInput(work_content=user_input)

        except Exception as e:
            logger.error(f"解析异常：{e}")
            return WeeklyReportInput(work_content=user_input)

    def generate_structured_report(self, report_input: WeeklyReportInput,
                                   template: str = "professional") -> str:
        """
        生成结构化周报

        Args:
            report_input: 结构化的输入数据
            template: 模板类型（professional, simple, detailed）

        Returns:
            生成的周报文本
        """
        templates = {
            "professional": self._get_professional_template(),
            "simple": self._get_simple_template(),
            "detailed": self._get_detailed_template()
        }

        template_prompt = templates.get(template, templates["professional"])

        generation_prompt = f"""
{template_prompt}

用户输入的工作信息：
- 工作内容：{report_input.work_content}
- 工作成果：{report_input.achievements if report_input.achievements else '无'}
- 下周计划：{report_input.next_plan if report_input.next_plan else '无'}
- 问题与解决：{report_input.issues if report_input.issues else '无'}
- 备注事项：{report_input.remarks if report_input.remarks else '无'}

请基于以上信息，生成一份专业、完整的周报。
"""

        try:
            response = llm_call(
                prompt=generation_prompt,
                system_prompt="你是一个专业的周报生成助手，擅长撰写结构化、专业的商务周报",
                model=DEFAULT_MODEL,
                retry_times=2
            )

            if isinstance(response, dict) and response.get('success'):
                logger.info("周报生成成功")
                return response.get('data', '')
            else:
                error_msg = response.get('error_message', '未知错误') if isinstance(response, dict) else '未知错误'
                logger.error(f"周报生成失败：{error_msg}")
                return f"周报生成失败：{error_msg}"

        except Exception as e:
            logger.error(f"生成异常：{e}")
            return f"周报生成异常：{str(e)}"

    def _get_professional_template(self) -> str:
        """获取专业版模板"""
        return """
请生成一份专业的周报，要求如下：

【周报结构】
一、本周工作总结
  - 详细列出完成的主要工作任务
  - 突出工作重点和优先级

二、工作成果与亮点
  - 量化成果（如完成率、提升百分比等）
  - 创新性工作或改进
  - 获得的认可或奖励

三、下周工作计划
  - 明确的工作目标
  - 具体的行动计划
  - 预期成果

四、遇到的问题与解决方案
  - 描述遇到的主要挑战
  - 已采取的解决措施
  - 需要协调的资源

五、重要事项提醒
  - 关键时间节点
  - 需要关注的事项

【写作要求】
- 使用正式的商务语气
- 语言简洁明了，条理清晰
- 重点突出，层次分明
- 适当使用数据支撑
"""

    def _get_simple_template(self) -> str:
        """获取简洁版模板"""
        return """
请生成一份简洁的周报，包含以下核心部分：

一、本周工作完成情况
二、主要成果
三、下周计划
四、问题与建议

要求：简洁明了，每条 1-2 句话，重点突出。
"""

    def _get_detailed_template(self) -> str:
        """获取详细版模板"""
        return """
请生成一份详细的周报，要求如下：

【详细周报结构】
一、本周工作概述
  1. 重点工作任务
  2. 常规工作任务
  3. 临时性工作任务

二、工作成果展示
  1. 定量成果（数据指标）
  2. 定性成果（质量提升）
  3. 创新与改进
  4. 团队协作贡献

三、问题分析与解决
  1. 遇到的主要问题
  2. 问题原因分析
  3. 解决方案与实施
  4. 经验总结与反思

四、下周详细计划
  1. 工作目标
  2. 具体行动步骤
  3. 资源配置
  4. 风险评估

五、资源需求与支持
  1. 需要的资源支持
  2. 需要协调的部门
  3. 时间要求

六、其他事项
  1. 重要通知
  2. 培训学习
  3. 团队建设

【写作要求】
- 内容详实，数据充分
- 逻辑严密，层次清晰
- 专业术语准确
- 具有可操作性和指导意义
"""

    def export_to_markdown(self, report_content: str, filename: str = None) -> str:
        """
        导出为 Markdown 格式

        Args:
            report_content: 周报内容
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"weekly_report_{date_str}"

        filepath = os.path.join(self.output_dir, f"{filename}.md")

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# 工作周报\n\n")
                f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("---\n\n")
                f.write(report_content)

            logger.info(f"Markdown 报告已保存：{filepath}")
            return filepath
        except Exception as e:
            logger.error(f"导出 Markdown 失败：{e}")
            raise

    def export_to_word(self, report_content: str, filename: str = None) -> str:
        """
        导出为 Word 格式（使用 HTML 转换）

        Args:
            report_content: 周报内容
            filename: 文件名（不含扩展名）

        Returns:
            保存的文件路径
        """
        if filename is None:
            date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"weekly_report_{date_str}"

        filepath = os.path.join(self.output_dir, f"{filename}.docx")

        try:
            # 简单的 HTML 格式 Word 文档
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: '微软雅黑', Arial, sans-serif; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 20px; }}
        h3 {{ color: #7f8c8d; }}
        p {{ margin: 10px 0; }}
        ul, ol {{ margin: 10px 0; padding-left: 30px; }}
        li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>工作周报</h1>
    <p><strong>生成时间：</strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <hr/>
    {self._convert_to_html(report_content)}
</body>
</html>
"""

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Word 报告已保存：{filepath}")
            return filepath
        except Exception as e:
            logger.error(f"导出 Word 失败：{e}")
            raise

    def _convert_to_html(self, text: str) -> str:
        """
        简单地将文本转换为 HTML

        Args:
            text: 原始文本

        Returns:
            HTML 格式的文本
        """
        lines = text.split('\n')
        html_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                html_lines.append('<br/>')
                continue

            # 处理标题
            if stripped.startswith('# '):
                html_lines.append(f'<h1>{stripped[2:]}</h1>')
            elif stripped.startswith('## '):
                html_lines.append(f'<h2>{stripped[3:]}</h2>')
            elif stripped.startswith('### '):
                html_lines.append(f'<h3>{stripped[4:]}</h3>')
            # 处理列表
            elif stripped.startswith('- ') or stripped.startswith('* '):
                html_lines.append(f'<li>{stripped[2:]}</li>')
            elif stripped.startswith('1. ') or stripped.startswith('2. ') or stripped.startswith('3. '):
                parts = stripped.split('. ', 1)
                if len(parts) == 2:
                    html_lines.append(f'<li>{parts[1]}</li>')
            # 处理加粗
            elif '**' in stripped:
                formatted = stripped.replace('**', '<strong>').replace('__', '<strong>')
                html_lines.append(f'<p>{formatted}</p>')
            # 普通段落
            else:
                html_lines.append(f'<p>{stripped}</p>')

        return '\n'.join(html_lines)

    def generate_and_export(self, user_input: str, template: str = "professional",
                            export_formats: List[str] = None) -> Dict[str, Any]:
        """
        一站式生成并导出周报

        Args:
            user_input: 用户输入
            template: 模板类型
            export_formats: 导出格式列表 ['markdown', 'word']

        Returns:
            包含生成结果和文件路径的字典
        """
        if export_formats is None:
            export_formats = ['markdown']

        result = {
            'success': False,
            'report_content': '',
            'exported_files': [],
            'error': None
        }

        try:
            # 1. 解析用户输入
            logger.info("开始解析用户输入...")
            report_input = self.parse_user_input(user_input)

            # 2. 生成结构化周报
            logger.info("正在生成周报...")
            report_content = self.generate_structured_report(report_input, template)
            result['report_content'] = report_content

            if report_content.startswith("周报生成失败") or report_content.startswith("周报生成异常"):
                result['error'] = report_content
                return result

            # 3. 导出文件
            for fmt in export_formats:
                if fmt.lower() == 'markdown':
                    filepath = self.export_to_markdown(report_content)
                    result['exported_files'].append({
                        'format': 'markdown',
                        'path': filepath
                    })
                elif fmt.lower() == 'word':
                    filepath = self.export_to_word(report_content)
                    result['exported_files'].append({
                        'format': 'word',
                        'path': filepath
                    })

            result['success'] = True
            logger.info(f"周报生成完成，导出 {len(result['exported_files'])} 个文件")

        except Exception as e:
            logger.error(f"周报生成过程异常：{e}")
            result['error'] = str(e)

        return result

