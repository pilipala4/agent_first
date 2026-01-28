import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件



class StructuredAgent:
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv('DASHSCOPE_API_KEY')
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def create_math_prompt(self, problem: str) -> str:
        """创建数学解题的结构化提示词"""
        return f"""
请使用思维链（Chain of Thought）方法解决以下数学问题，并以JSON格式返回结构化结果：

问题：{problem}

要求步骤：
1. 首先分析问题的关键信息
2. 列出解题思路和步骤
3. 逐步推导计算过程
4. 得出最终答案
5. 验证答案的合理性

请严格按照以下JSON格式返回：
{{
  "problem": "...",
  "analysis": "...",
  "steps": [
    {{
      "step_number": 1,
      "description": "...",
      "calculation": "..."
    }}
  ],
  "final_answer": "...",
  "verification": "..."
}}
"""

    def create_copywriting_prompt(self, requirements: str) -> str:
        """创建文案生成的结构化提示词"""
        return f"""
请使用思维链（Chain of Thought）方法生成满足以下要求的文案，并以JSON格式返回结构化结果：

需求：{requirements}

要求步骤：
1. 分析文案目标受众和目的
2. 确定文案风格和语调
3. 构思核心信息和要点
4. 组织文案结构
5. 撰写文案内容

请严格按照以下JSON格式返回：
{{
  "requirement": "...",
  "target_audience": "...",
  "style_tone": "...",
  "key_points": ["...", "..."],
  "structure_plan": "...",
  "generated_copy": "...",
  "call_to_action": "..."
}}
"""

    def chat_completion(self, prompt: str, model: str = "qwen3-32b") -> dict:
        """通用聊天完成方法"""
        completion = self.client.chat.completions.create(
            model="qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": """你是一个专业的AI助手，擅长使用思维链方法进行逻辑推理和内容创作。请严格按照JSON格式返回结构化输出。"""
                },
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "enable_thinking": False
            },
            response_format={"type": "json_object"}
        )

        try:
            # 尝试解析返回的JSON
            content = completion.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果解析失败，返回原始内容
            return {"raw_response": content}


def main():
    agent = StructuredAgent(api_key="your_actual_key")

    # 示例1：数学解题
    math_problem = "小明有15个苹果，吃了3个后又买了8个，然后平均分给4个朋友，每个朋友得到几个苹果？"
    print("=== 数学解题示例 ===")
    math_result = agent.chat_completion(agent.create_math_prompt(math_problem))
    print(json.dumps(math_result, ensure_ascii=False, indent=2))

    print("\n" + "=" * 50 + "\n")

    # 示例2：文案生成
    copywriting_req = "为一家咖啡店写一个促销文案，针对年轻白领，强调提神醒脑功能，活动是买一送一"
    print("=== 文案生成示例 ===")
    copy_result = agent.chat_completion(agent.create_copywriting_prompt(copywriting_req))
    print(json.dumps(copy_result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    agent = StructuredAgent()

    # 数学问题求解
    math_result = agent.chat_completion(
        agent.create_math_prompt("2x + 5 = 15, 求x的值")
    )
    print("数学解题结果:", json.dumps(math_result, ensure_ascii=False, indent=2))

    # 文案生成
    copy_result = agent.chat_completion(
        agent.create_copywriting_prompt("为新产品写推广文案")
    )
    print("文案生成结果:", json.dumps(copy_result, ensure_ascii=False, indent=2))