import os
from openai import OpenAI

client = OpenAI(
    api_key="Your_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-32b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "介绍一下千问模型的API免费调用额度?"},
    ],
    extra_body={
        "enable_thinking": False
    }
)

if __name__ == '__main__':
    print(completion.model_dump_json())
    print("炸憨")
