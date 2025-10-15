import requests
import json
import time
from typing import List, Dict
from utils.config import LLMConfig

def generate_single_sample(api_url: str, api_key: str, model_name: str) -> Dict:
    """生成单条记忆训练数据（支持自定义API地址和模型）"""
    prompt = """
    任务：生成用于训练大模型长期记忆能力的对话数据集。
    要求：
    1. 先生成一段多轮对话（2-10轮），主题随机（日常、工作、学习等）。
    2. 对话包含至少3个关键信息点（如时间、地点、事件、属性等）。
    3. 生成1个针对历史内容的记忆查询（例如：“我第2句话提到了什么？”“刚才说的XX是指什么？”）。
    4. 答案必须从对话历史中提取，不编造信息。
    5. 仅输出JSON，包含dialog_history（数组）、memory_query（字符串）、memory_answer（字符串）。
    示例输出：
    {
      "dialog_history": [
        "用户：我喜欢打篮球，每周六下午会和朋友去体育馆。",
        "助手：体育馆人多吗？需要提前预约吗？",
        "用户：是的，要提前一天预约，我们一般打2小时，然后去吃火锅。",
        "助手：你们常去的火锅店叫什么名字？",
        "用户：叫‘老地方火锅’，就在体育馆对面。"
      ],
      "memory_query": "我每周什么时候去打篮球？",
      "memory_answer": "你每周六下午去打篮球。"
    }
    """
    
    try:
        # 通用请求头（适配大多数模型API的认证方式）
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 通用请求体结构（可根据不同模型API调整）
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "timeout": 30
        }
        
        # 发送POST请求
        response = requests.post(
            url=api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()  # 检查HTTP错误状态
        response_data = response.json()
        
        # 提取生成内容（根据不同模型的响应结构调整此处）
        # 常见结构1: OpenAI/DeepSeek类 -> choices[0].message.content
        # 常见结构2: 其他模型可能使用 -> data[0].content 等
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        
        if not content:
            raise ValueError("模型返回内容为空")
            
        return json.loads(content)
    except Exception as e:
        print(f"生成失败：{e}")
        return None

def generate_dataset(
    num_samples: int, 
    output_file: str,
    api_url: str,
    api_key: str,
    model_name: str
) -> None:
    """批量生成数据集并保存为JSONL格式"""
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            print(f"生成第{i+1}/{num_samples}条数据...")
            sample = None
            # 重试机制（应对API波动）
            while sample is None:
                sample = generate_single_sample(api_url, api_key, model_name)
                if sample is None:
                    print("重试中...")
                    time.sleep(2)
            # 写入JSONL（每行一个JSON对象）
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            # 避免API速率限制
            time.sleep(1)
    print(f"数据集生成完成，保存至{output_file}")

if __name__ == "__main__":
    # 配置参数（可根据需要修改为不同模型的信息）
    llm_config = LLMConfig()
    config = {
        # DeepSeek API示例（请替换为实际可用的API地址和密钥）
        "api_url": "https://api.deepseek.com/v1/chat/completions",  # DeepSeek API地址
        # "api_url": "https://api.openai.com/v1/chat/completions",  # OpenAI API地址
        "api_key": llm_config.api_key,  # 替换为实际API密钥
        "model_name": "deepseek-chat",  # DeepSeek模型名称
        # "model_name": "gpt-3.5-turbo",  # OpenAI模型名称
        "num_samples": 5,
        "output_file": "memory_train.jsonl"
    }
    
    generate_dataset(
        num_samples=config["num_samples"],
        output_file=config["output_file"],
        api_url=config["api_url"],
        api_key=config["api_key"],
        model_name=config["model_name"]
    )