import requests
import json
import time
import os
from typing import List, Dict
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def generate_single_sample(api_url: str, api_key: str, model_name: str) -> Dict:
    """生成单条记忆训练数据（支持自定义API地址和模型）"""
    prompt = """
    Task: Generate a high-quality dialogue dataset for training large models' long-term memory capabilities.
    Requirements:
    1. Generate a coherent, multi-turn conversation (3-8 turns) on any random topic.
    2. Ensure the conversation has natural flow with logical transitions between turns.
    3. Include 4-6 key information points of varying types: numbers, dates, names, locations, descriptions, preferences, or specific events.
    4. The information should be distributed throughout the conversation history (not all in one message).
    5. Create a memory query of varying difficulty level (easy, medium, or hard):
       - Easy: Direct information retrieval from one message
       - Medium: Information requiring connection between multiple messages
       - Hard: Implicit information that requires inference from context
    6. The memory_answer must be strictly extracted from the dialogue history without any fabrication.
    7. Format: Output only valid JSON with exactly these fields: dialog_history (array of strings), memory_query (string), memory_answer (string).
    8. IMPORTANT: Do NOT include "User:" or "Assistant:" prefixes in the dialog_history entries. Just include the raw conversation content.
    9. CRITICAL: Make sure to output ONLY the JSON and nothing else (no markdown, no explanations, no tags).
    
    Example 1 (Easy Memory Query):
    {
      "dialog_history": [
        "I'm planning a trip to Paris next month. I'll stay at Hotel Grand Paris for 5 nights.",
        "That sounds wonderful! What are your main plans in Paris?",
        "I want to visit the Eiffel Tower, Louvre Museum, and take a Seine River cruise. I also booked tickets for a concert at the Opera Garnier.",
        "The Opera Garnier is beautiful! Which day is your concert?",
        "It's scheduled for July 15th at 8 PM."
      ],
      "memory_query": "Where will I stay during my trip to Paris?",
      "memory_answer": "You'll stay at Hotel Grand Paris."
    }
    
    Example 2 (Medium Memory Query):
    {
      "dialog_history": [
        "I have three cats named Max, Luna, and Oliver. Max is the oldest at 7 years old.",
        "That's a nice family of cats! What breeds are they?",
        "Max is a Maine Coon, Luna is a Siamese, and Oliver is a tabby with orange fur.",
        "Do they have any special habits or preferences?",
        "Luna loves sitting on windowsills and watching birds. Max enjoys napping on the couch, especially when it's sunny."
      ],
      "memory_query": "Which cat likes to watch birds?",
      "memory_answer": "Luna likes to watch birds."
    }
    
    Example 3 (Hard Memory Query):
    {
      "dialog_history": [
        "I started a new job at TechCorp last Monday. The office is on the 12th floor of the Glass Tower building.",
        "Congratulations on the new job! How's your commute?",
        "I take the subway every morning, which takes about 25 minutes. Sometimes I stop at the café near the station for coffee.",
        "Do you have any colleagues you've gotten to know yet?",
        "Yes, my team leader Sarah is very helpful. She showed me around on my first day and introduced me to everyone."
      ],
      "memory_query": "What did Sarah do to help on the first day of work?",
      "memory_answer": "Sarah showed me around on my first day and introduced me to everyone."
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
        
        # 简化版内容提取和解析
        # 尝试获取OpenAI/DeepSeek格式的响应内容
        try:
            content = response_data['choices'][0]['message']['content'].strip()
        except (KeyError, IndexError):
            # 尝试获取其他可能的响应格式
            try:
                content = response_data['data'][0]['content'].strip()
            except (KeyError, IndexError, TypeError):
                content = str(response_data)
                print(f"警告: 使用备用响应结构: {content[:100]}...")
        
        # 清理内容并提取JSON部分
        if content.startswith('```'):
            content = '\n'.join([line for line in content.split('\n') if not line.strip().startswith('```')])
        
        # 提取JSON对象
        json_start, json_end = content.find('{'), content.rfind('}')
        if json_start >= 0 and json_end > json_start:
            content = content[json_start:json_end+1]
        
        if not content:
            raise ValueError("模型返回内容为空")
        
        # 解析并验证JSON
        parsed_data = json.loads(content)
        required_fields = ['dialog_history', 'memory_query', 'memory_answer']
        if not all(field in parsed_data for field in required_fields):
            missing = [f for f in required_fields if f not in parsed_data]
            raise ValueError(f"缺少必要字段: {', '.join(missing)}")
        
        return parsed_data
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
    # 从环境变量获取API密钥，避免硬编码敏感信息
    api_key = os.getenv("LLM_API_KEY")
    
    if not api_key:
        print("警告: 未找到环境变量 LLM_API_KEY，请设置API密钥")
        # 提供一个交互式输入作为备选方案
        api_key = input("请输入您的API密钥: ")
    
    config = {
        # DeepSeek API示例（可根据需要修改为不同模型的信息）
        "api_url": os.getenv("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions"),  # 优先从环境变量获取
        "api_key": api_key,  # 从环境变量或用户输入获取
        "model_name": os.getenv("LLM_MODEL_NAME", "deepseek-chat"),  # 优先从环境变量获取
        "num_samples": 2300,
        "output_file": "memory_train.jsonl"
    }
    
    generate_dataset(
        num_samples=config["num_samples"],
        output_file=config["output_file"],
        api_url=config["api_url"],
        api_key=config["api_key"],
        model_name=config["model_name"]
    )