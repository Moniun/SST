from datasets import Dataset, load_dataset
import json

def load_and_preprocess_data(data_path, tokenizer, max_length=512):
    """
    加载并预处理训练数据
    data_path: 数据文件路径（JSONL格式）
    tokenizer: 分词器
    max_length: 最大序列长度
    """
    # 加载数据
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        # 支持Hugging Face数据集
        dataset = load_dataset(data_path)
        data = dataset["train"].to_list()

    # 转换为Dataset对象
    dataset = Dataset.from_list(data)
    
    # 预处理函数
    def preprocess_function(examples):
        # 处理不同格式的数据（支持对话格式和prompt-completion格式）
        prompts = []
        for item in examples:
            if "conversations" in item:
                # 对话格式: [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]
                prompt = "\n".join([f"{c['from']}: {c['value']}" for c in item["conversations"]])
            elif "prompt" in item and "completion" in item:
                # prompt-completion格式
                prompt = f"user: {item['prompt']}\nassistant: {item['completion']}"
            else:
                raise ValueError("数据格式不支持，请提供对话格式或prompt-completion格式")
            prompts.append(prompt)
        
        # 编码
        inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 设置标签（因果语言模型任务，标签与输入ID相同）
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    
    # 应用预处理
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names  # 移除原始列
    )
    
    return tokenized_dataset

# 示例数据生成（用于测试）
def generate_demo_data(output_path, num_samples=100):
    """生成示例训练数据（对话格式）"""
    import random
    topics = ["电影推荐", "旅游攻略", "美食制作", "科技新闻", "历史知识"]
    data = []
    
    for i in range(num_samples):
        topic = random.choice(topics)
        user_query = f"能介绍一下关于{topic}的内容吗？"
        assistant_answer = f"关于{topic}，有很多值得了解的信息...（示例回答{i}）"
        
        data.append({
            "conversations": [
                {"from": "user", "value": user_query},
                {"from": "assistant", "value": assistant_answer}
            ]
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"示例数据已生成: {output_path}")

if __name__ == "__main__":
    # 生成示例数据（运行此文件时执行）
    generate_demo_data("demo_training_data.jsonl")
    