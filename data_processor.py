from datasets import Dataset, load_dataset
import json
import torch

def load_and_preprocess_data(data_path, tokenizer, max_length=512):
    """
    加载并预处理训练数据，支持dialog_history、memory_query、memory_answer格式
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
    def preprocess_function(example):
        # 提取对话历史、记忆查询和答案
        dialog_history = example.get("dialog_history", [])
        memory_query = example.get("memory_query", "")
        memory_answer = example.get("memory_answer", "")
        
        # 为大模型准备输入：记忆查询 + 答案（用于训练）
        model_input = f"{memory_query}\n{memory_answer}"
        
        # 编码输入
        encoding = tokenizer(
            model_input,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 设置标签（仅答案部分参与损失计算）
        # 查找答案在输入中的起始位置
        query_encoding = tokenizer(
            memory_query + "\n",
            truncation=True,
            max_length=max_length
        )
        answer_start = len(query_encoding["input_ids"]) - 1  # -1 因为结尾有EOS token
        
        # 创建标签：query部分设为-100（不参与损失计算），answer部分正常
        labels = encoding["input_ids"].clone()
        labels[0, :answer_start] = -100
        
        # 返回处理后的数据
        return {
            "dialog_histories": dialog_history,
            "memory_queries": memory_query,
            "memory_answers": memory_answer,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels[0]
        }
    
    # 应用预处理
    # 注意：由于我们需要保留dialog_histories等非张量列，不使用remove_columns
    tokenized_dataset = dataset.map(preprocess_function)
    
    return tokenized_dataset

# 示例数据生成（用于测试）
def generate_demo_data(output_path, num_samples=100):
    """生成示例训练数据（对话格式）"""
    import random
    topics = ["电影推荐", "旅游攻略", "美食制作", "科技新闻", "历史知识"]
    data = []
    
    for i in range(num_samples):
        topic = random.choice(topics)
        # 使用新的数据格式
        user_query = f"能介绍一下关于{topic}的内容吗？"
        assistant_answer = f"关于{topic}，有很多值得了解的信息...（示例回答{i}）"
        memory_query = f"关于{topic}，你说了什么？"
        
        data.append({
            "dialog_history": [user_query, assistant_answer],
            "memory_query": memory_query,
            "memory_answer": assistant_answer
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"示例数据已生成: {output_path}")

if __name__ == "__main__":
    # 生成示例数据（运行此文件时执行）
    generate_demo_data("demo_training_data.jsonl")
    