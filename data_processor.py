from datasets import Dataset, load_dataset
import json
import torch

def load_and_preprocess_data(data_path, tokenizer, max_length=None):
    """
    加载并预处理训练数据，支持dialog_history、memory_query、memory_answer格式
    data_path: 数据文件路径（JSONL格式）
    tokenizer: 分词器
    max_length: 最大序列长度（默认为None，会自动从tokenizer获取模型最大长度）
    """
    # 如果未指定max_length，尝试从tokenizer或其配置中获取模型最大长度
    if max_length is None:
        # 尝试获取模型的最大上下文长度（不同模型有不同的配置属性）
        if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length > 0:
            max_length = tokenizer.model_max_length
        elif hasattr(tokenizer, 'max_model_input_sizes') and tokenizer.max_model_input_sizes:
            # 对于某些旧版本的tokenizer
            first_model = list(tokenizer.max_model_input_sizes.keys())[0]
            max_length = tokenizer.max_model_input_sizes[first_model]
        else:
            # 如果无法自动获取，使用一个合理的默认值
            max_length = 4096
        print(f"自动检测到模型最大长度: {max_length}")
    # 加载数据
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    else:
        # 支持Hugging Face数据集
        dataset = load_dataset(data_path)
        data = list(dataset["train"])  # 使用list()代替to_list()，更通用

    # 转换为Dataset对象
    dataset = Dataset.from_list(data)
    
    # 预处理函数
    def preprocess_function(example):
        # 提取对话历史、记忆查询和答案
        dialog_history = example.get("dialog_history", [])
        memory_query = example.get("memory_query", "")
        memory_answer = example.get("memory_answer", "")
        
        # 对对话历史中的每一句话进行编码
        encoded_dialog_history = []
        for history_item in dialog_history:
            if history_item:  # 确保不为空
                # 编码对话历史中的每个回合
                history_encoding = tokenizer(
                    history_item,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                # 将编码后的input_ids添加到列表中
                encoded_dialog_history.append(history_encoding["input_ids"][0])
        
        # 为大模型准备输入：只使用memory_query作为输入（减少计算量）
        model_input = memory_query
        
        # 编码输入（只包含查询）
        encoding = tokenizer(
            model_input,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 编码答案，用于生成标签
        answer_encoding = tokenizer(
            memory_answer,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # 创建标签：查询部分不参与损失计算，答案部分作为目标
        # 注意：这里假设答案长度不会超过max_length
        labels = torch.full_like(encoding["input_ids"][0], -100)
        # 获取答案的非填充部分长度
        answer_tokens = answer_encoding["input_ids"][0]
        # 找到第一个填充token的位置（假设使用0作为pad_token_id）
        non_pad_mask = answer_tokens != tokenizer.pad_token_id
        valid_answer_length = non_pad_mask.sum().item()
        # 确保不超过max_length
        answer_length = min(valid_answer_length, max_length)
        # 仅在标签的开始部分填入答案的有效token
        labels[:answer_length] = answer_encoding["input_ids"][0][:answer_length]
        
        # 返回处理后的数据
        return {
            "dialog_histories": encoded_dialog_history,
            "memory_queries": memory_query,
            "memory_answers": memory_answer,
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": labels
        }
    
    # 应用预处理
    # 注意：由于我们需要保留dialog_histories等非张量列，不使用remove_columns
    tokenized_dataset = dataset.map(preprocess_function)
    
    return tokenized_dataset