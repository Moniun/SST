from datasets import Dataset, load_dataset
import json
import torch

def load_and_preprocess_data(data_path, tokenizer, max_length=None, split=None, processed_data_path=None):
    """
    加载并预处理数据，支持dialog_history、memory_query、memory_answer格式
    
    Args:
        data_path: 数据文件路径（JSONL格式）或数据集名称
        tokenizer: 分词器
        max_length: 最大序列长度（默认为None，会自动从tokenizer获取模型最大长度）
        split: 数据集分割名称（用于Hugging Face数据集，如'train'、'validation'、'test'）
              对于JSONL文件，此参数无效
        processed_data_path: 预处理数据保存和加载的路径
                            如果该路径存在已处理好的数据，则直接使用；
                            如果不存在，则处理数据并保存到该路径
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
        # 直接读取JSONL文件，支持训练集和验证集文件
        print(f"正在加载JSONL文件: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        print(f"成功加载 {len(data)} 条数据")
    else:
        # 支持Hugging Face数据集
        print(f"正在加载数据集: {data_path}")
        dataset = load_dataset(data_path)
        # 根据指定的split选择数据集分割，默认为'train'
        split_name = split if split is not None else 'train'
        if split_name not in dataset:
            # 如果指定的分割不存在，使用第一个可用的分割
            available_splits = list(dataset.keys())
            print(f"警告: 分割 '{split_name}' 不存在，使用第一个可用分割 '{available_splits[0]}'")
            split_name = available_splits[0]
        data = list(dataset[split_name])
        print(f"成功加载数据集分割 '{split_name}'，共 {len(data)} 条数据")

    # 转换为Dataset对象
    dataset = Dataset.from_list(data)
    
    # 预处理函数
    def preprocess_function(example):
        # 提取对话历史、记忆查询和答案
        dialog_history = example.get("dialog_history", [])
        memory_query = example.get("memory_query", "")
        memory_answer = example.get("memory_answer", "")
        
        # 1. 固定最大对话轮数（根据需求调整，比如5）
        max_history_rounds = 12
        # 2. 获取pad_token_id（避免硬编码）
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 3. 初始化固定形状的二维张量（填充pad_token_id）
        encoded_dialog_history = torch.full((max_history_rounds, max_length), pad_token_id, dtype=torch.long)
        
        # 4. 编码有效对话轮次，填入张量
        for i, history_item in enumerate(dialog_history[:max_history_rounds]):  # 限制不超过最大轮数
            if not history_item:  # 跳过空字符串
                continue
            # 编码单轮对话
            history_encoding = tokenizer(
                history_item,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            # 填入二维张量的第i行（覆盖pad填充）
            encoded_dialog_history[i] = history_encoding["input_ids"][0]
        
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
    
    # 首先尝试从已保存的预处理数据加载（如果指定了保存路径且存在）
    if processed_data_path:
        import os
        if (processed_data_path.endswith('.json') or processed_data_path.endswith('.jsonl')) and os.path.exists(processed_data_path):
            print(f"从已保存的JSON文件加载预处理数据: {processed_data_path}")
            from datasets import load_dataset
            tokenized_dataset = load_dataset('json', data_files=processed_data_path)
            # 将数据集字典转换为Dataset对象
            tokenized_dataset = tokenized_dataset['train']
            print(f"成功加载已预处理数据，共 {len(tokenized_dataset)} 条")
            return tokenized_dataset
        elif os.path.exists(processed_data_path) and os.path.isdir(processed_data_path):
            print(f"从已保存的Dataset目录加载预处理数据: {processed_data_path}")
            from datasets import load_from_disk
            tokenized_dataset = load_from_disk(processed_data_path)
            print(f"成功加载已预处理数据，共 {len(tokenized_dataset)} 条")
            return tokenized_dataset
        else:
            print(f"未找到已保存的预处理数据，将进行预处理并保存到 {processed_data_path}")
    
    # 应用预处理
    print(f"开始预处理数据...")
    # 注意：由于我们需要保留dialog_histories等非张量列，不使用remove_columns
    tokenized_dataset = dataset.map(preprocess_function)
    print(f"数据预处理完成")
    
    # 保存预处理数据（如果指定了保存路径）
    if processed_data_path:
        print(f"保存预处理数据到: {processed_data_path}")
        import os
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(processed_data_path)), exist_ok=True)
        
        # 根据文件扩展名选择保存格式
        if processed_data_path.endswith('.json') or processed_data_path.endswith('.jsonl'):
            tokenized_dataset.to_json(processed_data_path)
        else:
            # 默认保存为HuggingFace的Dataset格式（目录）
            tokenized_dataset.save_to_disk(processed_data_path)
        print(f"数据保存成功")
    
    return tokenized_dataset


def load_train_val_data(train_path, val_path, tokenizer, max_length=None, train_processed_path=None, val_processed_path=None):
    """
    加载并预处理训练集和验证集数据
    
    Args:
        train_path: 训练集文件路径（JSONL格式）
        val_path: 验证集文件路径（JSONL格式）
        tokenizer: 分词器
        max_length: 最大序列长度
        train_processed_path: 训练集预处理数据保存和加载的路径
                             如果该路径存在已处理好的数据，则直接使用；
                             如果不存在，则处理数据并保存到该路径
        val_processed_path: 验证集预处理数据保存和加载的路径
                            如果该路径存在已处理好的数据，则直接使用；
                            如果不存在，则处理数据并保存到该路径
                            
    Returns:
        tuple: (tokenized_train_dataset, tokenized_val_dataset)
    """
    # 加载并预处理训练集（支持从已保存的预处理数据加载）
    print(f"处理训练集: {train_path}")
    train_dataset = load_and_preprocess_data(train_path, tokenizer, max_length, processed_data_path=train_processed_path)
    
    # 加载并预处理验证集（支持从已保存的预处理数据加载）
    print(f"处理验证集: {val_path}")
    val_dataset = load_and_preprocess_data(val_path, tokenizer, max_length, processed_data_path=val_processed_path)
    
    return train_dataset, val_dataset