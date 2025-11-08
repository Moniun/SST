# HIPPO: 对话历史记忆融合系统

HIPPO (History-Integrated Processing for Personalized Outputs) 是一个结合对话历史记忆功能的大语言模型增强系统。该系统通过引入专门的记忆模型（HippoModel），使大语言模型能够更好地理解和利用对话历史信息，提供更加连贯和个性化的回答。

## 项目概述

HIPPO系统的核心创新点在于：
- 将对话历史信息通过专门的记忆模型处理，生成记忆表示
- 将记忆表示与大语言模型的输出特征进行融合，实现记忆增强
- 采用非侵入式设计，不需要修改大语言模型的底层结构
- 支持阶段性训练策略，先训练记忆模型，再进行联合优化
- 实现差异化学习率，加速关键组件的收敛

## 系统架构

```
┌─────────────────┐      ┌─────────────────┐
│  对话历史输入   │ ────> │  HippoModel    │
│  (dialog_history)│      │ (记忆编码器)   │
└─────────────────┘      └──────────┬──────┘
                                    │
                                    ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  当前查询输入   │ ────> │  基础大语言模型 │ ────> │    特征融合     │ ────> 增强输出
│  (memory_query) │      │ (ModifiedQwen)  │      │                 │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.28+
- Datasets

### 安装依赖

```bash
pip install -r requirements.txt
```

### 生成示例数据

系统提供了生成示例训练数据的功能：

```bash
python data_processor.py
```

这将在当前目录生成`demo_training_data.jsonl`文件，包含100条示例对话数据。

### 模型训练

使用以下命令启动模型训练：

```bash
python train.py --model_name_or_path /path/to/qwen/model --data_path demo_training_data.jsonl --output_dir ./output --max_length 512 --per_device_train_batch_size 2
```

可选参数：
- `--model_name_or_path`: 预训练模型路径，默认为"Qwen/Qwen3-8B-Chat"
- `--data_path`: 训练数据路径，默认为"demo_training_data.jsonl"
- `--output_dir`: 模型保存路径，默认为"./qwen3-8b-custom-finetuned"
- `--max_length`: 最大序列长度，默认为512
- `--per_device_train_batch_size`: 每设备训练批次大小，默认为2
- `--gradient_accumulation_steps`: 梯度累积步数，默认为4
- `--learning_rate`: 基础学习率（会被内部差异化学习率覆盖），默认为2e-4
- `--num_train_epochs`: 训练轮数，默认为3

### 推理使用

训练完成后，可以使用`inference.py`进行推理：

```bash
python inference.py --model_path ./output --dialog_history "用户: 你好
助手: 你好！我是AI助手，有什么可以帮助你的吗？" --query "你刚才说你是谁？"
```

## 文件结构

```
├── data_processor.py    # 数据加载和预处理模块
├── generate_data.py     # 数据生成工具
├── hippo_model.py       # 记忆编码模型实现
├── model_customization.py # 大模型定制和特征融合
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖列表
└── README.md            # 项目文档（本文档）
```

## 核心组件说明

### 1. 数据处理模块 (data_processor.py)

负责加载和预处理训练数据，支持JSONL格式和Hugging Face数据集。预处理功能包括：
- 提取对话历史、记忆查询和答案
- 使用分词器处理输入文本
- 优化的标签生成策略：单独处理查询和答案，提高训练效率
- 保留对话历史信息用于Hippo模型的隐藏状态更新
- 合理设置标签掩码，确保损失计算的准确性

### 2. Hippo模型 (hippo_model.py)

专门设计的记忆编码模型，用于处理对话历史信息：
- 将对话历史编码为向量表示
- 捕获对话中的关键信息和上下文
- 生成可与大语言模型融合的特征表示
- 设计轻量级结构，平衡记忆能力和计算效率

### 3. 模型定制 (model_customization.py)

实现ModifiedQwen类，集成基础大语言模型和记忆融合功能：
- 继承自预训练Qwen模型，保持原有功能
- 在forward方法中集成HippoModel处理嵌入层输出
- 实现记忆特征与大模型输出的门控融合机制
- 优化的显存管理：使用torch.no_grad()减少推理时内存占用
- 智能的参数冻结/解冻控制，支持训练阶段管理

### 4. 训练模块 (train.py)

高级训练器实现，支持多种优化策略：
- 实现CustomTrainer类，增强的compute_loss方法
- 差异化学习率策略：Hippo模型使用5e-5，Transformer层使用1e-5
- 阶段性训练：第一个epoch只训练Hippo模型，后续epoch进行联合训练
- 自动层解冻机制：在第一个epoch完成后自动解冻Transformer层
- 动态优化器参数组更新，无需重启训练过程

## 数据格式

训练数据采用JSONL格式，每条记录包含以下字段：

```json
{
  "dialog_history": ["用户问题1", "助手回答1", "用户问题2", "助手回答2"...],
  "memory_query": "关于历史对话内容的查询",
  "memory_answer": "基于历史对话的正确回答"
}
```

## 使用场景

HIPPO系统特别适用于以下场景：
- 需要长对话记忆的客服机器人
- 个性化对话助手
- 需要保持对话连贯性的交互式应用
- 教育和辅导场景中的问答系统

## 性能优化特性

1. **阶段性训练策略**：
   - 第一个epoch只训练Hippo模型和门控机制
   - 后续epoch同时训练所有可训练组件
   - 有助于稳定训练过程和提升最终性能

2. **差异化学习率**：
   - Hippo模型和门控机制：5e-5
   - Transformer层：1e-5
   - 加速关键组件的收敛速度

3. **显存优化**：
   - 推理时自动启用torch.no_grad()
   - 根据训练状态和参数冻结情况智能控制梯度计算
   - 训练时选择性保存注意力权重

4. **其他优化**：
   - 混合精度训练(fp16)
   - 梯度累积支持
   - 8位AdamW优化器，降低显存占用

## 注意事项

1. 系统依赖于预训练的Qwen模型，请确保有正确的模型访问权限
2. 对话历史长度可能影响系统性能，建议合理控制
3. 训练数据质量对模型性能有显著影响，请确保数据质量
4. 训练过程中会自动执行阶段性优化，第一个epoch完成后会打印Transformer层解冻信息
5. 差异化学习率策略会自动应用，无需额外配置

## 许可证

[MIT License](LICENSE)

## 联系信息

如有问题或建议，请联系项目维护者。