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

### 数据准备

系统支持JSONL格式的训练数据。按照以下步骤准备数据：

1. 将训练数据和验证数据放在项目根目录的`data`文件夹中
2. 确保训练数据文件名为`data_train.jsonl`
3. 确保验证数据文件名为`data_val.jsonl`

数据格式要求：
```json
{
  "dialog_history": ["用户问题1", "助手回答1", "用户问题2", "助手回答2"...],
  "memory_query": "关于历史对话内容的查询",
  "memory_answer": "基于历史对话的正确回答"
}
```

### 模型训练

使用以下命令启动模型训练（使用默认参数时只需简单运行）：

```bash
python train.py
```

训练将自动加载`data/data_train.jsonl`作为训练集和`data/data_val.jsonl`作为验证集，并将模型保存到`./models`目录。

可选参数配置：
- `--model_name_or_path`: 预训练模型路径，默认为"Qwen/Qwen3-8B-Chat"
- `--data_path`: 训练数据路径，默认为"data/data_train.jsonl"
- `--test_data_path`: 验证数据路径，默认为"data/data_val.jsonl"
- `--output_dir`: 模型保存路径，默认为"./models"
- `--num_train_epochs`: 训练轮数，默认为3
- `--per_device_train_batch_size`: 每设备训练批次大小，默认为8
- `--gradient_accumulation_steps`: 梯度累积步数，默认为2
- `--learning_rate`: 学习率，默认为2e-4
- `--max_length`: 最大序列长度，默认为None（自动检测）
- `--evaluation_strategy`: 评估策略，默认为"steps"
- `--eval_steps`: 每多少步进行一次评估，默认为500
- `--load_best_model`: 是否在训练结束时加载最佳模型

训练过程中的关键特性：
- 第一个epoch只训练Hippo模型和门控机制
- 后续epoch自动解冻Transformer层进行联合训练
- 差异化学习率：Hippo模型使用5e-5，Transformer层使用1e-5
- 自动保存最佳模型检查点

### 推理使用

训练完成后，可以使用`inference.py`进行推理。系统现在使用隐藏状态机制维护对话上下文，不再需要显式提供对话历史。

#### 单次执行模式

使用`--prompt`参数直接提供输入内容：

```bash
python inference.py --model_path ./models --prompt "你好，请问有什么可以帮助你的？"
```

#### 交互式模式

直接运行`inference.py`即可进入交互式对话模式：

```bash
python inference.py --model_path ./models
```

在交互式模式中，系统提供以下命令：
- `exit` 或 `quit`: 退出程序
- `reset_hidden`: 重置HIPPO模型的隐藏状态

HIPPO模型会自动通过隐藏状态维护对话上下文，无需显式提供历史记录。这是系统的核心特性，允许模型在长对话中保持记忆。

## 文件结构

```
├── data/                # 数据文件夹
│   ├── data_train.jsonl # 训练数据集
│   └── data_val.jsonl   # 验证数据集
├── data_processor.py    # 数据加载和预处理模块
├── generate_data.py     # 数据生成工具
├── hippo_model.py       # 记忆编码模型实现
├── model_customization.py # 大模型定制和特征融合
├── train.py             # 训练脚本
├── inference.py         # 推理脚本
├── requirements.txt     # 依赖列表
└── README.md            # 项目文档
```

## 核心功能说明

### 1. 数据处理功能 (data_processor.py)

- 支持同时加载和预处理训练集和验证集
- 兼容JSONL格式和Hugging Face数据集
- 优化的标签生成策略，提高训练效率
- 保留对话历史信息用于记忆模型处理
- 详细的日志输出，方便监控数据加载状态

### 2. 记忆编码模型 (hippo_model.py)

- 将对话历史编码为向量表示
- 捕获对话中的关键信息和上下文关联
- 生成可与大语言模型融合的特征表示
- 轻量级设计，平衡记忆能力和计算效率

### 3. 模型集成与融合 (model_customization.py)

- 集成基础大语言模型和记忆融合功能
- 实现记忆特征与大模型输出的门控融合机制
- 智能的参数冻结/解冻控制，支持训练阶段管理
- 显存优化，降低推理时内存占用

### 4. 高级训练系统 (train.py)

- 阶段性训练策略：先训练记忆模型，再联合优化
- 差异化学习率自动应用，无需手动配置
- 自动层解冻机制，优化训练过程
- 动态评估和模型保存，跟踪训练进度

### 5. 记忆增强推理 (inference.py)

- 基于隐藏状态的上下文维护，无需显式提供对话历史
- 交互式对话模式，支持长对话记忆保持
- 自定义生成函数，精确控制隐藏状态更新
- 隐藏状态重置功能，灵活管理对话上下文

## 训练流程详解

1. **数据加载阶段**：
   - 自动加载`data/data_train.jsonl`和`data/data_val.jsonl`
   - 预处理对话历史、记忆查询和答案
   - 输出数据集大小统计信息

2. **模型初始化阶段**：
   - 加载基础Qwen模型
   - 初始化Hippo模型和门控机制
   - 冻结Transformer层，只保留Hippo模型参数可训练

3. **第一阶段训练**：
   - 只训练Hippo模型和门控机制
   - 让记忆模型先适应大模型的特征
   - 定期在验证集上评估性能

4. **自动阶段转换**：
   - 第一个epoch完成后，自动解冻Transformer层
   - 更新优化器参数组，应用差异化学习率
   - 继续训练所有可训练组件

5. **模型保存**：
   - 训练完成后将模型保存到`./models`目录
   - 可选保存最佳模型检查点

## 性能优化特性

1. **阶段性训练**：先训练记忆模型，再联合优化，稳定训练过程
2. **差异化学习率**：Hippo模型(5e-5)和Transformer层(1e-5)使用不同学习率
3. **显存优化**：混合精度训练(fp16)、梯度累积、智能梯度计算控制
4. **动态评估**：训练过程中定期在验证集上评估，监控模型性能

## 常见问题与解决方案

1. **显存不足**：
   - 减小`per_device_train_batch_size`
   - 增加`gradient_accumulation_steps`
   - 使用`--max_length`限制序列长度

2. **训练不稳定**：
   - 减小学习率
   - 确保数据质量，检查数据格式正确性
   - 使用`--load_best_model`参数保存最佳模型

3. **记忆效果不佳**：
   - 增加对话历史长度
   - 提高训练轮数
   - 优化训练数据，增加更多样化的记忆查询样本

## 许可证

[MIT License](LICENSE)

## 联系信息

如有问题或建议，请联系项目维护者。