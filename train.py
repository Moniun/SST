import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from model_customization import ModifiedQwen
from data_processor import load_train_val_data
import argparse
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 下载METEOR所需的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
# 初始化Sentence-BERT模型
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 多语言模型，支持中文

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B-Chat")
    parser.add_argument("--data_path", type=str, default="data/data_train.jsonl", help="训练数据路径")
    parser.add_argument("--test_data_path", type=str, default="data/data_val.jsonl", help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="评估时的batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=None, help="最大序列长度，默认为None，会自动从tokenizer获取模型最大长度")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500, help="每多少步进行一次评估")
    parser.add_argument("--eval_accumulation_steps", type=int, default=8, help="评估时的梯度累积步数")
    parser.add_argument("--test_sample_ratio", type=float, default=1.0, help="测试集采样比例，0.0-1.0，用于减少评估时的测试样本数量")
    parser.add_argument("--load_best_model", action="store_true", help="是否在训练结束时加载最佳模型")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练的路径")
    return parser.parse_args()

# 自定义Trainer类，用于处理对话历史的传递和优化的训练流程
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_epoch_completed = False
        self.transfomer_layers_unfrozen = False
        self.current_epoch = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 从输入中提取对话历史（只用于HippoModel的隐藏状态更新）
        dialog_histories = inputs.pop("dialog_histories", None)
        
        # 确保模型只使用memory_query作为输入（已经在数据预处理中处理）
        # 调用模型的forward方法，传入对话历史供HippoModel使用
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],  # 这里的labels已经是memory_answer的编码
            dialog_histories=dialog_histories  # 仅用于HippoModel的隐藏状态更新
        )
        
        # 直接使用模型计算的损失（基于memory_query的预测与memory_answer的对比）
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
    
    def _get_parameter_groups(self, model):
        """提取模型参数并按类型分组
        
        Args:
            model: 要提取参数的模型
            
        Returns:
            tuple: (transformer_params, hippo_params, 参数组列表)
        """
        # 收集当前可训练参数并按类型分组
        transformer_params = []
        hippo_params = []
        
        # 遍历所有参数，根据来源分组
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'hippo_model' in name or 'gate_mechanisms' in name:
                    # HippoModel和门控机制参数使用较大的学习率
                    hippo_params.append(param)
                elif 'transformer.h' in name:
                    # 解冻的Transformer层参数使用较小的学习率
                    transformer_params.append(param)
        
        # 创建参数组
        param_groups = []
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': 1e-5  # 解冻的Transformer层使用较小的学习率
            })
        
        if hippo_params:
            param_groups.append({
                'params': hippo_params,
                'lr': 5e-5  # HippoModel参数使用较大的学习率
            })
        
        return transformer_params, hippo_params, param_groups
    
    def create_optimizer(self):
        """创建使用差异化学习率的优化器"""
        if self.optimizer is None:
            # 使用辅助方法获取参数组
            _, _, optimizer_grouped_parameters = self._get_parameter_groups(self.model)
            
            # 如果没有参数组，添加所有可训练参数作为默认组
            if not optimizer_grouped_parameters:
                optimizer_grouped_parameters = [
                    {'params': [p for p in self.model.parameters() if p.requires_grad]}
                ]
            
            self.optimizer = self.optimizer_cls(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,  # 这个会被参数组中的lr覆盖
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                correct_bias=self.args.adam_beta2 < 1.0
            )
        return self.optimizer
    
    def compute_metrics(self, eval_pred):
        """计算评估指标，包括准确率、精确率、召回率、F1分数、METEOR和Sentence-BERT语义相似度"""
        predictions, labels = eval_pred
        
        # 获取预测的token IDs（取概率最大的token）
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
        
        # 创建掩码，只考虑非-100的位置（因为-100表示不参与损失计算的位置）
        mask = labels != -100
        
        # 计算准确率
        correct_predictions = (predictions == labels) & mask
        accuracy = correct_predictions.sum().item() / mask.sum().item()
        
        # 这里的实现假设所有非填充token都是正类，模型需要正确预测它们
        true_positives = correct_predictions.sum().item()
        predicted_positives = mask.sum().item()  # 所有被预测的位置都是正类
        actual_positives = mask.sum().item()  # 所有非填充位置都是实际正类
        
        # 避免除零错误
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 使用默认模型名称获取tokenizer
        # 在实际训练中，应该确保tokenizer与训练时使用的一致
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Chat")
        
        # 初始化METEOR和Sentence-BERT分数
        meteor_scores = []
        sbert_similarities = []
        
        # 对每个样本计算METEOR和Sentence-BERT分数
        for pred, label, msk in zip(predictions, labels, mask):
            # 过滤掉填充token，获取有效预测和标签
            valid_pred_ids = pred[msk].tolist()
            valid_label_ids = label[msk].tolist()
            
            # 解码为文本
            try:
                pred_text = tokenizer.decode(valid_pred_ids, skip_special_tokens=True)
                label_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
                
                # 计算METEOR分数
                # METEOR需要句子拆分为单词列表
                pred_tokens = pred_text.split()
                label_tokens = label_text.split()
                if len(pred_tokens) > 0 and len(label_tokens) > 0:
                    meteor = meteor_score([label_tokens], pred_tokens)  # METEOR接受参考句子列表和候选句子
                    meteor_scores.append(meteor)
                
                # 计算Sentence-BERT语义相似度
                if pred_text.strip() and label_text.strip():
                    # 生成句子嵌入
                    embeddings = sbert_model.encode([pred_text, label_text], convert_to_tensor=True)
                    # 计算余弦相似度
                    similarity = cos_sim(embeddings[0], embeddings[1]).item()
                    sbert_similarities.append(similarity)
            except Exception as e:
                # 如果解码或评分出错，跳过该样本
                continue
        
        # 计算平均METEOR分数和平均Sentence-BERT相似度
        avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        avg_sbert_similarity = sum(sbert_similarities) / len(sbert_similarities) if sbert_similarities else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "meteor": avg_meteor,
            "sbert_similarity": avg_sbert_similarity
        }
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """重写评估方法，添加更多信息输出，包括METEOR和Sentence-BERT分数"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        print(f"\n开始在测试集上进行评估 (Epoch {self.current_epoch})...")
        
        # 调用父类的评估方法
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # 输出详细的评估结果
        print(f"测试集评估结果 (Epoch {self.current_epoch}):")
        print(f"- 测试准确率: {metrics.get('eval_accuracy', 0):.4f}")
        print(f"- 精确率: {metrics.get('eval_precision', 0):.4f}")
        print(f"- 召回率: {metrics.get('eval_recall', 0):.4f}")
        print(f"- F1分数: {metrics.get('eval_f1', 0):.4f}")
        print(f"- METEOR分数: {metrics.get('eval_meteor', 0):.4f}")
        print(f"- Sentence-BERT相似度: {metrics.get('eval_sbert_similarity', 0):.4f}")
        print(f"- 测试损失值: {metrics.get('eval_loss', 0):.4f}")
        
        return metrics
    
    def on_epoch_begin(self, epoch, logs=None):
        """在每个epoch开始时执行"""
        self.current_epoch = epoch + 1  # epoch从0开始，转换为人类可读的从1开始
        print(f"\n===== 开始训练第 {self.current_epoch} 个epoch =====")
        
        # 检查是否完成了第一个epoch
        if self.current_epoch >= 2 and not self.first_epoch_completed:
            self.first_epoch_completed = True
            # 解冻需要训练的transformer层
            self._unfreeze_transformer_layers(self.model)
            # 更新优化器参数组以包含新解冻的参数
            self._update_optimizer_parameters()
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时的处理"""
        self.current_epoch = epoch + 1
        print(f"\n第 {self.current_epoch} 个epoch训练完成")
        
        return logs
    
    def training_step(self, model, inputs):
        """重写训练步骤，处理训练逻辑"""
        # 直接调用父类的训练步骤方法
        return super().training_step(model, inputs)
        
    def _unfreeze_transformer_layers(self, model):
        """解冻需要训练的transformer层"""
        if hasattr(model, 'advance_training_stage'):
            print(f"\n第一个epoch训练完成，准备解冻需要训练的transformer层...")
            # 调用模型的advance_training_stage方法来解冻层
            stage_info = model.advance_training_stage()
            print(stage_info)  # 打印模型返回的阶段信息
            self.transfomer_layers_unfrozen = True
    
    def _update_optimizer_parameters(self):
        """更新优化器参数组，包含新解冻的transformer层"""
        # 获取模型中实际解冻的Transformer层索引
        unfrozen_layer_indices = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'transformer.h' in name:
                # 从参数名中提取层索引，格式通常为 'transformer.h.{layer_idx}....'
                try:
                    layer_idx = int(name.split('transformer.h.')[1].split('.')[0])
                    unfrozen_layer_indices.add(layer_idx)
                except (IndexError, ValueError):
                    pass
        
        print(f"当前解冻的Transformer层索引: {sorted(unfrozen_layer_indices)}")
        
        # 使用辅助方法获取参数组
        transformer_params, hippo_params, new_optimizer_grouped_parameters = self._get_parameter_groups(self.model)
        
        # 更新优化器的参数组
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'param_groups'):
            self.optimizer.param_groups.clear()
            for param_group in new_optimizer_grouped_parameters:
                self.optimizer.add_param_group(param_group)
            
            # 打印更新后的参数组信息
            print(f"\n更新后的参数组信息：")
            print(f"- 解冻的Transformer层参数数量: {sum(p.numel() for p in transformer_params):,}")
            print(f"- HippoModel和门控机制参数数量: {sum(p.numel() for p in hippo_params):,}")
            print(f"- Transformer层学习率: 1e-5")
            print(f"- HippoModel学习率: 5e-5")
        else:
            print("警告: 优化器尚未创建或无效，无法更新参数组")
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """用于推理阶段"""
        # 从输入中提取对话历史
        dialog_histories = inputs.pop("dialog_histories", None)
        
        # 调用模型进行预测
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"] if "labels" in inputs else None,
                dialog_histories=dialog_histories
            )
        
        loss = outputs.loss
        
        # 确保logits的形状正确
        logits = outputs.logits
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, inputs["labels"] if "labels" in inputs else None)

def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    # 使用load_train_val_data同时加载训练集和验证集
    train_dataset, test_dataset = load_train_val_data(
        train_path=args.data_path,
        val_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # 根据采样比例减少测试集大小
    if args.test_sample_ratio < 1.0:
        import random
        random.seed(42)  # 设置随机种子以确保结果可复现
        sample_size = int(len(test_dataset) * args.test_sample_ratio)
        test_dataset = random.sample(test_dataset, sample_size)
        print(f"已对测试集进行采样，采样比例: {args.test_sample_ratio}, 采样后大小: {len(test_dataset)}")
    
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 初始化模型 - 如果从检查点恢复，则使用from_pretrained加载完整模型状态
    if args.resume_from_checkpoint is not None:
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        # 从检查点目录加载完整模型（包括自定义模块参数和训练状态）
        model = ModifiedQwen.from_pretrained(args.resume_from_checkpoint)
        print("已从检查点加载模型，包括自定义模块参数和训练阶段状态")
    else:
        print("从零开始训练新模型")
        # 初始化带自定义模块的模型
        model = ModifiedQwen(base_model_name_or_path=args.model_name_or_path)
        
        # 确保初始状态下只训练Hippo模型和门控机制，冻结所有transformer层
        # 这样第一个epoch只会训练Hippo模型，让它先适应大模型的特征
        model.freeze_all()  # 先冻结所有参数
        model.unfreeze_hippo_model()  # 只解冻Hippo模型和门控机制
        print("初始状态设置：只训练Hippo模型和门控机制，冻结所有transformer层")
    
    # 数据收集器（处理batch内的padding等）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型不需要掩码语言模型任务
    )
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,  # 学习率预热
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,  # 最多保存3个检查点
        fp16=True,  # 混合精度训练
        optim="adamw_torch",  # 使用标准AdamW优化器以支持梯度计算
        report_to="none",  # 不使用wandb等日志工具
        remove_unused_columns=False,
        label_names=["labels"],
        evaluation_strategy=args.evaluation_strategy,  # 评估策略
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,  # 每多少步评估
        eval_accumulation_steps=args.eval_accumulation_steps,  # 评估时的梯度累积步数
        load_best_model_at_end=args.load_best_model,  # 是否在训练结束时加载最佳模型
        metric_for_best_model="f1" if args.load_best_model else None,  # 使用F1分数作为最佳模型的指标
        greater_is_better=True  # F1分数越大越好
    )
    
    # 打印初始参数组信息
    hippo_params = []
    gate_params = []
    transformer_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'hippo_model' in name:
                hippo_params.append(param)
            elif 'gate_mechanisms' in name:
                gate_params.append(param)
            elif 'transformer.h' in name:
                transformer_params.append(param)
    
    print(f"初始训练参数配置：")
    print(f"- Hippo模型参数数量: {sum(p.numel() for p in hippo_params):,}")
    print(f"- 门控机制参数数量: {sum(p.numel() for p in gate_params):,}")
    print(f"- 解冻的Transformer层参数数量: {sum(p.numel() for p in transformer_params):,}")
    
    # 初始化自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,  # 添加测试数据集
        data_collator=data_collator,
        optimizers=(None, None),  # 不使用默认优化器，让Trainer根据参数组自动创建
        compute_metrics=CustomTrainer.compute_metrics  # 设置计算指标的方法
    )
    
    # 开始训练
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存最终模型
    print("训练完成，开始保存模型...")
    # 确保模型保存目录存在
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    # 使用模型的自定义保存方法，确保所有参数都能正确保存
    model.save_pretrained(args.output_dir)
    print(f"模型已保存到 {args.output_dir}！")

if __name__ == "__main__":
    main()
    