import torch
import os
import json
import tempfile
# 启用内存优化配置
torch.cuda.empty_cache()

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoModelForCausalLM
)
from torch.cuda.amp import autocast
from model_customization import ModifiedQwen
from data_processor import load_train_val_data
import argparse
import nltk
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 导入DeepSpeed
import deepspeed



# 确保NLTK数据可用
try:
    nltk.data.path.append("./nltk_data")
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 初始化Sentence-BERT模型用于评估语义相似度
sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # 多语言模型，支持中文

def parse_args():
    parser = argparse.ArgumentParser()
    # 模型和数据路径
    parser.add_argument("--model_name_or_path", type=str, default="Qwen")
    parser.add_argument("--data_path", type=str, default="data/data_train.jsonl", help="训练数据路径")
    parser.add_argument("--test_data_path", type=str, default="data/data_val.jsonl", help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="./models")
    parser.add_argument("--train_processed_path", type=str, default="./processed_data/train_dataset", help="训练数据预处理后保存和加载的路径")
    parser.add_argument("--val_processed_path", type=str, default="./processed_data/val_dataset", help="验证数据预处理后保存和加载的路径")
    
    # 训练参数
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="评估时的batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--eval_accumulation_steps", type=int, default=2, help="评估时的梯度累积步数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减参数")
    parser.add_argument("--transformer_learning_rate", type=float, default=1e-5, help="Transformer层的学习率")
    parser.add_argument("--hippo_learning_rate", type=float, default=5e-5, help="HippoModel的学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="学习率预热比例")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数，-1表示由epoch决定")
    
    # 混合精度和内存优化
    parser.add_argument("--fp16", action="store_true", default=True, help="是否启用混合精度训练(fp16)")
    parser.add_argument("--bf16", action="store_true", default=False, help="是否启用混合精度训练(bf16)")
    parser.add_argument("--max_split_size_mb", type=int, default=128, help="CUDA内存分割大小")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="是否启用梯度检查点")
    
    # 评估和日志
    parser.add_argument("--max_length", type=int, default=4096, help="最大序列长度")
    parser.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"], help="评估策略")
    parser.add_argument("--eval_steps", type=int, default=500, help="每多少步进行一次评估")
    parser.add_argument("--test_sample_ratio", type=float, default=1.0, help="测试集采样比例，0.0-1.0")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志打印步数")
    parser.add_argument("--save_steps", type=int, default=50, help="模型保存步数")
    parser.add_argument("--save_total_limit", type=int, default=3, help="最大保存检查点数量")
    
    # 训练控制
    parser.add_argument("--load_best_model", action="store_true", help="是否在训练结束时加载最佳模型")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从检查点恢复训练的路径")
    parser.add_argument("--metric_for_best_model", type=str, default="f1", help="用于选择最佳模型的指标")
    
    # DeepSpeed配置
    parser.add_argument("--deepspeed_zero_stage", type=int, default=2, choices=[0,1,2,3], help="DeepSpeed ZeRO优化阶段")
    parser.add_argument("--activation_checkpointing", action="store_true", default=True, help="是否启用激活检查点")
    
    return parser.parse_args()

# 自定义Trainer类，用于处理对话历史的传递和优化的训练流程
class CustomTrainer(Trainer):
    def __init__(self, *args, cmd_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cmd_args = cmd_args  # 保存命令行参数
        self.first_epoch_completed = False
        self.transformer_layers_unfrozen = False
        self.current_epoch = 0
        # 确保scaler属性存在，避免在prediction_step中引用时出错
        self.scaler = getattr(self, 'scaler', None)
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
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
        
        # 当gradient_accumulation_steps>1时，需要使用num_items_in_batch进行损失缩放
        if num_items_in_batch is not None and hasattr(self, "args") and self.args.gradient_accumulation_steps > 1:
            # 按批次中的实际项目数量缩放损失
            batch_size = inputs["input_ids"].shape[0]
            if num_items_in_batch < batch_size:
                loss = loss * (num_items_in_batch / batch_size)
        
        return (loss, outputs) if return_outputs else loss
    
    def _get_parameter_groups(self, model):
        """提取模型参数并按类型分组"""
        transformer_params = []
        hippo_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'hippo_model' in name or 'gate_mechanisms' in name:
                    hippo_params.append(param)
                elif 'model.layers' in name:
                    transformer_params.append(param)
        
        # 创建参数组
        param_groups = []
        transformer_lr = getattr(self.cmd_args, 'transformer_learning_rate', 1e-5)
        hippo_lr = getattr(self.cmd_args, 'hippo_learning_rate', 5e-5)
        
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': transformer_lr
            })
        
        if hippo_params:
            param_groups.append({
                'params': hippo_params,
                'lr': hippo_lr
            })
        
        return transformer_params, hippo_params, param_groups
    
    def create_optimizer(self):
        """创建使用差异化学习率的优化器 - DeepSpeed会接管优化器创建"""
        return self.optimizer
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 获取预测的token IDs（取概率最大的token）
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
        
        # 创建掩码，只考虑非-100的位置
        mask = labels != -100
        
        # 计算准确率
        correct_predictions = (predictions == labels) & mask
        accuracy = correct_predictions.sum().item() / mask.sum().item()
        
        # 计算精确率、召回率、F1
        true_positives = correct_predictions.sum().item()
        predicted_positives = mask.sum().item()
        actual_positives = mask.sum().item()
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 解码文本并计算METEOR和Sentence-BERT分数
        tokenizer = AutoTokenizer.from_pretrained(self.cmd_args.model_name_or_path)
        meteor_scores = []
        sbert_similarities = []
        
        for pred, label, msk in zip(predictions, labels, mask):
            valid_pred_ids = pred[msk].tolist()
            valid_label_ids = label[msk].tolist()
            
            try:
                pred_text = tokenizer.decode(valid_pred_ids, skip_special_tokens=True)
                label_text = tokenizer.decode(valid_label_ids, skip_special_tokens=True)
                
                # 计算METEOR分数
                pred_tokens = pred_text.split()
                label_tokens = label_text.split()
                if len(pred_tokens) > 0 and len(label_tokens) > 0:
                    meteor = meteor_score([label_tokens], pred_tokens)
                    meteor_scores.append(meteor)
                
                # 计算Sentence-BERT语义相似度
                if pred_text.strip() and label_text.strip():
                    embeddings = sbert_model.encode([pred_text, label_text], convert_to_tensor=True)
                    similarity = cos_sim(embeddings[0], embeddings[1]).item()
                    sbert_similarities.append(similarity)
            except Exception as e:
                continue
        
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
        """重写评估方法"""
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        print(f"\n开始在测试集上进行评估 (Epoch {self.current_epoch})...")
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
        self.current_epoch = epoch + 1
        print(f"\n===== 开始训练第 {self.current_epoch} 个epoch =====")
        
        # 检查是否完成了第一个epoch
        if self.current_epoch >= 2 and not self.first_epoch_completed:
            self.first_epoch_completed = True
            self._unfreeze_transformer_layers(self.model)
            self._update_optimizer_parameters()
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时的处理"""
        self.current_epoch = epoch + 1
        print(f"\n第 {self.current_epoch} 个epoch训练完成")
        return logs
    
    def training_step(self, model, inputs, num_items_in_batch = None):
        """重写训练步骤，使用DeepSpeed进行训练"""
        dialog_histories = inputs.pop("dialog_histories", None)
        model.train()
        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            dialog_histories=dialog_histories
        )
        loss = outputs.loss
        
        if loss is not None:
            loss = loss.mean()
        
        # 定期清理GPU缓存
        if self.state.global_step % 100 == 0:
            torch.cuda.empty_cache()
        
        return loss.detach()
        
    def _unfreeze_transformer_layers(self, model):
        """解冻需要训练的transformer层"""
        if hasattr(model, 'advance_training_stage'):
            print(f"\n第一个epoch训练完成，准备解冻需要训练的transformer层...")
            stage_info = model.advance_training_stage()
            print(stage_info)
            self.transformer_layers_unfrozen = True
    
    def _update_optimizer_parameters(self):
        """更新优化器参数组"""
        unfrozen_layer_indices = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'model.layers' in name:
                try:
                    layer_idx = int(name.split('model.layers.')[1].split('.')[0])
                    unfrozen_layer_indices.add(layer_idx)
                except (IndexError, ValueError):
                    pass
        
        print(f"当前解冻的Transformer层索引: {sorted(unfrozen_layer_indices)}")
        
        transformer_params, hippo_params, new_optimizer_grouped_parameters = self._get_parameter_groups(self.model)
        
        if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'param_groups'):
            self.optimizer.param_groups.clear()
            for param_group in new_optimizer_grouped_parameters:
                self.optimizer.add_param_group(param_group)
            
            transformer_lr = getattr(self.cmd_args, 'transformer_learning_rate', 1e-5)
            hippo_lr = getattr(self.cmd_args, 'hippo_learning_rate', 5e-5)
            print(f"\n更新后的参数组信息：")
            print(f"- 解冻的Transformer层参数数量: {sum(p.numel() for p in transformer_params):,}")
            print(f"- HippoModel和门控机制参数数量: {sum(p.numel() for p in hippo_params):,}")
            print(f"- Transformer层学习率: {transformer_lr}")
            print(f"- HippoModel学习率: {hippo_lr}")
        else:
            print("警告: 优化器尚未创建或无效，无法更新参数组")
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """用于推理阶段，添加内存优化"""
        dialog_histories = inputs.pop("dialog_histories", None)
        
        with torch.no_grad():
            if self.scaler is not None:
                with autocast():
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=inputs["labels"] if "labels" in inputs else None,
                        dialog_histories=dialog_histories
                    )
            else:
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"] if "labels" in inputs else None,
                    dialog_histories=dialog_histories
                )
        
        loss = outputs.loss
        logits = outputs.logits
        
        if prediction_loss_only:
            return (loss, None, None)
        
        return (loss, logits, inputs["labels"] if "labels" in inputs else None)

def get_deepspeed_config(args, train_dataset_size):
    """创建DeepSpeed配置，确保与命令行参数一致"""
    # 从args中提取关键参数
    batch_size = args.per_device_train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    transformer_learning_rate = args.transformer_learning_rate
    hippo_learning_rate = args.hippo_learning_rate
    weight_decay = args.weight_decay
    warmup_ratio = args.warmup_ratio
    max_steps = args.max_steps
    use_fp16 = args.fp16
    use_bf16 = args.bf16
    adam_betas = (0.9, 0.999)
    adam_epsilon = 1e-8
    
    # 估算总步数
    if max_steps > 0:
        total_steps = max_steps
    else:
        total_steps = train_dataset_size // (batch_size * gradient_accumulation_steps) * num_train_epochs
    
    # 计算预热步数
    warmup_steps = int(warmup_ratio * total_steps) if warmup_ratio > 0 else 0
    
    # 总训练批量大小
    train_batch_size = batch_size * gradient_accumulation_steps
    
    return {
        "train_batch_size": train_batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "fp16": {
            "enabled": use_fp16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "bf16": {
            "enabled": use_bf16
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": hippo_learning_rate,
                "betas": [adam_betas[0], adam_betas[1]],
                "eps": adam_epsilon,
                "weight_decay": weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": hippo_learning_rate,
                "warmup_num_steps": warmup_steps
            }
        },
        "zero_optimization": {
            "stage": args.deepspeed_zero_stage,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "activation_checkpointing": {
            "partition_activations": False,
            "cpu_checkpointing": args.activation_checkpointing,
            "profile": False
        },
        "gradient_clipping": 1.0,
        "steps_per_print": args.logging_steps,
        "wall_clock_breakdown": False,
        "communication_data_type": "fp16",
        "zero_allow_untested_optimizer": True
    }

def main():
    args = parse_args()
    
    # 设置CUDA内存优化参数
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"expandable_segments:True,max_split_size_mb:{args.max_split_size_mb}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    train_dataset, test_dataset = load_train_val_data(
        train_path=args.data_path,
        val_path=args.test_data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_processed_path=args.train_processed_path,
        val_processed_path=args.val_processed_path
    )
    
    # 采样测试集
    if args.test_sample_ratio < 1.0:
        import random
        random.seed(42)
        sample_size = int(len(test_dataset) * args.test_sample_ratio)
        test_dataset = random.sample(test_dataset, sample_size)
        print(f"已对测试集进行采样，采样比例: {args.test_sample_ratio}, 采样后大小: {len(test_dataset)}")
    
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    
    # 初始化模型
    if args.resume_from_checkpoint is not None:
        print(f"从检查点恢复训练: {args.resume_from_checkpoint}")
        model = ModifiedQwen.from_pretrained(args.resume_from_checkpoint)
        print("已从检查点加载模型")
    else:
        print("从零开始训练新模型")
        model = ModifiedQwen(base_model_name_or_path=args.model_name_or_path)
        model.freeze_all()
        model.unfreeze_hippo_model()
        print("初始状态设置：只训练Hippo模型和门控机制")
    
    # 自定义数据收集器
    class CustomDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            filtered_features = []
            for feature in features:
                filtered_feature = {
                    'input_ids': feature['input_ids'],
                    'attention_mask': feature['attention_mask'],
                    'labels': feature['labels'],
                    'dialog_histories': feature['dialog_histories']
                }
                filtered_features.append(filtered_feature)
            return super().__call__(filtered_features)
    
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 生成DeepSpeed配置文件
    deepspeed_config = get_deepspeed_config(args, len(train_dataset))
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        json.dump(deepspeed_config, f, indent=2)
        deepspeed_config_path = f.name
    
    # 计算预热步数
    if args.warmup_ratio > 0:
        if args.max_steps > 0:
            total_steps = args.max_steps
        else:
            total_steps = len(train_dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps) * args.num_train_epochs
        warmup_steps = int(args.warmup_ratio * total_steps)
    else:
        warmup_steps = 0
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.hippo_learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_steps=args.max_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16_full_eval=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        eval_accumulation_steps=args.eval_accumulation_steps,
        load_best_model_at_end=args.load_best_model,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=deepspeed_config_path  # 使用配置文件路径而非字典
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
            elif 'transformer.h' in name or 'model.layers' in name:
                transformer_params.append(param)
    
    print(f"初始训练参数配置：")
    print(f"- Hippo模型参数数量: {sum(p.numel() for p in hippo_params):,}")
    print(f"- 门控机制参数数量: {sum(p.numel() for p in gate_params):,}")
    print(f"- 解冻的Transformer层参数数量: {sum(p.numel() for p in transformer_params):,}")
    
    # 准备参数组
    def get_parameter_groups_with_different_lr(model, args):
        transformer_params = []
        hippo_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'hippo_model' in name or 'gate_mechanisms' in name:
                    hippo_params.append(param)
                elif 'model.layers' in name or 'transformer.h' in name:
                    transformer_params.append(param)
        
        param_groups = []
        if transformer_params:
            param_groups.append({
                'params': transformer_params,
                'lr': args.transformer_learning_rate
            })
        
        if hippo_params:
            param_groups.append({
                'params': hippo_params,
                'lr': args.hippo_learning_rate
            })
        
        if not param_groups:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad]}]
        
        return param_groups
    
    # 初始化分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9994'
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=get_parameter_groups_with_different_lr(model, args),
        config=deepspeed_config_path,  # 使用配置文件路径
        dist_init_required=True
    )
    
    # 初始化自定义Trainer
    trainer = CustomTrainer(
        model=model_engine.module if hasattr(model_engine, 'module') else model_engine,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        compute_metrics=CustomTrainer.compute_metrics,
        cmd_args=args
    )
    
    trainer.model_engine = model_engine
    
    # 开始训练
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # 保存模型
        torch.cuda.empty_cache()
        try:
            if hasattr(trainer, 'model_engine'):
                model_to_save = trainer.model_engine.module if hasattr(trainer.model_engine, 'module') else trainer.model_engine
            else:
                model_to_save = trainer.model
                if hasattr(model_to_save, 'module'):
                    model_to_save = model_to_save.module
            
            model_to_save.to('cpu')
            torch.cuda.empty_cache()
            
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"模型已成功保存到 {args.output_dir}")
            
            try:
                if hasattr(trainer, 'model_engine'):
                    trainer.model_engine.save_checkpoint(args.output_dir)
                    print(f"DeepSpeed检查点也已保存")
            except Exception as e:
                print(f"保存DeepSpeed检查点失败: {e}")
        except Exception as e:
            print(f"保存模型时出错: {e}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n内存不足错误！尝试以下解决方案：")
            print("1. 减小batch size")
            print("2. 增加gradient_accumulation_steps")
            print("3. 启用gradient_checkpointing")
            print("4. 使用更小的模型")
            print("5. 检查数据是否有超长样本")
            print(f"6. 尝试使用更高的ZeRO阶段（当前为ZeRO-{args.deepspeed_zero_stage}）")
        raise e
    finally:
        # 清理临时配置文件
        if os.path.exists(deepspeed_config_path):
            os.remove(deepspeed_config_path)

if __name__ == "__main__":
    main()