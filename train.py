import torch
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from model_customization import ModifiedQwen
from data_processor import load_and_preprocess_data
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-8B-Chat")
    parser.add_argument("--data_path", type=str, default="demo_training_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="./qwen3-8b-custom-finetuned")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    # 加载并预处理数据
    train_dataset = load_and_preprocess_data(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    
    # 初始化带自定义模块的模型
    model = ModifiedQwen(base_model_name_or_path=args.model_name_or_path)
    
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,  # 学习率预热
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,  # 最多保存3个检查点
        fp16=True,  # 混合精度训练
        optim="paged_adamw_8bit",  # 适配4-bit量化
        report_to="none",  # 不使用wandb等日志工具
        remove_unused_columns=False,
        label_names=["labels"]
    )
    
    # 自定义Trainer类，用于处理对话历史的传递
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            # 从输入中提取对话历史
            dialog_histories = inputs.pop("dialog_histories", None)
            
            # 调用模型的forward方法，传入对话历史
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
                dialog_histories=dialog_histories
            )
            
            # 计算损失
            loss = outputs.loss
            
            return (loss, outputs) if return_outputs else loss
    
    # 初始化自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    print(f"训练完成，模型已保存至: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
    