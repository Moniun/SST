import torch
from transformers import AutoTokenizer
from model_customization import ModifiedQwen
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen3-8b-custom-finetuned")
    parser.add_argument("--prompt", type=str, default="能介绍一下关于电影推荐的内容吗？")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = ModifiedQwen(base_model_name_or_path=args.model_path)
    model.eval()  # 切换到评估模式
    
    # 构建输入
    input_text = f"user: {args.prompt}\nassistant:"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码并打印结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果:")
    print(response.replace(input_text, "").strip())

if __name__ == "__main__":
    main()
    