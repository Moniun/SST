import torch
from model_customization import ModifiedQwen
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./qwen3-8b-custom-finetuned")
    parser.add_argument("--prompt", type=str, default="能介绍一下关于电影推荐的内容吗？")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--check_params", action="store_true", help="验证模型参数完整性")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载模型（使用自定义的from_pretrained方法确保所有参数正确加载）
    print(f"正在从 {args.model_path} 加载模型...")
    model = ModifiedQwen.from_pretrained(args.model_path)
    model.eval()  # 切换到评估模式
    
    # 获取分词器（已在from_pretrained中加载）
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # 验证参数完整性（可选）
    if args.check_params:
        print("\n验证模型参数完整性...")
        verify_model_parameters(model)
        print("验证完成！")
    
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

def verify_model_parameters(model):
    """验证模型自定义参数是否已正确加载"""
    # 检查hippo_model参数
    if hasattr(model, 'hippo_model'):
        param_values = []
        for param in model.hippo_model.parameters():
            param_values.extend(param.data.cpu().numpy().flatten())
        # 检查参数是否全为零或初始随机值（简单验证）
        param_mean = sum(param_values) / len(param_values) if param_values else 0
        print(f"  - Hippo模型参数平均值: {param_mean:.6f}")
    
    # 检查gate_mechanisms参数
    if hasattr(model, 'gate_mechanisms'):
        total_gates = len(model.gate_mechanisms)
        non_zero_gates = 0
        for gate in model.gate_mechanisms:
            for param in gate.parameters():
                if not torch.all(param.data == 0):
                    non_zero_gates += 1
                    break
        print(f"  - 门控机制参数状态: {non_zero_gates}/{total_gates} 个门控有非零参数")
    
    # 检查融合层参数
    if hasattr(model, 'fusion_layers'):
        param_values = []
        for layer in model.fusion_layers:
            for param in layer.parameters():
                param_values.extend(param.data.cpu().numpy().flatten())
        param_mean = sum(param_values) / len(param_values) if param_values else 0
        print(f"  - 融合层参数平均值: {param_mean:.6f}")

if __name__ == "__main__":
    main()
    