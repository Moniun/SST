import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from hippo_model import HippoModel

class ModifiedQwen(nn.Module):
    """包装Qwen3-8B并插入自定义模块"""
    def __init__(self, base_model_name_or_path):
        super().__init__()
        # 加载基础模型（4-bit量化）
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        custom_cache_dir = "qwen3-8b-custom-module-training"  # 替换为你的目标路径
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=custom_cache_dir  # 关键参数：指定自定义目录
        )
        
        # 获取模型配置
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        
        # 初始化自定义模块
        self.hippo_model = HippoModel(output_dim = self.hidden_size)
        
        # 定位目标MLP层（最后一个Transformer层的MLP）
        self.target_mlp_layer = self.base_model.transformer.h[-1]
        
        # 冻结基础模型参数
        self.freeze_base_model()
        # 解冻需要训练的层
        self.unfreeze_trainable_layers()

    def freeze_base_model(self):
        """冻结除目标MLP外的所有基础模型参数"""
        for name, param in self.base_model.named_parameters():
            # 只保留最后一层的MLP参数可训练
            if not name.startswith("transformer.h.-1.mlp"):
                param.requires_grad = False

    def unfreeze_trainable_layers(self):
        """确保自定义模块和目标MLP层可训练"""
        # 自定义模块参数
        for param in self.custom_module.parameters():
            param.requires_grad = True
        # 目标MLP层参数
        for param in self.target_mlp_layer.mlp.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取基础模型输出（包含隐藏状态）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )
        
        # 获取最后一层Transformer的输出（MLP的原始输入）
        last_hidden = outputs.hidden_states[-1]
        
        # 应用自定义模块
        hippo_output = self.hippo_model(output_dim = last_hidden)
        
        # 合并原始输入和自定义模块输出（残差连接）
        merged_input = last_hidden + hippo_output
        
        # 重新计算MLP输出（替换原始计算）
        ln_output = self.target_mlp_layer.ln_2(last_hidden)  # 原始LayerNorm
        mlp_output = self.target_mlp_layer.mlp(merged_input)  # 使用新输入
        final_output = ln_output + mlp_output  # 残差连接
        
        # 计算logits
        logits = self.base_model.lm_head(final_output)
        
        # 构造输出对象
        return type(outputs)(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def generate(self, *args, **kwargs):
        """包装生成函数，保持原始接口"""
        return self.base_model.generate(*args, **kwargs)
    