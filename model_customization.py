import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from hippo_model import HippoModel

class ModifiedQwen(nn.Module):
    """包装Qwen3-8B并插入自定义模块，支持对话历史记忆融合"""
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
        
        # 初始化Hippo模型，用于处理对话历史
        self.hippo_model = HippoModel(output_dim=self.hidden_size)
        
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
        """确保Hippo模型和目标MLP层可训练"""
        # Hippo模型参数
        for param in self.hippo_model.parameters():
            param.requires_grad = True
        # 目标MLP层参数
        for param in self.target_mlp_layer.mlp.parameters():
            param.requires_grad = True

    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None):
        """
        前向传播方法，实现对话历史记忆融合逻辑
        
        参数:
            input_ids: 大模型输入ID
            attention_mask: 注意力掩码
            labels: 标签（用于损失计算）
            dialog_histories: 对话历史列表，格式为[["句子1", "句子2"], ["句子3", "句子4", "句子5"], ...]
        """
        # 第一步：处理对话历史，通过HippoModel获取记忆表示
        if dialog_histories is not None and len(dialog_histories) > 0:
            # 将对话历史输入HippoModel，得到记忆表示
            # dialog_histories格式：[(batch_size,)]，每个元素是对话句子列表
            memory_representations = self.hippo_model(dialog_histories)
            # memory_representations形状: (batch_size, hidden_size)
        else:
            # 如果没有对话历史，使用零向量
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            memory_representations = torch.zeros(batch_size, self.hidden_size, 
                                               device=self.base_model.device, 
                                               dtype=torch.float32)
        
        # 第二步：将memory_query输入大模型，获取最后一层Transformer的输出
        if input_ids is not None:
            # 获取基础模型输出（包含隐藏状态）
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
            
            # 获取最后一层Transformer的输出（MLP的原始输入）
            last_hidden = outputs.hidden_states[-1]  # 形状: (batch_size, seq_len, hidden_size)
            
            # 第三步：在最后一个MLP层之前，融合HippoModel的输出
            # 将记忆表示扩展为序列长度，以便与每个位置的隐藏状态相加
            # (batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
            expanded_memory = memory_representations.unsqueeze(1).expand(-1, last_hidden.shape[1], -1)
            
            # 应用LayerNorm（使用模型现有的LayerNorm）
            ln_output = self.target_mlp_layer.ln_2(last_hidden)
            
            # 融合特征：将HippoModel的输出与MLP层的输入相加
            fused_input = ln_output + expanded_memory
            
            # 第四步：将融合后的输入送入MLP层
            mlp_output = self.target_mlp_layer.mlp(fused_input)
            
            # 应用残差连接
            final_output = last_hidden + mlp_output
            
            # 计算logits
            logits = self.base_model.lm_head(final_output)
            
            # 构造输出对象
            return type(outputs)(
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )
        else:
            # 如果只有对话历史输入，仅返回HippoModel的输出
            return memory_representations

    def generate(self, *args, **kwargs):
        """包装生成函数，保持原始接口"""
        return self.base_model.generate(*args, **kwargs)
    