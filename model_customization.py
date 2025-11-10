import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from hippo_model import HippoModel

class ModifiedQwen(nn.Module):
    """包装Qwen3-8B并插入单个HippoModel，在transformer层间进行门控融合"""
    def __init__(self, base_model_name_or_path, fusion_layers=None, cache_dir="qwen3-8b-custom-module-training"):
        super().__init__()
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name_or_path,
            trust_remote_code=True
        )
        
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16  # 使用bfloat16以减少显存占用
        )
        
        # 获取模型配置
        self.config = self.base_model.config
        self.hidden_size = self.config.hidden_size
        
        self.num_transformer_layers = self.config.num_hidden_layers
        
        # 默认融合层位置（均匀分布在模型层中）
        self.fusion_layers = fusion_layers if fusion_layers is not None else [0]
        
        # 添加融合层索引越界校验
        for layer_idx in self.fusion_layers:
            if not (0 <= layer_idx < self.num_transformer_layers):
                raise ValueError(
                    f"融合层索引{layer_idx}超出有效范围（0到{self.num_transformer_layers-1}）"
                )
        
        # 初始化单个Hippo模型，输入为第一层transformer的输入
        # 移除硬编码设备分配，让HippoModel自动匹配基础模型的设备
        self.hippo_model = HippoModel(
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            hippo_scale=0.1  # 添加缩放因子，提高训练稳定性
        ).to(self.base_model.device)  # 确保Hippo模型与基础模型在同一设备上
        
        # 初始化门控机制字典，为每个融合位置创建一个门控
        self.gate_mechanisms = nn.ModuleDict()
        
        # 为每个融合层创建门控机制
        for layer_idx in self.fusion_layers:
            # 创建门控机制 (使用sigmoid激活的线性层)
            self.gate_mechanisms[f"layer_{layer_idx}"] = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
        
        # 初始化门控机制参数
        self._initialize_gate_mechanisms()
        
        # 保存当前训练阶段
        self.current_stage = 0  # 0: 仅训练hippo模型, 1: 微调transformer相邻层
        
        # 冻结所有参数
        self.freeze_all()
        
        # 初始阶段：只解冻hippo模型
        self.unfreeze_hippo_model()

        # 初始化隐藏状态,用在推理时
        self.hidden_h = self.hippo_model.reset_h(batch_size=1)

    def freeze_all(self):
        """冻结所有参数"""
        # 冻结基础模型参数
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # 冻结Hippo模型参数
        for param in self.hippo_model.parameters():
            param.requires_grad = False
        
        # 冻结门控机制参数
        for param in self.gate_mechanisms.parameters():
            param.requires_grad = False
    
    def unfreeze_hippo_model(self):
        """解冻Hippo模型和门控机制参数"""
        for param in self.hippo_model.parameters():
            param.requires_grad = True
        for param in self.gate_mechanisms.parameters():
            param.requires_grad = True
    
    def unfreeze_adjacent_layers(self):
        """解冻与融合位置相邻的Transformer层参数"""
        for layer_idx in self.fusion_layers:
            # 解冻当前层
            if 0 <= layer_idx < len(self.base_model.transformer.h):
                for param in self.base_model.transformer.h[layer_idx].parameters():
                    param.requires_grad = True
            
            # 解冻下一层（如果存在）
            if layer_idx + 1 < len(self.base_model.transformer.h):
                for param in self.base_model.transformer.h[layer_idx + 1].parameters():
                    param.requires_grad = True
    
    def advance_training_stage(self):
        """进入下一训练阶段：Hippo模型和门控机制始终一起训练，只有transformer层在后续阶段解冻"""
        if self.current_stage == 0:
            # 冻结所有参数
            self.freeze_all()
            
            # 先解冻Hippo模型和门控机制（始终一起训练）
            self.unfreeze_hippo_model()
            # 然后解冻相邻的Transformer层
            self.unfreeze_adjacent_layers()
            
            self.current_stage = 1
            return f"已进入训练阶段1：同时训练Hippo模型、门控机制和相邻的Transformer层"
        else:
            return f"已经在最后训练阶段（阶段{self.current_stage}）"
    
    def _initialize_gate_mechanisms(self):
        """初始化门控机制参数，使门控偏向基础模型输出（初始阶段稳定训练）"""
        for gate_module in self.gate_mechanisms.values():
            linear_layer = gate_module[0]  # 获取第一个线性层
            
            # 门控机制初始化为偏向基础模型输出（权重较小）
            # 使用 Xavier 初始化，但降低权重方差
            nn.init.xavier_uniform_(linear_layer.weight, gain=0.1)  # 较小的gain，使初始输出接近0
            
            # 偏置初始化为负值，使sigmoid输出接近0，优先使用基础模型输出
            if linear_layer.bias is not None:
                nn.init.constant_(linear_layer.bias, -2.0)
    
    def get_trainable_params_info(self):
        """获取当前可训练参数信息"""
        trainable_params = 0
        total_params = 0
        
        # 计算基础模型可训练参数
        base_trainable = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
        base_total = sum(p.numel() for p in self.base_model.parameters())
        
        # 计算Hippo模型可训练参数
        hippo_trainable = sum(p.numel() for p in self.hippo_model.parameters() if p.requires_grad)
        hippo_total = sum(p.numel() for p in self.hippo_model.parameters())
        
        # 计算门控机制可训练参数
        gate_trainable = sum(p.numel() for p in self.gate_mechanisms.parameters() if p.requires_grad)
        gate_total = sum(p.numel() for p in self.gate_mechanisms.parameters())
        
        trainable_params = base_trainable + hippo_trainable + gate_trainable
        total_params = base_total + hippo_total + gate_total
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_ratio": trainable_params / total_params,
            "base_model_trainable": base_trainable,
            "hippo_model_trainable": hippo_trainable,
            "gate_mechanisms_trainable": gate_trainable,
            "current_stage": self.current_stage
        }

    def forward(self, input_ids=None, attention_mask=None, labels=None, dialog_histories=None, h_initial = None):
        """
        前向传播方法，使用单个HippoModel处理第一层输入，并在指定层间进行门控融合
        
        参数:
            input_ids: 大模型输入ID
            attention_mask: 注意力掩码
            labels: 标签（用于损失计算）
            dialog_histories: 对话历史列表（可选）
        """
        # 检查当前是否处于训练状态
        is_training = self.training
        
        # 获取嵌入层输出和位置编码
        # 嵌入层通常是冻结的，在任何状态下使用no_grad
        with torch.no_grad():
            hidden_states = self.base_model.transformer.wte(input_ids)
            batch_size, seq_length = input_ids.shape
            past_key_values_length = 0
            
            # 应用位置编码
            hidden_states = hidden_states + self.base_model.transformer.wpe(input_ids)

        # 当dialog_histories不为None时，使用对话历史更新HippoModel的隐藏状态
        if dialog_histories is not None and len(dialog_histories) > 0:
            # 证明模型是在训练
            is_training = True
            
            # 初始化隐藏状态
            h_initial = self.hippo_model.reset_h(batch_size)
            
            # 遍历对话历史中的每一句话，更新隐藏状态
            # dialog_histories是一个包含每轮对话历史的列表，每个元素是(batch_size, seq_len)的tensor
            for history_batch in dialog_histories:
                if history_batch is not None and history_batch.numel() > 0:
                    # 确保对话历史是二维的(batch_size, seq_len)格式
                    if len(history_batch.shape) == 1:
                        # 如果是一维，添加batch维度
                        history_batch = history_batch.unsqueeze(0)
                    
                    # 对话历史的嵌入转换和Hippo模型更新都不计算梯度
                    # 只需要获取隐藏状态h，不需要更新Hippo模型参数
                    with torch.no_grad():
                        history_embeds = self.base_model.transformer.wte(history_batch)
                        history_embeds = history_embeds + self.base_model.transformer.wpe(history_batch)
                        
                        # 使用对话历史更新HippoModel的隐藏状态
                        _, h_initial = self.hippo_model(history_embeds, h_initial)
                        self.hidden_h = h_initial

        # 使用Hippo模型处理第一层transformer的输入，传入更新后的隐藏状态
        # Hippo模型在训练时需要计算梯度，推理时不需要
        with torch.set_grad_enabled(is_training and not self._is_hippo_frozen()):
            hippo_output, self.hidden_h = self.hippo_model(hidden_states, self.hidden_h)
        
        # 初始化past_key_values存储
        past_key_values = []
        all_hidden_states = [hidden_states]
        all_attentions = []
        
        # 处理每一层transformer
        for layer_idx, layer in enumerate(self.base_model.transformer.h):
            # 检查当前层是否冻结
            is_layer_frozen = self._is_layer_frozen(layer_idx)
            
            # 对冻结的层在非训练状态下使用no_grad来节省显存
            with torch.set_grad_enabled(is_training and not is_layer_frozen):
                # 执行原始transformer层计算
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=is_training,  # 训练时才保存注意力权重，进一步节省显存
                    use_cache=True
                )
                
                # 更新hidden_states为当前层输出
                hidden_states = layer_outputs[0]
                past_key_value = layer_outputs[2]
                
                # 保存中间结果
                past_key_values.append(past_key_value)
                all_hidden_states.append(hidden_states)
                
                # 只在训练时保存注意力权重
                if is_training:
                    all_attentions.append(layer_outputs[1])
            
            # 检查当前层是否是需要融合Hippo输出的位置
            if layer_idx in self.fusion_layers:
                # 门控机制是否冻结
                is_gate_frozen = self._is_gate_frozen(layer_idx)
                
                with torch.set_grad_enabled(is_training and not is_gate_frozen):
                    # 获取当前层的门控机制
                    gate_mechanism = self.gate_mechanisms[f"layer_{layer_idx}"]
                    
                    # 计算门控权重
                    gate_input = torch.cat([hidden_states, hippo_output], dim=-1)
                    gate_weight = gate_mechanism(gate_input)
                    
                    # 门控加权融合
                    hidden_states = gate_weight * hippo_output + (1 - gate_weight) * hidden_states
        
        # 应用最终的层归一化
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        
        # 计算logits
        logits = self.base_model.lm_head(hidden_states)
        
        # 构造输出对象（已在顶部导入）
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values if self.base_model.transformer.use_cache else None,
            hidden_states=tuple(all_hidden_states),
            attentions=tuple(all_attentions)
        )
    
    def _is_hippo_frozen(self):
        """检查Hippo模型是否被冻结"""
        return not any(p.requires_grad for p in self.hippo_model.parameters())
    
    def _is_layer_frozen(self, layer_idx):
        """检查指定Transformer层是否被冻结"""
        if 0 <= layer_idx < len(self.base_model.transformer.h):
            return not any(p.requires_grad for p in self.base_model.transformer.h[layer_idx].parameters())
        return True
    
    def _is_gate_frozen(self, layer_idx):
        """检查指定层的门控机制是否被冻结"""
        if f"layer_{layer_idx}" in self.gate_mechanisms:
            return not any(p.requires_grad for p in self.gate_mechanisms[f"layer_{layer_idx}"].parameters())
        return True
    
    def generate(self, *args, **kwargs):
        """包装生成函数，确保使用修改后的前向传播逻辑，推理时使用no_grad节省显存"""
        # 设置为评估模式，确保所有冻结层都不计算梯度
        self.eval()
        
        # 在no_grad上下文中进行推理，大幅减少显存占用
        with torch.no_grad():
            # 调用基础模型的generate方法，它会使用我们重写的forward方法
            return self.base_model.generate(*args, **kwargs)
    
    def save_pretrained(self, save_directory):
        """自定义保存方法，确保所有子模块参数都能正确保存"""
        import os
        from transformers.trainer_utils import set_seed
        
        # 创建保存目录
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存基础模型
        self.base_model.save_pretrained(save_directory)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_directory)
        
        # 保存自定义模块参数
        custom_state_dict = {
            'hippo_model': self.hippo_model.state_dict(),
            'gate_mechanisms': self.gate_mechanisms.state_dict(),
            'fusion_layers': self.fusion_layers,
            'current_stage': self.current_stage
        }
        torch.save(custom_state_dict, os.path.join(save_directory, 'custom_modules.bin'))
        
        print(f"模型已保存至: {save_directory}")
        print(f"- 基础模型参数")
        print(f"- Hippo模型参数")
        print(f"- 门控机制参数")
        print(f"- 模型配置和分词器")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """自定义加载方法，确保所有子模块参数都能正确加载"""
        import os
        
        # 从kwargs中提取fusion_layers和cache_dir
        fusion_layers = kwargs.pop('fusion_layers', None)
        cache_dir = kwargs.pop('cache_dir', pretrained_model_name_or_path)
        
        # 初始化模型实例
        model = cls(
            base_model_name_or_path=pretrained_model_name_or_path,
            fusion_layers=fusion_layers,
            cache_dir=cache_dir,
            **kwargs
        )
        
        # 加载自定义模块参数
        custom_modules_path = os.path.join(pretrained_model_name_or_path, 'custom_modules.bin')
        if os.path.exists(custom_modules_path):
            # 优先使用GPU加载模型参数，如果可用的话
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            custom_state_dict = torch.load(custom_modules_path, map_location=device)
            
            # 加载Hippo模型参数
            model.hippo_model.load_state_dict(custom_state_dict['hippo_model'])
            
            # 加载门控机制参数
            model.gate_mechanisms.load_state_dict(custom_state_dict['gate_mechanisms'])
            
            # 恢复其他配置
            model.fusion_layers = custom_state_dict['fusion_layers']
            model.current_stage = custom_state_dict['current_stage']

            # 重置隐藏状态为初始值，确保模型加载后状态正确
            model.hidden_h = model.hippo_model.reset_h(batch_size=1)
            
            print(f"成功加载自定义模块参数")
            print(f"- Hippo模型参数")
            print(f"- 门控机制参数")
            print(f"- 当前训练阶段: {model.current_stage}")
        else:
            print(f"警告: 未找到自定义模块参数文件 {custom_modules_path}")
        
        return model

# 使用示例
if __name__ == "__main__":
    model = ModifiedQwen(
        base_model_name_or_path="Qwen/Qwen3-8B",
        fusion_layers=[0],
        cache_dir="qwen3-8b-custom-module-training"  # 可自定义路径
    )
    
    # 打印初始训练阶段信息
    print("初始训练参数信息:")
    params_info = model.get_trainable_params_info()
    print(f"当前阶段: {params_info['current_stage']}")
    print(f"可训练参数比例: {params_info['trainable_ratio']:.6f}")
    print(f"Hippo模型可训练参数: {params_info['hippo_model_trainable']:,}")
    print(f"门控机制可训练参数: {params_info['gate_mechanisms_trainable']:,}")
    
    # 模拟进入下一训练阶段
    stage_info = model.advance_training_stage()
    print(f"\n{stage_info}")
    
    # 打印新阶段参数信息
    params_info = model.get_trainable_params_info()
    print(f"当前阶段: {params_info['current_stage']}")
    print(f"可训练参数比例: {params_info['trainable_ratio']:.6f}")
    print(f"基础模型可训练参数: {params_info['base_model_trainable']:,}")
    print(f"Hippo模型可训练参数: {params_info['hippo_model_trainable']:,}")
    print(f"门控机制可训练参数: {params_info['gate_mechanisms_trainable']:,}")
    
    print("\n模型初始化完成，准备开始训练")