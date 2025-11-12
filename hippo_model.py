import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class HippoModel(nn.Module):
    def __init__(self, 
                 input_dim: int = 4096,  # 输入维度（通常为大模型隐藏层维度）
                 hidden_dim: int = 16,   # 隐藏状态维度
                 hippo_type: str = "LegS",  # Hippo矩阵类型
                 ffn_dim: int = 4096*4,     # 前馈网络中间层维度
                 output_dim: int = 4096,    # 输出维度
                 hippo_scale: float = 1.0): # Hippo矩阵缩放因子，默认为1.0（不缩放）
        """
        Hippo模型的基础实现，基于选择性状态空间模型(SSM)的序列建模
        
        参数:
            input_dim: 输入向量的维度
            hidden_dim: 隐藏状态h的维度
            hippo_type: Hippo矩阵类型，支持'LegT', 'LagT', 'LegS'
            ffn_dim: 前馈网络中间层维度
            output_dim: 最终输出维度
            hippo_scale: Hippo矩阵的缩放因子，控制状态更新速率。即使矩阵固定，适当的缩放
                        仍有助于提高数值稳定性和任务适应性。默认为1.0（保持原始值）
        """
        super().__init__()
        # 移除硬编码device，让模型可以自然地跟随pytorch的to(device)方法
        
        # 核心参数
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hippo_type = hippo_type
        self.ffn_dim = ffn_dim  # 使用传入的ffn_dim参数
        self.output_dim = output_dim
        self.hippo_scale = hippo_scale
        
        # 初始化Hippo矩阵（A）：用于隐藏状态的时序更新
        # 使用原始Mamba实现方式，将A矩阵设为固定参数（不可学习）
        A_np = self._create_hippo_matrix(hidden_dim, hippo_type)
        # 应用缩放因子，调节状态更新速率和数值稳定性
        A_np = A_np * self.hippo_scale
        # 使用register_buffer将其注册为非参数张量，不会参与梯度更新
        self.register_buffer('A', torch.tensor(A_np, dtype=torch.float32))
        
        # 基础投影层，用于动态生成SSM参数
        self.b_proj = nn.Linear(input_dim, hidden_dim)
        self.c_proj = nn.Linear(input_dim, hidden_dim * output_dim)
        self.d_proj = nn.Linear(input_dim, output_dim)
        
        # GLU门控机制
        self.b_gate = nn.Linear(input_dim, hidden_dim)
        self.c_gate = nn.Linear(input_dim, hidden_dim * output_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, input_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # 自定义初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """自定义权重初始化方法，提高模型性能和稳定性"""
        # 投影层使用Kaiming初始化，适合ReLU/GELU激活函数
        for module in [self.b_proj, self.c_proj, self.d_proj, self.b_gate, self.c_gate]:
            if isinstance(module, nn.Linear):
                # Kaiming初始化，适合GELU等激活函数
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 前馈网络使用GPT风格的初始化
        for i, module in enumerate(self.ffn):
            if isinstance(module, nn.Linear):
                if i == 0:  # 第一层
                    # 缩放的Kaiming初始化，对于GELU使用linear作为近似
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                    # 对第一层进行缩放，提高训练稳定性
                    module.weight.data *= 1.0 / (self.input_dim ** 0.25)
                else:  # 第二层
                    # 输出层使用较小的初始化值
                    nn.init.normal_(module.weight, mean=0.0, std=2.0 / self.ffn_dim)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _create_hippo_matrix(self, n: int, type_: str) -> np.ndarray:
        """
        创建基础的Hippo对角矩阵A，用于SSM计算
        
        参数:
            n: 矩阵维度（与hidden_dim一致）
            type_: 多项式类型
        返回:
            对角向量
        """
        # 基础对角线初始化
        diag = np.zeros(n, dtype=np.float32)
        
        if type_ == 'LegT':
            # Legendre Type T
            for k in range(1, n+1):
                diag[k-1] = -np.pi * (k - 0.5)**2 / 4.0
        elif type_ == 'LegS':
            # Legendre Type S
            for k in range(1, n+1):
                diag[k-1] = -k**2 * np.pi**2 / 4.0
        elif type_ == 'LagT':
            # Laguerre多项式
            for k in range(1, n+1):
                diag[k-1] = -k * 2.0
        else:
            raise ValueError(f"不支持的Hippo矩阵类型: {type_}，可选类型: 'LegT', 'LagT', 'LegS'")
        
        # 确保对角线元素都是负数，保证数值稳定性
        diag = np.minimum(diag, -1e-6)
        
        return diag

    def reset_h(self, batch_size: int) -> torch.Tensor:
        """
        初始化批次隐藏状态
        
        参数:
            batch_size: 批次大小
        返回:
            形状为(batch_size, hidden_dim)的初始隐藏状态
        """
        # 获取模型当前设备
        device = next(self.parameters()).device
        # 使用小随机初始化替代全零初始化，有助于打破对称性
        # 使用标准差为0.01的正态分布初始化，提供微小的随机性
        return torch.randn(
            batch_size, self.hidden_dim,
            dtype=torch.float32, 
            device=device
        ) * 0.01

    def selective_scan(self, B: torch.Tensor, C: torch.Tensor, A: torch.Tensor, h_initial: torch.Tensor) -> torch.Tensor:
        """
        Mamba风格的选择性扫描操作 - 使用递推公式实现O(L)复杂度
        
        此实现基于Mamba论文中的状态空间模型原理，使用递推关系高效计算
        时间复杂度：O(L)，空间复杂度：O(1)（除输出存储外）
        
        参数:
            B: 形状为(batch_size, seq_len, hidden_dim)的输入矩阵
            C: 形状为(batch_size, seq_len, hidden_dim, output_dim)的输出矩阵
            A: 形状为(hidden_dim,)的对角向量
            h_initial: 形状为(batch_size, hidden_dim)的初始隐藏状态
        返回:
            形状为(batch_size, seq_len, output_dim)的输出序列
        """
        device = B.device
        batch_size, seq_len, hidden_dim = B.shape
        
        # 计算指数衰减因子
        delta = torch.exp(A).view(1, 1, hidden_dim)  # (1, 1, hidden_dim)
        
        # 初始化隐藏状态
        h_current = h_initial.clone()  # (batch, hidden_dim)
        
        # 用于存储所有时间步的输出
        outputs = torch.zeros(batch_size, seq_len, C.shape[-1], device=device)
        
        # 递推计算 - O(L)复杂度实现
        for i in range(seq_len):
            # 更新隐藏状态: h_i = delta * h_{i-1} + B_i
            h_current = delta.squeeze(1) * h_current + B[:, i]
            
            # 计算输出: y_i = C_i^T * h_i
            outputs[:, i] = torch.einsum('bh, bho -> bo', h_current, C[:, i])
        
        return outputs


    def forward(self, 
                batch_vectors: torch.Tensor, 
                h_initial: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mamba模型的前向传播
        
        参数:
            batch_vectors: 形状为(batch_size, seq_len, hidden_size)的张量
        返回:
            元组：(输出序列, 最终隐藏状态)
        """
        # 获取模型当前设备
        device = next(self.parameters()).device
        
        # 输入验证与设备强制对齐，避免设备不匹配错误
        if not isinstance(batch_vectors, torch.Tensor) or batch_vectors.dim() != 3:
            raise ValueError("输入必须是形状为(batch_size, seq_len, input_dim)的张量")
        batch_vectors = batch_vectors.to(device)  # 强制输入到模型设备
        batch_size, seq_len, _ = batch_vectors.size()
        
        # 支持流式隐藏状态初始化：允许传入上一轮的隐藏状态
        if h_initial is None:
            h_initial = self.reset_h(batch_size)
        else:
            # 验证隐藏状态形状，增强鲁棒性
            if h_initial.shape != (batch_size, self.hidden_dim):
                raise ValueError(f"h_initial形状应为({batch_size}, {self.hidden_dim})，实际为{h_initial.shape}")
            h_initial = h_initial.to(device, dtype=batch_vectors.dtype)  # 设备和类型对齐
        
        # 层归一化
        x = self.norm1(batch_vectors)
        
        # 生成SSM参数（带GLU门控）
        B = self.b_proj(x) * torch.sigmoid(self.b_gate(x))  # (batch, seq_len, hidden_dim)
        C_proj = self.c_proj(x) * torch.sigmoid(self.c_gate(x))  # (batch, seq_len, hidden_dim*output_dim)
        # 将C_proj拆分为矩阵形状，匹配selective_scan的输入要求
        C = C_proj.reshape(batch_size, seq_len, self.hidden_dim, self.output_dim)
        D = self.d_proj(x)  # (batch, seq_len, output_dim)
        
        # 执行选择性扫描（向量化实现）
        ssm_output = self.selective_scan(B, C, self.A, h_initial)
        ssm_output = ssm_output + D  # 结合偏置项
        
        # 残差连接
        x = batch_vectors + ssm_output
        
        # 前馈网络处理
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = x + ffn_out  # 残差连接
        
        # 计算最终隐藏状态（用于流式处理）
        if seq_len == 1:
            # 单步情况，需要重新计算隐藏状态，因为selective_scan不返回中间状态
            # 执行单步的递推计算
            delta = torch.exp(self.A).view(1, -1)  # (1, hidden_dim)
            last_h = delta * h_initial + B[:, 0]
        else:
            # 多步情况，使用矩阵指数计算更高效的方式
            # 计算指数衰减因子的幂次
            exp_A = torch.exp(self.A)
            # 计算h_initial的衰减
            h_initial_decay = h_initial * (exp_A ** seq_len)
            
            # 创建衰减因子的几何序列
            # 对于第i个时间步的B，其权重为exp_A^(seq_len-1-i)
            decay_factors = exp_A.view(1, 1, -1) ** (seq_len - 1 - torch.arange(seq_len, device=device)).view(1, -1, 1)
            
            # 计算所有B的加权和
            b_contribution = torch.sum(B * decay_factors, dim=1)
            
            last_h = h_initial_decay + b_contribution
        
        # 返回输出和最终隐藏状态，支持链式流式调用
        return x, last_h


# 使用示例
if __name__ == "__main__":
    # 初始化模型
    model = HippoModel()
    # 创建示例张量输入：批次大小为2，序列长度为3，输入维度为4096
    device = next(model.parameters()).device
    batch_vectors = torch.randn(2, 3, 4096, device=device)
    # 前向传播
    outputs, last_h = model(batch_vectors)
    print("输出形状:", outputs.shape)  # 应输出 torch.Size([2, 3, 4096])
    print("最终隐藏状态形状:", last_h.shape)  # 应输出 torch.Size([2, 16])
    print("Mamba模型基础功能已成功实现")