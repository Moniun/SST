import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple

class HippoModel(nn.Module):
    def __init__(self, input_dim: int = 4096,
                 output_dim: int = 4096,
                 hidden_dim: int = 16,
                 seq_len: int = 1024,
                 ffn_dim: int = 16,
                 hippo_type: str = "LegS",
                 hippo_scale: float = 1.0,
                 dtype: torch.dtype = torch.float16):  # 保留 dtype 用于 A 矩阵缩放参考，但不用于模块初始化

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.hippo_type = hippo_type
        self.hippo_scale = hippo_scale
        self.model_dtype = dtype  # 仅作参考，不用于模块创建
    
        # 创建 Hippo 矩阵 A（始终用 float32 计算，避免数值问题）
        A_np = self._create_hippo_matrix(hidden_dim, hippo_type)
        A_np = A_np * self.hippo_scale
        # 注册为 float32 buffer，forward 时再转为输入 dtype
        self.register_buffer('A', torch.tensor(A_np, dtype=torch.float32))
    
        # 可学习参数：用 float32 初始化，训练时由 AMP 自动处理
        self.B = nn.Parameter(torch.randn(hidden_dim, seq_len) * 0.01)
        self.C = nn.Parameter(torch.randn(seq_len, hidden_dim) * 0.01)
    
        # D 网络：不要传 dtype！
        self.D = nn.Sequential(
            nn.Linear(seq_len, 16),
            nn.Linear(16, seq_len),
            nn.Sigmoid()
        )
    
        # FFN：同样不要传 dtype
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, output_dim)
        )
    
        # LayerNorm 也不传 dtype
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
    
        self._initialize_weights()
        
    def _init_weights(self, module):
        """统一权重初始化方法"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _initialize_weights(self):
        """自定义权重初始化"""
        self.D.apply(self._init_weights)
        self.ffn.apply(self._init_weights)
        nn.init.xavier_normal_(self.B.data)
        nn.init.xavier_normal_(self.C.data)
    
    def _create_hippo_matrix(self, n: int, type_: str) -> np.ndarray:
        """创建符合SSM稳定性要求的Hippo对角矩阵A"""
        diag = np.zeros(n, dtype=np.float32)
        
        if type_ == 'LegT':
            for k in range(1, n+1):
                raw_val = -np.pi * (k - 0.5)**2 / 4.0
                diag[k-1] = raw_val
        elif type_ == 'LegS':
            for k in range(1, n+1):
                raw_val = -k**2 * np.pi**2 / 4.0
                diag[k-1] = raw_val
        elif type_ == 'LagT':
            for k in range(1, n+1):
                raw_val = -k * 2.0
                diag[k-1] = raw_val
        else:
            raise ValueError(f"不支持的Hippo矩阵类型: {type_}")
        
        max_abs = np.max(np.abs(diag))
        if max_abs == 0:
            max_abs = 1e-6
        scale = 0.9 / max_abs
        diag = diag * scale
        diag = np.minimum(diag, -1e-6)
        return np.diag(diag)

    def reset_h(self, batch_size: int) -> torch.Tensor:
        device = next(self.parameters()).device
        # 返回 float32，forward 中会转为 input_dtype
        return torch.randn(batch_size, self.hidden_dim, self.input_dim, device=device) * 0.01

    def forward(self, batch_vectors: torch.Tensor,
                h_initial: Optional[torch.Tensor] = None,
                last_n_tokens: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    
        device = next(self.parameters()).device
        if not isinstance(batch_vectors, torch.Tensor) or batch_vectors.dim() != 3:
            raise ValueError("输入必须是形状为(batch_size, seq_len, input_dim)的张量")
    
        # 获取输入的实际 dtype（可能是 float16 或 float32）
        input_dtype = batch_vectors.dtype
        batch_vectors = batch_vectors.to(device)
    
        batch_size, seq_len, input_dim = batch_vectors.shape
    
        if last_n_tokens > 0 and last_n_tokens < seq_len:
            batch_vectors = batch_vectors[:, -last_n_tokens:, :]
            seq_len = last_n_tokens
    
        if h_initial is None:
            h_initial = self.reset_h(batch_size).to(device, dtype=input_dtype)
        else:
            h_initial = h_initial.to(device, dtype=input_dtype)
    
        x = self.norm1(batch_vectors.to(input_dtype))  # norm 自动适配
    
        # === 关键：将 A 转为当前计算 dtype ===
        A = self.A.to(dtype=input_dtype)  # 原来是 float32，现在转为 fp16 或 fp32
        B = self.B.to(dtype=input_dtype)
        C = self.C.to(dtype=input_dtype)
    
        hA = torch.matmul(A, h_initial)
        Bx = torch.matmul(B, x.to(input_dtype))
        h_new = hA + Bx
    
        Ch = torch.matmul(C, h_new)
    
        x_perm = x.permute(0, 2, 1)
        x_flat = x_perm.reshape(-1, seq_len)
        Dx_flat = self.D(x_flat.to(input_dtype))  # D 内部自动 cast
        Dx = Dx_flat.reshape(batch_size, input_dim, seq_len).permute(0, 2, 1)
    
        y = Ch + Dx
    
        x2 = self.norm2(y)
        ffn_out = self.ffn(x2)
        x_out = x2 + ffn_out
    
        return ffn_out, h_new