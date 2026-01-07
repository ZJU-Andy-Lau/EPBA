import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from typing import Tuple

# 假设 Rope 和 SelfAttentionBlock 已经实现在 model.encoder 中
from model.encoder import Rope, SelfAttentionBlock

class MotionTransformer(nn.Module):
    """
    基于 Transformer 的几何感知编码器 (Motion Transformer)
    用于替代原有的 CNN Encoder，通过全局交互捕捉复杂的仿射变换。
    """
    def __init__(self, 
                 input_dim: int, 
                 embed_dim: int = 128, 
                 num_layers: int = 4, 
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 1. Stem (特征投影)
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=8, num_channels=embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
        )
        
        # 2. Tokenization: [CLS] Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        
        # 3. RoPE 生成器 (复用 Encoder 中的 2D RoPE)
        # 每个 Head 的维度用于 RoPE 计算
        self.rope = Rope(d_model=embed_dim // num_heads)
        
        # 4. Transformer Layers
        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W] 拼接后的原始几何特征
        Returns:
            global_feature: [B, embed_dim] 聚合后的全局误差向量
        """
        B, _, H, W = x.shape
        
        # 1. Stem 投影
        x = self.stem(x) # [B, D, H, W]
        
        # 2. 生成标准的 2D Spatial RoPE
        # cos, sin: [1, 1, H*W, HeadDim]
        spatial_cos, spatial_sin = self.rope(x)
        
        # 3. 构建 Hybrid RoPE (针对 CLS Token)
        # CLS 部分强制为零旋转 (cos=1, sin=0)
        cls_cos = torch.ones(1, 1, 1, spatial_cos.shape[-1], device=x.device)
        cls_sin = torch.zeros(1, 1, 1, spatial_sin.shape[-1], device=x.device)
        
        # 拼接后的 RoPE: [1, 1, 1 + H*W, HeadDim]
        hybrid_cos = torch.cat([cls_cos, spatial_cos], dim=2)
        hybrid_sin = torch.cat([cls_sin, spatial_sin], dim=2)
        
        # 4. Tokenization
        # Flatten: [B, D, H, W] -> [B, H*W, D]
        x_flat = x.flatten(2).transpose(1, 2)
        
        # 插入 CLS Token: [B, 1 + H*W, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_flat), dim=1)
        
        # 5. Transformer 迭代
        for layer in self.layers:
            x_seq = layer(x_seq, hybrid_cos, hybrid_sin)
            
        # 6. 提取归一化后的 CLS Token
        x_seq = self.norm(x_seq)
        global_feature = x_seq[:, 0] # [B, embed_dim]
        
        return global_feature

class GRUBlock(nn.Module):
    def __init__(self, 
                 corr_levels=2, 
                 corr_radius=4, 
                 context_dim=128, 
                 hidden_dim=128,
                 use_mtf=True,
                 use_gru=True):
        """
        重构后的 GRU 迭代更新模块 (Transformer-based)
        """
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.use_mtf = use_mtf
        self.use_gru = use_gru

        # 1. 计算输入通道数
        self.corr_dim = corr_levels * (2 * corr_radius + 1)**2
        self.offset_dim = self.corr_dim * 2
        self.pos_dim = 2
        self.conf_dim = 1
        
        input_dim = self.corr_dim + self.offset_dim + self.context_dim + self.pos_dim + self.conf_dim
        
        # 2. 运动 Transformer (替换原有的 CNN Encoder)
        if self.use_mtf:
            self.encoder = MotionTransformer(
                input_dim=input_dim, 
                embed_dim=hidden_dim, 
                num_layers=4, 
                num_heads=8
            )
        else:
            self.encoder = nn.Sequential(
                # Layer 1: [B, input_dim, H, W] -> [B, 256, H/2, W/2]
                nn.Conv2d(input_dim, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                
                # Layer 2: [B, 256, H/2, W/2] -> [B, 192, H/4, W/4]
                nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                
                # Layer 3: [B, 192, H/4, W/4] -> [B, 128, H/8, W/8]
                nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        
        # 3. GRU 单元
        if self.use_gru:
            self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        else:
            self.gru = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU()
            )

        
        
        # 4. 解耦仿射头 (输入维度同步升级)
        self.head_trans = nn.Linear(hidden_dim, 2)
        self.head_linear = nn.Linear(hidden_dim, 4)
        
        # 5. 缩放因子
        self.register_buffer('scale_trans', torch.tensor(0.1))
        self.register_buffer('scale_linear', torch.tensor(0.0001))
       
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化输出头
        nn.init.normal_(self.head_trans.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_trans.bias, 0.0)
        
        nn.init.normal_(self.head_linear.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_linear.bias, 0.0)

    def forward(self, 
                corr_features: torch.Tensor, 
                corr_offsets: torch.Tensor, 
                context_features: torch.Tensor, 
                pos_features: torch.Tensor, 
                confidence_map: torch.Tensor, 
                hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播接口保持不变
        """
        # 1. 置信度门控
        masked_corr = corr_features * confidence_map
        masked_ctx = context_features * confidence_map
        
        # 2. 特征拼接
        x = torch.cat([masked_corr, corr_offsets, masked_ctx, pos_features, confidence_map], dim=1)
        
        # 3. 运动编码 (Transformer 提取全局特征)
        # x_global: [B, hidden_dim]
        x_global = self.encoder(x)
        
        # 4. GRU 状态更新
        if self.use_gru:
            new_hidden_state = self.gru(x_global, hidden_state)
        else:
            new_hidden_state = self.gru(x_global)
        
        # 5. 参数预测与缩放
        raw_trans = self.head_trans(new_hidden_state)   
        raw_linear = self.head_linear(new_hidden_state) 
        
        delta_trans = raw_trans * self.scale_trans
        delta_linear = raw_linear * self.scale_linear
        
        # 6. 构建仿射增量矩阵 [B, 2, 3]
        # 顺序: [da, db, dc, dd, dtx, dty]
        delta_affine_params = torch.cat([delta_linear.reshape(-1, 2, 2), delta_trans.unsqueeze(-1)], dim=-1)
        
        # 单位阵基准
        base_matrix = torch.eye(2, 3, device=x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        delta_affine = base_matrix + delta_affine_params
        
        return delta_affine, new_hidden_state