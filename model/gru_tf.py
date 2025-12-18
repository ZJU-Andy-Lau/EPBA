import torch
import torch.nn as nn
import torch.nn.functional as F

# 从同目录下的 encoder.py 导入必要的组件
# 注意：如果运行时报错找不到 model，请根据实际路径调整，例如 from .encoder import ...
try:
    from model.encoder import Rope, SelfAttentionBlock
except ImportError:
    from .encoder import Rope, SelfAttentionBlock

# -------------------------------------------------------------------------
# MotionTransformer: 几何感知运动特征编码器 (Transformer-based)
# -------------------------------------------------------------------------
class MotionTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, depth=4):
        """
        基于 Transformer 的运动特征编码器，旨在通过全局注意力捕捉大尺度几何变换。
        
        Args:
            input_dim (int): 输入特征通道数 (Corr + Offsets + Ctx)
            embed_dim (int): Transformer 内部特征维度 (Default: 256)
            num_heads (int): 注意力头数 (Default: 8)
            depth (int): Transformer 层数 (Default: 4, 精度优先)
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Stem: 特征融合与投影
        # 策略: 保持空间分辨率 (No Downsampling)，只做通道融合。
        # 原因: 16倍下采样后的特征图(32x32)通过 RoPE 可以保留极佳的亚像素几何精度。
        self.stem = nn.Sequential(
            nn.Conv2d(input_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(8, embed_dim), # GroupNorm 对 BatchSize 不敏感，训练更稳定
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=1),
            # 末端不加激活函数，直接作为 Embedding 输入
        )

        # 2. 位置编码 (RoPE)
        # 复用 Encoder 中的 2D RoPE。head_dim = 256 / 8 = 32
        self.rope = Rope(d_model=embed_dim // num_heads)
        
        # 3. Global Motion Token
        # 作为一个"容器"，在交互过程中主动聚合来自各个 Patch 的几何线索
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 4. Transformer Layers
        # 堆叠 Self-Attention 模块进行全图特征交互
        self.layers = nn.ModuleList([
            SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0) 
            for _ in range(depth)
        ])
        
        # 5. Output Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用截断正态分布初始化 CLS token，防止梯度爆炸或消失
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化 Stem (Conv2d 默认初始化通常够用，这里显式使用 Kaiming 以防万一)
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, input_dim, H, W]
        Returns:
            global_motion: [B, embed_dim]
        """
        B, _, H, W = x.shape
        device = x.device

        # --- 1. Stem Projection ---
        # [B, C, H, W] -> [B, embed_dim, H, W]
        feat = self.stem(x)
        
        # --- 2. Tokenization ---
        # Flatten: [B, embed_dim, H, W] -> [B, H*W, embed_dim]
        feat_flat = feat.flatten(2).transpose(1, 2)
        
        # 拼接 CLS Token: [B, 1 + H*W, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls_tokens, feat_flat), dim=1)
        
        # --- 3. Prepare RoPE ---
        # 计算 Patch 的位置编码: [1, 1, H*W, head_dim] (cos, sin)
        cos_patch, sin_patch = self.rope(feat)
        
        # 为 CLS token 构造"零旋转"编码 (cos=1, sin=0)
        # 这样 CLS token 不会受到位置干扰，纯粹用于聚合信息
        head_dim = cos_patch.shape[-1]
        cos_cls = torch.ones(1, 1, 1, head_dim, device=device)
        sin_cls = torch.zeros(1, 1, 1, head_dim, device=device)
        
        # 拼接 RoPE: [1, 1, 1 + H*W, head_dim]
        rope_cos = torch.cat([cos_cls, cos_patch], dim=2)
        rope_sin = torch.cat([sin_cls, sin_patch], dim=2)
        
        # --- 4. Transformer Interaction ---
        for layer in self.layers:
            # 传入 tokens 和 对应的 RoPE 编码
            tokens = layer(tokens, rope_cos, rope_sin)
            
        # --- 5. Extract Global Motion ---
        # 取出 CLS token 并归一化 [B, embed_dim]
        global_motion = self.norm(tokens[:, 0]) 
        
        return global_motion

# -------------------------------------------------------------------------
# GRUBlock: 迭代更新模块
# -------------------------------------------------------------------------
class GRUBlock(nn.Module):
    def __init__(self, 
                 corr_levels=2, 
                 corr_radius=4, 
                 context_dim=128, 
                 tf_embed_dim=256,
                 tf_heads=8,
                 tf_depth=4):
        """
        基于 Transformer 和 GRU 的迭代更新模块。
        
        Args:
            corr_levels (int): 相关性金字塔层数
            corr_radius (int): 查表半径
            context_dim (int): 上下文特征通道数
            tf_embed_dim (int): MotionTransformer 的嵌入维度，同时也是 GRU 的新隐藏层维度
            tf_heads (int): Transformer 头数
            tf_depth (int): Transformer 层数
        """
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.context_dim = context_dim
        
        # 1. 计算输入通道数
        # Correlation channels = levels * (2*r + 1)^2
        self.corr_dim = corr_levels * (2 * corr_radius + 1)**2
        
        # Offset channels = Correlation channels * 2 (每个查表点对应 dy, dx)
        self.offset_dim = self.corr_dim * 2
        
        # 总输入通道数: Corr + Offsets + Context
        input_dim = self.corr_dim + self.offset_dim + self.context_dim
        
        # 2. 运动编码器 (Motion Encoder) - 使用新的 Transformer 架构
        self.motion_encoder = MotionTransformer(
            input_dim=input_dim,
            embed_dim=tf_embed_dim,
            num_heads=tf_heads,
            depth=tf_depth
        )
        
        # 3. GRU 单元 (Recurrent Unit)
        # 为了最大化利用 Transformer 的特征，我们将 GRU 的维度对齐到 tf_embed_dim (256)
        self.gru_dim = tf_embed_dim
        self.gru = nn.GRUCell(input_size=self.gru_dim, hidden_size=self.gru_dim)
        
        # 4. 解耦仿射头 (Decoupled Affine Heads)
        # 将隐藏状态映射为参数增量
        self.head_trans = nn.Linear(self.gru_dim, 2) # tx, ty
        self.head_linear = nn.Linear(self.gru_dim, 4) # a, b, c, d
        
        # 5. 注册缩放因子 (Buffers)
        # 平移部分: 允许较大更新 (0.1 对应归一化坐标下的 5% 图像宽度)
        self.register_buffer('scale_trans', torch.tensor(0.1))
        # 线性部分: 强约束，抑制旋转/缩放震荡
        self.register_buffer('scale_linear', torch.tensor(0.0001))
       
        # 6. 参数初始化 (仅初始化 Heads，Encoder 已内部初始化)
        self._reset_head_parameters()

    def _reset_head_parameters(self):
        """
        初始化输出头，确保初始输出为 0。
        """
        nn.init.normal_(self.head_trans.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_trans.bias, 0.0)
        
        nn.init.normal_(self.head_linear.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.head_linear.bias, 0.0)

    def forward(self, corr_features, corr_offsets, context_features, confidence_map, hidden_state):
        """
        前向传播
        
        Args:
            corr_features: [B, C_corr, H, W] - 相关性查表特征 (Values)
            corr_offsets:  [B, C_offset, H, W] - 相关性查表偏移 (Geometry)
            context_features: [B, 128, H, W] - 上下文特征
            confidence_map: [B, 1, H, W] - 置信度图 (0~1)
            hidden_state: [B, 256] - 上一时刻的 GRU 状态 (维度需匹配 tf_embed_dim)
            
        Returns:
            delta_affine: [B, 6] - 仿射参数增量 [da, db, dc, dd, dtx, dty]
            new_hidden_state: [B, 256] - 更新后的 GRU 状态
        """
        B, _, H, W = corr_features.shape
        
        # 1. 显式置信度门控 (Explicit Confidence Gating)
        # 利用置信度图抑制噪声区域的特征
        masked_corr = corr_features * confidence_map
        masked_ctx = context_features * confidence_map
        
        # 2. 特征拼接
        # Input: [B, C_corr + C_offset + C_ctx, H, W]
        x = torch.cat([masked_corr, corr_offsets, masked_ctx], dim=1)
        
        # 3. 运动编码 (Motion Encoding)
        # [B, input_dim, H, W] -> [B, 256]
        # 通过 MotionTransformer 提取全局几何特征
        global_motion_feat = self.motion_encoder(x)
        
        # 4. GRU 状态更新
        # 输入和隐藏状态均为 [B, 256]
        new_hidden_state = self.gru(global_motion_feat, hidden_state)
        
        # 5. 参数预测 (Decoding)
        # 分别预测平移和线性部分
        raw_trans = self.head_trans(new_hidden_state)   # [B, 2]
        raw_linear = self.head_linear(new_hidden_state) # [B, 4]
        
        # 6. 应用差分缩放 (Differential Scaling)
        delta_trans = raw_trans * self.scale_trans
        delta_linear = raw_linear * self.scale_linear
        
        # 7. 拼接输出
        # 输出顺序: [da, db, dc, dd, dtx, dty]
        # 对应矩阵: [[1+da, db, dtx], [dc, 1+dd, dty]]
        delta_affine = torch.cat([delta_linear.reshape(-1,2,2), delta_trans.unsqueeze(-1)], dim=-1) # B,2,3

        base_matrix = torch.tensor([
            [
                [1.0,0.0,0.0],
                [0.0,1.0,0.0]
            ]
        ] * delta_affine.shape[0], device=delta_affine.device, dtype=delta_affine.dtype) # B,2,3

        delta_affine = base_matrix + delta_affine
        
        return delta_affine, new_hidden_state