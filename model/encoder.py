import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# --------------------------------------------------------
# 1. RoPE 实现 (严格 2D)
# --------------------------------------------------------
class Rope(nn.Module):
    """
    严格的 2D 旋转位置编码。
    将特征维度一分为二，一半编码 X 坐标，一半编码 Y 坐标。
    """
    def __init__(self, d_model, base=10000.0):
        super().__init__()
        self.d_model = d_model
        assert d_model % 4 == 0, "For strict 2D RoPE, head_dim must be divisible by 4"
        
        self.base = base
        self.dim_half = d_model // 2
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim_half, 2).float() / self.dim_half))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        """
        x: [Batch, Channels, Height, Width] 
        返回: cos, sin 张量 [1, 1, SeqLen, HeadDim]
        """
        h, w = x.shape[-2], x.shape[-1]
        device = x.device
        
        # 生成网格坐标
        t_x = torch.arange(w, device=device).type_as(self.inv_freq)
        t_y = torch.arange(h, device=device).type_as(self.inv_freq)

        # 外积生成 2D 频率
        freqs_x = torch.einsum('i,j->ij', t_x, self.inv_freq) 
        freqs_x = freqs_x.unsqueeze(0).repeat(h, 1, 1) 
        
        freqs_y = torch.einsum('i,j->ij', t_y, self.inv_freq)
        freqs_y = freqs_y.unsqueeze(1).repeat(1, w, 1)
        
        # 拼接 X, Y 频率
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)
        freqs = freqs.flatten(0, 1) # [SeqLen, D/2]
        
        # 准备 sin/cos (对应 rotate_half)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting [1, 1, L, D]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    将 RoPE 应用于 Query 和 Key。
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    # cos, sin 自动广播
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --------------------------------------------------------
# 2. 自注意力模块 (Self Attention)
# --------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        # x: [B, L, D]
        residual = x
        x = self.norm1(x)
        
        B, L, D = x.shape
        
        # QKV
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape [B, Heads, L, HeadDim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # RoPE
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Attention
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        
        x = x.transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.out_proj(x)
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x

# --------------------------------------------------------
# 3. 交叉注意力模块 (Cross Attention)
# --------------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        # Source -> Query
        self.to_q = nn.Linear(embed_dim, embed_dim)
        # Target -> Key, Value
        self.to_kv = nn.Linear(embed_dim, embed_dim * 2)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, rope_cos, rope_sin):
        """
        x: Source features (Query) [B, L, D]
        context: Target features (Key/Value) [B, L, D]
        rope_cos, rope_sin: 位置编码，用于编码相对位置偏差
        """
        residual = x
        
        # Norm
        x_norm = self.norm_q(x)
        ctx_norm = self.norm_kv(context)
        
        B, L, D = x.shape
        
        # Generate Q, K, V
        q = self.to_q(x_norm)
        kv = self.to_kv(ctx_norm)
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        # 在 Cross Attention 中，RoPE 编码的是 Query 像素与 Key 像素的空间位置关系
        # 如果 Query(0,0) 关注 Key(0,0)，相对位置为 0，旋转为 0
        q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)
        
        # Attention
        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = residual + self.out_proj(attn_out)
        
        # FFN
        x = x + self.mlp(self.norm_ffn(x))
        
        return x

# --------------------------------------------------------
# 4. 双流 Adapter (Two-Stream Adapter)
# --------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, input_dim, embed_dim=256, ctx_dim=128, num_layers=2, num_heads=4):
        """
        num_layers: 交互层数。每一层包含一次 Self Attention 和一次 Cross Attention。
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 特征融合器 (共享)
        self.fuser = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(input_dim // 4, embed_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
        )

        # RoPE 生成器
        self.rope = Rope(d_model=embed_dim // num_heads)
        
        # 上下文和置信度头 (在交互前生成，保持原始语义)
        self.ctx_proj = nn.Conv2d(embed_dim, ctx_dim, 1)
        self.conf_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 4, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        
        # 交互层 (Interleaved Self & Cross Attention)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self': SelfAttentionBlock(embed_dim, num_heads),
                'cross': CrossAttentionBlock(embed_dim, num_heads)
            }))

    def forward(self, feat0, feat1):
        """
        输入:
            feat0: [B, C_in, H, W]
            feat1: [B, C_in, H, W]
        """
        # 1. 特征融合 (共享权重)
        f0 = self.fuser(feat0) # [B, embed_dim, H, W]
        f1 = self.fuser(feat1)
        
        # 2. 提取 Context 和 Confidence (Pre-interaction)
        # 这一步很重要：保持 Context 的纯洁性，不混入另一张图的信息
        ctx0 = self.ctx_proj(f0)
        conf0 = self.conf_head(f0)
        
        ctx1 = self.ctx_proj(f1)
        conf1 = self.conf_head(f1)
        
        # 3. 准备 Transformer 输入
        B, D, H, W = f0.shape
        
        # Flatten: [B, D, H, W] -> [B, H*W, D]
        f0_flat = f0.flatten(2).transpose(1, 2)
        f1_flat = f1.flatten(2).transpose(1, 2)
        
        # 生成 RoPE
        rope_cos, rope_sin = self.rope(f0)
        
        # 4. 交互循环
        for i, layer in enumerate(self.layers):
            # --- Self Attention ---
            f0_flat = layer['self'](f0_flat, rope_cos, rope_sin)
            f1_flat = layer['self'](f1_flat, rope_cos, rope_sin)
            
            # --- Cross Attention (Bidirectional) ---
            # f0 从 f1 获取信息
            f0_new = layer['cross'](f0_flat, f1_flat, rope_cos, rope_sin)
            # f1 从 f0 获取信息
            f1_new = layer['cross'](f1_flat, f0_flat, rope_cos, rope_sin)
            
            f0_flat, f1_flat = f0_new, f1_new
            
        # 5. 恢复形状并归一化
        match0 = f0_flat.transpose(1, 2).view(B, D, H, W)
        match1 = f1_flat.transpose(1, 2).view(B, D, H, W)
        
        match0 = F.normalize(match0, p=2, dim=1)
        match1 = F.normalize(match1, p=2, dim=1)
        
        return match0, ctx0, conf0, match1, ctx1, conf1

# --------------------------------------------------------
# 5. Siamese Encoder (主入口)
# --------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, dino_weight_path, embed_dim=256, ctx_dim=128, layers=[5, 11, 17, 23],use_adapter = True, use_conf = True, verbose=1):
        super().__init__()
        self.verbose = verbose
        self.layers = layers
        self.embed_dim = embed_dim
        self.ctx_dim = ctx_dim

        # 加载 DINOv3
        self.backbone = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        # 输入通道数 = DINO每层通道 * 层数
        input_channels = 1024 * len(layers) 
        self.adapter = Adapter(input_dim=input_channels, embed_dim=embed_dim, ctx_dim=ctx_dim)

        self.use_adapter = use_adapter
        self.use_conf = use_conf

    def _extract_dino_features(self, x):
        """
        提取多层特征并拼接
        """
        B, _, H, W = x.shape
        feat_multilayers = self.backbone.get_intermediate_layers(x=x, n=self.layers)
        feat_backbone = torch.cat(feat_multilayers, dim=-1)
        feat_backbone = feat_backbone.reshape(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
        return feat_backbone

    def forward(self, img0, img1):
        """
        孪生输入: 同时接收 img0 和 img1
        """
        # 1. 批量提取 Backbone 特征 (Batching for efficiency)
        # 拼接成 [2B, C, H, W] 进 DINO，比跑两次快
        imgs = torch.cat([img0, img1], dim=0)
        feats_all = self._extract_dino_features(imgs)
        
        # 拆分
        feat0, feat1 = feats_all.chunk(2, dim=0)
        
        # 2. Adapter 交互提取
        if self.use_adapter or self.use_conf:
            m0, c0, conf0, m1, c1, conf1 = self.adapter(feat0, feat1)
        if not self.use_adapter:
            m0,m1 = feat0.detach(),feat1.detach()
            c0,c1 = feat0[:,:self.ctx_dim].detach(),feat1[:,:self.ctx_dim].detach()
        if not self.use_conf:
            conf0 = torch.ones((feat0.shape[0],1,feat0.shape[2],feat0.shape[3]),dtype=feat0.dtype,device=feat0.device).detach()
            conf1 = torch.ones((feat1.shape[0],1,feat1.shape[2],feat1.shape[3]),dtype=feat1.dtype,device=feat1.device).detach()
        
        # 返回两个元组
        return (m0, c0, conf0), (m1, c1, conf1)
    
    def unfreeze_backbone(self, layers: List[int] = []):
        parameters = []
        for layer in layers:
            block = self.backbone.blocks[layer]
            block.requires_grad_(True)
            parameters.extend(list(block.parameters()))
            if self.verbose > 0:
                print(f"Unfreeze backbone layer {layer}")
        return parameters
    
    def load_adapter(self, adapter_path: str):
        self.adapter.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(adapter_path, map_location='cpu').items()}, strict=True)
    
    def save_adapter(self, output_path: str):
        state_dict = {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}
        torch.save(state_dict, output_path)

    def save_backbone(self, output_path: str):
        state_dict = {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}
        torch.save(state_dict, output_path)