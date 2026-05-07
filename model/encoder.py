import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from torchvision.models import resnet50, ResNet50_Weights
from model.backbones.satmae import SatMAEBackboneWrapper, parse_satmae_layers

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
    def __init__(self, input_dim, embed_dim=256, ctx_dim=128, num_layers=2, num_heads=4, use_adapter=True, use_conf=True):
        """
        num_layers: 交互层数。每一层包含一次 Self Attention 和一次 Cross Attention。
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.ctx_dim = ctx_dim
        self.use_adapter = use_adapter
        self.use_conf = use_conf

        # 特征融合器 (共享)
        self.fuser = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 4, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(input_dim // 4, embed_dim, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1, 1, 0),
        )

        if self.use_adapter:
            # RoPE 生成器
            self.rope = Rope(d_model=embed_dim // num_heads)
            
            # 上下文和置信度头 (在交互前生成，保持原始语义)
            self.ctx_proj = nn.Conv2d(embed_dim, ctx_dim, 1)
            
            # 交互层 (Interleaved Self & Cross Attention)
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(nn.ModuleDict({
                    'self': SelfAttentionBlock(embed_dim, num_heads),
                    'cross': CrossAttentionBlock(embed_dim, num_heads)
                }))

        if self.use_conf:
            self.conf_head = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 2, embed_dim // 4, 1, 1, 0),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )

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
        
        if self.use_conf:
            conf0 = self.conf_head(f0)
            conf1 = self.conf_head(f1)
        else:
            conf0 = torch.ones((feat0.shape[0],1,feat0.shape[2],feat0.shape[3]),dtype=feat0.dtype,device=feat0.device).detach()
            conf1 = torch.ones((feat1.shape[0],1,feat1.shape[2],feat1.shape[3]),dtype=feat1.dtype,device=feat1.device).detach()
        
        if self.use_adapter:
            ctx0 = self.ctx_proj(f0)
            ctx1 = self.ctx_proj(f1)
            
            
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
        
        else:
            match0,match1 = feat0.detach(),feat1.detach()
            ctx0,ctx1 = feat0[:,:self.ctx_dim].detach(),feat1[:,:self.ctx_dim].detach()
        
        return match0, ctx0, conf0, match1, ctx1, conf1

# --------------------------------------------------------
# 5. Siamese Encoder (主入口)
# --------------------------------------------------------
class DINOv3BackboneWrapper(nn.Module):
    def __init__(self, dino_weight_path, layers=None):
        super().__init__()
        self.layers = layers if layers is not None else [5, 11, 17, 23]
        self.backbone = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.output_dim = 1024 * len(self.layers)

    def forward(self, x):
        B, _, H, W = x.shape
        feat_multilayers = self.backbone.get_intermediate_layers(x=x, n=self.layers)
        feat_backbone = torch.cat(feat_multilayers, dim=-1)
        feat_backbone = feat_backbone.reshape(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
        return feat_backbone


class ResNet50BackboneWrapper(nn.Module):
    _valid_layers = ("layer1", "layer2", "layer3", "layer4")
    _layer_dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}

    def __init__(self, resnet_weight_path=None, resnet_weights="IMAGENET1K_V2", resnet_layers=None):
        super().__init__()
        self.resnet_layers = resnet_layers if resnet_layers is not None else ["layer1", "layer2", "layer3"]
        for layer in self.resnet_layers:
            if layer not in self._valid_layers:
                raise ValueError(f"Invalid resnet layer '{layer}'. Valid options: {self._valid_layers}")

        if resnet_weight_path:
            model = resnet50(weights=None)
            state_dict = torch.load(resnet_weight_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
        else:
            weights = ResNet50_Weights[resnet_weights] if isinstance(resnet_weights, str) else resnet_weights
            model = resnet50(weights=weights)

        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.output_dim = sum(self._layer_dims[k] for k in self.resnet_layers)

    def forward(self, x):
        x = self.stem(x)
        feats = {}
        x = self.layer1(x); feats["layer1"] = x
        x = self.layer2(x); feats["layer2"] = x
        x = self.layer3(x); feats["layer3"] = x
        x = self.layer4(x); feats["layer4"] = x

        target_h = x.shape[-2] * 2
        target_w = x.shape[-1] * 2
        aligned = []
        for name in self.resnet_layers:
            f = feats[name]
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            aligned.append(f)
        return torch.cat(aligned, dim=1)


class Encoder(nn.Module):
    def __init__(self, dino_weight_path=None, embed_dim=256, ctx_dim=128, layers=[5, 11, 17, 23],
                 use_adapter=True, use_conf=True, verbose=1, backbone="dinov3",
                 resnet_weight_path=None, resnet_weights="IMAGENET1K_V2", resnet_layers="layer1,layer2,layer3",
                 satmae_weight_path=None, satmae_layers="5,11,17,23", satmae_img_size=512,
                 satmae_patch_size=16, satmae_model="vit_large_patch16", satmae_ckpt_key=None,
                 satmae_apply_norm=True, freeze_backbone=True):
        super().__init__()
        self.verbose = verbose
        self.layers = layers
        self.embed_dim = embed_dim
        self.ctx_dim = ctx_dim

        self.backbone_name = backbone.lower()
        if isinstance(resnet_layers, str):
            resnet_layers = [i.strip() for i in resnet_layers.split(",") if i.strip()]

        if self.backbone_name == "dinov3":
            self.backbone = DINOv3BackboneWrapper(dino_weight_path=dino_weight_path, layers=layers)
        elif self.backbone_name == "resnet50":
            self.backbone = ResNet50BackboneWrapper(
                resnet_weight_path=resnet_weight_path,
                resnet_weights=resnet_weights,
                resnet_layers=resnet_layers,
            )
        elif self.backbone_name == "satmae":
            self.backbone = SatMAEBackboneWrapper(
                weight_path=satmae_weight_path,
                layers=parse_satmae_layers(satmae_layers),
                img_size=satmae_img_size,
                patch_size=satmae_patch_size,
                model_name=satmae_model,
                ckpt_key=satmae_ckpt_key,
                apply_norm=satmae_apply_norm,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'. Use 'dinov3', 'resnet50', or 'satmae'.")

        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        input_channels = self.backbone.output_dim
        self.adapter = Adapter(input_dim=input_channels, embed_dim=embed_dim, ctx_dim=ctx_dim, use_adapter=use_adapter,use_conf=use_conf)

        self.use_adapter = use_adapter
        self.use_conf = use_conf

    def _extract_backbone_features(self, x):
        if self.freeze_backbone:
            with torch.no_grad():
                return self.backbone(x)
        return self.backbone(x)

    def forward(self, img0, img1):
        """
        孪生输入: 同时接收 img0 和 img1
        """
        # 1. 批量提取 Backbone 特征 (Batching for efficiency)
        # 拼接成 [2B, C, H, W] 进 DINO，比跑两次快
        imgs = torch.cat([img0, img1], dim=0)
        feats_all = self._extract_backbone_features(imgs)
        
        # 拆分
        feat0, feat1 = feats_all.chunk(2, dim=0)
        
        # 2. Adapter 交互提取
        m0, c0, conf0, m1, c1, conf1 = self.adapter(feat0, feat1)
        
        # 返回两个元组
        return (m0, c0, conf0), (m1, c1, conf1)
    
    def unfreeze_backbone(self, layers: List[int] = []):
        if self.backbone_name != "dinov3":
            raise RuntimeError("unfreeze_backbone currently only supports dinov3 backbone.")
        parameters = []
        for layer in layers:
            block = self.backbone.backbone.blocks[layer]
            block.requires_grad_(True)
            parameters.extend(list(block.parameters()))
            if self.verbose > 0:
                print(f"Unfreeze backbone layer {layer}")
        return parameters
    
    def load_adapter(self, adapter_path: str):
        checkpoint = torch.load(adapter_path, map_location='cpu')
        if isinstance(checkpoint, dict) and "adapter" in checkpoint:
            metadata = checkpoint.get("metadata", {})
            ckpt_backbone = metadata.get("backbone_name")
            ckpt_output_dim = metadata.get("backbone_output_dim")
            if ckpt_backbone is not None and ckpt_backbone != self.backbone_name:
                raise RuntimeError(
                    f"Adapter checkpoint backbone mismatch: checkpoint backbone='{ckpt_backbone}', "
                    f"current backbone='{self.backbone_name}'. Train/load an adapter for the selected backbone."
                )
            if ckpt_output_dim is not None and int(ckpt_output_dim) != int(self.backbone.output_dim):
                raise RuntimeError(
                    f"Adapter checkpoint backbone_output_dim mismatch: checkpoint={ckpt_output_dim}, "
                    f"current={self.backbone.output_dim}."
                )
            raw_state = checkpoint["adapter"]
        else:
            raw_state = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in raw_state.items()}
        try:
            self.adapter.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load adapter checkpoint '{adapter_path}'. "
                f"Current backbone='{self.backbone_name}' expects adapter input_dim={self.backbone.output_dim}. "
                f"Checkpoint is likely incompatible with current backbone/input_dim (e.g., DINO adapter vs ResNet adapter). "
                f"Please train a dedicated adapter for the selected backbone. Original error: {e}"
            ) from e
    
    def save_adapter(self, output_path: str):
        state_dict = {k: v.detach().cpu() for k, v in self.adapter.state_dict().items()}
        torch.save({
            "adapter": state_dict,
            "metadata": {
                "backbone_name": self.backbone_name,
                "backbone_output_dim": int(self.backbone.output_dim),
            },
        }, output_path)

    def save_backbone(self, output_path: str):
        state_dict = {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()}
        torch.save(state_dict, output_path)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
