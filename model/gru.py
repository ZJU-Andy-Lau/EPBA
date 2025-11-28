import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUBlock(nn.Module):
    def __init__(self, 
                 corr_levels=2, 
                 corr_radius=4, 
                 context_dim=128, 
                 hidden_dim=128):
        """
        基于图的 GRU 迭代更新模块 (Graph-based GRU Update Block)
        
        Args:
            corr_levels (int): 相关性金字塔层数 (默认 2)
            corr_radius (int): 查表半径 (默认 4)
            context_dim (int): 上下文特征通道数 (默认 128)
            hidden_dim (int): GRU 隐藏状态维度 (默认 128)
        """
        super().__init__()
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        
        # 1. 计算输入通道数
        # Correlation channels = levels * (2*r + 1)^2
        self.corr_dim = corr_levels * (2 * corr_radius + 1)**2
        
        # [新增] Offset channels = Correlation channels * 2 (每个查表点对应 dy, dx)
        self.offset_dim = self.corr_dim * 2
        
        # self.flow_dim = 2  # u, v
        
        # [修改] 总输入通道数: Corr + Offsets + Context + Flow
        # 例如: 324 + 648 + 128 = 1100
        input_dim = self.corr_dim + self.offset_dim + self.context_dim #+ self.flow_dim
        
        # 2. 运动编码器 (Motion Encoder)
        # 将空间误差特征图压缩为全局误差向量
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
        
        # 3. GRU 单元 (Recurrent Unit)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        
        # 4. 解耦仿射头 (Decoupled Affine Heads)
        # 将隐藏状态映射为参数增量
        self.head_trans = nn.Linear(hidden_dim, 2) # tx, ty
        self.head_linear = nn.Linear(hidden_dim , 4) # a, b, c, d
        
        # 5. 注册缩放因子 (Buffers)
        # 平移部分: 允许较大更新 (0.1 对应归一化坐标下的 5% 图像宽度)
        self.register_buffer('scale_trans', torch.tensor(0.1))
        # 线性部分: 强约束，抑制旋转/缩放震荡
        self.register_buffer('scale_linear', torch.tensor(0.0001))
        
        # 6. 参数初始化
        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化策略：
        1. 卷积层使用 Kaiming 初始化。
        2. 输出头 (Heads) 使用零初始化 (Zero Init)，确保初始输出为 0。
        """
        # 初始化卷积层
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 初始化输出头
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
            hidden_state: [B, 128] - 上一时刻的 GRU 状态
            
        Returns:
            delta_affine: [B, 6] - 仿射参数增量 [da, db, dc, dd, dtx, dty]
            new_hidden_state: [B, 128] - 更新后的 GRU 状态
        """
        B, _, H, W = corr_features.shape
        
        # 1. 显式置信度门控 (Explicit Confidence Gating)
        # 利用置信度图抑制噪声区域的特征
        # Flow 和 Offsets 代表客观几何信息，通常不进行 Masking，保留结构感知
        masked_corr = corr_features * confidence_map
        masked_ctx = context_features * confidence_map
        
        # 2. 特征拼接
        # Input: [B, C_corr + C_offset + C_ctx, H, W]
        # 将 masked_corr, corr_offsets, masked_ctx 在通道维度拼接
        x = torch.cat([masked_corr, corr_offsets, masked_ctx], dim=1)
        
        # 3. 运动编码 (Motion Encoding)
        # CNN 下采样: [B, input_dim, H, W] -> [B, 128, H/8, W/8]
        x = self.encoder(x)
        
        # 全局平均池化 (GAP): [B, 128, H/8, W/8] -> [B, 128, 1, 1]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        
        # 展平: [B, 128]
        # 这是本次迭代的 "Global Error Vector"
        x = x.view(B, -1)
        
        # 4. GRU 状态更新
        new_hidden_state = self.gru(x, hidden_state)
        
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
        
        return delta_affine, new_hidden_state