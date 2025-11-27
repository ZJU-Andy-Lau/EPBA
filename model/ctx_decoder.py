import torch
import torch.nn as nn
class ContextDecoder(nn.Module):
    def __init__(self, ctx_dim=128):
        super().__init__()
        # 简单的上采样网络
        self.decoder = nn.Sequential(
            nn.Conv2d(ctx_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 3, kernel_size=3, padding=1), # 输出 RGB
        )

    def forward(self, ctx_feat):
        return self.decoder(ctx_feat)