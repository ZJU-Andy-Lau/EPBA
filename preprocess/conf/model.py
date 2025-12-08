import torch
import torch.nn as nn

class ConfHead(nn.Module):
    def __init__(self,dino_weight_path:str):
        super().__init__()
        self.backbone = torch.hub.load('./dinov3', 'dinov3_vitl16', source='local', weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1, 1, 0),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        B, _, H, W = x.shape
        feat_backbone = self.backbone.get_intermediate_layers(x)
        feat_backbone = feat_backbone.reshape(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
        conf = self.head(feat_backbone)
        return conf
    
    def save_head(self,path):
        state_dict = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        torch.save(state_dict, path)