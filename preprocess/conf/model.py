import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2

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
        self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    def forward(self,x):
        B, _, H, W = x.shape
        feat_backbone = self.backbone.get_intermediate_layers(x)[0]
        feat_backbone = feat_backbone.reshape(B, H // 16, W // 16, -1).permute(0, 3, 1, 2)
        conf = self.head(feat_backbone)
        return conf
    
    def save_head(self,path):
        state_dict = {k: v.detach().cpu() for k, v in self.head.state_dict().items()}
        torch.save(state_dict, path)

    def load_head(self,path):
        state_dict = {k: v.detach().cpu() for k, v in torch.load(path).items()}
        self.head.load_state_dict(state_dict)

    @torch.no_grad()
    def pred(self,imgs:np.ndarray):
        if imgs.ndim == 2:
            imgs = np.stack([imgs] * 3,axis=-1)

        if imgs.ndim == 3:
            imgs = imgs[None]

        device = next(self.parameters()).device

        B,_,H,W = imgs.shape
        input_data = torch.stack([self.transform(img) for img in imgs],dim=0)
        input_data = input_data.to(device)

        conf = self.forward(input_data)
        conf = conf.squeeze().detach().cpu().numpy() # B,h,W

        conf_resize = np.zeros((B,H,W),dtype=np.float32)
        for b in range(B):
            conf_resize[b] = cv2.resize(conf[b],(W,H),interpolation=cv2.INTER_LINEAR)
        
        if B == 1:
            return conf_resize[0]
        else:
            return conf_resize
        

