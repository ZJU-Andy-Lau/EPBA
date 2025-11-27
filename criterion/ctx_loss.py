import torch
import torch.nn as nn
import torch.nn.functional as F

class CtxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self,img_pred,img_gt):
        """
        img_pred: B,3,H,W
        img_gt: B,3,H,W
        """
        img_gt = (img_gt - img_gt.min()) / (img_gt.max() - img_gt.min())
        ctx_loss = torch.abs(img_pred - img_gt).mean()
        return ctx_loss
