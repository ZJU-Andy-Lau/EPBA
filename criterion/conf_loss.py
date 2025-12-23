import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import residual_to_conf

class ConfLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self,conf,residual):
        """
        conf: B,1,h,w
        residual: B,1,h,w
        """
        conf_label = residual_to_conf(residual) # B,1,h,w
        loss_conf = self.bce(conf,conf_label)
        return loss_conf
