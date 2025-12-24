import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from .utils import residual_to_conf

class ConfLoss(nn.Module):
    def __init__(self, parallax_border = (2.,8.)):
        super().__init__()
        self.bce = nn.BCELoss()
        self.get_conf_label = partial(residual_to_conf,left = parallax_border[0],right = parallax_border[1]) 

    def forward(self,conf,residual):
        """
        conf: B,1,h,w
        residual: B,1,h,w
        """
        conf_label = self.get_conf_label(residual) # B,1,h,w
        loss_conf = self.bce(conf,conf_label)
        return loss_conf
