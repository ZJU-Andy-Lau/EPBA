import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
    
    def get_conf_label(self,residual):
        conf_label = torch.full(residual.shape,.5,device=residual.device,dtype=residual.dtype)
        valid_residual = residual[residual >= 0]
        res_mid = torch.median(valid_residual)
        conf_label[residual > res_mid] = .1
        conf_label[(residual <= res_mid) & (residual >= 0)] = .9
        conf_label[residual < 0] = .1
        return conf_label
        

    def forward(self,conf,residual):
        """
        conf: B,1,h,w
        residual: B,1,h,w
        """
        conf_label = self.get_conf_label(residual) # B,1,h,w
        loss_conf = self.bce(conf,conf_label)
        return loss_conf
