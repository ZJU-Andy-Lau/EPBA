import torch.nn as nn
from functools import partial
from .sim_loss import SimLoss
from .conf_loss import ConfLoss
from .ctx_loss import CtxLoss
from .utils import residual_to_conf

class Loss(nn.Module):
    def __init__(self, downsample_factor = 16,temperature = 0.07, parallax_border = (2.,10.)):
        super().__init__()
        self.sim_loss = SimLoss(downsample_factor = downsample_factor,
                                temperature = temperature)
        self.conf_loss = ConfLoss()

        self.ctx_loss = CtxLoss()

        self.get_conf_weights = partial(residual_to_conf,left = parallax_border[0],right = parallax_border[1])
    
    def forward(self,input):
        match_feats_1, _, confs_1 = input['feats_1']
        match_feats_2, _, confs_2 = input['feats_2']

        loss_conf_1 = self.conf_loss(conf = confs_1, residual = input['residual_1'])
        loss_conf_2 = self.conf_loss(conf = confs_2, residual = input['residual_2'])
        loss_conf = .5 * loss_conf_1 + .5 * loss_conf_2

        conf_weights = self.get_conf_weights(input['residual_1']) * self.get_conf_weights(input['residual_2'])
        conf_weights = conf_weights.squeeze().mean(dim=(1,2))
        conf_weights = conf_weights.detach() / conf_weights.detach().mean()

        loss_sim = self.sim_loss(feats_a = match_feats_1,
                                 feats_b = match_feats_2,
                                 Hs_a = input['Hs_a'],
                                 Hs_b = input['Hs_b'],
                                 M_a_b = input['M_a_b'],
                                 conf_weights = conf_weights)
        
        loss_ctx_1 = self.ctx_loss(input['imgs_pred_1'],input['imgs_1'])
        loss_ctx_2 = self.ctx_loss(input['imgs_pred_2'],input['imgs_2'])
        loss_ctx = .5 * loss_ctx_1 + .5 * loss_ctx_2
        
        loss = loss_sim + loss_conf + loss_ctx
        loss_details = {
            'loss':loss.clone().detach(),
            'loss_sim':loss_sim.clone().detach(),
            'loss_conf':loss_conf.clone().detach(),
            'loss_ctx':loss_ctx.clone().detach()
        }


        return loss,loss_details
        