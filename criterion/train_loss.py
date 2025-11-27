import torch
import torch.nn as nn
import torch.nn.functional as F
from .sim_loss import SimLoss
from .conf_loss import ConfLoss
from .affine_loss import AffineLoss
from .consist_loss import ConsistLoss
from .utils import invert_affine_matrix

class Loss(nn.modules):
    def __init__(self,img_size = (512,512), downsample_factor = 16,temperature = 0.07,decay_rate = 0.8,reg_weight = 0.001,device = 'cuda'):
        self.sim_loss = SimLoss(downsample_factor = downsample_factor,
                                temperature = temperature)
        self.conf_loss = ConfLoss()
        self.affine_loss = AffineLoss(img_size = img_size,
                                      grid_stride = downsample_factor,
                                      decay_rate = decay_rate,
                                      reg_weight = reg_weight,
                                      device = device)
        self.consist_loss = ConsistLoss(img_size = img_size,
                                        grid_stride = downsample_factor,
                                        decay_rate = decay_rate)
    
    def forward(self,input):
        match_feats_1, ctx_feats_1, confs_1 = input['feats_1']
        match_feats_2, ctx_feats_2, confs_2 = input['feats_2']

        loss_sim = self.sim_loss(feats_a = match_feats_1,
                                 feats_b = match_feats_2,
                                 Hs_a = input['Hs_a'],
                                 Hs_b = input['Hs_b'],
                                 M_a_b = input['M_a_b'])
        
        loss_conf_1 = self.conf_loss(conf = confs_1, residual = input['residual_1'])
        loss_conf_2 = self.conf_loss(conf = confs_2, residual = input['residual_1'])
        loss_conf = .5 * loss_conf_1 + .5 * loss_conf_2

        loss_affine_1 = self.affine_loss(delta_affines = input['preds_1'],
                                         Hs_a = input['Hs_a'],
                                         Hs_b = input['Hs_b'],
                                         M_a_b = input['M_a_b'])
        
        loss_affine_2 = self.affine_loss(delta_affines = input['preds_2'],
                                         Hs_a = input['Hs_b'],
                                         Hs_b = input['Hs_a'],
                                         M_a_b = invert_affine_matrix(input['M_a_b']))
        
        loss_affine = .5 * loss_affine_1 + .5 * loss_affine_2

        loss_consist = self.consist_loss(delta_affine_a = input['preds_1'],
                                         delta_affine_b = input['preds_2'])
        
        loss = loss_sim + loss_conf + loss_affine + loss_consist
        loss_details = {
            'loss':loss.clone().detach(),
            'loss_sim':loss_sim.clone().detach(),
            'loss_conf':loss_conf.clone().detach(),
            'loss_affine':loss_affine.clone().detach(),
            'loss_consist':loss_consist.clone().detach()
        }


        return loss,loss_details
        