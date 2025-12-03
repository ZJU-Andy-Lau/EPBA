import torch
import torch.nn as nn
import torch.nn.functional as F
from .sim_loss import SimLoss
from .conf_loss import ConfLoss
from .affine_loss import AffineLoss
from .consist_loss import ConsistLoss
from .ctx_loss import CtxLoss
from utils import invert_affine_matrix

class Loss(nn.Module):
    def __init__(self,img_size = (512,512), downsample_factor = 16,temperature = 0.07,decay_rate = 0.8,reg_weight = 0.001,device = 'cuda'):
        super().__init__()
        self.sim_loss = SimLoss(downsample_factor = downsample_factor,
                                temperature = temperature)
        self.conf_loss = ConfLoss()
        self.affine_loss = AffineLoss(img_size = img_size,
                                      grid_stride = 64,
                                      decay_rate = decay_rate,
                                      reg_weight = reg_weight,
                                      device = device)
        self.consist_loss = ConsistLoss(img_size = img_size,
                                        grid_stride = downsample_factor,
                                        decay_rate = decay_rate,
                                        device = device)
        self.ctx_loss = CtxLoss()
    
    def forward(self,input, return_details=False): # [修改] 增加 return_details 参数
        match_feats_1, _, confs_1 = input['feats_1']
        match_feats_2, _, confs_2 = input['feats_2']

        loss_sim = self.sim_loss(feats_a = match_feats_1,
                                 feats_b = match_feats_2,
                                 Hs_a = input['Hs_a'],
                                 Hs_b = input['Hs_b'],
                                 M_a_b = input['M_a_b'])
        
        loss_conf_1 = self.conf_loss(conf = confs_1, residual = input['residual_1'])
        loss_conf_2 = self.conf_loss(conf = confs_2, residual = input['residual_1'])
        loss_conf = .5 * loss_conf_1 + .5 * loss_conf_2

        # [修改] 处理 Affine Loss 的可视化返回
        if return_details:
            # 仅对 loss_affine_1 (A->B) 进行可视化采集，避免信息过载
            loss_affine_1, loss_affine_last_1, affine_details = self.affine_loss(delta_affines = input['preds_1'],
                                                                                Hs_a = input['Hs_a'],
                                                                                Hs_b = input['Hs_b'],
                                                                                M_a_b = input['M_a_b'],
                                                                                norm_factor = input['norm_factor_a'],
                                                                                return_details = True)
        else:
            loss_affine_1, loss_affine_last_1 = self.affine_loss(delta_affines = input['preds_1'],
                                                                Hs_a = input['Hs_a'],
                                                                Hs_b = input['Hs_b'],
                                                                M_a_b = input['M_a_b'],
                                                                norm_factor = input['norm_factor_a'])
            affine_details = None
        
        # 反向过程 (B->A) 暂时不采集可视化信息
        loss_affine_2, loss_affine_last_2 = self.affine_loss(delta_affines = input['preds_2'],
                                                            Hs_a = input['Hs_b'],
                                                            Hs_b = input['Hs_a'],
                                                            M_a_b = invert_affine_matrix(input['M_a_b']),
                                                            norm_factor = input['norm_factor_b'])
        
        loss_affine = .5 * loss_affine_1 + .5 * loss_affine_2
        loss_affine_last = .5 * loss_affine_last_1 + .5 * loss_affine_last_2

        loss_consist = self.consist_loss(delta_affine_a = input['preds_1'],
                                         delta_affine_b = input['preds_2'])
        
        loss_ctx_1 = self.ctx_loss(input['imgs_pred_1'],input['imgs_1'])
        loss_ctx_2 = self.ctx_loss(input['imgs_pred_2'],input['imgs_2'])
        loss_ctx = .5 * loss_ctx_1 + .5 * loss_ctx_2
        
        loss = loss_sim + loss_conf + loss_affine + loss_consist * 0. + loss_ctx
        loss_details = {
            'loss':loss.clone().detach(),
            'loss_sim':loss_sim.clone().detach(),
            'loss_conf':loss_conf.clone().detach(),
            'loss_affine':loss_affine.clone().detach(),
            'loss_affine_last':loss_affine_last.clone().detach(),
            'loss_consist':loss_consist.clone().detach(),
            'loss_ctx':loss_ctx.clone().detach()
        }

        # [修改] 构造 extra_info 返回包
        extra_info = {}
        if affine_details is not None:
            extra_info['affine_details'] = affine_details

        return loss,loss_details,extra_info # [修改] 返回三个值