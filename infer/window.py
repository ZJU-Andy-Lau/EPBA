import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os

from infer.utils import feats_type

from shared.rpc import RPCModelParameterTorch



class Window():
    def __init__(self,img:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,H:np.ndarray):
        self.img = img
        self.dem = dem
        self.rpc = rpc
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.match_feats,self.ctx_feats,self.confs = None,None,None
    
    def load_feats(self,feats:feats_type):
        self.match_feats,self.ctx_feats,self.confs = feats
    
    def clear(self):
        del self.img
        del self.dem
        del self.H
        del self.H_inv
        del self.match_feats
        del self.ctx_feats
        del self.confs

