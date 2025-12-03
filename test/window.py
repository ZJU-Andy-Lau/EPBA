import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os

from utils import feats_type

from shared.rpc import RPCModelParameterTorch



class Window():
    def __init__(self,img:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,H:np.ndarray):
        self.img = img,
        self.dem = dem
        self.rpc = rpc
        self.H = H
        self.H_inv = np.linalg.inv(H)
        self.match_feats,self.ctx_feats,self.confs = None,None,None
    
    def load_feats(self,feats:feats_type):
        self.match_feats,self.ctx_feats,self.confs = feats

