import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os
from typing import List
from copy import deepcopy

from shared.rpc import RPCModelParameterTorch
from rs_image import RSImage
from utils import find_intersection,find_squares,extract_features,get_coord_mat,apply_H,apply_M,solve_weighted_affine,haversine_distance,quadsplit_diags,avg_downsample,check_invalid_tensors
from shared.utils import get_current_time
from window import Window
from model.encoder import Encoder
from model.gru import GRUBlock
from solve.solve_windows import WindowSolver

default_configs = {
    'max_window_num':-1,
    'min_window_size':500,
    'max_window_size':8000,
    'min_area_ratio':0.5,
    'output_path':'./result'
}

class Pair():
    def __init__(self,rs_image_a:RSImage,rs_image_b:RSImage,id_a:int,id_b:int,configs = {},device:str = 'cuda'):
        self.rs_image_a = rs_image_a
        self.rs_image_b = rs_image_b
        self.id_a = id_a
        self.id_b = id_b
        self.configs = {**default_configs,**configs}
        self.device = device
        self.window_pairs:List[WindowPair] = []
        self.window_size = -1
        self.solver_ab = Solver(rs_image_a=rs_image_a,
                                rs_image_b=rs_image_b,
                                configs=self.configs,
                                device=device)
        self.solver_ba = Solver(rs_image_a=rs_image_b,
                                rs_image_b=rs_image_a,
                                configs=self.configs,
                                device=device)

    def solve_affines(self,encoder:Encoder,gru:GRUBlock):
        affine_ab = self.solver_ab.solve_affine(encoder,gru)
        affine_ba = self.solver_ba.solve_affine(encoder,gru)
        return affine_ab,affine_ba
    
    
    def check_error(self):
        lines_i = self.rs_image_a.tie_points[:,0]
        samps_i = self.rs_image_b.tie_points[:,1]
        heights_i = self.rs_image_a.dem[lines_i,samps_i]
        lats_i, lons_i = self.rs_image_a.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
        coords_i = np.stack([lats_i, lons_i], axis=-1)
        
        lines_j = self.rs_image_b.tie_points[:,0]
        samps_j = self.rs_image_b.tie_points[:,1]
        heights_j = self.rs_image_b.dem[lines_j,samps_j]
        lats_j, lons_j = self.rs_image_b.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
        coords_j = np.stack([lats_j, lons_j], axis=-1)

        distances = haversine_distance(coords_i, coords_j)
        return distances
    

class Solver():
    def __init__(self,rs_image_a:RSImage,rs_image_b:RSImage,configs:dict,device:str = 'cuda'):
        self.rs_image_a = rs_image_a
        self.rs_image_b = rs_image_b
        self.rpc_a = deepcopy(rs_image_a.rpc)
        self.rpc_b = deepcopy(rs_image_b.rpc)
        self.configs = configs
        self.device = device
        self.window_pairs:List[WindowPair] = []
        self.window_size = -1

    def init_window_pairs(self,a_max = 8000,a_min = 500,area_ratio = 0.5):
        corners_a = self.rs_image_a.__get_corner_xys__() # (4,2)
        corners_b = self.rs_image_b.__get_corner_xys__()

        polygon_corners = find_intersection(np.stack([corners_a,corners_b],axis=0))
        window_diags = find_squares(polygon_corners,a_max,a_min,area_ratio) # N,2,2

        self.build_window_pairs(window_diags)

    def quadsplit_windows(self):
        diags_old = []
        for window_pair in self.window_pairs:
            diag = window_pair.diag
            diags_old.append(diag)
        diags_old = np.stack(diags_old,axis=0)
        diags_new = quadsplit_diags(diags_old)

        self.build_window_pairs(diags_new)
        
    def build_window_pairs(self,window_diags):
        if self.configs['max_window_num'] > 0 and window_diags.shape[0] > self.configs['max_window_num']:
            idxs = np.random.choice(range(window_diags.shape[0]),self.configs['max_window_num'])
            window_diags = window_diags[idxs]
        self.window_size = np.abs(window_diags[0,0,1] - window_diags[0,0,0])

        data_a,data_b = self.get_data_by_diags(window_diags)
        self.window_pairs = self.generate_window_pairs(data_a,data_b,window_diags)
        self.window_pairs_num = len(self.window_pairs)
    
    def get_data_by_diags(self,diags):
        corners_linesamps_a = self.rs_image_a.convert_diags_to_corners(diags,self.rpc_a)
        corners_linesamps_b = self.rs_image_b.convert_diags_to_corners(diags,self.rpc_b)
        
        imgs_a,dems_a,Hs_a = self.rs_image_a.crop_windows(corners_linesamps_a)
        imgs_b,dems_b,Hs_b = self.rs_image_b.crop_windows(corners_linesamps_b)
        
        return (imgs_a,dems_a,Hs_a),(imgs_b,dems_b,Hs_b)
    
    def generate_window_pairs(self,data_a,data_b,diags):
        imgs_a,dems_a,Hs_a = data_a
        imgs_b,dems_b,Hs_b = data_b
        N = imgs_a.shape[0]
        window_pairs = []
        for i in range(N):
            window_a = Window(imgs_a[i],dems_a[i],self.rpc_a,Hs_a[i])
            window_b = Window(imgs_b[i],dems_b[i],self.rpc_b,Hs_b[i])
            window_pair = WindowPair(window_a,window_b,diags[i])
            window_pairs.append(window_pair)
        return window_pairs
    
    def collect_imgs(self):
        """
        Returns:
            (imgs_a,imgs_b) -> (N,H,W,3),(N,H,W,3)
        """
        imgs_a = []
        imgs_b = []
        for window_pair in self.window_pairs:
            img_a = window_pair.window_a.img
            img_b = window_pair.window_b.img
            imgs_a.append(img_a)
            imgs_b.append(img_b)
        return np.stack(imgs_a,axis=0),np.stack(imgs_b,axis=0)
    
    def collect_dems(self,to_tensor = False):
        """
        Returns:
            (dems_a,dems_b) -> (N,H,W),(N,H,W)
        """
        dems_a = []
        dems_b = []
        for window_pair in self.window_pairs:
            dem_a = window_pair.window_a.dem
            dem_b = window_pair.window_b.dem
            dems_a.append(dem_a)
            dems_b.append(dem_b)
        dems_a = np.stack(dems_a,axis=0)
        dems_b = np.stack(dems_b,axis=0)
        if to_tensor:
            dems_a = torch.from_numpy(dems_a).to(device=self.device,dtype=torch.float32)
            dems_b = torch.from_numpy(dems_b).to(device=self.device,dtype=torch.float32)
        return dems_a,dems_b
    
    def collect_Hs(self,to_tensor = False):
        """
        Returns:
            (Hs_a,Hs_b) -> (N,3,3),(N,3,3)
        """
        Hs_a = []
        Hs_b = []
        for window_pair in self.window_pairs:
            H_a = window_pair.window_a.H
            H_b = window_pair.window_b.H
            Hs_a.append(H_a)
            Hs_b.append(H_b)
        Hs_a = np.stack(Hs_a,axis=0)
        Hs_b = np.stack(Hs_b,axis=0)
        if to_tensor:
            Hs_a = torch.from_numpy(Hs_a).to(device=self.device,dtype=torch.float32)
            Hs_b = torch.from_numpy(Hs_b).to(device=self.device,dtype=torch.float32)
        return Hs_a,Hs_b

    def get_window_affines(self,encoder:Encoder,gru:GRUBlock):
        imgs_a,imgs_b = self.collect_imgs()
        dems_a,dems_b = self.collect_dems(to_tensor=True)
        Hs_a,Hs_b = self.collect_Hs(to_tensor=True)
        B,H,W = imgs_a.shape[:3]
        feats_a,feats_b = extract_features(encoder,imgs_a,imgs_b,device=self.device)
        for i in range(imgs_a.shape[0]):
            cv2.imwrite(os.path.join(self.configs['output_path'],f"{get_current_time()}_{i}_{self.window_size}_a.png"),imgs_a[i])
            cv2.imwrite(os.path.join(self.configs['output_path'],f"{get_current_time()}_{i}_{self.window_size}_b.png"),imgs_b[i])
        height = avg_downsample(dems_a,16)
        check_invalid_tensors([dems_a,Hs_a,Hs_b,feats_a[0],feats_a[1],feats_a[2]],height)
        solver = WindowSolver(B,H,W,
                                gru=gru,
                                feats_a=feats_a,feats_b=feats_b,
                                H_as=Hs_a,H_bs=Hs_b,
                                rpc_a=self.rpc_a,rpc_b=self.rpc_b,
                                height=height)
        
        preds = solver.solve(flag = 'ab',final_only=True)

        _,_,confs_a = feats_a
        _,_,confs_b = feats_b
        scores_a = confs_a.reshape(B,-1).mean(dim=1) # B,
        scores_b = confs_b.reshape(B,-1).mean(dim=1)
        scores = torch.sqrt(scores_a * scores_b) # B,
        
        return preds,scores
    
    def merge_affines(self,affines:torch.Tensor,Hs:torch.Tensor,scores:torch.Tensor):
        """
        Args:
            affines: torch.Tensor, (B,2,3)
            Hs: torch.Tensor, (B,3,3)
            scores: torch.Tensor, (B,)
        Returns:
            affine: torch.Tensor, (2,3)
        """
        coords_mat = get_coord_mat(32,32,Hs.shape[0],16,self.device) # (B,32,32,2)
        coords_mat_flat = coords_mat.flatten(1,2) # (B,1024,2)
        coords_src = apply_H(coords=coords_mat_flat,Hs=torch.linalg.inv(Hs),device=self.device) # (B,1024,2) 大图坐标系下的坐标
        coords_dst = apply_M(coords=coords_src,Ms=affines,device=self.device) # (B,1024,2) 对每个窗口应用其仿射变换
        coords_src = coords_src.reshape(-1,2) # B*1024,2
        coords_dst = coords_dst.reshape(-1,2) # B*1024,2

        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        scores_norm = scores_norm.unsqueeze(-1).expand(-1,1024).reshape(-1) # B*1024

        merged_affine = solve_weighted_affine(coords_src,coords_dst,scores_norm)

        return merged_affine

    def solve_level_affine(self,encoder:Encoder,gru:GRUBlock):
        """
        Returns:
            affine: torch.Tensor, (2,3)
        """
        Hs_a,Hs_b = self.collect_Hs(to_tensor=True)
        preds,scores = self.get_window_affines(encoder,gru)
        affine = self.merge_affines(preds,Hs_a,scores)
        print(f"\n{affine}\n")
        self.rpc_a.Update_Adjust(affine)

        return affine
    
    def solve_affine(self,encoder:Encoder,gru:GRUBlock):
        self.init_window_pairs(a_max=self.configs['max_window_size'],
                               a_min=self.configs['min_window_size'],
                               area_ratio=self.configs['min_area_ratio'])
        while self.window_size >= self.configs['min_window_size']:
            self.solve_level_affine(encoder,gru)
            self.quadsplit_windows()
        return self.rpc_a.adjust_params

        
class WindowPair():
    def __init__(self,window_a:Window,window_b:Window,diag:np.ndarray):
        self.window_a = window_a
        self.window_b = window_b
        self.diag = diag