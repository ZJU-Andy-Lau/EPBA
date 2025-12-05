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
from utils import find_intersection,find_squares,extract_features,get_coord_mat,apply_H,apply_M,solve_weighted_affine,haversine_distance,quadsplit_diags,affine_xy_to_rowcol
from shared.utils import get_current_time,check_invalid_tensors
from shared.visualize import make_checkerboard
import shared.visualize as visualizer
from criterion.utils import invert_affine_matrix
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
                                configs={
                                    **self.configs,
                                    **{
                                        'output_path':os.path.join(self.configs['output_path'],'solve_ab')
                                    },
                                    
                                },
                                device=device)
        self.solver_ba = Solver(rs_image_a=rs_image_b,
                                rs_image_b=rs_image_a,
                                configs={
                                    **self.configs,
                                    **{
                                        'output_path':os.path.join(self.configs['output_path'],'solve_ba')
                                    },
                                    
                                },
                                device=device)

    def solve_affines(self,encoder:Encoder,gru:GRUBlock):
        print("solve ab")
        affine_ab = self.solver_ab.solve_affine(encoder,gru)
        print("solve ba")
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
        os.makedirs(self.configs['output_path'],exist_ok=True)

    def init_window_pairs(self,a_max = 8000,a_min = 500,area_ratio = 0.5):
        corners_a = self.rs_image_a.__get_corner_xys__() # (4,2)
        corners_b = self.rs_image_b.__get_corner_xys__()

        polygon_corners = find_intersection(np.stack([corners_a,corners_b],axis=0))
        window_diags = find_squares(polygon_corners,a_max,a_min,area_ratio) # N,2,2

        self.build_window_pairs(window_diags)

    def quadsplit_windows(self):
        new_diags = []
        scores = []
        for window_pair in self.window_pairs:
            new_diag,score = window_pair.quadsplit()
            new_diags.append(new_diag)
            scores.append(score)
            window_pair.clear()
        
        new_diags = np.concatenate(new_diags,axis=0) # （4*N,2,2)
        scores = np.concatenate(scores,axis=0) #(4*N,)

        self.build_window_pairs(new_diags,scores)
        
    def build_window_pairs(self,window_diags,scores = None):
        if self.configs['max_window_num'] > 0 and window_diags.shape[0] > self.configs['max_window_num']:
            if scores is None:
                idxs = np.random.choice(range(window_diags.shape[0]),self.configs['max_window_num'])
                window_diags = window_diags[idxs]
            else:
                sorted_idxs = np.argsort(-scores)[:self.configs['max_window_num']] #从大到小
                window_diags = window_diags[sorted_idxs]
        self.window_size = np.abs(window_diags[0,1,0] - window_diags[0,0,0])

        data_a,data_b = self.get_data_by_diags(window_diags)
        self.window_pairs = self.generate_window_pairs(data_a,data_b,window_diags)
        self.window_pairs_num = len(self.window_pairs)
    
    def get_data_by_diags(self,diags,rpc_a = None,rpc_b = None):
        if rpc_a is None:
            rpc_a = self.rpc_a
        if rpc_b is None:
            rpc_b = self.rpc_b
        corners_linesamps_a = self.rs_image_a.convert_diags_to_corners(diags,rpc_a)
        corners_linesamps_b = self.rs_image_b.convert_diags_to_corners(diags,rpc_b)
        
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

    def distribute_feats(self,feats_a,feats_b):
        match_feats_a,ctx_feats_a,confs_a = feats_a
        match_feats_b,ctx_feats_b,confs_b = feats_b
        for idx, window_pair in enumerate(self.window_pairs):
            window_pair.window_a.load_feats((match_feats_a[idx],ctx_feats_a[idx],confs_a[idx]))
            window_pair.window_b.load_feats((match_feats_b[idx],ctx_feats_b[idx],confs_b[idx]))

    def check_adjust(self):
        ori_rpc = deepcopy(self.rs_image_a.rpc)
        test_diag = self.window_pairs[0].diag[None].copy()
        test_diag[:,1,:] = test_diag[:,0,:] + [500,-500]
        data_ori_a,data_b = self.get_data_by_diags(test_diag,rpc_a=ori_rpc)
        data_a,_ = self.get_data_by_diags(test_diag)
        img_ori,img_a,img_b = data_ori_a[0][0],data_a[0][0],data_b[0][0]
        checker_ori_a = make_checkerboard(img_ori,img_a,num_tiles=8)
        checker_ori_b = make_checkerboard(img_ori,img_b,num_tiles=8)
        checker_a_b = make_checkerboard(img_a,img_b,num_tiles=8)
        return checker_ori_a,checker_ori_b,checker_a_b,img_a,img_b
    
    def check_rpc(self,rpc:RPCModelParameterTorch):
        ori_rpc = deepcopy(self.rs_image_a.rpc)
        test_diag = self.window_pairs[0].diag[None].copy()
        data_ori_a,data_b = self.get_data_by_diags(test_diag,rpc_a=ori_rpc)
        data_a,_ = self.get_data_by_diags(test_diag,rpc_a=rpc)
        img_ori,img_a,img_b = data_ori_a[0][0],data_a[0][0],data_b[0][0]
        checker_ori_a = make_checkerboard(img_ori,img_a,num_tiles=8)
        checker_ori_b = make_checkerboard(img_ori,img_b,num_tiles=8)
        checker_a_b = make_checkerboard(img_a,img_b,num_tiles=8)
        return checker_ori_a,checker_ori_b,checker_a_b,img_a,img_b

    def get_window_affines(self,encoder:Encoder,gru:GRUBlock):
        imgs_a,imgs_b = self.collect_imgs()
        dems_a,dems_b = self.collect_dems(to_tensor=True)
        Hs_a,Hs_b = self.collect_Hs(to_tensor=True)
        B,H,W = imgs_a.shape[:3]
        feats_a,feats_b = extract_features(encoder,imgs_a,imgs_b,device=self.device)
        self.distribute_feats(feats_a,feats_b)
        # check_invalid_tensors([dems_a,Hs_a,Hs_b,feats_a[0],feats_a[1],feats_a[2],height])
        solver = WindowSolver(B,H,W,
                              gru=gru,
                              feats_a=feats_a,feats_b=feats_b,
                              H_as=Hs_a,H_bs=Hs_b,
                              rpc_a=self.rpc_a,rpc_b=self.rpc_b,
                              height=dems_a,
                              test_imgs_a=imgs_a,test_imgs_b=imgs_b)
        
        preds,vis = solver.solve(flag = 'ab',final_only=True,return_vis=True)
        
        cv2.imwrite(os.path.join(self.configs['output_path'],f"{self.window_size}_pyr_lvl0.png"),vis['level_0'])
        cv2.imwrite(os.path.join(self.configs['output_path'],f"{self.window_size}_pyr_lvl1.png"),vis['level_1'])
        for i in range(vis['test']['imgs_a'].shape[0]):
            cv2.imwrite(os.path.join(self.configs['output_path'],f"{self.window_size}_test_img_{i}_a.png"),vis['test']['imgs_a'][i])
            cv2.imwrite(os.path.join(self.configs['output_path'],f"{self.window_size}_test_img_{i}_b.png"),vis['test']['imgs_b'][i])
            cv2.imwrite(os.path.join(self.configs['output_path'],f"{self.window_size}_test_img_{i}_ab.png"),make_checkerboard(vis['test']['imgs_a'][i],
                                                                                                                              vis['test']['imgs_b'][i]))
        rpc_a_test = deepcopy(self.rpc_a)
        rpc_a_test.Update_Adjust(invert_affine_matrix(preds[0]))
        output_path = os.path.join(self.configs['output_path'],f"check_rpc_level_{self.window_size}")
        os.makedirs(output_path,exist_ok=True)
        checker_ori_a,checker_ori_b,checker_a_b,img_a,img_b = self.check_rpc(rpc_a_test)
        cv2.imwrite(os.path.join(output_path,f"a.png"),img_a)
        cv2.imwrite(os.path.join(output_path,f"b.png"),img_b)
        cv2.imwrite(os.path.join(output_path,f"ori_a.png"),checker_ori_a)
        cv2.imwrite(os.path.join(output_path,f"ori_b.png"),checker_ori_b)
        cv2.imwrite(os.path.join(output_path,f"a_b.png"),checker_a_b)


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
        for affine in affines:
            print(f"{affine.detach().cpu().numpy()}\n")
        coords_mat = get_coord_mat(32,32,Hs.shape[0],16,self.device) # (B,32,32,2)
        coords_mat_flat = coords_mat.flatten(1,2) # (B,1024,2)
        coords_src = apply_H(coords=coords_mat_flat,Hs=torch.linalg.inv(Hs),device=self.device) # (B,1024,2) 大图坐标系下的坐标
        coords_dst = apply_M(coords=coords_src,Ms=affines,device=self.device) # (B,1024,2) 对每个窗口应用其仿射变换
        coords_src = coords_src.reshape(-1,2) # B*1024,2
        coords_dst = coords_dst.reshape(-1,2) # B*1024,2

        if torch.abs(scores.max() - scores.min()) < 1e-4:
            scores_norm = torch.ones(size=scores.shape,device=scores.device,dtype=scores.dtype)
        else:
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min())

        scores_norm = scores_norm.unsqueeze(-1).expand(-1,1024).reshape(-1) # B*1024
        
        merged_affine = solve_weighted_affine(coords_src,coords_dst,scores_norm)
        print(f"merged:\n{merged_affine.detach().cpu().numpy()}\n")
        # check_invalid_tensors([affines,coords_mat_flat,coords_src,coords_dst,scores_norm,merged_affine],"[merge affines]: ")

        return merged_affine

    def solve_level_affine(self,encoder:Encoder,gru:GRUBlock):
        """
        Returns:
            affine: torch.Tensor, (2,3)
        """
        Hs_a,Hs_b = self.collect_Hs(to_tensor=True)
        preds,scores = self.get_window_affines(encoder,gru)
        # check_invalid_tensors([preds,scores],"[solve level affine]: ")
        affine = self.merge_affines(preds,Hs_a,scores)
        self.rpc_a.Clear_Adjust()
        self.rpc_a.Update_Adjust(affine)
        print(f"accumulate:\n{self.rpc_a.adjust_params.detach().cpu().numpy()}\n")

        output_path = os.path.join(self.configs['output_path'],f"check_adjust_level_{self.window_size}")
        os.makedirs(output_path,exist_ok=True)
        checker_ori_a,checker_ori_b,checker_a_b,img_a,img_b = self.check_adjust()
        cv2.imwrite(os.path.join(output_path,f"a.png"),img_a)
        cv2.imwrite(os.path.join(output_path,f"b.png"),img_b)
        cv2.imwrite(os.path.join(output_path,f"ori_a.png"),checker_ori_a)
        cv2.imwrite(os.path.join(output_path,f"ori_b.png"),checker_ori_b)
        cv2.imwrite(os.path.join(output_path,f"a_b.png"),checker_a_b)
        self.window_pairs[0].visualize(os.path.join(output_path,f'feats_vis_{self.window_size}'))

        return affine
    
    @torch.no_grad()
    def solve_affine(self,encoder:Encoder,gru:GRUBlock):
        self.init_window_pairs(a_max=self.configs['max_window_size'],
                               a_min=self.configs['min_window_size'],
                               area_ratio=self.configs['min_area_ratio'])
        while self.window_size >= self.configs['min_window_size']:
            print(f"Solve level {self.window_size} m")
            self.solve_level_affine(encoder,gru)
            self.quadsplit_windows()
        return self.rpc_a.adjust_params

        
class WindowPair():
    def __init__(self,window_a:Window,window_b:Window,diag:np.ndarray):
        self.window_a = window_a
        self.window_b = window_b
        self.diag = diag
    
    def _get_score(self,tlrc,brrc):
        confs_a = self.window_a.confs[...,tlrc[0]:brrc[0],tlrc[1]:brrc[1]].reshape(-1)
        confs_b = self.window_b.confs[...,tlrc[0]:brrc[0],tlrc[1]:brrc[1]].reshape(-1)
        scores = torch.sqrt(confs_a * confs_b).mean()
        return scores.item()

    def quadsplit(self):
        """
        Return:
        new_diags: np.ndarray (4,2,2)
        scores: np.ndarray (4,)
        """
        tlx,tly = self.diag[0]
        brx,bry = self.diag[1]
        mid_x = (tlx + brx) * 0.5
        mid_y = (tly + bry) * 0.5
        H,W = self.window_a.confs.shape[-2:]
        mid_row = H // 2
        mid_col = W // 2
        new_diags = np.array([
            [
                [tlx,tly],
                [mid_x,mid_y]
            ],
            [
                [mid_x,tly],
                [brx,mid_y]
            ],
            [
                [tlx,mid_y],
                [mid_x,bry]
            ],
            [
                [mid_x,mid_y],
                [brx,bry]
            ]
        ])
        scores = np.array([
            self._get_score([0,0],[mid_row,mid_col]),
            self._get_score([0,mid_col],[mid_row,W]),
            self._get_score([mid_row,0],[H,mid_col]),
            self._get_score([mid_row,mid_col],[H,W]),
        ])
        return new_diags,scores
    
    def visualize(self,output_path:str):
        match_feat_a,ctx_feat_a,conf_a = self.window_a.match_feats.permute(1,2,0),self.window_a.ctx_feats.permute(1,2,0),self.window_a.confs.permute(1,2,0)
        match_feat_b,ctx_feat_b,conf_b = self.window_b.match_feats.permute(1,2,0),self.window_b.ctx_feats.permute(1,2,0),self.window_b.confs.permute(1,2,0)
        img_a,img_b = self.window_a.img,self.window_b.img
        match_feats_vis = visualizer.feats_pca(torch.stack([match_feat_a,match_feat_b],dim=0).cpu().numpy())
        ctx_feats_vis = visualizer.feats_pca(torch.stack([ctx_feat_a,ctx_feat_b],dim=0).cpu().numpy())
        img_match_vis = visualizer.vis_sparse_match(img_a,img_b,match_feat_a.permute(2,0,1).cpu().numpy(),match_feat_b.permute(2,0,1).cpu().numpy(),conf_a.squeeze().cpu().numpy())
        pyr_res_vis = visualizer.vis_pyramid_response(match_feat_a.permute(2,0,1).cpu().numpy(),match_feat_b.permute(2,0,1).cpu().numpy(),level_num=2)
        conf_vis = visualizer.vis_confidence_overlay(img_a,conf_a.squeeze().cpu().numpy())
        os.makedirs(output_path,exist_ok=True)
        cv2.imwrite(os.path.join(output_path,'match_feat_a.png'),match_feats_vis[0])
        cv2.imwrite(os.path.join(output_path,'match_feat_b.png'),match_feats_vis[1])
        cv2.imwrite(os.path.join(output_path,'ctx_feat_a.png'),ctx_feats_vis[0])
        cv2.imwrite(os.path.join(output_path,'ctx_feat_b.png'),ctx_feats_vis[1])
        cv2.imwrite(os.path.join(output_path,'img_match.png'),img_match_vis)
        cv2.imwrite(os.path.join(output_path,'pyr_res.png'),pyr_res_vis)
        cv2.imwrite(os.path.join(output_path,'conf.png'),conf_vis)        

    def clear(self):
        self.window_a.clear()
        self.window_b.clear()