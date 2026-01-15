import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from shared.rpc import RPCModelParameterTorch,project_linesamp
from infer.utils import apply_H
from kornia.feature import LoFTR

class MatchSolver():
    def __init__(self,
                 imgs_a:np.ndarray,imgs_b:np.ndarray,
                 H_as:torch.Tensor,H_bs:torch.Tensor,
                 rpc_a:RPCModelParameterTorch = None,rpc_b:RPCModelParameterTorch = None,
                 height:torch.Tensor = None,
                 method = 'loftr',
                 device = 'cuda',
                 reporter = None):
        self.imgs_a = imgs_a
        self.imgs_b = imgs_b
        self.H_as = H_as
        self.H_bs = H_bs
        self.rpc_a = rpc_a
        self.rpc_b = rpc_b
        self.height = height
        self.method = method
        self.device = device
        self.reporter = reporter
        self.N,self.H,self.W = imgs_a.shape[:3]
        if method == 'loftr':
            self.matcher = LoFTR(pretrained=None)
            ckpt = torch.load('weights/loftr_outdoor.ckpt')['state_dict']
            self.matcher.load_state_dict(ckpt)
            self.matcher.to(device).eval()
    
    def run_sift_matching(self, img_a_np, img_b_np):
        sift = cv2.SIFT_create()

        gray_a = cv2.equalizeHist(cv2.cvtColor(img_a_np, cv2.COLOR_RGB2GRAY))
        gray_b = cv2.equalizeHist(cv2.cvtColor(img_b_np, cv2.COLOR_RGB2GRAY))

        kp1, des1 = sift.detectAndCompute(gray_a, None)
        kp2, des2 = sift.detectAndCompute(gray_b, None)

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return np.empty((0, 2)), np.empty((0, 2))

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
            
        pts_a = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp2[m.trainIdx].pt for m in good])
        
        return pts_a,pts_b
    
    @torch.no_grad()
    def run_loftr_matching(self, img_a_np, img_b_np):
        img0 = cv2.cvtColor(img_a_np, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img_b_np, cv2.COLOR_RGB2GRAY)
        
        img0 = torch.from_numpy(img0).float().to(self.device) / 255.0
        img1 = torch.from_numpy(img1).float().to(self.device) / 255.0
        
        batch = {'image0': img0.unsqueeze(0).unsqueeze(0), 'image1': img1.unsqueeze(0).unsqueeze(0)}
        
        correspondences = self.matcher(batch)
        
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()

        return mkpts0, mkpts1

    def solve(self):
        results = []
        for i in range(self.N):
            img_a,img_b = self.imgs_a[i],self.imgs_b[i]
            if self.method == 'loftr':
                pts_a,pts_b = self.run_loftr_matching(img_a,img_b)
            else:
                pts_a,pts_b = self.run_sift_matching(img_a,img_b)

            if pts_a.shape[0] < 10 or pts_b.shape[0] < 10:
                results.append(np.array([
                    [1.0,0.0,0.0],
                    [0.0,1.0,0.0]
                ]))
                continue

            pts_a_rc,pts_b_rc = pts_a[:,[1,0]],pts_b[:,[1,0]]

            pts_a_rc_t = torch.from_numpy(pts_a_rc).unsqueeze(0).to(self.device)
            pts_b_rc_t = torch.from_numpy(pts_b_rc).unsqueeze(0).to(self.device)
            
            H_a_inv = torch.inverse(self.H_as[i]).unsqueeze(0).to(self.device, dtype=torch.float32)
            H_b_inv = torch.inverse(self.H_bs[i]).unsqueeze(0).to(self.device, dtype=torch.float32)
            
            pts_a_global = apply_H(pts_a_rc_t, H_a_inv, self.device).squeeze(0).cpu().numpy()
            pts_b_global = apply_H(pts_b_rc_t, H_b_inv, self.device).squeeze(0).cpu().numpy()

            idxs_b = pts_b_rc.astype(int)
            heights = self.height[i,idxs_b[:,0],idxs_b[:,1]]

            pts_b_global_to_a = np.stack(project_linesamp(self.rpc_b,self.rpc_a,
                                                          pts_b_global[:,0],pts_b_global[:,1],heights,'numpy'),
                                        axis=-1)
            
            M,_ = cv2.estimateAffine2D(pts_a_global,pts_b_global_to_a,ransacReprojThreshold=3.0)
            
            results.append(M)
        
        results = torch.from_numpy(np.stack(results,axis=0)).to(device=self.device,dtype=torch.float32)

        return results
            
