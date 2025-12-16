import warnings

warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2


from utils import warp_quads
from shared.utils import project_mercator,mercator2lonlat,bilinear_interpolate,resample_from_quad
from shared.visualize import make_checkerboard
from shared.rpc import RPCModelParameterTorch,project_linesamp
from tqdm import tqdm,trange
import rasterio
from typing import Tuple

class RSImage():
    def __init__(self,options,root:str,id:int,device:str='cuda'):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = options
        self.root = root
        self.id = id
        self.image = cv2.imread(os.path.join(root,'image.png'),cv2.IMREAD_GRAYSCALE)
        self.image = np.stack([self.image] * 3,axis=-1)
        self.dem = np.load(os.path.join(root,'dem.npy'))
        if os.path.exists(os.path.join(root,'tie_points.txt')):
            self.tie_points = self.__load_tie_points__(os.path.join(root,'tie_points.txt'))
        else:
            self.tie_points = None
        self.device = device
        self.H,self.W = self.image.shape[:2]
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(root,'rpc.txt'))
        self.rpc.to_gpu(device=device)
        
        self.corner_xys = self.__get_corner_xys__()

    def __load_tie_points__(self,path) -> np.ndarray:
        tie_points = np.loadtxt(path,dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1,-1)
        elif tie_points.shape[1] != 2:
            print("tie points format error")
            return None
        return tie_points
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,br,bl] [x,y] np.ndarray
        """
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ([0.,self.W-1.,self.W-1.,0],
                                                     [0.,0.,self.H - 1.,self.H - 1.],
                                                     [self.dem[0,0],self.dem[0,-1],self.dem[-1,-1],self.dem[-1,0]]),dim=-1)
        xys = project_mercator(latlons)
        return xys.cpu().numpy()[:,[1,0]] # y,x -> x,y
    
    def dem_interp(self,sampline:np.ndarray):
        if sampline.ndim == 1:
            sampline = sampline[None]
        return bilinear_interpolate(self.dem,sampline)

    @torch.no_grad()
    def xy_to_sampline(self,xy:np.ndarray,max_iter = 100,rpc:RPCModelParameterTorch = None) -> np.ndarray:
        """
        args:
            xy : (N,2) (x,y) ndarray
        return :
            sampline : (N,2) (samp,line) ndarray
        """
        if xy.ndim == 1:
            xy = xy[None]
        if rpc is None:
            rpc = self.rpc
        latlon = mercator2lonlat(xy[:,[1,0]])
        sampline = np.array([self.W,self.H],dtype=np.float32) * (xy - self.corner_xys[0]) / (self.corner_xys[3] - self.corner_xys[0])
        dem = self.dem_interp(sampline)
        invalid_mask = np.full(dem.shape,True,dtype=bool)
        for iter in range(max_iter):
            sampline_new = np.stack(rpc.RPC_OBJ2PHOTO(latlon[invalid_mask,0],latlon[invalid_mask,1],dem[invalid_mask],'numpy'),axis=-1)
            dis = np.linalg.norm(sampline_new - sampline[invalid_mask],axis=-1)
            sampline[invalid_mask] = sampline_new
            invalid_mask[invalid_mask] = dis > 1.
            if invalid_mask.sum() == 0:
                break
        return sampline.squeeze()
    
    def convert_diags_to_corners(self,diags:np.ndarray,rpc:RPCModelParameterTorch = None):
        """
        Args:
            diags: ndarray, (N,2,2), (x,y)
        Return:
            corners: ndarray, (N,4,2), (line,samp)
        """
        if diags.ndim < 3:
            diags = diags[None]
        N = diags.shape[0]
        corners_xy = np.zeros((N,4,2),dtype=diags.dtype)
        corners_xy[:,0,:] = diags[:,0,:]
        corners_xy[:,1,0] = diags[:,1,0]
        corners_xy[:,1,1] = diags[:,0,1]
        corners_xy[:,2,:] = diags[:,1,:]
        corners_xy[:,3,0] = diags[:,0,0]
        corners_xy[:,3,1] = diags[:,1,1]
        corners_xy_flat = corners_xy.reshape(-1,2) # N*4,2
        corners_samplines_flat = self.xy_to_sampline(corners_xy_flat,rpc=rpc)
        corners_linesamps = corners_samplines_flat.reshape(N,4,2)[...,[1,0]]
        return corners_linesamps



    def crop_windows(self,corners:np.ndarray,output_size=(512, 512)):
        """
        根据给定的四边形顶点坐标，对图像/数组进行透视变换裁切。

        Args:
            corners (np.ndarray): 形状为 (B, 4, 2) 的数组。
                                存储 B 个四边形的顶点坐标，格式为 (row, col)。
                                顺序: 左上, 右上, 右下, 左下 (对应输出的四个角)。
            output_size (tuple): 目标输出尺寸 (target_h, target_w)。默认为 (512, 512)。

        Returns:
            warped_images (B,H,W,3) ,
            warped_dems (B,H,W) ,
            Hs (np.ndarray): 形状为 (B, 3, 3) 的单应变换矩阵。
                            该矩阵是在 (row, col) 坐标系下的变换。
        """
        warped_res,Hs = warp_quads(corners,[self.image,self.dem],output_size)
        warped_imgs,warped_dems = warped_res
        return warped_imgs,warped_dems,Hs













def vis_registration(image_a:RSImage,image_b:RSImage,output_path:str,window_size = (2048,2048),device = 'cuda'):
    H,W = window_size
    center_line,center_samp = image_a.H // 2,image_a.W // 2
    diag = np.array([
            [center_line - H // 2, center_samp - W // 2],
            [center_line + H // 2, center_samp + W // 2]
        ])
    img_a = image_a.image[diag[0,0]:diag[1,0],diag[0,1]:diag[1,1]]
    heights = image_a.dem[diag[0,0]:diag[1,0],diag[0,1]:diag[1,1]]
    heights_flat = heights.reshape(-1) # H*W

    #生成img_a每个像素点的像素坐标
    rows,cols = np.arange(diag[0,0],diag[1,0]),np.arange(diag[0,1],diag[1,1])
    coords = np.stack(np.meshgrid(rows,cols,indexing='ij'),axis=-1) # 512,512,2
    coords_flat_in_a = coords.reshape(-1,2)

    #将坐标投影到b上        
    lines_in_b,samps_in_b = project_linesamp(image_a.rpc,image_b.rpc,
                                             coords_flat_in_a[:,0],coords_flat_in_a[:,1],heights_flat)
    
    #采样
    lines_in_b_norm = 2.0 * lines_in_b.to(torch.float32) / (image_b.H - 1) - 1.0
    samps_in_b_norm = 2.0 * samps_in_b.to(torch.float32) / (image_b.W - 1) - 1.0
    sample_coords = torch.stack([samps_in_b_norm,lines_in_b_norm],dim=-1).reshape(1,H,W,2)
    input_img = torch.from_numpy(image_b.image).to(dtype=torch.float32,device=device)[None].permute(0,3,1,2) # 1,3,H,W
    sampled_img = F.grid_sample(input_img,sample_coords,mode='bilinear',padding_mode='zeros',align_corners=True) # 1,3,H,W
    img_b = sampled_img[0].permute(1,2,0).cpu().numpy()

    img_a_b = make_checkerboard(img_a,img_b)

    cv2.imwrite(os.path.join(output_path,f'registration_{image_a.id}_{image_b.id}.png'),img_a_b)

    
    

    # @torch.no_grad()
    # def get_image_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
    #     tl_sampline = np.array(tl_sampline)
    #     br_sampline = np.array(br_sampline)
    #     H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
    #     W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
    #     line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
    #     samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
    #     tl_sampline = np.array([samp_start,line_start],dtype=int)
    #     br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
    #     return self.image[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]
    
    # @torch.no_grad()
    # def get_dem_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
    #     tl_sampline = np.array(tl_sampline)
    #     br_sampline = np.array(br_sampline)
    #     H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
    #     W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
    #     line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
    #     samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
    #     tl_sampline = np.array([samp_start,line_start],dtype=int)
    #     br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
    #     return self.dem[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]

    # @torch.no_grad()
    # def get_image_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
    #     """
    #     return: crop_img,tl_sampline,br_sampline
    #     """
    #     tlxy = np.array(tlxy)
    #     brxy = np.array(brxy)
    #     tl_sampline = self.xy_to_sampline(tlxy)
    #     br_sampline = self.xy_to_sampline(brxy)
    #     return self.get_image_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    # @torch.no_grad()
    # def get_dem_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
    #     """
    #     return: crop_dem,tl_sampline,br_sampline
    #     """
    #     tlxy = np.array(tlxy)
    #     brxy = np.array(brxy)
    #     tl_sampline = self.xy_to_sampline(tlxy)
    #     br_sampline = self.xy_to_sampline(brxy)
    #     return self.get_dem_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    # @torch.no_grad()
    # def resample_image_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
    #     img_resampled,local_hw2 = resample_from_quad(self.image,corner_samplines[:,[1,0]],target_shape)
    #     if need_local:
    #         return img_resampled,local_hw2
    #     else:
    #         return img_resampled
    
    # @torch.no_grad()
    # def resample_dem_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
    #     dem_resampled,local_hw2 = resample_from_quad(self.dem,corner_samplines[:,[1,0]],target_shape)
    #     if need_local:
    #         return dem_resampled,local_hw2
    #     else:
    #         return dem_resampled
        
    # def vis_grid(self,diags:list[np.ndarray],output_path:str = None):
    #     vis_img = self.image.copy()
    #     for diag in diags:
    #         min_x,min_y,max_x,max_y = diag[:,0].min(),diag[:,1].min(),diag[:,0].max(),diag[:,1].max()
    #         corners = [self.xy_to_sampline(np.array([min_x,min_y])),
    #                    self.xy_to_sampline(np.array([max_x,min_y])),
    #                    self.xy_to_sampline(np.array([max_x,max_y])),
    #                    self.xy_to_sampline(np.array([min_x,max_y])),
    #                    self.xy_to_sampline(np.array([min_x,min_y]))]
    #         for i in range(len(corners) - 1):
    #             cv2.line(vis_img,(int(corners[i][0]),int(corners[i][1])),(int(corners[i+1][0]),int(corners[i+1][1])),(0,0,255),5)
        
    #     if not output_path is None:
    #         cv2.imwrite(output_path,vis_img)
        
    #     return vis_img