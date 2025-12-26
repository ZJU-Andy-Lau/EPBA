import warnings

warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2


from infer.utils import warp_quads,create_grid_img,solve_weighted_affine
from shared.utils import project_mercator,mercator2lonlat,bilinear_interpolate,resample_from_quad
from shared.visualize import make_checkerboard
from shared.rpc import RPCModelParameterTorch,project_linesamp
from tqdm import tqdm,trange
import rasterio
from typing import Tuple

class RSImageMeta():
    def __init__(self,options,root:str,id:int,device:str='cuda'):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = options
        self.root = root
        self.id = id
        if os.path.exists(os.path.join(root,'dem.npy')):
            self.dem = np.load(os.path.join(root,'dem.npy'),mmap_mode='r')
        elif os.path.exists(os.path.join(root,'dem.tif')):
            with rasterio.open(os.path.join(root,'dem.tif')) as f:
                self.dem = f.read(1)
        else:
            raise ValueError(f"DEM not found in root:{root}")
        self.device = device
        self.H,self.W = self.dem.shape[:2]
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(root,'rpc.txt'))
        self.rpc.to_gpu(device=device)
        
        self.corner_xys = self.__get_corner_xys__()

        del self.rpc
        del self.dem
        self.rpc = None
        self.dem = None
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,br,bl] [x,y] np.ndarray
        """
        heights = [self.dem[0,0],self.dem[0,-1],self.dem[-1,-1],self.dem[-1,0]]
        heights = [i if not np.isnan(i) else 0 for i in heights]
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ([0.,self.W-1.,self.W-1.,0],
                                                     [0.,0.,self.H - 1.,self.H - 1.],
                                                     heights),dim=-1)
        xys = project_mercator(latlons)
        return xys.cpu().numpy()[:,[1,0]] # y,x -> x,y

class RSImage():
    def __init__(self,options,root:str,id:int,device:str='cuda'):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = options
        self.root = root
        self.id = id
        self.device = device
        self.initialize()
    
    def __init__(self,meta:RSImageMeta,device:str = None):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = meta.options
        self.root = meta.root
        self.id = meta.id
        self.device = meta.device if device is None else device
        self.initialize()
    
    def initialize(self):
        with rasterio.open(os.path.join(self.root,'image.png'),'r') as f:
            data = f.read()
            if data.shape[0] > 1:
                weights = np.array([0.299, 0.587, 0.114])
                data = np.tensordot(weights,data,axes=1).astype(np.uint8)
            else:
                data = data[0]
        self.image = np.stack([data] * 3,axis=-1)
            
        # self.image = cv2.imread(os.path.join(self.root,'image.png'),cv2.IMREAD_GRAYSCALE)
        # self.image = np.stack([self.image] * 3,axis=-1)
        if os.path.exists(os.path.join(self.root,'dem.npy')):
            self.dem = np.load(os.path.join(self.root,'dem.npy'),mmap_mode='r')
        elif os.path.exists(os.path.join(self.root,'dem.tif')):
            with rasterio.open(os.path.join(self.root,'dem.tif')) as f:
                self.dem = f.read(1)
        else:
            raise ValueError(f"DEM not found in root:{self.root}")
        if os.path.exists(os.path.join(self.root,'tie_points.txt')):
            self.tie_points = self.__load_tie_points__(os.path.join(self.root,'tie_points.txt'))
        else:
            self.tie_points = None
        
        self.H,self.W = self.image.shape[:2]
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(self.root,'rpc.txt'))
        self.rpc.to_gpu(device=self.device)

        self.affine_list = []
        
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
        heights = [self.dem[0,0],self.dem[0,-1],self.dem[-1,-1],self.dem[-1,0]]
        heights = [i if not np.isnan(i) else 0 for i in heights]
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ([0.,self.W-1.,self.W-1.,0],
                                                     [0.,0.,self.H - 1.,self.H - 1.],
                                                     heights),dim=-1)
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

    def merge_affines(self):
        if len(self.affine_list) == 0:
            return None
        if len(self.affine_list) == 1:
            return self.affine_list[0]
        rows = torch.arange(0, self.H, max(self.H // 32,1), dtype=torch.float32)
        cols = torch.arange(0, self.W, max(self.W // 32,1), dtype=torch.float32)
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing='ij')
        grid_row = grid_row.flatten()
        grid_col = grid_col.flatten()
        ones = torch.ones_like(grid_row)
        src_grid = torch.stack([grid_row, grid_col, ones], dim=0) # 3,N
        all_src_list = []
        all_dst_list = []
        
        for affine_mat in self.affine_list:
            dst_grid = torch.mm(affine_mat.to(torch.float32), src_grid.to(torch.float32).clone())
            all_src_list.append(src_grid[:2])
            all_dst_list.append(dst_grid) 


        
        X = torch.cat(all_src_list, dim=1).T # (N,2)
        Y = torch.cat(all_dst_list, dim=1).T # (N,2)
        scores = torch.ones(len(X),device=X.device)
        solution = solve_weighted_affine(X,Y,scores)

        return solution

    def vis_checkpoints(self,ref_points:np.ndarray = None):
        """
        ref_points: (N,3) (lon,lat,height) np.ndarray
        """
        if not ref_points is None:
            samps,lines = self.rpc.RPC_OBJ2PHOTO(ref_points[:,1],ref_points[:,0],ref_points[:,2],'numpy')
            samps,lines = np.round(samps).astype(int),np.round(lines).astype(int)
            ref_points = np.stack([lines,samps],axis=-1)
        grid_imgs = []
        s = 256
        for i,p in enumerate(self.tie_points):
            cl,cs = p
            cl = min(self.H - s // 2,max(s // 2, cl))
            cs = min(self.W - s // 2,max(s // 2, cs))
            pl = p[0] - cl + s // 2
            ps = p[1] - cs + s // 2
            img = self.image[cl - s // 2 : cl + s // 2, cs - s // 2 : cs + s // 2]
            cv2.circle(img,(ps,pl),1,(0,255,0),-1)
            if not ref_points is None:
                rp = ref_points[i]
                rpl = rp[0] - p[0] + pl
                rps = rp[1] - p[1] + ps
                cv2.circle(img,(rps,rpl),1,(0,0,255),-1)

            grid_imgs.append(img)

        vis_img = create_grid_img(grid_imgs)
        return vis_img

    def check_error(self,ref_points:np.ndarray):
        """
        ref_points: (N,3) (lon,lat,height) np.ndarray
        """
        samps,lines = self.rpc.RPC_OBJ2PHOTO(ref_points[:,1],ref_points[:,0],ref_points[:,2],'numpy')
        ref = np.stack([lines,samps],axis=-1)
        dis = np.linalg.norm(ref - self.tie_points,axis=-1)
        return dis

    def get_ref_points(self,heights = None):
        """
        如果作为ref_image,提供自身的tie_points作为ref_points
        """
        if heights is None:
            heights = self.dem[self.tie_points[:,0],self.tie_points[:,1]]
        lats,lons = self.rpc.RPC_PHOTO2OBJ(self.tie_points[:,1],self.tie_points[:,0],heights,'numpy')
        ref_points = np.stack([lons,lats,heights],axis=-1)
        return ref_points

class RSImage_Error_Check():
    def __init__(self,meta:RSImageMeta,device:str = None):
        self.options = meta.options
        self.root = meta.root
        self.id = meta.id
        self.device = meta.device if device is None else device
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(self.root,'rpc.txt'))
        self.rpc.to_gpu(self.device)
        self.tie_points_path = os.path.join(self.root,'tie_points.txt')
        if os.path.exists(os.path.join(self.root,'dem.npy')):
            self.dem = np.load(os.path.join(self.root,'dem.npy'),mmap_mode='r')
        elif os.path.exists(os.path.join(self.root,'dem.tif')):
            with rasterio.open(os.path.join(self.root,'dem.tif')) as f:
                self.dem = f.read(1)

        if os.path.exists(os.path.join(self.root,'tie_points.txt')):
            self.tie_points = self.__load_tie_points__(os.path.join(self.root,'tie_points.txt'))
        else:
            self.tie_points = None
        self.heights = self.get_heights_for_tie_points()
    
    def __load_tie_points__(self,path) -> np.ndarray:
        tie_points = np.loadtxt(path,dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1,-1)
        elif tie_points.shape[1] != 2:
            print("tie points format error")
            return None
        return tie_points

    def get_heights_for_tie_points(self) -> np.ndarray:
        """
        公开接口：获取所有 tie_points 对应的高程值。
        """
        if self.tie_points is None:
            return np.array([])
        
        lines = self.tie_points[:, 0]
        samps = self.tie_points[:, 1]
        return self.dem[lines,samps]

    def check_error(self,ref_points:np.ndarray):
        """
        ref_points: (N,3) (lon,lat,height) np.ndarray
        """
        samps,lines = self.rpc.RPC_OBJ2PHOTO(ref_points[:,1],ref_points[:,0],ref_points[:,2],'numpy')
        ref = np.stack([lines,samps],axis=-1)
        dis = np.linalg.norm(ref - self.tie_points,axis=-1)
        return dis

    def get_ref_points(self,heights = None):
        """
        如果作为ref_image,提供自身的tie_points作为ref_points
        """
        if heights is None:
            heights = self.heights
        lats,lons = self.rpc.RPC_PHOTO2OBJ(self.tie_points[:,1],self.tie_points[:,0],heights,'numpy')
        ref_points = np.stack([lons,lats,heights],axis=-1)
        return ref_points
    

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
    min_line,max_line,min_samp,max_samp = int(max(lines_in_b.min(),0)),int(min(lines_in_b.max(),image_b.H)) + 1,int(max(samps_in_b.min(),0)),int(min(samps_in_b.max(),image_b.W)) + 1
    img_b = image_b.image[min_line:max_line,min_samp:max_samp]
    lines_in_b -= min_line
    samps_in_b -= min_samp
    
    #采样
    lines_in_b_norm = 2.0 * lines_in_b.to(torch.float32) / (img_b.shape[0] - 1) - 1.0
    samps_in_b_norm = 2.0 * samps_in_b.to(torch.float32) / (img_b.shape[1] - 1) - 1.0
    sample_coords = torch.stack([samps_in_b_norm,lines_in_b_norm],dim=-1).reshape(1,H,W,2)
    input_img = torch.from_numpy(img_b).to(dtype=torch.float32,device=device)[None].permute(0,3,1,2) # 1,3,H,W
    sampled_img = F.grid_sample(input_img,sample_coords,mode='bilinear',padding_mode='zeros',align_corners=True) # 1,3,H,W
    img_b = sampled_img[0].permute(1,2,0).cpu().numpy()

    img_a_b = make_checkerboard(img_a,img_b)

    cv2.imwrite(os.path.join(output_path,f'registration_{image_a.id}_{image_b.id}.png'),img_a_b)

    del sample_coords
    del input_img
    del sampled_img
    sample_coords = None
    input_img = None
    sample_coords = None