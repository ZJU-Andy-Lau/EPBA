import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2

from infer.utils import warp_quads
from shared.utils import project_mercator,mercator2lonlat,bilinear_interpolate,project_linesamp
from shared.rpc import RPCModelParameterTorch
from shared.visualize import make_checkerboard

class RSImage():
    def __init__(self, options, root:str, id:int, device:str='cuda', lazy:bool=False):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        lazy: If True, only load metadata (RPC, corners, shape).
              DEM corners are read via mmap to avoid full load.
              Image shape is read via cv2 (loading headers if possible, or full decode then del).
        """
        self.options = options
        self.root = root
        self.id = id
        self.device = device
        self.lazy = lazy

        # 1. RPC 必须加载 (文件很小)
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(root, 'rpc.txt'))
        
        # 2. 获取图像尺寸 (不读取像素数据)
        img_path = os.path.join(root, 'image.png')
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        # 快速读取形状
        _tmp_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if _tmp_img is None:
             raise RuntimeError(f"Failed to read image: {img_path}")
        self.H, self.W = _tmp_img.shape[:2]
        del _tmp_img # 立即释放

        # 3. 处理 DEM 和 角点计算 (Lazy 模式使用 mmap 加速)
        dem_path = os.path.join(root, 'dem.npy')
        if not os.path.exists(dem_path):
             raise FileNotFoundError(f"DEM not found at {dem_path}")
        
        # 使用 mmap_mode='r' 读取角点，避免加载整个数组
        _dem_mmap = np.load(dem_path, mmap_mode='r')
        h_tl = _dem_mmap[0, 0]
        h_tr = _dem_mmap[0, -1]
        h_br = _dem_mmap[-1, -1]
        h_bl = _dem_mmap[-1, 0]
        
        # 计算 corner_xys (需要在 CPU 上进行，避免多进程初始化 CUDA 冲突)
        # 将 RPC 临时转到 CPU
        self.rpc.to_gpu('cpu')
        
        # 构造角点经纬度输入
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ(
            [0., self.W-1., self.W-1., 0.],
            [0., 0., self.H-1., self.H-1.],
            [h_tl, h_tr, h_br, h_bl]
        ), dim=-1)
        
        xys = project_mercator(latlons)
        self.corner_xys = xys.cpu().numpy()[:, [1, 0]] # y,x -> x,y

        # 加载 Tie Points (通常很小，Lazy 模式下也保留，用于 check_error)
        tp_path = os.path.join(root, 'tie_points.txt')
        if os.path.exists(tp_path):
            self.tie_points = self.__load_tie_points__(tp_path)
        else:
            self.tie_points = None

        # 4. 根据 Lazy 模式决定是否保留数据
        if self.lazy:
            self.image = None
            self.dem = None
            # Lazy 模式下 RPC 保持在 CPU
        else:
            # 非 lazy 模式，正常加载全量数据到内存
            self.image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            self.image = np.stack([self.image] * 3, axis=-1)
            self.dem = np.load(dem_path) # 全量加载
            self.rpc.to_gpu(device)
            
        # 释放 mmap 句柄
        del _dem_mmap

    def load_heavy_data(self):
        """
        Worker 进程在接收到任务后调用此方法，加载真正的像素数据和 DEM。
        """
        if self.image is None:
            self.image = cv2.imread(os.path.join(self.root, 'image.png'), cv2.IMREAD_GRAYSCALE)
            self.image = np.stack([self.image] * 3, axis=-1)
        
        if self.dem is None:
            self.dem = np.load(os.path.join(self.root, 'dem.npy'))
        
        # 将 RPC 移动到指定的计算设备 (GPU)
        self.rpc.to_gpu(self.device)

    def load_dem_only(self):
        """
        Rank 0 在进行全局平差/误差报告时调用，只加载 DEM 用于几何投影，不加载图像像素以节省内存。
        """
        if self.dem is None:
            self.dem = np.load(os.path.join(self.root, 'dem.npy'))
        # 确保 RPC 在正确的设备上
        self.rpc.to_gpu(self.device)

    def __load_tie_points__(self, path) -> np.ndarray:
        tie_points = np.loadtxt(path, dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1, -1)
        elif tie_points.size == 0:
            return None
        return tie_points
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,br,bl] [x,y] np.ndarray
        """
        return self.corner_xys
    
    def dem_interp(self, sampline: np.ndarray):
        if sampline.ndim == 1:
            sampline = sampline[None]
        return bilinear_interpolate(self.dem, sampline)

    @torch.no_grad()
    def xy_to_sampline(self, xy: np.ndarray, max_iter=100, rpc: RPCModelParameterTorch=None) -> np.ndarray:
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
        latlon = mercator2lonlat(xy[:, [1, 0]])
        sampline = np.array([self.W, self.H], dtype=np.float32) * (xy - self.corner_xys[0]) / (self.corner_xys[3] - self.corner_xys[0])
        dem = self.dem_interp(sampline)
        invalid_mask = np.full(dem.shape, True, dtype=bool)
        for iter in range(max_iter):
            sampline_new = np.stack(rpc.RPC_OBJ2PHOTO(latlon[invalid_mask, 0], latlon[invalid_mask, 1], dem[invalid_mask], 'numpy'), axis=-1)
            dis = np.linalg.norm(sampline_new - sampline[invalid_mask], axis=-1)
            sampline[invalid_mask] = sampline_new
            invalid_mask[invalid_mask] = dis > 1.
            if invalid_mask.sum() == 0:
                break
        return sampline.squeeze()
    
    def convert_diags_to_corners(self, diags: np.ndarray, rpc: RPCModelParameterTorch=None):
        """
        Args:
            diags: ndarray, (N,2,2), (x,y)
        Return:
            corners: ndarray, (N,4,2), (line,samp)
        """
        if diags.ndim < 3:
            diags = diags[None]
        N = diags.shape[0]
        corners_xy = np.zeros((N, 4, 2), dtype=diags.dtype)
        corners_xy[:, 0, :] = diags[:, 0, :]
        corners_xy[:, 1, 0] = diags[:, 1, 0]
        corners_xy[:, 1, 1] = diags[:, 0, 1]
        corners_xy[:, 2, :] = diags[:, 1, :]
        corners_xy[:, 3, 0] = diags[:, 0, 0]
        corners_xy[:, 3, 1] = diags[:, 1, 1]
        corners_xy_flat = corners_xy.reshape(-1, 2)
        corners_samplines_flat = self.xy_to_sampline(corners_xy_flat, rpc=rpc)
        corners_linesamps = corners_samplines_flat.reshape(N, 4, 2)[..., [1, 0]]
        return corners_linesamps

    def crop_windows(self, corners: np.ndarray, output_size=(512, 512)):
        """
        根据给定的四边形顶点坐标，对图像/数组进行透视变换裁切。
        """
        warped_res, Hs = warp_quads(corners, [self.image, self.dem], output_size)
        warped_imgs, warped_dems = warped_res
        return warped_imgs, warped_dems, Hs

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