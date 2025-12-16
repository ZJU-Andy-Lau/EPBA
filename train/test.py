import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import json
import numpy as np
import h5py
import cv2

import torch
from torchvision import transforms

from load_data import generate_affine_matrices,xy2rc_mat
from model.encoder import Encoder
from model.gru import GRUBlock
from shared.utils import get_current_time,load_model_state_dict,load_config
import shared.visualize as visualizer
from solve.solve_windows import WindowSolver

def load_data(args):
    """
    return:
        imgs_a, imgs_b : (N,3,H,W) torch.Tensor
        Hs_a, Hs_b: (N,3,3) torch.Tensor
        Ms:(N,2,3) torch.Tensor
    """
    file = h5py.File(args.dataset_path,'r')
    all_keys = list(file.keys())
    select_keys = []
    if not args.dataset_select is None:
        idxs = [int(i) for i in args.dataset_select.split(',')]
        select_keys = [all_keys[i] for i in idxs]
    else:
        idxs = np.random.choice(len(all_keys),args.dataset_num)
        select_keys = [all_keys[i] for i in idxs]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    imgs_a = []
    imgs_b = []
    Hs_a = []
    Hs_b = []
    Ms = []

    dsize = (512,512)

    for key in select_keys:
        img_num = len(file[key]['images'])
        idx1,idx2 = np.random.choice(img_num,2)
        if idx1 == idx2:
            idx2 = (idx1 + 1) % img_num
        img_1_full = file[key]['images'][f'{idx1}'][:]
        img_2_full = file[key]['images'][f'{idx2}'][:]
        img_1_full = np.stack([img_1_full]*3,axis=-1)
        img_2_full = np.stack([img_2_full]*3,axis=-1)
        H,W = img_1_full.shape[:2]

        Hs_a_xy,Hs_b_xy,M_xy = generate_affine_matrices((H,W),(256,2048),dsize,1)
        img_2_full = cv2.warpAffine(img_2_full,M_xy,(W,H),flags=cv2.INTER_LINEAR)
        
        img_1_warp = cv2.warpPerspective(img_1_full,Hs_a_xy[0],dsize,flags=cv2.INTER_LINEAR)
        img_2_warp = cv2.warpPerspective(img_2_full,Hs_b_xy[0],dsize,flags=cv2.INTER_LINEAR)

        H_a_rc = xy2rc_mat(Hs_a_xy)[0]
        H_b_rc = xy2rc_mat(Hs_b_xy)[0]
        M_rc = xy2rc_mat(M_xy)

        imgs_a.append(img_1_warp)
        imgs_b.append(img_2_warp)
        Hs_a.append(H_a_rc)
        Hs_b.append(H_b_rc)
        Ms.append(M_rc)
    
    imgs_a = torch.stack([transform(img) for img in imgs_a],dim=0).to(device=args.device,dtype=torch.float32)
    imgs_b = torch.stack([transform(img) for img in imgs_b],dim=0).to(device=args.device,dtype=torch.float32)
    Hs_a = torch.from_numpy(np.stack(Hs_a,axis=0)).to(device=args.device,dtype=torch.float32)
    Hs_b = torch.from_numpy(np.stack(Hs_b,axis=0)).to(device=args.device,dtype=torch.float32)
    Ms = torch.from_numpy(np.stack(Ms,axis=0)).to(device=args.device,dtype=torch.float32)

    return imgs_a,imgs_b,Hs_a,Hs_b,Ms

def load_models(args):
    model_configs = load_config(args.model_config_path)

    encoder = Encoder(dino_weight_path=args.dino_path,
                      embed_dim=model_configs['encoder']['embed_dim'],
                      ctx_dim=model_configs['encoder']['ctx_dim'])
    
    encoder.load_adapter(args.adapter_path)
    
    gru = GRUBlock(corr_levels=model_configs['gru']['corr_levels'],
                   corr_radius=model_configs['gru']['corr_radius'],
                   context_dim=model_configs['gru']['ctx_dim'],
                   hidden_dim=model_configs['gru']['hidden_dim'])
    
    load_model_state_dict(gru,args.gru_path)
    
    args.gru_iter_num = model_configs['gru']['iter_num']
    
    encoder = encoder.to(args.device).eval()
    gru = gru.to(args.device).eval()

    print("Models Loaded")

    return encoder,gru

@torch.no_grad()
def solve(args,encoder:Encoder,gru:GRUBlock,data):
    imgs_a,imgs_b,Hs_a,Hs_b,Ms = data
    B,_,H,W = imgs_a.shape

    feats_1,feats_2 = encoder(imgs_a,imgs_b)

    solver = WindowSolver(B,H,W,gru,feats_1,feats_2,Hs_a,Hs_b,gru_max_iter=args.gru_iter_num)

    M_a_b_pred = solver.solve(flag='ab',final_only=True) # N,2,3

    report(args,M_a_b_pred,Ms,imgs_a,imgs_b)

def report(args,Ms_pred:torch.Tensor,data):
    imgs_a,imgs_b,Hs_a,Hs_b,Ms_gt = data
    B, _, H, W = imgs_a.shape
    device = imgs_a.device
    N = H * W
    
    batch_metrics = []

    # -------------------------------------------------------------------------
    # Part 1: 定量精度评估 (Quantitative Evaluation)
    # -------------------------------------------------------------------------
    y_range = torch.arange(0, H, dtype=torch.float32, device=device)
    x_range = torch.arange(0, W, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing='ij')
    
    # (3, N) -> [row, col, 1]
    grid_local = torch.stack([
        grid_y.reshape(-1),
        grid_x.reshape(-1),
        torch.ones(N, dtype=torch.float32, device=device)
    ], dim=0)
    
    # 扩展为 Batch 维度: (B, 3, N)
    grid_local_batch = grid_local.unsqueeze(0).expand(B, -1, -1)

    # Coords_large = H_a^-1 * Grid_local
    try:
        Hs_a_inv = torch.linalg.inv(Hs_a)
    except RuntimeError:
        Hs_a_inv = torch.inverse(Hs_a)
        
    coords_large_homo = torch.bmm(Hs_a_inv, grid_local_batch) # (B, 3, N)
    
    # 透视除法 (Perspective Division) 归一化齐次坐标
    z = coords_large_homo[:, 2:3, :] + 1e-7
    coords_large = coords_large_homo / z # (B, 3, N) -> [row_g, col_g, 1]
    # Coords_pred = Ms_pred * Coords_large -> (B, 2, N)
    coords_pred = torch.bmm(Ms_pred, coords_large)
    # Coords_gt = Ms_gt * Coords_large -> (B, 2, N)
    coords_gt = torch.bmm(Ms_gt, coords_large)

    # Euclidean distance per point: (B, N)
    dist_errors = torch.norm(coords_pred - coords_gt, dim=1)
    
    for i in range(B):
        errors_i = dist_errors[i]
        metrics = {
            'mean': f"{errors_i.mean().item():.3f} px",
            'median': f"{errors_i.median().item():.3f} px",
            'max': f"{errors_i.max().item():.3f} px"
        }
        batch_metrics.append(metrics)

    # -------------------------------------------------------------------------
    # Part 2: 可视化配准精度 (Visualization)
    # -------------------------------------------------------------------------    
    Ms_pred_np = Ms_pred.detach().cpu().numpy()
    Ms_gt_np = Ms_gt.detach().cpu().numpy()
    
    Hs_a_np = Hs_a.detach().cpu().numpy()
    Hs_b_np = Hs_b.detach().cpu().numpy()
    
    # 单位变换矩阵 (2, 3)
    identity_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    for i in range(B):
        img_a_vis = visualizer.denormalize_image(imgs_a[i])
        img_b_vis = visualizer.denormalize_image(imgs_b[i])

        target_wh = (W, H) # 输出尺寸 (Width, Height)

        # (1) Raw: 单位变换 (仅靠 Crop 自身的位置关系)
        img_a_raw_warp = visualizer.warp_image_by_global_affine(
            img_a_vis, Hs_a_np[i], Hs_b_np[i], identity_M, target_wh
        )
        
        # (2) GT: 真值变换
        img_a_gt_warp = visualizer.warp_image_by_global_affine(
            img_a_vis, Hs_a_np[i], Hs_b_np[i], Ms_gt_np[i], target_wh
        )
        
        # (3) Pred: 预测变换
        img_a_pred_warp = visualizer.warp_image_by_global_affine(
            img_a_vis, Hs_a_np[i], Hs_b_np[i], Ms_pred_np[i], target_wh
        )
        
        # 2.4 生成棋盘格 (Checkerboard)
        # 将 Warp 后的图 A 与 图 B 进行棋盘格混合
        vis_raw = visualizer.make_checkerboard(img_a_raw_warp, img_b_vis, box_size=32)
        vis_gt = visualizer.make_checkerboard(img_a_gt_warp, img_b_vis, box_size=32)
        vis_pred = visualizer.make_checkerboard(img_a_pred_warp, img_b_vis, box_size=32)
        
        # 2.5 保存图像
        # 路径结构: output_path/sample_{batch}_{i}/
        save_dir = os.path.join(args.output_path, f"sample_{i}")
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(save_dir, "checkerboard_raw.png"), cv2.cvtColor(vis_raw, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "checkerboard_gt.png"), cv2.cvtColor(vis_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_dir, "checkerboard_pred.png"), cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR))

        print(f"Data:{i}")
        print(batch_metrics[i])
        with open(os.path.join(save_dir,'metrics.json'),'w') as f:
            json.dump(batch_metrics[i],f)    
        

    return batch_metrics
    

def main(args):
    data = load_data(args)
    encoder,gru = load_models(args)
    
    solve(args,encoder,gru,data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=10)
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--adapter_path',type=str,default=None)
    parser.add_argument('--gru_path',type=str,default=None)
    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')
    parser.add_argument('--output_path',type=str,default='./results')
    parser.add_argument('--device',type=str,default='cuda')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path,get_current_time())
    os.makedirs(args.output_path,exist_ok=True)

    print("==============================configs==============================")
    for k,v in vars(args).items():
        print(f"{k}:{v}")
    print("===================================================================")
    main(args)