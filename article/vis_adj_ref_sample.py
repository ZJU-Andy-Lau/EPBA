import numpy as np
import cv2
import h5py
import os
from tqdm import tqdm
import argparse
from train.load_data import generate_affine_matrices,xy2rc_mat
from shared.visualize import make_checkerboard,warp_image_by_global_affine

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--output_path',type=str,default='./results/vis_adj_ref_sample')
    parser.add_argument('--num',type=int,default=10)

    args = parser.parse_args()

    os.makedirs(args.output_path,exist_ok=True)

    database = h5py.File(os.path.join(args.root,'train_data.h5'),'r')
    keys = list(database.keys())
    selected_keys = [keys[i] for i in np.random.choice(len(keys),args.num,replace=False)]

    H,W = database[keys[0]]['images']['0'][:].shape[:2]
    dsize = (1024,1024)

    H_as_xy, H_bs_xy, M_a_b_xy = generate_affine_matrices((H,W),(256,1024),dsize,args.num)

    for i,key in enumerate(tqdm(selected_keys)):
        img1_full = database[key]['images']['0'][:]
        img2_full = database[key]['images']['1'][:]
        img2_full_aff = cv2.warpAffine(img2_full, M_a_b_xy, (W, H), flags=cv2.INTER_LINEAR)
        img1 = cv2.warpPerspective(img1_full, H_as_xy[i], dsize, flags=cv2.INTER_LINEAR)
        img2 = cv2.warpPerspective(img2_full_aff, H_bs_xy[i], dsize, flags=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(args.output_path,f'{key}_1.png'),img1)
        cv2.imwrite(os.path.join(args.output_path,f'{key}_2.png'),img2)
        img_1_warp = warp_image_by_global_affine(img1,xy2rc_mat(H_as_xy[i]),xy2rc_mat(H_bs_xy[i]),xy2rc_mat(M_a_b_xy),dsize)
        ckb = make_checkerboard(img_1_warp,img2)
        cv2.imwrite(os.path.join(args.output_path,f'{key}_ckb.png'),ckb)