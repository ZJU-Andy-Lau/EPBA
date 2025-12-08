import torch
import torch.nn as nn
from torchvision import transforms

import os
import numpy as np
import cv2
import argparse
import h5py
from preprocess.conf.model import ConfHead
from shared.visualize import vis_confidence_overlay

@torch.no_grad()
def main(args):
    dataset = h5py.File(args.dataset_path,'r')
    if not args.img_select is None:
        keys = args.img_select.split(',')
    else:
        keys = list(dataset.keys())
        keys = [keys[i] for i in np.random.choice(len(dataset.keys()),10)]

    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    
    imgs = torch.stack([transform(dataset[key]['images']['image_0'][:]) for key in keys],dim=0)
    img_num = imgs.shape[0]

    conf_head = ConfHead(args.dino_weight_path)
    conf_head.load_head(args.conf_head_path)

    conf_head = conf_head.cuda().eval()
    imgs = imgs.cuda()

    confs = conf_head(imgs)
    confs = confs.detach().cpu().numpy()


    for i in range(img_num):
        vis_conf = vis_confidence_overlay(dataset[keys[i]]['images']['image_0'][:],confs[i])
        cv2.imwrite(os.path.join(args.output_path,f'{keys[i]}.png'),vis_conf)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--output_path',type=str)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--conf_head_path',type=str,default=None)
    parser.add_argument('--img_select',type=str,default=None)

    args = parser.parse_args()

    os.makedirs(args.output_path,exist_ok=True)
    main(args)
