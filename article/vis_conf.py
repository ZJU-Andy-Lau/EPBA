import argparse
from model.encoder import Encoder
import cv2
from infer.utils import extract_features
from shared.utils import load_config
from shared.visualize import vis_confidence_overlay
import os

def main(args):
    img = cv2.imread(args.img_path,cv2.IMREAD_GRAYSCALE)
    img_full = cv2.imread(args.img_path)

    model_configs = load_config(args.model_config_path)
    encoder = Encoder(dino_weight_path=args.dino_path,
                      embed_dim=model_configs['encoder']['embed_dim'],
                      ctx_dim=model_configs['encoder']['ctx_dim'])
    encoder.load_adapter(args.adapter_path)

    feat,_ = extract_features(encoder,img[None],img[None])

    _,_,conf = feat
    conf = conf.squeeze().detach().cpu().numpy()

    conf_overlay = vis_confidence_overlay(img_full,conf)
    cv2.imwrite(os.path.join(args.output_path,args.img_path.split('/')[-1].replace('.png','_conf.png')),conf_overlay)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_path',type=str)

    parser.add_argument('--output_path',type=str,default='./results/article_conf_vis')

    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')

    parser.add_argument('--dino_path',type=str,default='./weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')

    parser.add_argument('--adapter_path',type=str)

    args = parser.parse_args()