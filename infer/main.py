import warnings
warnings.filterwarnings("ignore")
import os
import argparse
import numpy as np
from scipy import datasets
import h5py
import datetime
import time
from tqdm import tqdm,trange
from skimage.transform import AffineTransform
from skimage.measure import ransac
import random
from functools import partial
import cv2
from copy import deepcopy
from typing import List
import itertools
import yaml
import threading
import queue # 标准库 queue

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset,DataLoader,DistributedSampler

from torchvision import transforms

from model.encoder import Encoder
from model.gru import GRUBlock
from model.ctx_decoder import ContextDecoder
from shared.utils import str2bool,get_current_time,load_model_state_dict,load_config
from utils import is_overlap,convert_pair_dicts_to_solver_inputs,get_error_report,Reporter
from pair import Pair
from solve.global_affine_solver import GlobalAffineSolver,TopologicalAffineSolver
from rs_image import RSImage,vis_registration

def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images(args) -> List[RSImage] :
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs] #按照字典序的idx
    images = []
    for idx,folder in enumerate(img_folders):
        img_path = os.path.join(base_path,folder)
        images.append(RSImage(args,img_path,idx,args.device))
        # print(f"Loaded Image {idx} from {folder}")
    # print(f"Totally {len(images)} Images Loaded")
    args.image_num = len(images)
    return images

def build_pairs(args,images:List[RSImage]) -> List[Pair]:
    images_num = len(images)
    configs = {
        'max_window_num':args.max_window_num,
        'min_window_size':args.min_window_size,
        'max_window_size':args.max_window_size,
        'min_area_ratio':args.min_cover_area_ratio,
    }
    pairs = []
    for i,j in itertools.combinations(range(images_num),2):
        if is_overlap(images[i],images[j],args.min_window_size ** 2):
            configs['output_path'] = os.path.join(args.output_path,f"pair_{images[i].id}_{images[j].id}")
            pair = Pair(images[i],images[j],images[i].id,images[j].id,configs,device=args.device)
            pairs.append(pair)
    # print(f"Totally {len(pairs)} Pairs")
    return pairs

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
    
    encoder = encoder.to(args.device).eval().half()
    gru = gru.to(args.device).eval()

    # encoder = torch.compile(encoder,mode='max-autotune')
    # gru = torch.compile(gru,mode="max-autotune")

    # print("Models Loaded")

    return encoder,gru

# ==========================================
# [新增] 监控线程
# ==========================================
class MonitorThread(threading.Thread):
    def __init__(self, msg_queue, nprocs):
        super().__init__()
        self.msg_queue = msg_queue
        self.nprocs = nprocs
        self.status_table = {i: {"status": "WAITING", "pair": "--", "prog": "--", "dir": "--", "level": "--", "step": "--"} for i in range(nprocs)}
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        # 清屏并隐藏光标
        os.system('cls' if os.name=='nt' else 'clear')
        print("\033[?25l", end="") 
        
        while self.running:
            try:
                # 非阻塞获取消息，超时刷新
                while True:
                    msg = self.msg_queue.get_nowait()
                    if msg == 'STOP':
                        self.running = False
                        break
                    
                    rank = msg['rank']
                    if 'status' in msg: self.status_table[rank]['status'] = msg['status']
                    if 'pair_id' in msg: self.status_table[rank]['pair'] = msg['pair_id']
                    if 'progress' in msg: self.status_table[rank]['prog'] = msg['progress']
                    if 'direction' in msg: self.status_table[rank]['dir'] = msg['direction']
                    if 'level' in msg: self.status_table[rank]['level'] = msg['level']
                    if 'step' in msg: self.status_table[rank]['step'] = msg['step']
            except queue.Empty:
                pass
            
            self._draw()
            time.sleep(0.1)
        
        # 恢复光标
        print("\033[?25h", end="")

    def _draw(self):
        # 移动光标到顶部
        print("\033[H", end="")
        print("="*90)
        print(f"| {'Rank':^4} | {'Status':^10} | {'Pair (A-B)':^12} | {'Progress':^10} | {'Dir':^6} | {'Level':^8} | {'Step':^12} |")
        print("-" * 90)
        for i in range(self.nprocs):
            s = self.status_table[i]
            print(f"| {i:^4} | {s['status']:^10} | {s['pair']:^12} | {s['prog']:^10} | {s['dir']:^6} | {s['level']:^8} | {s['step']:^12} |")
        print("="*90)
        print("\n\n") # 留出空间给Log

# ==========================================
# [修改] Worker 主函数
# ==========================================
def main_worker(rank, world_size, args, msg_queue):
    # 1. 环境初始化
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        # Fallback to gloo if nccl fails (e.g. no GPU)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
    torch.cuda.set_device(rank)
    args.device = f'cuda:{rank}'
    
    # 2. 启动监控线程 (仅 Rank 0)
    monitor_thread = None
    if rank == 0:
        monitor_thread = MonitorThread(msg_queue, world_size)
        monitor_thread.start()

    # 3. 创建 Reporter
    reporter = Reporter(msg_queue, rank)
    reporter.update(status="LOADING")

    # 4. 加载数据 (所有 Rank 都加载全量，以保证 Pair 顺序一致)
    images = load_images(args)
    pairs = build_pairs(args,images)
    
    # 5. 加载模型
    encoder, gru = load_models(args)

    # 6. 任务分配
    my_pairs_indices = [i for i in range(len(pairs)) if i % world_size == rank]
    my_pairs = [pairs[i] for i in my_pairs_indices]
    
    local_results = []
    
    # 7. 推理循环
    reporter.update(status="RUNNING")
    for idx, pair in enumerate(my_pairs):
        global_idx = my_pairs_indices[idx]
        pair_id_str = f"{pair.id_a}-{pair.id_b}"
        progress_str = f"{idx+1}/{len(my_pairs)}"
        
        reporter.update(pair_id=pair_id_str, progress=progress_str, status="RUNNING")
        
        # [关键] 设置确定性随机种子，保证与单卡运行一致
        # 种子与 Pair ID 绑定，不受处理顺序和 Rank 影响
        deterministic_seed = args.random_seed + pair.id_a * 100000 + pair.id_b
        init_random_seed(deterministic_seed)
        
        # 执行推理
        try:
            affine_ab, affine_ba = pair.solve_affines(encoder, gru, reporter=reporter)
            local_results.append({
                pair.id_a: affine_ab,
                pair.id_b: affine_ba
            })
        except Exception as e:
            reporter.update(status="ERROR")
            # 在实际工程中建议记录日志到文件
            # print(f"Rank {rank} Error on pair {pair_id_str}: {e}")
    
    reporter.update(status="GATHERING")
    
    # 8. 结果汇总
    # all_results_lists will range(world_size) list of lists
    all_results_lists = [None for _ in range(world_size)]
    dist.all_gather_object(all_results_lists, local_results)
    
    # 9. Global Solve (仅 Rank 0)
    if rank == 0:
        reporter.update(status="SOLVING")
        # 展平列表
        final_results = []
        # 按 Round-Robin 顺序重组结果 (可选，其实顺序不影响 Global Solve)
        # 这里简单展平
        for sublist in all_results_lists:
            final_results.extend(sublist)
            
        print("\n\nGlobal Solving...")
        
        solver_configs = load_config(args.solver_config_path)
        solver = GlobalAffineSolver(images=images,
                                    device=args.device,
                                    anchor_indices=[0],
                                    max_iter=100,
                                    converge_tol=1e-6)
        
        Ms = solver.solve(final_results)
        
        # 更新 RPC
        for image in images:
            M = Ms[image.id]
            print(f"Affine Matrix of Image {image.id}\n{M}\n")
            image.rpc.Update_Adjust(M)
            image.rpc.Merge_Adjust()
        
        # 可视化与报告
        reporter.update(status="VISUALIZING")
        for i,j in itertools.combinations(range(len(images)),2):
            vis_registration(image_a=images[i],image_b=images[j],output_path=args.output_path,device=args.device)
        
        report = get_error_report(pairs)
        print("\n" + "--- Global Error Report (Summary) ---")
        print(f"Total tie points checked: {report['count']}")
        print(f"Mean Error:   {report['mean']:.4f} m")
        print(f"Median Error: {report['median']:.4f} m")
        print(f"Max Error:    {report['max']:.4f} m")
        print(f"RMSE:         {report['rmse']:.4f} m")
        print(f"< 1.0 m: {report['<1m_percent']:.2f} %")
        print(f"< 3.0 m: {report['<3m_percent']:.2f} %")
        print(f"< 5.0 m: {report['<5m_percent']:.2f} %")
        
        reporter.update(status="DONE")
        msg_queue.put('STOP')
        monitor_thread.join()

    dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #==============================数据相关设置=====================================

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    
    parser.add_argument('--select_imgs',type=str,default='0,1')

    #==============================================================================


    #==============================模型相关设置=====================================

    parser.add_argument('--dino_path', type=str, default='weights')

    parser.add_argument('--adapter_path', type=str, default='weights/adapter.pth')

    parser.add_argument('--gru_path', type=str, default='weights/gru.pth')

    parser.add_argument('--model_config_path', type=str, default='configs/model_config.yaml')

    #==============================================================================


    #==============================求解相关设置=====================================

    parser.add_argument('--max_window_size', type=int, default=8000)

    parser.add_argument('--min_window_size', type=int, default=500)

    parser.add_argument('--max_window_num', type=int, default=256)

    parser.add_argument('--min_cover_area_ratio', type=float, default=0.5)

    parser.add_argument('--solver_config_path', type=str, default='configs/global_solver_config.yaml')

    #================================================================================


    #==============================输出相关设置=====================================
    
    parser.add_argument('--output_path', type=str, default='results')
    
    parser.add_argument('--vis_resolution', type=float, default=1.0, 
                        help='Resolution (in meters) for output orthophotos and checkerboards.')
    
    #===============================================================================


    #==============================实验相关设置======================================
    
    parser.add_argument('--experiment_id', type=str, default=None)

    parser.add_argument('--random_seed',type=int,default=42)
   
    parser.add_argument('--world_size', type=int, default=8, help='Number of GPUs/Processes to use')

    #==============================================================================


    args = parser.parse_args()

    if args.experiment_id is None:
        args.experiment_id = get_current_time()
    
    if '[time]' in args.experiment_id:
        args.experiment_id = args.experiment_id.replace('[time]',get_current_time())
    
    args.output_path = os.path.join(args.output_path,args.experiment_id)
    os.makedirs(args.output_path,exist_ok=True)

    # 必要的 MP 设置
    mp.set_start_method('spawn', force=True)
    
    # [修复] 设置分布式通信所需的环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 可以选择任意一个空闲端口

    # 创建共享消息队列
    manager = mp.Manager()
    queue = manager.Queue()
    
    # 获取可用 GPU 数量
    world_size = min(torch.cuda.device_count(), args.world_size)
    print(f"Spawning {world_size} processes for inference...")
    
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args, queue))