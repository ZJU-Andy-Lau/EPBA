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
from typing import List, Any
import itertools
import yaml
import threading
import queue # 标准库 queue
import sys

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

# ==========================================
# [新增] 线程安全的日志打印与监控
# ==========================================
class MonitorThread(threading.Thread):
    def __init__(self, msg_queue, log_queue, nprocs, max_logs=20):
        super().__init__()
        self.msg_queue = msg_queue
        self.log_queue = log_queue
        self.nprocs = nprocs
        self.status_table = {i: {"status": "INIT", "pair": "--", "prog": "--", "dir": "--", "level": "--", "step": "--"} for i in range(nprocs)}
        self.running = True
        self.logs = [] # 存储最近的日志
        self.max_logs = max_logs

    def run(self):
        # 初始清屏
        os.system('cls' if os.name=='nt' else 'clear')
        
        while self.running:
            # 1. 处理状态更新
            try:
                while True:
                    msg = self.msg_queue.get_nowait()
                    if msg == 'STOP':
                        self.running = False
                        break
                    rank = msg['rank']
                    for k, v in msg.items():
                        if k != 'rank': self.status_table[rank][k] = v
            except queue.Empty:
                pass

            # 2. 处理日志更新
            try:
                while True:
                    log_msg = self.log_queue.get_nowait()
                    self.logs.append(log_msg)
                    if len(self.logs) > self.max_logs:
                        self.logs.pop(0)
            except queue.Empty:
                pass
            
            # 3. 绘制界面 (Double Buffering concept using ANSI)
            self._draw()
            time.sleep(0.1) # 10 FPS 刷新率

    def _draw(self):
        # ANSI控制码：\033[H 回到左上角, \033[J 清除屏幕下方内容
        buffer = "\033[H\033[J" 
        
        # 1. 绘制最近的日志区域
        buffer += "--- Recent Logs ---\n"
        for log in self.logs:
            buffer += f"{log}\n"
        buffer += "-" * 90 + "\n\n"

        # 2. 绘制监控表格
        buffer += f"| {'Rank':^4} | {'Status':^10} | {'Pair':^12} | {'Prog':^10} | {'Dir':^6} | {'Level':^8} | {'Step':^12} |\n"
        buffer += "|" + "-"*88 + "|\n"
        
        for i in range(self.nprocs):
            s = self.status_table[i]
            # 截断过长的字符串防止表格错位
            step_str = str(s['step'])[:12]
            pair_str = str(s['pair'])[:12]
            
            line = f"| {i:^4} | {s['status']:^10} | {pair_str:^12} | {s['prog']:^10} | {s['dir']:^6} | {s['level']:^8} | {step_str:^12} |\n"
            buffer += line
        
        buffer += "=" * 90
        
        # 一次性打印，减少闪烁
        sys.stdout.write(buffer)
        sys.stdout.flush()

class SafeLogger:
    """替代 print，将输出重定向到监控线程的日志队列"""
    def __init__(self, queue, rank):
        self.queue = queue
        self.rank = rank
        
    def info(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}][Rank {self.rank}] {msg}"
        if self.queue:
            self.queue.put(formatted_msg)
        else:
            print(formatted_msg)

# ==========================================
# 辅助函数
# ==========================================

def init_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_images_metadata(args, logger=None) -> List[dict]:
    """
    [修改] 仅加载元数据，不读取大图像文件。
    返回构造 RSImage 所需的参数字典列表。
    """
    base_path = os.path.join(args.root, 'adjust_images')
    if logger: logger.info(f"Scanning images in {base_path}...")
    
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs]
    
    images_meta = []
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        # 仅存储路径和ID，不实例化 RSImage，避免读取图片
        images_meta.append({
            'img_path': img_path,
            'id': idx,
            'device': args.device
        })
    
    if logger: logger.info(f"Found {len(images_meta)} images metadata.")
    return images_meta

def instantiate_rsimage(meta_dict, device) -> RSImage:
    """Worker 在本地根据元数据实例化 RSImage"""
    # 这里的关键是 RSImage 的 __init__ 是否会读图
    # 如果会读图，这里会在 Worker 进程发生 IO，这是预期的（分散 IO）
    # 如果 __init__ 只是存路径，那就是完美的懒加载
    # 假设 RSImage.__init__ 会读图，我们就在这里触发它
    return RSImage(options=None, # RSImage 内部可能不需要 args，或者我们可以 mock 一个
                   img_dir=meta_dict['img_path'], 
                   img_id=meta_dict['id'], 
                   device=device)

def build_pairs_metadata(args, images_meta: List[dict], logger=None) -> List[dict]:
    """
    [修改] 基于元数据构建 Pair 配置。
    此时不进行重叠检测（因为需要加载图片），或者需要一种轻量级的重叠检测方法。
    
    假设：为了不加载所有图片，我们可能无法在 Rank 0 准确判断重叠。
    策略调整：
    Rank 0 生成所有可能的 Pair 组合 (itertools.combinations)。
    Worker 收到 Pair 任务后，先加载图片头信息判断重叠，如果不重叠则跳过。
    
    为了保持逻辑一致性，如果必须在 Rank 0 过滤重叠，Rank 0 必须加载所有 RPC 信息。
    """
    # 鉴于无法修改 RSImage，我们假设 Rank 0 必须加载所有图片才能判断重叠。
    # 为了解决 IO 问题，我们让 Rank 0 加载所有图片（这是瓶颈，但比 Scatter 大对象好）。
    # 或者，我们接受 Worker 可能会收到不重叠的无效 Pair 并在运行时过滤。
    
    # 方案 B：Rank 0 仅分发索引 (i, j)，Worker 加载 image[i] 和 image[j] 后自行判断。
    
    images_num = len(images_meta)
    all_combinations = list(itertools.combinations(range(images_num), 2))
    
    if logger: logger.info(f"Generated {len(all_combinations)} potential pairs.")
    
    # 构建轻量级任务描述
    tasks = []
    for id_a, id_b in all_combinations:
        task = {
            'id_a': id_a,
            'id_b': id_b,
            'output_base': args.output_path,
            'config': {
                'max_window_num': args.max_window_num,
                'min_window_size': args.min_window_size,
                'max_window_size': args.max_window_size,
                'min_area_ratio': args.min_cover_area_ratio,
            }
        }
        tasks.append(task)
        
    return tasks

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

    return encoder,gru

# ==========================================
# Worker 主逻辑
# ==========================================
def main_worker(rank, world_size, args, msg_queue, log_queue):
    # 1. 环境初始化
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
    torch.cuda.set_device(rank)
    args.device = f'cuda:{rank}'
    
    # 2. 启动监控线程 (仅 Rank 0)
    monitor_thread = None
    if rank == 0:
        monitor_thread = MonitorThread(msg_queue, log_queue, world_size)
        monitor_thread.start()

    # 3. 创建 Reporter 和 Logger
    reporter = Reporter(msg_queue, rank)
    logger = SafeLogger(log_queue, rank)
    
    reporter.update(status="INIT")

    # -----------------------------------------------------------
    # [核心修改] 阶段 1: 元数据分发 (Metadata Broadcast & Scatter)
    # -----------------------------------------------------------
    
    # 容器定义
    # images_meta_wrapper: 存放所有图片的元数据列表 (List[Dict])
    # my_tasks_wrapper: 存放分配给当前 Rank 的任务列表 (List[Dict])
    images_meta_wrapper = [None]
    my_tasks_wrapper = [None]
    
    if rank == 0:
        logger.info("Rank 0: Preparing metadata...")
        # Rank 0 生成元数据列表 (非常小，仅包含路径)
        images_meta_list = load_images_metadata(args, logger)
        images_meta_wrapper[0] = images_meta_list
        
        # Rank 0 生成所有任务列表 (id_a, id_b)
        all_tasks = build_pairs_metadata(args, images_meta_list, logger)
        
        # 切分任务
        scatter_list = [[] for _ in range(world_size)]
        for i, task in enumerate(all_tasks):
            scatter_list[i % world_size].append(task)
        logger.info(f"Rank 0: Assigned {len(all_tasks)} pairs to {world_size} workers.")
    else:
        scatter_list = None

    # A. 广播图像元数据 (所有 Rank 都需要知道有哪些图片)
    dist.broadcast_object_list(images_meta_wrapper, src=0)
    all_images_meta = images_meta_wrapper[0] # List[Dict]
    
    # B. 分发任务列表 (每个 Rank 只拿到自己的索引组合)
    dist.scatter_object_list(my_tasks_wrapper, scatter_list, src=0)
    my_tasks = my_tasks_wrapper[0] # List[Dict]
    
    logger.info(f"Received {len(my_tasks)} potential pairs.")

    # -----------------------------------------------------------
    # [核心修改] 阶段 2: 本地加载与过滤 (Local Loading & Filtering)
    # -----------------------------------------------------------
    reporter.update(status="LOADING_IMG")
    
    # 优化策略：分析 my_tasks，找出当前 Rank 真正需要加载哪些图片
    # 避免加载整个数据集，只加载涉及到的图片
    needed_img_ids = set()
    for task in my_tasks:
        needed_img_ids.add(task['id_a'])
        needed_img_ids.add(task['id_b'])
    
    # 实例化需要的 RSImage 对象 (这里会触发 IO，但是分散并发的)
    local_image_cache = {}
    
    # 如果需要在 Rank 0 做全局平差，Rank 0 需要加载所有图片
    # 如果内存不够，Rank 0 可以只在最后再加载所有图片
    # 这里为了简单，Worker 只加载需要的，Rank 0 如果也是 Worker，它也先只加载需要的
    
    for img_id in needed_img_ids:
        # 找到对应的元数据
        meta = next(m for m in all_images_meta if m['id'] == img_id)
        try:
            # 实例化 (IO 操作)
            rs_img = instantiate_rsimage(meta, args.device)
            local_image_cache[img_id] = rs_img
        except Exception as e:
            logger.info(f"Failed to load image {img_id}: {e}")

    # -----------------------------------------------------------
    # 阶段 3: 构建本地 Pair 对象
    # -----------------------------------------------------------
    reporter.update(status="BUILD_PAIR")
    local_pairs = []
    
    for task in my_tasks:
        id_a = task['id_a']
        id_b = task['id_b']
        
        if id_a not in local_image_cache or id_b not in local_image_cache:
            continue
            
        img_a = local_image_cache[id_a]
        img_b = local_image_cache[id_b]
        
        # 此时进行重叠检测 (利用已加载的 RPC/角点信息)
        if is_overlap(img_a, img_b, args.min_window_size ** 2):
            task_config = task['config']
            task_config['output_path'] = os.path.join(task['output_base'], f"pair_{id_a}_{id_b}")
            
            pair = Pair(img_a, img_b, id_a, id_b, task_config, device=args.device)
            local_pairs.append(pair)
    
    logger.info(f"Valid overlapping pairs: {len(local_pairs)}")

    # -----------------------------------------------------------
    # 阶段 4: 模型加载与推理 (与之前相同)
    # -----------------------------------------------------------
    reporter.update(status="LOAD_MODEL")
    encoder, gru = load_models(args)
    
    local_results = []
    reporter.update(status="RUNNING")
    
    for idx, pair in enumerate(local_pairs):
        pair_id_str = f"{pair.id_a}-{pair.id_b}"
        progress_str = f"{idx+1}/{len(local_pairs)}"
        
        reporter.update(pair_id=pair_id_str, progress=progress_str, status="RUNNING")
        
        # 确定性随机种子
        deterministic_seed = args.random_seed + pair.id_a * 100000 + pair.id_b
        init_random_seed(deterministic_seed)
        
        try:
            affine_ab, affine_ba = pair.solve_affines(encoder, gru, reporter=reporter)
            local_results.append({
                pair.id_a: affine_ab,
                pair.id_b: affine_ba
            })
        except Exception as e:
            reporter.update(status="ERROR")
            logger.info(f"Error processing pair {pair_id_str}: {str(e)}")
            # import traceback
            # logger.info(traceback.format_exc())

    # -----------------------------------------------------------
    # 阶段 5: 结果收集
    # -----------------------------------------------------------
    reporter.update(status="GATHERING")
    all_results_lists = [None for _ in range(world_size)]
    dist.all_gather_object(all_results_lists, local_results)
    
    # -----------------------------------------------------------
    # 阶段 6: 全局平差 (仅 Rank 0)
    # -----------------------------------------------------------
    if rank == 0:
        reporter.update(status="SOLVING")
        final_results = []
        for sublist in all_results_lists:
            final_results.extend(sublist)
            
        logger.info(f"Gathered {len(final_results)} pairwise results. Starting Global Solve...")
        
        # 为了全局平差，Rank 0 需要持有所有 Image 对象
        # 检查缓存中是否缺图，如果缺则补充加载
        needed_all_ids = set([m['id'] for m in all_images_meta])
        missing_ids = needed_all_ids - set(local_image_cache.keys())
        
        if missing_ids:
            logger.info(f"Rank 0: Loading remaining {len(missing_ids)} images for global solver...")
            for img_id in tqdm(missing_ids, desc="Loading Images"):
                meta = next(m for m in all_images_meta if m['id'] == img_id)
                local_image_cache[img_id] = instantiate_rsimage(meta, args.device)
        
        # 构建符合 GlobalSolver 接口的 images 列表 (按 id 排序)
        sorted_images = [local_image_cache[i] for i in sorted(local_image_cache.keys())]
        
        solver_configs = load_config(args.solver_config_path)
        solver = GlobalAffineSolver(images=sorted_images,
                                    device=args.device,
                                    anchor_indices=[0],
                                    max_iter=100,
                                    converge_tol=1e-6)
        
        Ms = solver.solve(final_results)
        
        # 更新 RPC 并输出
        for image in sorted_images:
            M = Ms[image.id]
            logger.info(f"Affine Matrix Image {image.id}:\n{M.cpu().numpy()}") # 简化日志
            image.rpc.Update_Adjust(M)
            image.rpc.Merge_Adjust()
        
        reporter.update(status="DONE")
        time.sleep(2) # 让监控面板展示一会儿 DONE 状态
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

    parser.add_argument('--device',type=str,default='cuda')
    
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
    msg_queue = manager.Queue()
    log_queue = manager.Queue()
    
    world_size = min(torch.cuda.device_count(), args.world_size)
    print(f"Spawning {world_size} processes for inference...")
    
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args, msg_queue, log_queue))