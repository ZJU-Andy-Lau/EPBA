import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.transforms import v2

import numpy as np
import cv2
from shapely.geometry import Polygon, box
import os
import typing
from typing import List,Tuple
import time


if typing.TYPE_CHECKING:
    from model.encoder import Encoder
    from rs_image import RSImage
    from pair import Pair

def warp_quads(corners, values:List[np.ndarray], output_size=(512, 512)):
        
        # 1. 基础参数解析
        batched = True
        if corners.ndim < 3:
            corners = corners[None]
            batched = False
        B = corners.shape[0]
        target_h, target_w = output_size

        dst_pts_rc = np.array([
            [0, 0],                     # Top-Left
            [0, target_w - 1],          # Top-Right
            [target_h - 1, target_w - 1], # Bottom-Right
            [target_h - 1, 0]           # Bottom-Left
        ], dtype=np.float32)

            
        warped_values = []
        Hs = []

        # 4. 循环处理每个四边形
        for i in range(B):
            src_pts_rc = corners[i].astype(np.float32)
            src_pts_xy = src_pts_rc[:, ::-1] # (row, col) -> (col, row) / (y, x) -> (x, y)
            dst_pts_xy = dst_pts_rc[:, ::-1] 
            H_xy = cv2.getPerspectiveTransform(src_pts_xy, dst_pts_xy)
            
            warped_value_i = []

            for value in values:
                warped_value = cv2.warpPerspective(
                    value, 
                    H_xy, 
                    (target_w, target_h), 
                    flags=cv2.INTER_LINEAR
                )
                warped_value_i.append(warped_value)

            H_rc = H_xy.copy()
            H_rc[:, [0, 1]] = H_rc[:, [1, 0]]
            H_rc[[0, 1], :] = H_rc[[1, 0], :]
            
            warped_values.append(warped_value_i)
            Hs.append(H_rc)

        values_res = []
        for i in range(len(values)):
            values_res.append(
                 np.stack([
                      warped_values[j][i] for j in range(len(warped_values))
                 ],axis=0)
            )
        Hs = np.stack(Hs,axis=0)

        if not batched:
            values_res = values_res[0]
            Hs = Hs[0]

        return values_res,Hs

def find_intersection(corners):
    """
    计算多个凸四边形的共同重叠区域（交集）。

    Args:
        corners (np.ndarray): 形状为 (B, 4, 2) 的数组。
                              存储 B 个凸四边形的顶点坐标，格式为 (x, y)。
                              顺序可以是顺时针或逆时针。

    Returns:
        intersection_coords (np.ndarray): 形状为 (N, 2) 的数组，记录重叠区域多边形的顶点。
                                          坐标格式保持为 (x, y)。
                                          如果无重叠，返回空数组 (0, 2)。
    """
    B = corners.shape[0]
    if B == 0:
        return np.empty((0, 2))

    # 1. 将第一个四边形作为初始交集区域
    # shapely 接受的输入本来就是 (x, y) 序列，现在语义完全匹配
    current_poly = Polygon(corners[0])

    if not current_poly.is_valid:
        # 尝试修复无效的多边形（例如自相交）
        current_poly = current_poly.buffer(0)

    # 2. 迭代与其余四边形求交集
    for i in range(1, B):
        next_poly = Polygon(corners[i])
        
        if not next_poly.is_valid:
            next_poly = next_poly.buffer(0)

        # 计算交集
        current_poly = current_poly.intersection(next_poly)

        # 如果交集变为空，可以提前结束
        if current_poly.is_empty:
            return np.empty((0, 2))

    # 3. 解析结果
    
    if current_poly.is_empty:
        return np.empty((0, 2))
    
    if current_poly.geom_type == 'Polygon':
        # exterior.coords 返回的是一系列点
        coords = np.array(current_poly.exterior.coords)
        # 去除最后一个重复的闭合点
        if len(coords) > 0 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        return coords
    
    elif current_poly.geom_type in ['Point', 'MultiPoint', 'LineString']:
        # 如果交集退化为点或线
        if hasattr(current_poly, 'coords'):
             return np.array(current_poly.coords)
        elif hasattr(current_poly, 'geoms'): # MultiPoint
             all_coords = []
             for geom in current_poly.geoms:
                 all_coords.extend(geom.coords)
             return np.array(all_coords)
        else:
             return np.empty((0, 2))
    else:
        return np.empty((0, 2))
    
def convert_diags_to_tlbr(diags: np.ndarray) -> np.ndarray:    
    tlbr = diags.copy() 
    tlbr[:, 1, 1] = diags[:, 0, 1]
    tlbr[:, 0, 1] = diags[:, 1, 1]
    return tlbr

def find_squares(corners, a_max, a_min=1.0, target_area_ratio = 0.5):
    """
    在凸多边形内划分正方形，通过迭代缩小边长 a，直到填充面积超过多边形面积的一定比例。

    Args:
        corners (np.ndarray): 形状为 (N, 2) 的数组，记录多边形顶点 (x, y)。
        a_max (float): 正方形边长的初始迭代值。
        a_min (float): 正方形边长的最小限制，当 current_a < a_min 时停止迭代。
        target_area_ratio (float): 目标面积比例。

    Returns:
        squares (np.ndarray): 形状为 (M, 2, 2) 的数组。
                              M 为正方形数量。
                              每个正方形由 [左上角(x,y), 右下角(x,y)] 组成。
    """
    # 1. 创建多边形并计算目标面积
    poly = Polygon(corners)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    poly_area = poly.area
    target_area = target_area_ratio * poly_area
    
    # 获取边界框
    min_x, min_y, max_x, max_y = poly.bounds
    
    current_a = float(a_max)
    
    # 用于显示的迭代计数器
    iteration = 0
    
    # 迭代循环：只要当前边长大于等于最小边长，就继续尝试
    while current_a >= a_min:
        best_squares_for_this_a = []
        max_count_for_this_a = -1
        
        # --- 改进部分：网格偏移搜索 ---
        # 不仅仅从 min_x, min_y 开始，而是尝试在 [0, a) 范围内平移网格
        search_steps = 10 
        
        # 生成偏移量数组
        offsets_x = np.linspace(0, current_a, search_steps, endpoint=False)
        offsets_y = np.linspace(0, current_a, search_steps, endpoint=False)
        
        for off_x in offsets_x:
            for off_y in offsets_y:
                current_candidates = []
                
                # 基于当前偏移量生成网格起始点
                start_x = min_x + off_x
                start_y = min_y + off_y
                
                # 生成网格
                # 注意范围要稍微大一点以覆盖整个多边形
                epsilon = 1e-9
                x_range = np.arange(start_x, max_x + epsilon, current_a)
                y_range = np.arange(start_y, max_y + epsilon, current_a)
                
                for x in x_range:
                    for y in y_range:
                        # 快速预筛选：如果正方形完全在边界框外，直接跳过 (优化性能)
                        if x > max_x or y > max_y:
                            continue
                            
                        # 构建正方形
                        sq_poly = box(x, y, x + current_a, y + current_a)
                        
                        # 严格检查包含关系
                        if poly.contains(sq_poly):
                            current_candidates.append([
                                [x, y], 
                                [x + current_a, y + current_a]
                            ])
                
                # 如果当前偏移找到的正方形更多，则更新最佳方案
                if len(current_candidates) > max_count_for_this_a:
                    max_count_for_this_a = len(current_candidates)
                    best_squares_for_this_a = current_candidates

        # 计算当前最佳覆盖率
        current_coverage = max_count_for_this_a * (current_a ** 2)
        
        # 4. 检查是否满足停止条件
        if current_coverage > target_area:
            return convert_diags_to_tlbr(np.array(best_squares_for_this_a))
        
        # 5. 更新迭代参数
        current_a /= 2.0
        iteration += 1
        
    print(f"Warning: Minimum edge length ({a_min}) reached without meeting area threshold.")
    return np.array([])

def quadsplit_diags(diags:np.ndarray) -> np.ndarray:
    diags = diags.astype(float)
    x_tl = diags[:, 0, 0]  # (N,)
    y_tl = diags[:, 0, 1]  # (N,)
    x_br = diags[:, 1, 0]  # (N,)
    y_br = diags[:, 1, 1]  # (N,)
    x_mid = (x_tl + x_br) / 2.0  # (N,)
    y_mid = (y_tl + y_br) / 2.0  # (N,)
    q1 = np.stack([
        np.stack([x_tl, y_tl], axis=1), 
        np.stack([x_mid, y_mid], axis=1)
    ], axis=1) # (N, 2, 2)
    q2 = np.stack([
        np.stack([x_mid, y_tl], axis=1), 
        np.stack([x_br, y_mid], axis=1)
    ], axis=1) # (N, 2, 2)
    q3 = np.stack([
        np.stack([x_tl, y_mid], axis=1), 
        np.stack([x_mid, y_br], axis=1)
    ], axis=1) # (N, 2, 2)
    q4 = np.stack([
        np.stack([x_mid, y_mid], axis=1), 
        np.stack([x_br, y_br], axis=1)
    ], axis=1) # (N, 2, 2)

    new_diags = np.stack([q1, q2, q3, q4], axis=1).reshape(-1,2,2)
    
    return new_diags

feats_type = Tuple[torch.Tensor,torch.Tensor,torch.Tensor]

@torch.no_grad()
def extract_features(encoder:'Encoder',imgs_a:np.ndarray,imgs_b:np.ndarray,device:str = 'cuda') -> Tuple[feats_type,feats_type]:
    """
    Args:
        encoder: Encoder
        imgs_a: np.ndarray, (N,H,W,3)
        imgs_b: np.ndarray, (N,H,W,3)
        device: str

    Returns:
        feats
    """
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #     ])
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True), # [0, 255] -> [0.0, 1.0] (在GPU上极快)
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    input_a = torch.from_numpy(imgs_a).permute(0,3,1,2).contiguous().to(device, non_blocking=True)
    input_b = torch.from_numpy(imgs_b).permute(0,3,1,2).contiguous().to(device, non_blocking=True)
    input_a = transform(input_a)
    input_b = transform(input_b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"========transform data time:{t1 - t0}s")
    encoder = encoder.to(device).eval().half()
    input_a = input_a.to(device).half()
    input_b = input_b.to(device).half()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    print(f"========to device time:{t2 - t1}s")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        feats_a,feats_b = encoder(input_a,input_b)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    print(f"========encoder forward time:{t3 - t2}s")
    feats_a = [i.to(torch.float32) for i in feats_a]
    feats_b = [i.to(torch.float32) for i in feats_b]

    return feats_a,feats_b

def get_coord_mat(h,w,b = 0,ds = 1,device = 'cuda'):
    """
    Args:
        h: 下采样后的高度
        w: 下采样后的宽度
        b: batchsize
        ds: 下采样倍率
        device: str
    
    Returns:
        coords: torch.Tensor, (b,h,w,2), (row,col)
    """
    grid_row, grid_col = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    coords_row = grid_row * ds + ds / 2.0
    coords_col = grid_col * ds + ds / 2.0
    coords = torch.stack([coords_row,coords_col],dim=-1) # h,w,2
    if b > 0:
        coords = torch.stack([coords] * b,dim=0) # b,h,w,2
    return coords

def apply_H(coords:torch.Tensor,Hs:torch.Tensor,device:str = 'cpu'):
    """
    coords: B,N,2
    Hs: B,3,3

    return: B,N,2
    """
    B,N = coords.shape[:2]
    coords = coords.permute(0,2,1) # B,2,N
    ones = torch.ones(B, 1, N, device=device)
    coords_homo = torch.cat([coords,ones],dim=1) # B,3,N
    coords_trans_homo = torch.bmm(Hs,coords_homo) # (B,3,3) @ (B,3,N) -> (B,3,N)
    eps = 1e-7
    z = coords_trans_homo[:, 2:3, :]
    coords_trans = coords_trans_homo[:, :2, :] / (z + eps) # B,2,N
    return coords_trans.permute(0,2,1) # B,N,2

def apply_M(coords:torch.Tensor,Ms:torch.Tensor,device:str = 'cpu'):
    """
    coords: B,N,2
    Ms: B,2,3

    return: B,N,2
    """
    B,N = coords.shape[:2]
    coords = coords.permute(0,2,1) # B,2,N
    ones = torch.ones(B, 1, N, device=device)
    coords_homo = torch.cat([coords,ones],dim=1) # B,3,N
    coords_trans = torch.bmm(Ms,coords_homo) # (B,2,3) @ (B,3,N) -> (B,2,N)
    return coords_trans.permute(0,2,1) # B,N,2

def solve_weighted_affine(src: torch.Tensor, dst: torch.Tensor, scores: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    使用加权最小二乘法计算从 src 到 dst 的仿射变换矩阵。

    Args:
        src (torch.Tensor): 源点坐标，形状为 (N, 2)。
        dst (torch.Tensor): 目标点坐标，形状为 (N, 2)。
        scores (torch.Tensor): 每个点的置信度权重，形状为 (N,)。
        eps (float): 防止除零或数值不稳定的微小值。

    Returns:
        torch.Tensor: 仿射变换矩阵，形状为 (2, 3)。
                      格式为 [[a, b, tx], [c, d, ty]]。
    """
    # 1. 检查输入形状
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"src shape must be (N, 2), got {src.shape}")
    if dst.ndim != 2 or dst.shape[1] != 2:
        raise ValueError(f"dst shape must be (N, 2), got {dst.shape}")
    if scores.ndim != 1 or scores.shape[0] != src.shape[0]:
        raise ValueError(f"scores shape must be (N,), got {scores.shape}")
    
    # 2. 确保数据类型为浮点型 (lstsq 需要 float) 且设备一致
    # 如果输入是 int，必须转为 float
    if not src.is_floating_point():
        src = src.float()
    if not dst.is_floating_point():
        dst = dst.float()
    if not scores.is_floating_point():
        scores = scores.float()

    # 确保所有张量在同一设备
    device = src.device
    if dst.device != device or scores.device != device:
        raise RuntimeError("All input tensors (src, dst, scores) must be on the same device.")

    N = src.shape[0]

    # 3. 构建增广矩阵 (Augmented Matrix)
    # src_aug = [x, y, 1]
    ones = torch.ones(N, 1, device=device, dtype=src.dtype)
    src_aug = torch.cat([src, ones], dim=1)  # 形状: (N, 3)

    # 4. 应用权重 (Weighted Least Squares)
    # 核心原理: 最小化 sum( w_i * ||Ax_i - b_i||^2 )
    # 等价于求解线性方程: sqrt(W) * A * X = sqrt(W) * B
    
    # 计算权重的平方根
    # 注意: 这里的 eps 很重要，防止权重为0时出现数值问题
    weights_sqrt = torch.sqrt(torch.clamp(scores, min=0) + eps).view(-1, 1) # 形状: (N, 1)

    # 利用广播机制对矩阵的每一行进行加权
    A_weighted = src_aug * weights_sqrt  # 形状: (N, 3)
    B_weighted = dst * weights_sqrt      # 形状: (N, 2)

    # 5. 求解线性方程组
    # 求解 A_weighted @ X = B_weighted
    # torch.linalg.lstsq 在 GPU 上也能高效运行
    # driver='gels' 是求解一般矩阵最小二乘的标准驱动
    try:
        result = torch.linalg.lstsq(A_weighted, B_weighted, driver='gels')
    except RuntimeError:
        # 如果 gels 失败（极少数情况），回退到 gelsd (基于SVD，更稳健但稍慢)
        result = torch.linalg.lstsq(A_weighted, B_weighted, driver='gelsd')
        
    X = result.solution  # 形状: (3, 2)

    # 6. 格式化输出 (3, 2) -> (2, 3)
    affine_matrix = X.t()

    return affine_matrix

def is_overlap(image_a:'RSImage',image_b:'RSImage',min_area:float = 0.):
    poly1 = Polygon(image_a.corner_xys)
    poly2 = Polygon(image_b.corner_xys)
    overlap_area = poly1.intersection(poly2).area
    return overlap_area >= min_area

def convert_pair_dicts_to_solver_inputs(
    pair_list,
    device = None,
    dtype = torch.float32,
):
    if device is None:
        first_tensor = None
        for d in pair_list:
            for v in d.values():
                first_tensor = v
                break
            if first_tensor is not None:
                break
        if first_tensor is not None and isinstance(first_tensor, torch.Tensor):
            device = first_tensor.device
        else:
            device = torch.device("cpu")

    edge_src_list: List[int] = []
    edge_dst_list: List[int] = []
    A_list: List[torch.Tensor] = []
    t_list: List[torch.Tensor] = []

    # 3. 遍历每个无向边 dict，拆成两条有向边
    for d in pair_list:
        if len(d) != 2:
            raise ValueError(
                f"pair_list 中的 dict 期望恰好有两个 key（对应一对节点），但实际 len={len(d)}。内容={d}"
            )

        # 例如 d = {i: M_i_j, j: M_j_i}
        ids = list(d.keys())
        i, j = ids[0], ids[1]

        M_i_j = d[i]
        M_j_i = d[j]

        # 检查输入 tensor 形状
        if not (isinstance(M_i_j, torch.Tensor) and M_i_j.shape == (2, 3)):
            raise ValueError(f"M_{i}_to_{j} 的形状不是 (2,3)，实际为 {M_i_j.shape}")
        if not (isinstance(M_j_i, torch.Tensor) and M_j_i.shape == (2, 3)):
            raise ValueError(f"M_{j}_to_{i} 的形状不是 (2,3)，实际为 {M_j_i.shape}")

        # 统一到指定 device / dtype
        M_i_j = M_i_j.to(device=device, dtype=dtype)
        M_j_i = M_j_i.to(device=device, dtype=dtype)

        # 从 (2,3) 中拆出 A (2,2) 和 t (2,)
        A_i_j = M_i_j[:, :2]  # 线性部分
        t_i_j = M_i_j[:, 2]   # 平移部分

        A_j_i = M_j_i[:, :2]
        t_j_i = M_j_i[:, 2]

        # 有向边 i -> j
        edge_src_list.append(i)
        edge_dst_list.append(j)
        A_list.append(A_i_j)
        t_list.append(t_i_j)

        # 有向边 j -> i
        edge_src_list.append(j)
        edge_dst_list.append(i)
        A_list.append(A_j_i)
        t_list.append(t_j_i)

    # 4. 转为统一的 tensor 形式
    edge_src = torch.tensor(edge_src_list, dtype=torch.long, device=device)
    edge_dst = torch.tensor(edge_dst_list, dtype=torch.long, device=device)
    A_ij = torch.stack(A_list, dim=0).to(device=device, dtype=dtype)   # (E,2,2)
    t_ij = torch.stack(t_list, dim=0).to(device=device, dtype=dtype)   # (E,2)

    # 5. 权重先全部设为 1.0
    w_ij = torch.ones(edge_src.shape[0], dtype=dtype, device=device)

    return edge_src, edge_dst, A_ij, t_ij, w_ij

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """计算两组 (lat, lon) 坐标之间的 Haversine 距离 (米)"""
    R = 6371000 
    lat1 = coords1[:, 0]
    lon1 = coords1[:, 1]
    lat2 = coords2[:, 0]
    lon2 = coords2[:, 1]

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def get_error_report(pairs:List['Pair']):
    all_distances_list = []
    for pair in pairs:
        distances = pair.check_error()
        all_distances_list.append(distances)
    all_distances = np.concatenate(all_distances_list)
    total_points = len(all_distances)
    report = {
        'mean': float(np.mean(all_distances)),
        'median': float(np.median(all_distances)),
        'max': float(np.max(all_distances)),
        'rmse': float(np.sqrt(np.mean(all_distances**2))),
        'count': int(total_points),
        '<1m_percent': float(((all_distances < 1.0).sum() / total_points) * 100),
        '<3m_percent': float(((all_distances < 3.0).sum() / total_points) * 100),
        '<5m_percent': float(((all_distances < 5.0).sum() / total_points) * 100),
    }
    return report

def affine_xy_to_rowcol(matrix):
    """
    将 (x, y) 坐标系下的仿射变换矩阵转换为 (row, col) 坐标系。
    
    Args:
        matrix (torch.Tensor): 形状为 (N, 2, 3) 或 (2, 3) 的 tensor。
                               原矩阵形式为 [[sx, shy, tx], [shx, sy, ty]] (对应 x, y)
    
    Returns:
        torch.Tensor: 转换后的矩阵，形状与输入相同。
                      新矩阵形式为 [[sy, shx, ty], [shy, sx, tx]] (对应 row, col)
    """
    is_batch = matrix.dim() == 3
    if not is_batch:
        matrix = matrix.unsqueeze(0)

    permuted_rows = matrix[:, [1, 0], :]
    result = permuted_rows[:, :, [1, 0, 2]]

    if not is_batch:
        result = result.squeeze(0)
        
    return result