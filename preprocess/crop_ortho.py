import os
import glob
import math
import numpy as np
import cv2
import rasterio
from rasterio import features, Affine
from rasterio.warp import reproject, Resampling, transform_geom
from rasterio.transform import from_origin
from rasterio.crs import CRS
from shapely.geometry import shape, box, Polygon
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class OrthophotoGridProcessor:
    def __init__(self, input_folder: str, output_folder: str, grid_size_meters: float, target_pixel_size: int):
        """
        初始化处理器
        :param input_folder: 存放TIF影像的输入文件夹路径
        :param output_folder: 结果输出文件夹路径
        :param grid_size_meters: 网格的物理尺寸（米），即 x * x 中的 x
        :param target_pixel_size: 输出影像的像素尺寸，即 y * y 中的 y
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.grid_size = grid_size_meters
        self.target_px = target_pixel_size
        
        # 获取所有tif文件
        self.image_paths = glob.glob(os.path.join(input_folder, "*.tif"))
        if not self.image_paths:
            raise FileNotFoundError(f"在 {input_folder} 中未找到 .tif 文件")
            
        self.image_polygons = {} # 存储每张图的有效范围 {filename: shapely_polygon} (统一投影坐标系下)
        self.intersection_polygon = None # 公共交集区域
        self.valid_grids = [] # 存储生成的网格 [(grid_id, shapely_box), ...]
        
        # 处理用的统一投影坐标系 (Projected CRS)
        self.processing_crs = None 

        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)

    def _estimate_utm_crs(self, src) -> CRS:
        """
        如果影像为地理坐标系，根据中心经纬度估算合适的 UTM 投影坐标系
        """
        # 获取影像中心点经纬度
        left, bottom, right, top = src.bounds
        center_lon = (left + right) / 2
        center_lat = (bottom + top) / 2
        
        # 计算 UTM 带号
        zone_number = int((center_lon + 180) / 6) + 1
        
        # 判断南北半球 (北半球: 326xx, 南半球: 327xx)
        if center_lat >= 0:
            epsg_code = 32600 + zone_number
        else:
            epsg_code = 32700 + zone_number
            
        return CRS.from_epsg(epsg_code)

    def _get_valid_footprint(self, src, target_crs: CRS) -> Polygon:
        """
        提取单张影像的有效像素范围（优化版）
        包含：降采样读取、形态学去噪、坐标转换、几何简化
        """
        # 1. 降采样设置 (加速 I/O)
        # 使用 1/10 的分辨率进行轮廓提取，对于边界计算通常足够精确
        decimation = 10 
        new_height = max(1, int(src.height / decimation))
        new_width = max(1, int(src.width / decimation))
        
        # 2. 读取降采样后的 Mask
        try:
            # out_shape 自动触发重采样
            mask = src.read_masks(1, out_shape=(new_height, new_width))
        except Exception:
            data = src.read(1, out_shape=(new_height, new_width))
            mask = (data != 0).astype('uint8') * 255
            
        # 3. 形态学处理 (去噪)
        # 开运算去除孤立噪点，闭运算填充内部小孔
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. 计算降采样后的 Transform
        # 原始 transform 将像素坐标映射到地理坐标
        # 新 transform 需要适应缩小的像素网格
        # new_transform * (x_small, y_small) = old_transform * (x_small * decimation, y_small * decimation)
        downsample_transform = src.transform * Affine.scale(src.width / new_width, src.height / new_height)

        # 5. 矢量化
        shapes = features.shapes(mask, transform=downsample_transform)
        polygons = [shape(geom) for geom, val in shapes if val > 0]
        
        if not polygons:
            raise ValueError(f"影像 {src.name} 中没有检测到有效像素！")
            
        # 6. 合并多边形
        multi_poly = unary_union(polygons)
        # 修复拓扑
        valid_poly = multi_poly.buffer(0)
        
        # 7. 几何简化 (Simplification) - 极大提升后续求交集速度
        # tolerance=1 表示允许 1 米（或度）的误差，对于边界来说可以忽略不计
        # 如果是经纬度坐标，tolerance 需要很小 (如 1e-5)，但在后续投影转换后再简化更安全
        # 这里先做轻微简化
        valid_poly = valid_poly.simplify(tolerance=0.0 if src.crs.is_geographic else 1.0, preserve_topology=True)

        # 8. 坐标转换 (若原图与处理坐标系不同)
        if src.crs != target_crs:
            # rasterio.warp.transform_geom 可以处理 GeoJSON 风格的 geometry
            # 将 Shapely 对象转为 GeoJSON mapping
            geom_mapping = valid_poly.__geo_interface__
            
            # 执行投影转换
            transformed_geom = transform_geom(
                src_crs=src.crs,
                dst_crs=target_crs,
                geom=geom_mapping
            )
            valid_poly = shape(transformed_geom)

        # 9. 二次简化 (投影后基于米的单位简化)
        # 再次简化，去除重投影可能产生的微小节点，容差设为 1 米
        valid_poly = valid_poly.simplify(tolerance=1.0, preserve_topology=True)
        
        # 再次修复可能因简化产生的拓扑问题
        return valid_poly.buffer(0)

    def step1_calculate_intersection(self):
        """
        计算所有影像的公共有效重叠区域 (自适应坐标系)
        """
        print(f"[Step 1] 开始计算 {len(self.image_paths)} 张影像的公共重叠区域...")
        
        intersection = None

        # 预扫描第一张图，确定统一的处理坐标系 (Processing CRS)
        with rasterio.open(self.image_paths[0]) as first_src:
            if first_src.crs.is_geographic:
                print("  -> 检测到地理坐标系 (经纬度)，正在自动计算适合的 UTM 投影...")
                self.processing_crs = self._estimate_utm_crs(first_src)
                print(f"  -> 已选定投影坐标系: {self.processing_crs['init'] if 'init' in self.processing_crs else self.processing_crs.to_string()}")
            else:
                print("  -> 检测到投影坐标系 (米)，将直接使用。")
                self.processing_crs = first_src.crs

        for idx, img_path in enumerate(self.image_paths):
            filename = os.path.basename(img_path)
            print(f"  -> 处理影像 ({idx+1}/{len(self.image_paths)}): {filename}")
            
            with rasterio.open(img_path) as src:
                # 提取有效范围 (包含降采样、去噪、投影转换)
                poly = self._get_valid_footprint(src, self.processing_crs)
                self.image_polygons[filename] = poly
                
                # 求交集
                if intersection is None:
                    intersection = poly
                else:
                    intersection = intersection.intersection(poly)
                    
                # 快速检查
                if intersection.is_empty:
                    raise RuntimeError(f"在处理 {filename} 时交集变为空！影像之间可能没有重叠。")

        self.intersection_polygon = intersection
        print("[Step 1] 公共区域计算完成。")

    def step2_generate_grids(self):
        """
        通过滑动窗口搜索最佳网格对齐位置，最大化网格数量
        """
        print(f"[Step 2] 正在搜索最佳网格铺设位置 (Grid Size: {self.grid_size}m)...")
        
        minx, miny, maxx, maxy = self.intersection_polygon.bounds
        
        # 定义搜索步长 (例如网格的 10%)
        # 步长越小，对齐越精确，但计算时间越长
        step_percent = 0.1 
        search_step = self.grid_size * step_percent
        
        best_grids = []
        max_count = -1
        best_offset = (0, 0)
        
        # 使用 numpy 生成偏移量序列
        # 只需要在一个 grid_size 的周期内搜索即可涵盖所有对齐情况
        offsets = np.arange(0, self.grid_size, search_step)
        total_iterations = len(offsets) ** 2
        print(f"  -> 将尝试 {total_iterations} 种偏移组合...")

        for off_x in offsets:
            for off_y in offsets:
                current_grids = []
                grid_id_counter = 0
                
                # 计算当前偏移下的网格起点
                # X轴向右偏移，Y轴向下偏移(注意 maxy 是上方)
                start_x = minx + off_x
                start_y = maxy - off_y
                
                # 生成网格
                curr_y = start_y
                while curr_y - self.grid_size >= miny:
                    curr_x = start_x
                    while curr_x + self.grid_size <= maxx:
                        # 构建网格几何体
                        b = box(curr_x, curr_y - self.grid_size, curr_x + self.grid_size, curr_y)
                        
                        # 严格判定：必须完全包含在公共区域内
                        if self.intersection_polygon.contains(b):
                            current_grids.append((grid_id_counter, b))
                            grid_id_counter += 1
                        
                        curr_x += self.grid_size
                    curr_y -= self.grid_size
                
                # 更新最优解
                if len(current_grids) > max_count:
                    max_count = len(current_grids)
                    best_grids = current_grids
                    best_offset = (off_x, off_y)
        
        # 保存最终结果 (重新编号 ID)
        self.valid_grids = [(i, g[1]) for i, g in enumerate(best_grids)]
        
        print(f"[Step 2] 搜索完成。")
        print(f"  -> 最佳偏移: X+{best_offset[0]:.1f}m, Y-{best_offset[1]:.1f}m")
        print(f"  -> 最终生成 {len(self.valid_grids)} 个有效完全重叠网格。")

    def step3_visualize(self):
        """
        绘制可视化示意图
        注意：此时所有坐标均为投影坐标 (米)
        """
        print("[Step 3] 生成可视化预览图...")
        fig, ax = plt.subplots(figsize=(12, 12))
        
        colors = plt.cm.get_cmap('tab10')
        
        # 1. 绘制影像轮廓
        for i, (name, poly) in enumerate(self.image_polygons.items()):
            if poly.geom_type == 'MultiPolygon':
                for geom in poly.geoms:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=colors(i%10), linestyle='--', linewidth=1, label=f'Img: {name}' if i < 5 else None)
            else:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=colors(i%10), linestyle='--', linewidth=1, label=f'Img: {name}' if i < 5 else None)

        # 2. 绘制交集
        if self.intersection_polygon.geom_type == 'MultiPolygon':
            for geom in self.intersection_polygon.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, color='gray', alpha=0.3, label='Intersection')
        else:
            x, y = self.intersection_polygon.exterior.xy
            ax.fill(x, y, color='gray', alpha=0.3, label='Intersection')

        # 3. 绘制网格
        for gid, poly_box in self.valid_grids:
            x, y = poly_box.exterior.xy
            ax.plot(x, y, color='red', linewidth=1.5)
            # 标注ID (稍微简化显示，避免太密)
            cx, cy = poly_box.centroid.x, poly_box.centroid.y
            ax.text(cx, cy, str(gid), color='black', fontsize=8, ha='center', va='center', weight='bold')

        ax.set_title(f"Grid Layout Preview (Projected CRS)\nTotal Grids: {len(self.valid_grids)}")
        ax.set_xlabel("Easting (Meters)")
        ax.set_ylabel("Northing (Meters)")
        ax.axis('equal')
        
        # 图例去重
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        preview_path = os.path.join(self.output_folder, "grid_layout_preview.png")
        plt.savefig(preview_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Step 3] 预览图已保存至: {preview_path}")

    def step4_process_and_export(self):
        """
        批量裁切与输出
        关键点：使用 processing_crs 作为目标坐标系，rasterio 会自动处理重投影
        """
        print("[Step 4] 开始批量裁切与输出...")
        
        # 目标分辨率 (米/像素)
        target_res = self.grid_size / self.target_px
        
        for grid_id, grid_box in self.valid_grids:
            grid_dir = os.path.join(self.output_folder, f"Grid_{grid_id}")
            os.makedirs(grid_dir, exist_ok=True)
            
            minx, miny, maxx, maxy = grid_box.bounds
            
            # 构建目标变换矩阵 (基于 Processing CRS，单位米)
            dst_transform = from_origin(minx, maxy, target_res, target_res)
            
            for img_path in self.image_paths:
                filename = os.path.basename(img_path)
                save_name = os.path.splitext(filename)[0] + ".png"
                save_path = os.path.join(grid_dir, save_name)
                
                with rasterio.open(img_path) as src:
                    destination = np.zeros((self.target_px, self.target_px), dtype=np.float32)
                    
                    # 执行重投影 (Reproject)
                    # src_crs: 原图坐标系 (可能是经纬度)
                    # dst_crs: 处理坐标系 (米)
                    # rasterio 会自动处理两者之间的转换和重采样
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=destination,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=self.processing_crs, # 指定为统一的投影坐标系
                        resampling=Resampling.bilinear
                    )
                    
                    # --- 归一化优化：使用 2% - 98% 百分比截断拉伸 ---
                    # 避免因极值(outliers)或噪点导致图像整体过暗
                    
                    # 获取有效像素值 (排除0值背景，如果有的话)
                    valid_pixels = destination[destination > 0]
                    
                    if valid_pixels.size > 0:
                        # 计算 2% 和 98% 分位数
                        p2, p98 = np.percentile(valid_pixels, 2), np.percentile(valid_pixels, 98)
                    else:
                        # 如果全黑或全是0，退化为 min/max
                        p2, p98 = destination.min(), destination.max()
                    
                    if p98 - p2 > 0:
                        # 截断数据到 [p2, p98] 区间
                        img_clipped = np.clip(destination, p2, p98)
                        # 线性拉伸到 0-255
                        norm_img = ((img_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
                    else:
                        norm_img = np.zeros_like(destination, dtype=np.uint8)
                        
                    cv2.imwrite(save_path, norm_img)
                    
            if grid_id % 10 == 0:
                print(f"  -> 已处理网格 Grid_{grid_id} ...")

        print("[Step 4] 全部处理完成！")

    def run(self):
        self.step1_calculate_intersection()
        self.step2_generate_grids()
        
        if not self.valid_grids:
            print("警告：没有生成任何有效网格！请检查影像重叠情况或缩小网格尺寸。")
            return
            
        self.step3_visualize()
        self.step4_process_and_export()


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 配置参数
    INPUT_DIR = "../datasets/shandong_ortho/2"   # 输入文件夹 
    OUTPUT_DIR = "../datasets/shandong_ortho_1500/2"    # 输出文件夹 
    
    GRID_SIZE = 1500.0  
    OUTPUT_PIXELS = 3000 

    try:
        processor = OrthophotoGridProcessor(
            input_folder=INPUT_DIR, 
            output_folder=OUTPUT_DIR, 
            grid_size_meters=GRID_SIZE, 
            target_pixel_size=OUTPUT_PIXELS
        )
        processor.run()
    except Exception as e:
        print(f"程序运行出错: {e}")