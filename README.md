# EPBA：无控/有控正射影像联合平差框架（中文使用说明）

本文档面向**源码使用者与工程部署者**，详细说明 EPBA 项目的数据构建、训练流程与两类推理（EPBA / EPA）使用方式。

---

## 1. 项目概览

EPBA 代码仓库的核心流程可概括为：

1. 使用 `preprocess/generate_train_dataset.py` 从多景已配准且有重叠的正射影像中构建训练样本（HDF5）。
2. 先进行预训练（`train/pretrain.py`），再进行正式训练（`train/train.py`）。
3. 推理阶段分为：
   - **EPBA 平差**：无参考影像，仅依赖待平差影像之间的重叠关系（`infer/main.py`）。
   - **EPA 平差**：有参考影像，通过“待平差影像 ↔ 参考影像”关系估计校正（`infer/main_ref.py`）。

---

## 2. 环境与依赖（建议）

> 仓库中未提供统一顶层 `requirements.txt`，请按代码实际依赖准备环境。

### 2.1 Python 与基础库
建议：

- Python 3.10+
- PyTorch（需 CUDA）
- torchvision
- numpy / scipy
- opencv-python
- rasterio
- shapely
- h5py
- tqdm
- omegaconf
- scikit-image
- tensorboard

### 2.2 DDP 运行前置
训练与推理脚本均使用 `torch.distributed`（NCCL）初始化，推荐使用 `torchrun` 启动。

示例：

```bash
torchrun --nproc_per_node=4 train/pretrain.py ...
```

---

## 3. 构建训练数据

你提出的流程是正确的，下面给出完整落地说明。

### 3.1 输入数据组织

将**一组有重叠区域、且已完成配准**的正射影像（`.tif/.tiff`）放在同一文件夹中，例如：

```text
datasets/
└── scene_A_raw/
    ├── img_001.tif
    ├── img_002.tif
    ├── img_003.tif
    └── ...
```

> 脚本会扫描该目录下所有 `.tif/.tiff` 文件并按文件名字典序处理。

### 3.2 执行数据生成脚本

```bash
python preprocess/generate_train_dataset.py \
  --input_dir datasets/scene_A_raw \
  --output_h5 datasets/train_data.h5 \
  --window_size_m 1500 \
  --output_size_px 3000 \
  --min_cover 2 \
  --device cuda:0
```

### 3.3 输出数据结构（HDF5）

脚本会向目标 HDF5 文件追加 group（`0/1/2/...`），每个 group 包含：

- `images/{k}`：`uint8` 灰度图块（窗口裁切后重采样）
- `parallax/{k}`：`float32` 视差图
- group attributes（关键元信息）
  - `processing_crs`
  - `window_bounds`
  - `window_size_m`
  - `output_size_px`
  - `resolution_m_per_px`
  - `global_indices`
  - `source_paths`
  - `complete`

### 3.4 关键参数说明（`generate_train_dataset.py`）

#### 几何与采样参数

- `--window_size_m`：地面窗口边长（米）
- `--output_size_px`：输出像素尺寸（正方形）
- `--grid_step_m`：窗口滑动步长（米），默认等于 `window_size_m`
- `--min_cover`：一个窗口至少被多少景影像覆盖（建议 `>=2`）
- `--coverage_threshold`：单影像对窗口覆盖比例阈值

#### 影像读取与坐标参数

- `--band`：读取波段索引（从 1 开始）
- `--processing_crs`：可手动指定处理 CRS（如 `EPSG:32650`）
- `--mask_decimation`：足迹提取时掩膜降采样倍率
- `--footprint_simplify_m`：足迹简化阈值

#### CasP 相关参数（用于匹配/视差）

- `--casp_config`：CasP 配置文件路径
- `--casp_weights`：CasP 权重路径
- `--casp_window_size`：CasP 匹配窗口尺寸
- `--casp_threshold`：CasP 匹配阈值

#### 运行与质量控制参数

- `--device`：计算设备，如 `cuda:0` / `cpu`
- `--allow_cpu_fallback`：CUDA 不可用时是否自动回退 CPU
- `--min_match_points_warning`：匹配点数低于该值时警告
- `--norm_percentile_low / --norm_percentile_high`：归一化分位数
- `--min_valid_ratio`：重投影后有效像素比例下限
- `--max_groups`：最多追加多少新 group

#### HDF5 压缩参数

- `--h5_compression`：`none/gzip/lzf`
- `--h5_compression_opts`：压缩等级（如 gzip level）

---

## 4. 训练

训练分为两阶段：**预训练** 与 **正式训练**。

> 两个脚本都会从 `--dataset_path/train_data.h5` 读取数据。

### 4.1 训练数据目录约定

假设你生成了 `datasets/train_data.h5`，则训练时传入：

```bash
--dataset_path ./datasets
```

### 4.2 预训练（`train/pretrain.py`）

#### 启动示例

```bash
torchrun --nproc_per_node=4 train/pretrain.py \
  --dataset_path ./datasets \
  --dataset_num 2000 \
  --dino_weight_path ./weights \
  --batch_size 8 \
  --max_epoch 300 \
  --lr_encoder_max 1e-3 \
  --model_save_path ./weights/pretrain_run
```

#### 作用

预训练阶段主要优化 encoder adapter 与 context decoder，为后续正式训练提供更稳定初始化。

### 4.3 正式训练（`train/train.py`）

#### 启动示例

```bash
torchrun --nproc_per_node=4 train/train.py \
  --dataset_path ./datasets \
  --dataset_num 2000 \
  --dino_weight_path ./weights \
  --adapter_path ./weights/pretrain_run/adapter.pth \
  --decoder_path ./weights/pretrain_run/decoder.pth \
  --batch_size 8 \
  --predictor_max_iter 10 \
  --max_epoch 500 \
  --lr_encoder_max 1e-3 \
  --lr_predictor_max 1e-3 \
  --model_save_path ./weights/train_run
```

#### 作用

正式训练在预训练基础上，联合优化编码、预测与仿射求解相关模块，面向最终平差性能。

### 4.4 训练脚本参数（重点）

#### 数据选择

- `--dataset_path`：数据集目录（内部应有 `train_data.h5`）
- `--dataset_num`：随机选取多少个 group 参与训练
- `--dataset_select`：手动指定 group id（逗号分隔）

#### 权重加载与保存

- `--dino_weight_path`：DINO 权重目录/路径
- `--adapter_path` / `--predictor_path` / `--decoder_path`：各模块预加载权重
- `--model_save_path`：最终模型保存目录
- `--checkpoints_path`：中间 checkpoint 保存目录
- `--resume_training`：是否断点续训

#### 训练超参数

- `--batch_size`
- `--max_epoch`
- `--parallax_border_left` / `--parallax_border_right`
- `--predictor_max_iter`（仅正式训练）
- 学习率：
  - 预训练：`--lr_encoder_min/max`
  - 正式训练：`--lr_encoder_min/max` + `--lr_predictor_min/max`

#### 网络开关（正式训练）

- `--use_adapter`
- `--use_conf`
- `--use_mtf`

#### 运行/日志

- `--log_prefix`
- `--local_rank`（由 `torchrun` 注入）

---

## 5. EPBA 平差（无参考影像）

对应脚本：`infer/main.py`。

### 5.1 待平差数据组织格式

`--root` 目录下至少包含 `adjust_images/`，每个子目录代表一景影像：

```text
infer_data_epba/
├── adjust_images/
│   ├── img_000/
│   │   ├── image.png
│   │   ├── dem.npy            # 或 dem_usgs.npy（开启 usgs_dem 时）
│   │   ├── rpc.txt
│   │   └── tie_points.txt     # 可选
│   ├── img_001/
│   │   ├── image.png
│   │   ├── dem.npy
│   │   └── rpc.txt
│   └── ...
```

文件说明：

- `image.png`：正射影像（脚本内部会转灰度后复制为 3 通道）
- `dem.npy`：DEM（若 `--usgs_dem=True` 则优先 `dem_usgs.npy`）
- `rpc.txt`：RPC 参数
- `tie_points.txt`：可选，用于误差统计（像素点对）

### 5.2 推理调用示例

```bash
torchrun --nproc_per_node=4 infer/main.py \
  --root ./infer_data_epba \
  --select_imgs -1 \
  --dino_path ./weights \
  --adapter_path ./weights/train_run/adapter.pth \
  --predictor_path ./weights/train_run/predictor.pth \
  --model_config_path ./configs/model_config.yaml \
  --output_path ./results \
  --experiment_id epba_case_01 \
  --output_rpc True
```

### 5.3 主要参数说明（EPBA）

#### 数据与模型

- `--root`：数据根目录
- `--select_imgs`：选择参与平差的影像索引，`-1` 表示全部
- `--dino_path` / `--adapter_path` / `--predictor_path`
- `--model_config_path`

#### 窗口与匹配

- `--max_window_size` / `--min_window_size`
- `--max_window_num`
- `--min_cover_area_ratio`
- `--quad_split_times`
- `--predictor_iter_num`（默认从配置读取）
- `--match`（自定义匹配策略参数）
- `--mutual`：是否双向估计 pair

#### 全局求解

- `--sample_points_num`：全局仿射求解采样点数
- `--fixed_id`：固定某景为参考（避免漂移）
- `--solver_max_iter`：全局求解迭代上限

#### 输出

- `--output_path`
- `--experiment_id`
- `--output_rpc`：是否输出融合后 RPC 文件
- `--vis` / `--vis_resolution`
- `--results_csv`

---

## 6. EPA 平差（基于参考影像）

对应脚本：`infer/main_ref.py`。

### 6.1 待平差数据组织格式

`--root` 下需同时包含 `adjust_images/` 与 `ref_images/`：

```text
infer_data_epa/
├── adjust_images/
│   ├── img_000/
│   │   ├── image.png
│   │   ├── dem.npy
│   │   └── rpc.txt
│   └── ...
└── ref_images/
    ├── ref_000/
    │   ├── image.png
    │   ├── dem.npy
    │   └── rpc.txt
    └── ...
```

每个子目录文件要求与 EPBA 相同（`image.png + DEM + rpc.txt`，可选 tie points）。

### 6.2 推理调用示例

```bash
torchrun --nproc_per_node=4 infer/main_ref.py \
  --root ./infer_data_epa \
  --select_adjust_imgs -1 \
  --select_ref_imgs -1 \
  --dino_path ./weights \
  --adapter_path ./weights/train_run/adapter.pth \
  --predictor_path ./weights/train_run/predictor.pth \
  --model_config_path ./configs/model_config.yaml \
  --output_path ./results \
  --experiment_id epa_case_01 \
  --output_rpc True
```

### 6.3 主要参数说明（EPA）

- `--select_adjust_imgs`：选择待平差影像（索引）
- `--select_ref_imgs`：选择参考影像（索引）
- `--max_window_size / --min_window_size / --max_window_num`
- `--min_cover_area_ratio / --quad_split_times`
- `--predictor_iter_num`
- `--match`
- `--output_path / --experiment_id / --results_csv / --output_rpc`
- `--usgs_dem`：切换 DEM 文件名到 `dem_usgs.npy`

EPA 会为每景待平差影像搜索重叠参考影像，估计多个仿射后融合更新 RPC。

---

## 7. 推荐执行顺序（端到端）

1. 准备一组已配准、重叠正射影像（tif）。
2. 运行 `generate_train_dataset.py` 生成 `train_data.h5`。
3. 运行 `train/pretrain.py` 预训练，得到 adapter/decoder 初始权重。
4. 运行 `train/train.py` 正式训练，得到最终 adapter/predictor（及 decoder）权重。
5. 无参考场景：运行 `infer/main.py`（EPBA）。
6. 有参考场景：运行 `infer/main_ref.py`（EPA）。

---

## 8. 常见问题与排查

### 8.1 `train_data.h5` 找不到
请确认训练时 `--dataset_path` 指向包含 `train_data.h5` 的目录，而不是文件本身。

### 8.2 推理时报 `DEM not found`
检查每景目录是否存在：

- 默认 `dem.npy`
- 若 `--usgs_dem=True`，则需要 `dem_usgs.npy`

### 8.3 分布式启动失败
请确认：

- 使用 `torchrun`
- CUDA 与 NCCL 可用
- 多卡机器上 `--nproc_per_node` 不超过可见 GPU 数量

### 8.4 输出 RPC 文件为空或异常
确认：

- `--output_rpc=True`
- 每景 `rpc.txt` 格式正确且可被 `RPCModelParameterTorch` 读取

---

## 9. 目录速览（核心）

```text
preprocess/
  generate_train_dataset.py   # 训练样本构建
train/
  pretrain.py                 # 预训练
  train.py                    # 正式训练
infer/
  main.py                     # EPBA（无参考）
  main_ref.py                 # EPA（有参考）
  rs_image.py                 # 输入影像目录文件规范（image/dem/rpc）
configs/
  model_config.yaml           # 模型结构参数
```

---

如果你愿意，我下一步可以继续为你补一版“**可直接运行的最小配置模板**”（按单机 1 卡 / 4 卡分别给出完整命令），以及“**参数调优建议表**”（不同分辨率、重叠率、影像数量场景下的推荐参数）。
