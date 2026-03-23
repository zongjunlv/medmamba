# MedMamba

一个面向 3D 医学体数据分类的 MedMamba 实验仓库，当前主要用于 ABUS 数据的二分类研究。仓库包含训练、测试、数据缓存和 Grad-CAM 可解释性分析流程，并已经改成适合公开发布的参数化用法。

## 项目目标

当前版本主要用于：

- 3D ABUS 体数据分类
- 基于 `npy` 缓存的高效数据加载
- MedMamba3D 与 VideoMamba 变体实验
- 训练过程评估与最优权重保存
- Grad-CAM 可解释性分析

## 当前项目结构

```text
MedMamba/
├── configs/                     # 配置文件
├── data/                        # 数据集读取
├── models/                      # 3D 模型与层实现
├── trainer/                     # 训练循环
├── utils/                       # 评估、缓存预处理等工具
├── grad_cam/                    # 可解释性可视化
├── assets/                      # 日志、可视化、输出资源
├── train.py                     # 训练入口
├── test.py                      # 测试入口
└── README.md
```

## 环境依赖

建议使用 Linux + CUDA 环境。

安装基础依赖：

```bash
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging timm==0.4.12 pytest chardet yacs termcolor submitit tensorboardX
pip install triton==2.0.0 scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy monai==1.3.0 nibabel==5.1.0
```

如果使用 VideoMamba / SS3D 相关变体，还需要补充：

```bash
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

## 数据组织

缓存脚本假定数据根目录形如：

```text
your_dataset_root/
├── labels_train.csv
├── labels_val.csv
├── labels_test.csv
├── train/
│   └── imagesTr_origin/
├── val/
│   └── imagesVal_origin/
└── test/
    └── imagesTs_origin/
```

其中 `labels_*.csv` 至少需要包含这些字段：

- `case_id`
- `data_path`
- `label`

标签当前在缓存阶段会被映射为：

- `B -> 0`
- `M -> 1`

## 数据预缓存

训练和验证推荐使用缓存后的 `npy` 文件，而不是每轮直接读取原始 DICOM/医学影像。

运行：

```bash
python utils/pre_cache_binary_tdsc.py --root /path/to/TDSC
```

该脚本会：

- 读取 `labels_train.csv / labels_val.csv / labels_test.csv`
- 加载原始医学体数据
- 做强度归一化与尺寸统一
- 保存为 `npy`
- 生成新的缓存索引文件：
  - `labels_train_cache.csv`
  - `labels_val_cache.csv`
  - `labels_test_cache.csv`

当前默认 resize 到：

```text
(256, 256, 128)
```

## 训练

当前 [train.py](/Users/jungle/Documents/MedMamba/medmamba/train.py) 支持单卡与 `torchrun` 单机多卡训练，并通过命令行参数传入数据路径与训练参数。

最小训练示例：

```bash
python train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --model medmamba3d_tiny \
  --disable-swanlab
```

单卡完整示例：

```bash
python train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --batch-size 4 \
  --epochs 200 \
  --lr 1e-5 \
  --model medmamba3d_small \
  --num-classes 2 \
  --disable-swanlab
```

单机双卡示例：

```bash
torchrun --nproc_per_node=2 train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --batch-size 4 \
  --epochs 200 \
  --lr 1e-5 \
  --model medmamba3d_small \
  --num-classes 2 \
  --find-unused-parameters \
  --disable-swanlab
```

多 seed 批量训练示例：

```bash
python train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --model medmamba3d_tiny \
  --seeds 3407 42 1234 \
  --disable-swanlab
```

训练脚本当前行为包括：

- 固定随机种子
- 使用 `AdamW`
- warmup + cosine 学习率调度
- 每轮验证
- 按 `MCC` 保存最优模型
- 记录训练过程中出现的 `AUC` 上限
- 输出 `sensitivity / specificity` 以及两者差值，辅助判断是否偏科
- 早停控制
- 本地持续保存 `csv` 日志与训练曲线图
- 可选 `swanlab` 记录指标

如果不传 `--save-path`，默认保存为：

```text
assets/checkpoints_3d/best_model_<model>.pth
```

如果使用 `--seeds` 进行多 seed 训练，则会自动保存为：

```text
assets/checkpoints_3d/best_model_<model>_seed<seed>.pth
```

并额外导出：

```text
assets/checkpoints_3d/best_model_<model>_multiseed_summary.csv
```

如果不传 `--log-dir`，训练日志默认保存为：

```text
assets/training_logs/train_log_<model>.csv
assets/training_logs/train_curves_<model>.png
```

## 测试

当前 [test.py](/Users/jungle/Documents/MedMamba/medmamba/test.py) 通过命令行选择测试集、模型类型和 checkpoint。

最小测试示例：

```bash
python test.py \
  --test-csv /path/to/TDSC/labels_test_cache.csv \
  --model medmamba3d_tiny \
  --disable-swanlab
```

完整示例：

```bash
python test.py \
  --test-csv /path/to/TDSC/labels_test_cache.csv \
  --checkpoint assets/checkpoints_3d/best_model_medmamba3d_small.pth \
  --model medmamba3d_small \
  --num-classes 2 \
  --batch-size 4 \
  --num-workers 8 \
  --disable-swanlab
```

支持的测试模型：

- `medmamba3d_tiny`
- `medmamba3d_small`
- `medmamba3d_base`
- `medmamba3d_large`
- `resnet34`
- `r2plus1d`
- `r3d`
- `unet_encoder`
- `swin3d`

如果不传 `--checkpoint`，测试脚本会自动按下面的规则查找：

```text
assets/checkpoints_3d/best_model_<model>.pth
```

## 模型实现

当前仓库里和主实验最相关的模型文件有：

- [models/medmamba3d.py](/Users/jungle/Documents/MedMamba/medmamba/models/medmamba3d.py)
- [models/medmamba3d_videomamba.py](/Users/jungle/Documents/MedMamba/medmamba/models/medmamba3d_videomamba.py)
- [models/layers/ss3d.py](/Users/jungle/Documents/MedMamba/medmamba/models/layers/ss3d.py)
- [models/layers/ss3d_videomamba.py](/Users/jungle/Documents/MedMamba/medmamba/models/layers/ss3d_videomamba.py)
- [models/layers/vss3d_layer.py](/Users/jungle/Documents/MedMamba/medmamba/models/layers/vss3d_layer.py)
- [models/layers/vss3d_layer_videomamba.py](/Users/jungle/Documents/MedMamba/medmamba/models/layers/vss3d_layer_videomamba.py)

## Grad-CAM

`grad_cam/` 目录用于做模型可解释性分析，帮助观察模型在 2D/3D 图像中关注的区域。

当前仓库中比较关键的文件：

- [grad_cam/abus3d_cam.py](/Users/jungle/Documents/MedMamba/medmamba/grad_cam/abus3d_cam.py)
- [grad_cam/abus3d_cam_eval.py](/Users/jungle/Documents/MedMamba/medmamba/grad_cam/abus3d_cam_eval.py)

## 注意事项

- `dataset/` 不纳入 Git 仓库
- 训练和测试都需要显式传入 csv 或 checkpoint 路径
- `test.py` 的模型结构必须和 checkpoint 对应
- 推荐优先查看本地 `csv/png` 日志；如无联网环境可直接使用 `--disable-swanlab`

## 建议的后续整理

- 统一训练与测试的模型选择逻辑
- 把数据类别数从代码中抽离
- 为 `README` 补充一份真实的 TDSC csv 示例
- 把更多实验配置沉淀到 `configs/`
