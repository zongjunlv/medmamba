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
pip install -r requirements.txt
```

如果缺少 3D 医学图像相关依赖，可补充安装：

```bash
pip install monai SimpleITK scikit-image medpy scikit-learn pandas tqdm swanlab
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

当前 [train.py](/data02/workspace/LZJ_SPACE/MedMamba/train.py) 默认使用 `models.medmamba3d_videomamba.create_medmamba3d_tiny`，并通过命令行参数传入数据路径与训练参数。

最小训练示例：

```bash
python train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --model medmamba3d_tiny
```

完整示例：

```bash
python train.py \
  --train-csv /path/to/TDSC/labels_train_cache.csv \
  --val-csv /path/to/TDSC/labels_val_cache.csv \
  --batch-size 4 \
  --epochs 200 \
  --lr 1e-5 \
  --model medmamba3d_small \
  --num-classes 2
```

训练脚本当前行为包括：

- 固定随机种子
- 使用 `AdamW`
- warmup + cosine 学习率调度
- 每轮验证
- 按 AUC 保存最优模型
- 早停控制
- 可选 `swanlab` 记录指标

如果不传 `--save-path`，默认保存为：

```text
assets/checkpoints_3d/best_model_<model>.pth
```

## 测试

当前 [test.py](/data02/workspace/LZJ_SPACE/MedMamba/test.py) 通过命令行选择测试集、模型类型和 checkpoint。

最小测试示例：

```bash
python test.py \
  --test-csv /path/to/TDSC/labels_test_cache.csv \
  --model medmamba3d_tiny
```

完整示例：

```bash
python test.py \
  --test-csv /path/to/TDSC/labels_test_cache.csv \
  --checkpoint assets/checkpoints_3d/best_model_resnet34.pth \
  --model resnet34 \
  --num-classes 2 \
  --test-runs 5
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

- [models/medmamba3d.py](/data02/workspace/LZJ_SPACE/MedMamba/models/medmamba3d.py)
- [models/medmamba3d_videomamba.py](/data02/workspace/LZJ_SPACE/MedMamba/models/medmamba3d_videomamba.py)
- [models/layers/ss3d.py](/data02/workspace/LZJ_SPACE/MedMamba/models/layers/ss3d.py)
- [models/layers/ss3d_videomamba.py](/data02/workspace/LZJ_SPACE/MedMamba/models/layers/ss3d_videomamba.py)
- [models/layers/vss3d_layer.py](/data02/workspace/LZJ_SPACE/MedMamba/models/layers/vss3d_layer.py)
- [models/layers/vss3d_layer_videomamba.py](/data02/workspace/LZJ_SPACE/MedMamba/models/layers/vss3d_layer_videomamba.py)

## Grad-CAM

`grad_cam/` 目录用于做模型可解释性分析，帮助观察模型在 2D/3D 图像中关注的区域。

当前仓库中比较关键的文件：

- [grad_cam/abus3d_cam.py](/data02/workspace/LZJ_SPACE/MedMamba/grad_cam/abus3d_cam.py)
- [grad_cam/abus3d_cam_eval.py](/data02/workspace/LZJ_SPACE/MedMamba/grad_cam/abus3d_cam_eval.py)

## 注意事项

- `dataset/` 不纳入 Git 仓库
- 训练和测试都需要显式传入 csv 或 checkpoint 路径
- `test.py` 的模型结构必须和 checkpoint 对应
- `swanlab` 已接入，如果本机未登录，运行时可能需要先配置

## 建议的后续整理

- 统一训练与测试的模型选择逻辑
- 把数据类别数从代码中抽离
- 为 `README` 补充一份真实的 TDSC csv 示例
- 把更多实验配置沉淀到 `configs/`
