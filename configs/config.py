"""
Configuration file for MedMamba training
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data configuration"""
    train_root: str = "the path of your train set"  # 训练集根目录
    val_root: str = "the path of your validation set"  # 验证集根目录
    batch_size: int = 32  # 每次迭代送入模型的样本数
    num_workers: int = 0  # DataLoader 子进程数，0 表示主进程加载
    img_size: Tuple[int, int] = (224, 224)  # 输入图像尺寸 (H, W)
    
    # Medical image processing options
    use_medical_transforms: bool = False  # 是否启用更适合医学图像的变换流程
    augmentation_level: str = "light"  # 数据增强强度，可选 "light"、"medium"、"heavy"
    
    
    # Data transforms parameters
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # 归一化均值
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # 归一化标准差


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "medmamba"  # 模型名称或模型类型标识
    num_classes: int = 6  # 分类类别数
    
    # MedMamba specific parameters
    patch_size: int = 4  # patch 划分大小，越大下采样越激进
    in_chans: int = 3  # 输入通道数，灰度图通常为 1
    depths: List[int] = field(default_factory=lambda: [2, 2, 4, 2])  # 每个 stage 的 block 数
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 768])  # 每个 stage 的特征维度
    d_state: int = 16  # Mamba/SSM 的状态维度，影响序列表达能力
    drop_rate: float = 0.0  # 普通 dropout 比例
    attn_drop_rate: float = 0.0  # selective scan/注意力相关 dropout 比例
    drop_path_rate: float = 0.1  # 随机深度 drop path 比例
    use_checkpoint: bool = False  # 是否启用梯度检查点以节省显存


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100  # 总训练轮数
    learning_rate: float = 0.0001  # 初始学习率
    weight_decay: float = 0.0  # 权重衰减系数，用于正则化
    optimizer: str = "adam"  # 优化器类型
    
    # Checkpointing
    save_dir: str = "./checkpoints"  # checkpoint 保存目录
    save_best_only: bool = True  # 是否只保存最佳模型
    
    # Early stopping
    use_early_stopping: bool = True  # 是否启用早停
    early_stopping_patience: int = 15  # 连续多少个 epoch 无提升后停止
    early_stopping_metric: str = "f1"  # 早停监控指标，如 "accuracy" 或 "f1"
    
    # Device
    device: str = "cuda:0"  # 训练设备
    
    # Logging
    print_freq: int = 10  # 日志打印间隔
    

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        # Ensure save directory exists
        os.makedirs(self.training.save_dir, exist_ok=True)
        
        # Auto-detect number of workers if not specified
        if self.data.num_workers == 8:
            self.data.num_workers = min([
                os.cpu_count(), 
                self.data.batch_size if self.data.batch_size > 1 else 0, 
                8
            ])


def get_config() -> Config:
    """Get default configuration"""
    return Config()


def get_medmamba_tiny_config() -> Config:
    """Get MedMamba-Tiny configuration"""
    config = Config()
    config.model.depths = [2, 2, 4, 2]
    config.model.dims = [96, 192, 384, 768]
    return config


def get_medmamba_small_config() -> Config:
    """Get MedMamba-Small configuration"""
    config = Config()
    config.model.depths = [2, 2, 8, 2]
    config.model.dims = [96, 192, 384, 768]
    return config


def get_medmamba_base_config() -> Config:
    """Get MedMamba-Base configuration"""
    config = Config()
    config.model.depths = [2, 2, 12, 2]
    config.model.dims = [128, 256, 512, 1024]
    return config


def get_medical_image_config() -> Config:
    """Get optimized configuration for medical image analysis"""
    config = Config()
    
    # Optimize for medical images
    config.data.use_medical_transforms = True
    config.data.augmentation_level = "light"  # Conservative augmentation
    config.data.img_size = 224
    config.data.batch_size = 16  # Smaller batch for medical images
    
    # Medical-friendly normalization (for CT/MRI)
    config.data.normalize_mean = (0.485,)  # Single channel medical images
    config.data.normalize_std = (0.229,)
    
    # Model adjustments for medical data
    config.model.in_chans = 1  # Grayscale medical images
    config.model.drop_path_rate = 0.05  # Reduced dropout for medical data
    
    # Training adjustments
    config.training.learning_rate = 0.0001
    config.training.weight_decay = 1e-4
    
    return config


