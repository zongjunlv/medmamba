"""Model package exports.

部分历史模型依赖特定版本的 MONAI / 第三方扩展，环境不满足时不应阻塞
当前 videomamba 变体或基线模型的导入。
"""

__all__ = []

try:
    from .medmamba3d import Resnet18, VSSM3D as VSSM3D

    __all__ += ["Resnet18", "VSSM3D"]
except Exception:
    Resnet18 = None
    VSSM3D = None

try:
    from .medmamba3d_videomamba import VSSM3D as VSSM3DVideo

    __all__ += ["VSSM3DVideo"]
except Exception:
    VSSM3DVideo = None

try:
    from .model import (
        ResNet3D,
        SwinClassifier3D,
        ViTClassifier3D,
        R2Plus1DClassifier,
        R3DClassifier,
        UNetEncoderClassifier,
    )

    __all__ += [
        "ResNet3D",
        "SwinClassifier3D",
        "ViTClassifier3D",
        "R2Plus1DClassifier",
        "R3DClassifier",
        "UNetEncoderClassifier",
    ]
except Exception:
    ResNet3D = None
    SwinClassifier3D = None
    ViTClassifier3D = None
    R2Plus1DClassifier = None
    R3DClassifier = None
    UNetEncoderClassifier = None

from . import layers
