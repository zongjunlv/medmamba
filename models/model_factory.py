from . import R2Plus1DClassifier, R3DClassifier, ResNet3D, SwinClassifier3D, UNetEncoderClassifier
from .medmamba3d_videomamba import (
    create_medmamba3d_base,
    create_medmamba3d_large,
    create_medmamba3d_small,
    create_medmamba3d_tiny,
)

MODEL_CHOICES = [
    "medmamba3d_tiny",
    "medmamba3d_small",
    "medmamba3d_base",
    "medmamba3d_large",
    "resnet34",
    "r2plus1d",
    "r3d",
    "unet_encoder",
    "swin3d",
]


def build_model(model_name, num_classes):
    if model_name == "medmamba3d_tiny":
        return create_medmamba3d_tiny(num_classes=num_classes)
    if model_name == "medmamba3d_small":
        return create_medmamba3d_small(num_classes=num_classes)
    if model_name == "medmamba3d_base":
        return create_medmamba3d_base(num_classes=num_classes)
    if model_name == "medmamba3d_large":
        return create_medmamba3d_large(num_classes=num_classes)
    if model_name == "resnet34":
        return ResNet3D(variant="resnet34", num_classes=num_classes, in_chans=1, pretrained=True)
    if model_name == "r2plus1d":
        return R2Plus1DClassifier(num_classes=num_classes, in_chans=1, pretrained=True)
    if model_name == "r3d":
        return R3DClassifier(num_classes=num_classes, in_chans=1, pretrained=True)
    if model_name == "unet_encoder":
        return UNetEncoderClassifier(num_classes=num_classes, in_chans=1)
    if model_name == "swin3d":
        return SwinClassifier3D(num_classes=num_classes, in_chans=1, pretrained=False)
    raise ValueError(f"Unsupported model: {model_name}")
