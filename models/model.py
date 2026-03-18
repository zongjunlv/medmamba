import os
import torch
import torch.nn as nn
from monai.networks import nets as monai_nets
from monai.networks.nets.resnet import get_medicalnet_pretrained_resnet_args
from monai.networks.nets.swin_unetr import SwinTransformer
from monai.networks.nets.vit import ViT
from monai.networks.nets import UNet
import torchvision.models.video as tv_video

__all__ = [
    "ResNet3D",
    "DenseNet3D",
    "SwinClassifier3D",
    "ViTClassifier3D",
    "R2Plus1DClassifier",
    "R3DClassifier",
    "UNetEncoderClassifier",
]


def _get_monai_builder(name: str):
    builder = getattr(monai_nets, name, None)
    if builder is None:
        raise ValueError(f"{name} 未在 monai.networks.nets 中提供，请检查 MONAI 版本或更换可用变体。")
    return builder


class ResNet3D(nn.Module):
    """
    3D ResNet 分类封装，默认使用 MedicalNet 的预训练权重（仅在 spatial_dims=3 且 in_chans=1 时可用）。
    variant 可选：resnet18/resnet34/resnet50/resnet101/resnet152。
    """

    def __init__(self, variant: str = "resnet34", num_classes: int = 3, in_chans: int = 1, pretrained: bool = True, **kwargs):
        super().__init__()
        self.variant = variant
        supported = {"resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}
        if variant not in supported:
            raise ValueError(f"不支持的 ResNet 变体：{variant}，可选 {sorted(supported)}。")
        depth = int(variant.replace("resnet", ""))
        if pretrained:
            bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(depth)
        else:
            # 使用 MONAI 默认
            bias_downsample, shortcut_type = True, "B"

        builder = _get_monai_builder(variant)
        base_kwargs = dict(
            spatial_dims=3,
            n_input_channels=in_chans,
            num_classes=num_classes,  # 此处仅为保持接口一致，实际 feed_forward=False 时 fc 被去掉
            feed_forward=False,  # 为了兼容 MedicalNet 预训练，统一去掉原始 fc，外部自建分类头
            bias_downsample=bias_downsample,
            shortcut_type=shortcut_type,
            pretrained=pretrained,
        )
        base_kwargs.update(kwargs)
        self.model = builder(**base_kwargs)
        self.out_channels = 512 if variant in ("resnet18", "resnet34") else 2048
        self.head = nn.Linear(self.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)  # (B, C_out)
        return self.head(feats)



def _adapt_video_first_conv(model: nn.Module, in_chans: int):
    """将 torchvision.video 预训练权重的首层 3 通道卷积适配为 in_chans 通道。"""
    if in_chans == 3:
        return model
    conv1 = model.stem[0] if hasattr(model, "stem") else getattr(model, "conv1", None)
    if conv1 is None:
        return model
    new_conv = nn.Conv3d(
        in_channels=in_chans,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        if in_chans == 1:
            w = conv1.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(w)
        else:
            # 随机初始化非预训练通道，其余沿用原权重
            new_conv.weight.zero_()
            repeat = torch.ceil(torch.tensor(in_chans / 3)).int().item()
            w = conv1.weight.repeat(1, repeat, 1, 1, 1)[:, :in_chans]
            new_conv.weight.copy_(w / repeat)
        if conv1.bias is not None:
            new_conv.bias.copy_(conv1.bias)
    if hasattr(model, "stem"):
        model.stem[0] = new_conv
    else:
        model.conv1 = new_conv
    return model


class R2Plus1DClassifier(nn.Module):
    """
    torchvision R(2+1)D-18 分类封装，支持 Kinetics400 预训练。
    """

    def __init__(self, num_classes: int = 3, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = tv_video.r2plus1d_18(weights=weights)
        self.backbone = _adapt_video_first_conv(self.backbone, in_chans)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.variant = "r2plus1d_18"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class R3DClassifier(nn.Module):
    """
    torchvision R3D-18 分类封装，可作为 I3D 类基线（Kinetics400 预训练）。
    """

    def __init__(self, num_classes: int = 3, in_chans: int = 1, pretrained: bool = True):
        super().__init__()
        weights = tv_video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.backbone = tv_video.r3d_18(weights=weights)
        self.backbone = _adapt_video_first_conv(self.backbone, in_chans)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        self.variant = "r3d_18"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class UNetEncoderClassifier(nn.Module):
    """
    基于 MONAI 3D UNet encoder 的分类封装：将 UNet 输出特征全局池化后接线性层。
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        channels=(8,16,32,64,128),
        strides=(2, 2, 2, 2),
        norm="instance",
    ):
        super().__init__()
        self.backbone = UNet(
            spatial_dims=3,
            in_channels=in_chans,
            out_channels=channels[-1],
            channels=channels,
            strides=strides,
            num_res_units=2,
            norm=norm,
            
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(channels[-1], num_classes)
        self.variant = "unet_encoder"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = self.avgpool(feats).flatten(1)
        return self.head(feats)


class SwinClassifier3D(nn.Module):
    """
    基于 MONAI SwinTransformer 编码器的简单 3D 分类头：取最后一层特征做全局平均池化 + 全连接。
    预训练权重需自行加载（MONAI 不附带分类预训练）。
    """

    def __init__(
        self,
        num_classes: int = 3,
        in_chans: int = 1,
        embed_dim: int = 96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        patch_size=(4, 4, 4),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        pretrained_url: str | None = None,
    ):
        super().__init__()
        self.backbone = SwinTransformer(
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        # MONAI 的 SwinTransformer 在默认 downsample="merging" 下最后一层仍会执行 PatchMerging，
        # 因此输出通道数是 num_features 的 2 倍（例如 1536 而不是 768）。这里按最后一层是否存在 downsample 自适应确定维度。
        layer_lists = [self.backbone.layers1, self.backbone.layers2, self.backbone.layers3, self.backbone.layers4]
        last_layer = layer_lists[self.backbone.num_layers - 1][0]
        out_channels = self.backbone.num_features * (2 if getattr(last_layer, "downsample", None) is not None else 1)
        self.out_channels = out_channels
        self.norm = nn.LayerNorm(out_channels)
        self.head = nn.Linear(out_channels, num_classes)
        if pretrained:
            self._load_pretrained(pretrained_path, pretrained_url)

    def _load_pretrained(self, pretrained_path: str | None, pretrained_url: str | None) -> None:
        if pretrained_path is None and pretrained_url is None:
            raise ValueError("未提供预训练权重路径或 URL。")

        if pretrained_path is None:
            filename = os.path.basename(pretrained_url)
            pretrained_path = os.path.join("assets", "pretrained", filename)

        if not os.path.exists(pretrained_path):
            if pretrained_url is None:
                raise FileNotFoundError(f"预训练权重不存在：{pretrained_path}")
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            try:
                from monai.apps import download_url
                download_url(pretrained_url, pretrained_path)
            except Exception as exc:
                raise RuntimeError(
                    f"下载预训练权重失败，请手动下载到 {pretrained_path}"
                ) from exc

        ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        backbone_state = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            if k.startswith("swinViT."):
                k = k[len("swinViT."):]
            # SSL 权重里层级形如 layers1.0.0.blocks.*，需去掉多余的 ".0."
            if k.startswith("layers"):
                k = k.replace(".0.0.", ".0.")
            if k in backbone_state and backbone_state[k].shape == v.shape:
                filtered[k] = v

        incompatible = self.backbone.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(backbone_state)
        print(f"[SwinClassifier3D] 预训练权重加载：{loaded}/{total} 个参数匹配。")
        if loaded == 0:
            print("[SwinClassifier3D] 警告：未匹配到任何权重，请检查配置是否一致。")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[-1]  # 取最高层输出
        features = self.avgpool(features).flatten(1)
        features = self.norm(features)
        return self.head(features)


class ViTClassifier3D(nn.Module):
    """
    MONAI ViT 的 3D 分类封装。img_size/patch_size 需与输入体素大小匹配。
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        patch_size=(16, 16, 16),
        num_classes: int = 3,
        in_chans: int = 1,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        pretrained: bool = False,
        pretrained_path: str | None = None,
        pretrained_url: str | None = None,
    ):
        super().__init__()
        self.backbone = ViT(
            in_channels=in_chans,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            spatial_dims=3,
            qkv_bias=qkv_bias,
        )
        if pretrained:
            self._load_pretrained(pretrained_path, pretrained_url)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.backbone(x)
        return logits

    def _load_pretrained(self, pretrained_path: str | None, pretrained_url: str | None) -> None:
        if pretrained_path is None and pretrained_url is None:
            raise ValueError("未提供预训练权重路径或 URL。")

        if pretrained_path is None:
            filename = os.path.basename(pretrained_url)
            pretrained_path = os.path.join("assets", "pretrained", filename)

        if not os.path.exists(pretrained_path):
            if pretrained_url is None:
                raise FileNotFoundError(f"预训练权重不存在：{pretrained_path}")
            os.makedirs(os.path.dirname(pretrained_path), exist_ok=True)
            try:
                from monai.apps import download_url
                download_url(pretrained_url, pretrained_path)
            except Exception as exc:
                raise RuntimeError(
                    f"下载预训练权重失败，请手动下载到 {pretrained_path}"
                ) from exc

        ckpt = torch.load(pretrained_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            state = ckpt

        backbone_state = self.backbone.state_dict()
        filtered = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
            if k.startswith("encoder."):
                k = k[len("encoder."):]
            if k in backbone_state and backbone_state[k].shape == v.shape:
                filtered[k] = v

        self.backbone.load_state_dict(filtered, strict=False)
        loaded = len(filtered)
        total = len(backbone_state)
        print(f"[ViTClassifier3D] 预训练权重加载：{loaded}/{total} 个参数匹配。")
        if loaded == 0:
            print("[ViTClassifier3D] 警告：未匹配到任何权重，请检查配置是否一致。")
