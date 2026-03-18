import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from monai.networks.nets import DenseNet121, resnet18
from monai.networks.blocks.cablock import FeedForward


from .layers import (
    PatchEmbed3D, PatchMerging3D, VSS3DLayer,
    SS3D
)


class VSSM3D(nn.Module):
    def __init__(self, 
                 patch_size=4,
                 in_chans=1, 
                 num_classes=3, 
                 depths=[2, 2, 4, 2],
                 dims=[96, 192, 384, 768], 
                 d_state=16, # d_state 越大，序列在“选择性扫描”(selective scan) 时保留的记忆越长、能表达的动态越丰富。
                 drop_rate=0., 
                 attn_drop_rate=0.,    # VSS3DLayer/SS3D 内部 selective scan 的“注意力/状态更新 dropout”
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False, # use_checkpoint用来在前向时保存更少的中间激活、以反向重算换取显存省
                 scan_directions=6,
                 **kwargs   # **kwargs 让模块在未来需要额外配置时无需改函数签名。
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]        # 告诉 PatchEmbed3D 该把输入映射到多少通道，也作为后续层的起始通道
        self.num_features = dims[-1]    # 特征维度是 dims 列表的最后一个值，直接决定分类头的输入大小。
        self.dims = dims
        self.scan_directions = scan_directions

        
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        # 有些场景（例如你想自己在后续层统一做 BN/LN，或担心额外归一化影响分布）会选择关闭 patch-level 的 norm，
        # 所以这里加判断来保持灵活。默认开启能让小批量/初始训练更稳定，不需要时再关掉即可。
        
        # 正则化
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 生成一个长度等于所有 block 总数的线性序列，从 0 均匀递增到 drop_path_rate
        # 后面每一层在构造 VSS3DLayer 里会取这段区间对应的片段，赋给各个 block 的 DropPath
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSS3DLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                # drop_path从dpr里取，越往后的层数越大
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # 持续下采样至最后一层
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                attn_drop=attn_drop_rate, 
                scan_directions=scan_directions,
            )
            self.layers.append(layer)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool3d(1)  # 3D global average pooling
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        self.apply(self._init_weights)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # 3D convolutions
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _init_weights(self, m: nn.Module):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_backbone(self, x):
        # Patch embedding: (B, C, D, H, W) -> (B, D', H', W', embed_dim)
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            
        return x

    def forward(self, x):
        # Get backbone features
        x = self.forward_backbone(x)  # (B, D', H', W', C)
        
        # Convert to (B, C, D', H', W') for pooling
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        
        # Global average pooling: (B, C, D', H', W') -> (B, C, 1, 1, 1)
        x = self.avgpool(x)
        
        # Flatten: (B, C, 1, 1, 1) -> (B, C)
        x = torch.flatten(x, start_dim=1)
        
        # Classification head
        x = self.head(x)
        
        return x


# Factory functions for different model sizes
def create_medmamba3d_tiny(num_classes=3, **kwargs):
    """Create MedMamba3D-Tiny model"""
    model = VSSM3D(
        depths=[2, 2, 4, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba3d_small(num_classes=3, **kwargs):
    """Create MedMamba3D-Small model"""
    model = VSSM3D(
        depths=[2, 2, 8, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        **kwargs
    )
    return model


def create_medmamba3d_base(num_classes=3):
    """Create MedMamba3D-Base model"""
    model = VSSM3D(
        depths=[2, 2, 12, 2],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes
    )
    return model


def create_medmamba3d_large(num_classes=3, **kwargs):
    """Create MedMamba3D-Large model"""
    model = VSSM3D(
        depths=[2, 2, 16, 2],
        dims=[192, 384, 768, 1536],
        num_classes=num_classes,
        **kwargs
    )
    return model


