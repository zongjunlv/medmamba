import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from timm.models.layers import DropPath
from einops import rearrange

from .ss3d import SS3D


def channel_shuffle_3d(x: Tensor, groups: int) -> Tensor:
    batch_size, depth, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape: [B, D, H, W, C] -> [B, D, H, W, groups, channels_per_group]
    x = x.view(batch_size, depth, height, width, groups, channels_per_group)

    # 调换 groups 和 channels_per_group 这两个维度，
    # 把“第 i 组里的第 j 个通道”换成“第 j 组里的第 i 个通道”
    # 让前面“Conv 路径”和“SSM 路径”输出的通道在残差相加前重新穿插，有助于不同路径之间的信息交换
    x = torch.transpose(x, 4, 5).contiguous()

    # flatten back to original shape
    x = x.view(batch_size, depth, height, width, -1)

    return x


class SS3D_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        scan_directions: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 均分两个分支
        self.ln_1 = norm_layer(hidden_dim // 2)
        
        
        self.self_attention = SS3D(
            d_model=hidden_dim // 2, 
            dropout=attn_drop_rate, 
            d_state=d_state,
            scan_directions=scan_directions,
            **kwargs
        )
        
        
        self.drop_path = DropPath(drop_path)

        # 3D Convolution path for local features
        self.conv3d_path = nn.Sequential(
            nn.BatchNorm3d(hidden_dim // 2),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=1,
                stride=1
            ),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor):
        # Split input into two parts for parallel processing
        input_up, input_down = input.chunk(2, dim=-1)
        
        
        # (B, D, H, W, C) -> (B, C, D, H, W) 
        input_up = input_up.permute(0, 4, 1, 2, 3).contiguous()
        conv_out = self.conv3d_path(input_up)
        # back to (B, D, H, W, C)
        conv_out = conv_out.permute(0, 2, 3, 4, 1).contiguous()

        ssm_out = self.drop_path(self.self_attention(self.ln_1(input_down)))
        
       
        output = torch.cat((conv_out, ssm_out), dim=-1)
        
        # Apply 3D channel shuffle for better feature interaction
        output = channel_shuffle_3d(output, groups=2)
        
        # Residual connection
        return output + input


class VSS3DLayer(nn.Module):

    def __init__(
        self, 
        dim: int,
        depth: int,
        attn_drop: float = 0.,
        drop_path: float = 0., 
        norm_layer = nn.LayerNorm, 
        downsample = None, 
        use_checkpoint: bool = False, 
        d_state: int = 16,
        scan_directions: int = 6,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        
        self.blocks = nn.ModuleList([
            SS3D_Conv_SSM(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                scan_directions=scan_directions,
            )
            for i in range(depth)
        ])
        
        # 遍历整个 VSS3DLayer 及其子模块的参数
        self.apply(self._init_weights)

        if downsample is not None:
            # 如果不是最后一层，进行PatchMerging3D（dim, norm_layer）
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def _init_weights(self, module: nn.Module):
        for name, p in module.named_parameters():
            if name in ["out_proj.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x):
        # Process through all blocks
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        # Optional downsampling
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
            
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # 3D Convolution for patch projection
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size           # 将图像分辨率缩小patch_size
        )
        
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # (B, C, D, H, W) -> (B, D', H', W', embed_dim)
        x = self.proj(x).permute(0, 2, 3, 4, 1)
        
        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
            
        return x
# 3D Patch operations for downsampling/upsampling in 3D

class PatchMerging3D(nn.Module):

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # 8 patches in 3D
        self.norm = norm_layer(8 * dim)

    def forward(self, x):

        B, D, H, W, C = x.shape

        # Handle odd dimensions
        pad_input = (D % 2 == 1) or (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))

        # Split into 8 patches (2x2x2)
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        # Concatenate all patches
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, D/2, H/2, W/2, 8*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, D/2, H/2, W/2, 2*C)

        return x


class PatchExpand3D(nn.Module):

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, (dim_scale ** 3) * self.dim, bias=False)
        self.norm = norm_layer(self.dim // (dim_scale ** 3))

    def forward(self, x):

        B, D, H, W, C = x.shape
        x = self.expand(x)
        
        # Reshape to expand spatial dimensions
        x = rearrange(
            x, 
            'b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c', 
            p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale,
            c=C // (self.dim_scale ** 3)
        )
        x = self.norm(x)

        return x
