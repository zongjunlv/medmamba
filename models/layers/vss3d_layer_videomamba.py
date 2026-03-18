import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from timm.models.layers import DropPath

from .ss3d_videomamba import SS3DVideoMamba


def channel_shuffle_3d(x: Tensor, groups: int) -> Tensor:
    batch_size, depth, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, depth, height, width, groups, channels_per_group)
    x = torch.transpose(x, 4, 5).contiguous()
    x = x.view(batch_size, depth, height, width, -1)
    return x


def _resolve_groups(channels: int, target_groups: int) -> int:
    if target_groups <= 1:
        return 1
    if channels % target_groups == 0:
        return target_groups
    for g in range(min(channels, target_groups), 1, -1):
        if channels % g == 0:
            return g
    return 1


class SS3D_Conv_SSM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        scan_type: str = "Axis-parallel",
        group_type: str = "Cube",
        k_group: int = None,
        use_ssm: bool = True,
        use_group_conv: bool = False,
        group_conv_groups: int = 4,
        use_shuffle: bool = True,
        use_gate: bool = True,
        fusion_mode: str = "sum",
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_ssm = use_ssm
        self.use_shuffle = use_shuffle

        self.ln_1 = norm_layer(hidden_dim // 2)

        self.self_attention = SS3DVideoMamba(
            d_model=hidden_dim // 2,
            dropout=attn_drop_rate,
            d_state=d_state,
            scan_type=scan_type,
            group_type=group_type,
            k_group=k_group,
            use_gate=use_gate,
            fusion_mode=fusion_mode,
            **kwargs,
        )

        self.drop_path = DropPath(drop_path)
        conv_groups = _resolve_groups(hidden_dim // 2, group_conv_groups) if use_group_conv else 1

        self.conv3d_path = nn.Sequential(
            nn.BatchNorm3d(hidden_dim // 2),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=conv_groups,
            ),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=conv_groups,
            ),
            nn.BatchNorm3d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=hidden_dim // 2,
                out_channels=hidden_dim // 2,
                kernel_size=1,
                stride=1,
            ),
            nn.ReLU(),
        )

    def forward(self, input: torch.Tensor):
        input_up, input_down = input.chunk(2, dim=-1)

        input_up = input_up.permute(0, 4, 1, 2, 3).contiguous()
        conv_out = self.conv3d_path(input_up)
        conv_out = conv_out.permute(0, 2, 3, 4, 1).contiguous()

        if self.use_ssm:
            ssm_out = self.drop_path(self.self_attention(self.ln_1(input_down)))
        else:
            ssm_out = torch.zeros_like(input_down)

        output = torch.cat((conv_out, ssm_out), dim=-1)
        if self.use_shuffle:
            output = channel_shuffle_3d(output, groups=2)
        return output + input


class VSS3DLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint: bool = False,
        d_state: int = 16,
        scan_type: str = "Axis-parallel",
        group_type: str = "Cube",
        k_group: int = None,
        use_ssm: bool = True,
        use_group_conv: bool = False,
        group_conv_groups: int = 4,
        use_shuffle: bool = True,
        use_gate: bool = True,
        fusion_mode: str = "sum",
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SS3D_Conv_SSM(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    attn_drop_rate=attn_drop,
                    d_state=d_state,
                    scan_type=scan_type,
                    group_type=group_type,
                    k_group=k_group,
                    use_ssm=use_ssm,
                    use_group_conv=use_group_conv,
                    group_conv_groups=group_conv_groups,
                    use_shuffle=use_shuffle,
                    use_gate=use_gate,
                    fusion_mode=fusion_mode,
                )
                for i in range(depth)
            ]
        )

        self.apply(self._init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def _init_weights(self, module: nn.Module):
        for name, p in module.named_parameters():
            if name in ["out_proj.weight"]:
                p = p.clone().detach_()
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x
