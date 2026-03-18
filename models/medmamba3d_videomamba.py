import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from monai.networks.nets import resnet18


from .layers import PatchMerging3D
from .layers.vss3d_layer_videomamba import VSS3DLayer

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.Resnet18 = resnet18(
            spatial_dims = 3,
            n_input_channels = 1,
            num_classes = 3,
            pretrained= True,
            feed_forward = False,
            shortcut_type = 'A',
            bias_downsample = True
        )
        self.fc = nn.Linear(512, 3)
    def forward(self, x):
        x = self.Resnet18(x)
        x = self.fc(x)
        return x



class VSSM3D(nn.Module):
    def __init__(self, 
                 patch_size=4,  # 预留的 patch 大小参数；当前 VideoMamba 版本主要由 3D Conv 做 token 化
                 in_chans=1,  # 输入体数据通道数，单模态医学图像通常为 1
                 num_classes=3,  # 分类类别数
                 depths=[2, 2, 4, 2],  # 每个 stage 堆叠多少个 VSS3D block
                 dims=[96, 192, 384, 768],  # 每个 stage 的特征维度
                 d_state=16, # d_state 越大，序列在“选择性扫描”(selective scan) 时保留的记忆越长、能表达的动态越丰富。
                 conv3D_channel=32,  # 初始 3D 卷积输出通道数，即 token 化前的局部特征维度
                 conv3D_kernel=(4, 4, 4),  # 3D 卷积核大小，决定局部感受野
                 conv3D_stride=(4, 4, 4),  # 3D 卷积步长，决定初始下采样倍率
                 conv3D_padding=0,  # 3D 卷积填充
                 drop_rate=0.,  # 输入 token 后的普通 dropout
                 attn_drop_rate=0.,    # VSS3DLayer/SS3D 内部 selective scan 的“注意力/状态更新 dropout”
                 drop_path_rate=0.1,  # 随机深度比例，越靠后层通常越大
                 norm_layer=nn.LayerNorm,  # 各 stage 默认归一化层
                 patch_norm=True,  # 预留参数，表示是否在 patch/token 级别做归一化
                 use_checkpoint=False, # use_checkpoint用来在前向时保存更少的中间激活、以反向重算换取显存省
                 scan_type="Axis-parallel",  # 扫描方式，如沿坐标轴并行扫描
                 group_type="Cube",  # 分组方式，决定局部 token 如何组成 group
                 k_group=None,  # group 数或每组大小的附加控制参数，None 表示走默认策略
                 use_ssm=True,  # 是否启用状态空间建模分支
                 use_group_conv=False,  # 是否在分组特征交互时使用 group convolution
                 group_conv_groups=4,  # group convolution 的分组数
                 use_shuffle=True,  # 是否在分组后做通道/token shuffle，增强跨组信息交换
                 use_gate=True,  # 是否启用门控机制控制多分支融合
                 fusion_mode="sum",  # 多分支融合方式，如求和或其他融合策略
                 tokenization_mode="patch",  # token 组织方式：patch、slice 或 patch+cross-slice
                 use_token_fusion=True,  # 混合 token 模式下是否真的执行融合
                 token_fusion_mode="gated",  # token 融合策略：concat、weighted、gated
                 **kwargs   # **kwargs 让模块在未来需要额外配置时无需改函数签名。
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dims[0]        # 告诉 PatchEmbed3D 该把输入映射到多少通道，也作为后续层的起始通道
        self.num_features = dims[-1]    # 特征维度是 dims 列表的最后一个值，直接决定分类头的输入大小。
        self.dims = dims
        self.scan_type = scan_type
        self.group_type = group_type
        self.k_group = k_group
        self.tokenization_mode = tokenization_mode
        self.use_token_fusion = use_token_fusion
        self.token_fusion_mode = token_fusion_mode
        if self.token_fusion_mode not in {"concat", "weighted", "gated"}:
            raise ValueError(f"Unsupported token_fusion_mode: {self.token_fusion_mode}")
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(
                in_chans,
                out_channels=conv3D_channel,
                kernel_size=conv3D_kernel,
                stride=conv3D_stride,
                padding=conv3D_padding,
            ),
            nn.BatchNorm3d(conv3D_channel),
            nn.ReLU(),
        )
        self.embedding_spatial_spectral = nn.Sequential(nn.Linear(conv3D_channel, self.embed_dim))
        self.token_fusion_weights = nn.Parameter(torch.zeros(2))  # weighted 融合模式下的可学习权重，softmax 后分别作用于 patch token 和 slice token
        self.token_fusion_gate = nn.Linear(self.embed_dim, 2, bias=True)
        self.token_fusion_concat = nn.Linear(self.embed_dim * 2, self.embed_dim, bias=False)
        
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

    def _fuse_tokens(self, patch_tokens: torch.Tensor, slice_tokens: torch.Tensor) -> torch.Tensor:
        if self.token_fusion_mode == "concat":
            return self.token_fusion_concat(torch.cat([patch_tokens, slice_tokens], dim=-1))
        if self.token_fusion_mode == "weighted":
            weights = torch.softmax(self.token_fusion_weights, dim=0)
            return patch_tokens * weights[0] + slice_tokens * weights[1]
        if self.token_fusion_mode == "gated":
            context = patch_tokens.mean(dim=(1, 2, 3))
            gate = torch.softmax(self.token_fusion_gate(context), dim=-1)
            return (
                patch_tokens * gate[:, 0].view(-1, 1, 1, 1, 1)
                + slice_tokens * gate[:, 1].view(-1, 1, 1, 1, 1)
            )
        raise ValueError(f"Unsupported token_fusion_mode: {self.token_fusion_mode}")

    def forward_backbone(self, x):
        x = self.conv3d_features(x)
        patch_tokens = rearrange(x, 'b c d h w -> b d h w c')
        patch_tokens = self.embedding_spatial_spectral(patch_tokens)

        if self.tokenization_mode == "patch":
            x = patch_tokens
        else:
            slice_tokens = x.mean(dim=(3, 4), keepdim=True).expand(-1, -1, -1, x.size(3), x.size(4))
            slice_tokens = rearrange(slice_tokens, 'b c d h w -> b d h w c')
            slice_tokens = self.embedding_spatial_spectral(slice_tokens)

            if self.tokenization_mode == "slice":
                x = slice_tokens
            elif self.tokenization_mode in {"patch+cross-slice", "patch_cross_slice"}:
                if self.use_token_fusion:
                    x = self._fuse_tokens(patch_tokens, slice_tokens)
                else:
                    x = patch_tokens
            else:
                raise ValueError(f"Unsupported tokenization_mode: {self.tokenization_mode}")

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


def create_medmamba3d_base(num_classes=3, **kwargs):
    """Create MedMamba3D-Base model"""
    model = VSSM3D(
        depths=[2, 2, 12, 2],
        dims=[128, 256, 512, 1024],
        num_classes=num_classes,
        **kwargs
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


class MedMamba3DClassifier(nn.Module):
    def __init__(self,
                 model_size='base',
                 num_classes=3,
                 in_chans=1,
                 input_size=(64, 64, 64),
                 normalization='instance',
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Input normalization
        if normalization == 'instance':
            self.input_norm = nn.InstanceNorm3d(in_chans)
        elif normalization == 'batch':
            self.input_norm = nn.BatchNorm3d(in_chans)
        elif normalization == 'group':
            self.input_norm = nn.GroupNorm(num_groups=1, num_channels=in_chans)
        else:
            self.input_norm = nn.Identity()
        
        # Create backbone model
        self.backbone = VSSM3D(
            depths=[2, 2, 12, 2],
            dims=[128, 256, 512, 1024],
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=dropout_rate,
            **kwargs
        )
        
    def preprocess(self, x):
        # Normalize input
        x = self.input_norm(x)
        
        # Clip extreme values (common in medical imaging)
        x = torch.clamp(x, min=-3, max=3)
        
        return x
    
    def forward(self, x):
        x = self.preprocess(x)
        
        # Forward through backbone
        logits = self.backbone(x)
        
        return logits
