"""
State Space 3D (SS3D) implementation for MedMamba
Extended from SS2D to support 3D volumetric data (medical volumes, videos, etc.)
"""
import time
import math
from functools import partial
from typing import Optional, Callable
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass




def flops_selective_scan_ref_3d(B=1, L=4096, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    FLOPS calculation for 3D selective scan
    Similar to 2D version but with 3D spatial dimensions
    
    Args:
        B: batch size
        L: sequence length (D*H*W for 3D)
        D: model dimension 
        N: state dimension
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    
    assert not with_complex

    flops = 0
    
    # Core einsum operations for 3D
    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    
    # Sequential processing
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 

    # Optional components
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    
    return flops


class SS3D(nn.Module):
    def __init__(
        self,
        d_model,        #输入/输出通道数，也是状态空间块的主维度；与上一层的 embedding 维一致。
        d_state=16,     #状态空间的隐状态维度（记忆槽数量），越大能捕捉的长程依赖越丰富
        d_conv=3,       #深度可分离 3D 卷积的核大小，用于在进入状态空间前做局部混合
        expand=2,       #输入线性扩张的倍率，内部实际通道 d_inner = expand * d_model。
        dt_rank="auto", #Δt 投影的低秩大小；"auto" 时会取 ceil(d_model/16)，控制时间步参数的近似精度与开销
        dt_min=0.001,   #初始化时希望 softplus(dt_bias) 落在的范围，避免 Δt 过大或过小导致不稳定
        dt_max=0.1,
        dt_init="random",   #Δt 权重初始化策略，当前支持 "random"（默认均匀）或 "constant"。
        dt_scale=1.0,       #调整 Δt 权重初始化幅度的缩放因子。
        dt_init_floor=1e-4, #Δt 采样的下限，防止初始化为 0。
        dropout=0.,         #SS3D 输出后的 dropout 概率。
        conv_bias=True,     #控制深度卷积是否带 bias。
        bias=False,         #输入/输出线性层是否使用 bias。
        device=None,
        dtype=None,
        scan_directions=6,  #决定 create_3d_scan_sequences 的序列数。
        **kwargs,           #向下传递到 selective scan/其它子模块的附加参数，便于扩展。
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)  # d_inner就是B*L*D中的D
        # 每 16 个通道共享一个 rank，保证近似精度与开销之间的平衡
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_directions = scan_directions  

        
        # 把输入通道从 d_model 投影到 2 * d_inner，并在前向里用 x, z = xz.chunk(2, dim=-1) 拆成两半：
        # 一半进入状态空间主干（x），另一半作为后续的门控/调制量（z）
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            # 这层卷积的输入输出通道都是 self.d_inner，
            # groups=self.d_inner 把卷积核按通道拆成 self.d_inner 组，每组只有 1 个通道的输入和输出。
            groups=self.d_inner,            # 让不同通道互不干扰，保持 SSM 推断时的通道独立性
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()    #给后续特征加一个 SiLU（Swish）激活，避免模型退化成纯线性系统

        
        
        self.x_proj = tuple(
            # 输入向量长度self.d_inner扩展到输出长度 self.dt_rank + 2*self.d_state。
            # dt_rank用于 Δt 投影， 两个d_state分别对应SSM的B，C矩阵
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.scan_directions)
        )
        #  把 6 组权重沿新维度堆叠成一个张量，这样就能在前向里用 torch.einsum 同时处理所有方向。
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj # 权重已经提取到x_proj_weight里了，不再保留，释放内存

        # dt_proj 本质上就是一个 nn.Linear(dt_rank, d_inner)，
        # 所以只包含两组可训练参数：weight（形状 d_inner × dt_rank）和 bias（长度 d_inner）。
        self.dt_projs = tuple(
            # 为每个扫描方向各自创建一条 Δt 投影层，并将维度扩展到 D大小
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.scan_directions)
        )

        # 将weight ，bias 堆叠成一个张量
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        # State parameters
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.scan_directions, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.scan_directions, merge=True)

        # Forward core function
        self.forward_core = self.forward_core

        # Output layers
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", 
                dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):

        # 将长度 r 的低秩表示恢复到 D        
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # 设定权重初始化的标准差上限，避免 Δt 投影在初始阶段输出过大或过小
        dt_init_std = dt_rank**-0.5 * dt_scale  
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)# 常数初始化，Δt 投影层会以固定权重开始工作
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)# 权重在 ±dt_init_std 之间均匀随机初始化
        else:
            raise NotImplementedError

        #先 torch.rand(d_inner) 生成 [0,1) 的均匀分布；
        #乘上 log(dt_max) - log(dt_min) 再加 log(dt_min)，等价于在 log 空间做线性插值；
        #再 torch.exp(...) 把结果从 log 域还原回数值域，
        #这样 dt 就是服从 log-uniform 分布的随机数（每个输出通道一个），范围在 [dt_min, dt_max]；
        #最后 .clamp(min=dt_init_floor) 防止数值过小。
        #这样得到的 dt 会被转成 softplus 反函数，填到线性层 bias，确保初始 Δt 既覆盖合理区间、又不至于为 0。
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        #这段是把刚采样的 dt 映射回 softplus 的反函数，以便直接写进线性层的 bias。
        #因为后续计算 Δt 会走 F.softplus(bias)，想让初始 Δt 恰好等于采样的 dt，就需要 bias = softplus^{-1}(dt)。
        #torch.expm1(-dt) 是 e^{-dt} - 1，torch.log(-torch.expm1(-dt)) 正好是 softplus 的解析逆函数
        #所以 inv_dt = dt + log(-expm1(-dt)) 相当于 softplus^{-1}(dt)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # dt_proj.bias._no_reinit = True 是打一个标记，
        # 避免后续调用 model.apply(init_fn) 时再次把 bias 归零，保证这个特殊初始化不会被覆盖
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):

        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def reconstruct_from_sequences(self, ys, B, D, H, W):
        K, C, L = ys.shape[1], ys.shape[2], ys.shape[3]
        
        # Process each scanning direction
        y1 = ys[:, 0].view(B, C, D, H, W)  # Forward depth-first
        y2 = torch.flip(ys[:, 1], dims=[-1]).view(B, C, D, H, W)  # Reverse depth-first
        
        # Height-first sequences
        y3_hdw = ys[:, 2].view(B, C, H, D, W)
        y3 = y3_hdw.permute(0, 1, 3, 2, 4).contiguous()  # H->D->W back to D->H->W
        
        y4_hdw = torch.flip(ys[:, 3], dims=[-1]).view(B, C, H, D, W) 
        y4 = y4_hdw.permute(0, 1, 3, 2, 4).contiguous()
        
        # Width-first sequences  
        y5_wdh = ys[:, 4].view(B, C, W, D, H)
        y5 = y5_wdh.permute(0, 1, 3, 4, 2).contiguous()  # W->D->H back to D->H->W
        
        y6_wdh = torch.flip(ys[:, 5], dims=[-1]).view(B, C, W, D, H)
        y6 = y6_wdh.permute(0, 1, 3, 4, 2).contiguous()
        
        # Combine all directions
        y_combined = y1 + y2 + y3 + y4 + y5 + y6
        
        # Convert to (B, D, H, W, C) format
        y = y_combined.permute(0, 2, 3, 4, 1).contiguous()
        
        return y

    def forward_core(self, x: torch.Tensor):
       
        self.selective_scan = selective_scan_fn
        
        B, C, D, H, W = x.shape
        L = D * H * W
        K = self.scan_directions

        # 生成6条序列，长/宽/高正反向扫描
        # 深度方向（D->H->W）
        seq_d = x.view(B, -1, L)
        seq_d_rev = torch.flip(seq_d, dims=[-1])

        # 高度方向（H->D->W）：先交换 D/H，再展平
        seq_h = x.permute(0, 1, 3, 2, 4).contiguous().view(B, -1, L)
        seq_h_rev = torch.flip(seq_h, dims=[-1])

        # 宽度方向（W->D->H）：交换到 W 在前，再展平
        seq_w = x.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, L)
        seq_w_rev = torch.flip(seq_w, dims=[-1])

        # 堆成 K=6 条扫描序列
        xs = torch.stack([seq_d, seq_d_rev, seq_h, seq_h_rev, seq_w, seq_w_rev], dim=1)  # (B, K, C, L)

        # 在进入 selective scan 之前，为每个扫描方向生成状态空间模型所需的 Δ、B、C 参数
        # xs(batch, direction, channel, sequence_length)
        # self.x_proj_weight 是事先堆好的 (K, dt_rank + 2*d_state, d_inner) 张量
        # einsum 按照方向 k 和通道 c 做矩阵乘法，把每个方向的 C 维特征映射到 D 维，保持批次 b 和序列位置 l
        x_dbl = torch.einsum("b k c l, k d c -> b k d l", xs, self.x_proj_weight)

        # 在第 2 维（索引从 0 开始）按照 [dt_rank, d_state, d_state] 这三个长度依次切分：
        # 先取 dt_rank 个通道作为 dts，接着取 d_state 个作为 Bs，最后再取 d_state 个作为 Cs
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        
        
        # dts 的形状是 (B, K, R, L)，self.dt_projs_weight 是 (K, C, R)，
        # 按 r 维做乘法并对 r 求和，得到 (B, K, C, L)。
        dts = torch.einsum("b k r l, k c r -> b k c l", dts, self.dt_projs_weight)

        
        xs = xs.float().view(B, -1, L)  # (b, k * c, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * c, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * c)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * c, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * c)

        # out_y = (B, K, d_inner, L)
        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        # y =  (B, D, H, W, C)
        y = self.reconstruct_from_sequences(out_y, B, D, H, W)
        
        return y


    def forward(self, x: torch.Tensor, **kwargs):
        # x.shape = （B, D, H, W, C）   C = d_model，

        # Input projection
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # 转换成(B, C, D, H, W) 
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))
        
        
        y = self.forward_core(x)  # (b, d, h, w, d_inner)
        # 在扫描算法里，X的形状是（B，L，D）B就是b，L是d*h*w，D就是d_model
        
        # Normalize and apply gating
        y = self.out_norm(y)
        y = y * F.silu(z)
        
        # Final projection
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
            
        return out



