import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:
    selective_scan_fn = None


_SELECTIVE_SCAN_WARNED = False


class SS3DVideoMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        scan_type="Axis-parallel",
        group_type="Cube",
        k_group=None,
        use_gate=True,
        fusion_mode="sum",
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.scan_type = scan_type
        self.group_type = group_type
        self.k_group = self._resolve_k_group(scan_type, k_group) 
        self.route_count = self._route_count(scan_type)
        self.use_gate = use_gate
        self.fusion_mode = fusion_mode
        if self.fusion_mode not in {"sum", "weighted", "gated", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = tuple(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.k_group)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.k_group)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.k_group, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.k_group, merge=True)

        if self.route_count > 1:
            if self.fusion_mode == "weighted":
                self.route_weights = nn.Parameter(torch.zeros(self.route_count))
            elif self.fusion_mode == "gated":
                self.route_gate = nn.Linear(self.d_inner, self.route_count, bias=True)
            elif self.fusion_mode == "concat":
                self.route_concat_proj = nn.Linear(self.d_inner * self.route_count, self.d_inner, bias=False)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod   
    # 根据 scan_type 推断 k_group 的值，决定要构造多少条扫描序列。
    def _resolve_k_group(scan_type: str, k_group: Optional[int]) -> int:
        if scan_type == "Axis-parallel":
            expected = 6
        elif scan_type in {"Axis-xz", "Axis-yz", "Parallel spectral-spatial"}:
            expected = 4
        elif scan_type in {
            "Axis-z",
            "Order-z-xy",
            "Order-xy-z",
            "Spectral-priority",
            "Spatial-priority",
            "Cross spectral-spatial",
            "Cross spatial-spectral",
        }:
            expected = 2
        else:
            raise ValueError(f"Unsupported scan_type: {scan_type}")
        if k_group is None:
            return expected
        if k_group != expected:
            raise ValueError(f"scan_type {scan_type} expects k_group={expected}, got {k_group}")
        return k_group

    @staticmethod
    # 根据 scan_type 推断 route_count 的值，决定要融合多少条扫描路径的结果。
    def _route_count(scan_type: str) -> int:
        if scan_type == "Axis-parallel":
            return 3
        if scan_type in {"Axis-xz", "Axis-yz", "Parallel spectral-spatial"}:
            return 2
        if scan_type in {
            "Axis-z",
            "Order-z-xy",
            "Order-xy-z",
            "Spectral-priority",
            "Spatial-priority",
        }:
            return 1
        if scan_type in {"Cross spectral-spatial", "Cross spatial-spectral"}:
            return 2
        raise ValueError(f"Unsupported scan_type: {scan_type}")

    @staticmethod
    # 初始化Mamba里的delta t投影，这是selective scan 里时间步长/离散化相关的参数
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        # 把一个低秩表示 dt_rank 投影到每个通道的 d_inner。
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # 初始化权重和偏置，如果是constant，就初始化为一个固定值；如果是random，就在一个范围内随机初始化。
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # 在对数空间 里随机采样一个 dt，范围大致在 [dt_min, dt_max]。
        # 这样做的目的通常是：让不同通道初始时有不同时间尺度。
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # 把 scan 的“步长参数”初始化到稳定、合理的数值区间。
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    # A_log_init 和 D_init：初始化状态空间参数
    # 在 forward_core 里会看到：真正用的时候是 A = -exp(A_log)。
    # 这能保证 A 是负数，常见目的是让状态转移更稳定。
    # 如果有多条扫描序列（比如 k_group=6），每条扫描都需要一份参数，所以这里会复制多份。
    # 如果 merge=True，就把 [k_group, d_inner, ...] 展平成更便于后面计算的形状。
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D 是 SSM 里的 skip / residual-like 项，初始成 1 比较常见。
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # 4 个 flatten_*：把 3D 张量按不同顺序拉平成 1D
    # 输入是x: [B, C, D, H, W]，输出都是 [B, C, L]，L=D*H*W，但展平顺序不同。
    # flatten_spectral_spatial是先把 (h,w) 合成一个维度，再和 d 合并。
    # 先遍历空间位置，再在每个空间位置上遍历 D
    def flatten_spectral_spatial(self, x):
        x = rearrange(x, "b c d h w -> b c (h w) d")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    # flatten_spectral_spectral是先保留 d 在外层，再把 (h,w) 拉平。
    # 先遍历 D，再在每个 D slice 里遍历所有空间位置
    def flatten_spatial_spectral(self, x):
        x = rearrange(x, "b c d h w -> b c d (h w)")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    # flatten_x_priority 是先把 (d,h) 合成一个维度，再和 w 合并。
    # 优先沿 x/W 方向连续扫描
    def flatten_x_priority(self, x):
        x = rearrange(x, "b c d h w -> b c (d h) w")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    # flatten_y_priority 是先把 (d,w) 合成一个维度，再和 h 合并。
    # 优先沿 y/H 方向连续扫描
    def flatten_y_priority(self, x):
        x = rearrange(x, "b c d h w -> b c (d w) h")
        x = rearrange(x, "b c n m -> b c (n m)")
        return x

    # reshape_* 是 flatten_* 的逆操作，把 [B, C, L] 还原成 [B, C, D, H, W]，顺序要和 flatten 时对应上。
    # y的形状是 [B, C, L]，要还原成 [B, C, D, H, W]，中间会有一些转置和重排来调整维度顺序。
    def reshape_spectral_spatial(self, y, B, D, H, W):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, H * W, D, -1)
        y = rearrange(y, "b o d c -> b d o c")
        y = y.view(B, D, H, W, -1)
        return y

    def reshape_spatial_spectral(self, y, B, D, H, W):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, D, H, W, -1)
        return y

    def reshape_x_priority(self, y, B, D, H, W):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, D, H, W, -1)
        return y

    # reshape_y_priority 多了一次 permute，因为它之前 flatten 的时候把 H/W 的顺序换过，所以恢复时要再换回来
    def reshape_y_priority(self, y, B, D, H, W):
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, D, W, H, -1)
        y = y.permute(0, 1, 3, 2, 4).contiguous()
        return y

    # 根据 scan_type 构造多条扫描序列
    # 先生成 4 种展平序列：形状都是[B, C, L]，然后按照 scan_type 不同，堆成：xs: [B, K, C, L]，k=k_group
    def scan(self, x):
        x_x = self.flatten_x_priority(x)
        x_y = self.flatten_y_priority(x)
        x_z = self.flatten_spectral_spatial(x)
        x_xy = self.flatten_spatial_spectral(x)

        if self.scan_type == "Axis-parallel":
            xs = torch.stack(
                [x_x, torch.flip(x_x, dims=[-1]), x_y, torch.flip(x_y, dims=[-1]), x_z, torch.flip(x_z, dims=[-1])],
                dim=1,
            )
        elif self.scan_type == "Axis-z":
            xs = torch.stack([x_z, torch.flip(x_z, dims=[-1])], dim=1)
        elif self.scan_type == "Axis-xz":
            xs = torch.stack([x_x, torch.flip(x_x, dims=[-1]), x_z, torch.flip(x_z, dims=[-1])], dim=1)
        elif self.scan_type == "Axis-yz":
            xs = torch.stack([x_y, torch.flip(x_y, dims=[-1]), x_z, torch.flip(x_z, dims=[-1])], dim=1)
        elif self.scan_type in {"Order-xy-z", "Spectral-priority"}:
            xs = torch.stack([x_z, torch.flip(x_z, dims=[-1])], dim=1)
        elif self.scan_type in {"Order-z-xy", "Spatial-priority"}:
            xs = torch.stack([x_xy, torch.flip(x_xy, dims=[-1])], dim=1)
        elif self.scan_type == "Cross spectral-spatial":
            xs = torch.stack([x_z, torch.flip(x_xy, dims=[-1])], dim=1)
        elif self.scan_type == "Cross spatial-spectral":
            xs = torch.stack([x_xy, torch.flip(x_z, dims=[-1])], dim=1)
        elif self.scan_type == "Parallel spectral-spatial":
            xs = torch.stack([x_z, torch.flip(x_z, dims=[-1]), x_xy, torch.flip(x_xy, dims=[-1])], dim=1)
        else:
            raise ValueError(f"Unsupported scan_type: {self.scan_type}")
        return xs

    # _fuse_routes：多路输出怎么融合
    def _fuse_routes(self, routes, out_y):
        if len(routes) == 1:
            return routes[0]
        # sum直接逐元素相加
        if self.fusion_mode == "sum":
            return sum(routes)
        # weighted 给每条路一个可学习的权重，先做 softmax 归一化，然后加权求和。
        if self.fusion_mode == "weighted":
            weights = torch.softmax(self.route_weights, dim=0)
            fused = 0.0
            for idx, route in enumerate(routes):
                fused = fused + route * weights[idx]
            return fused
        # out_y 是扫描后的序列输出[B, K, C, L]，mean(dim=(1, 3))后得到[B, C],相当于抽一个全局上下文表示。
        # 过 route_gate 得到每个样本自己的路由权重：[B, route_count],于是不同样本可以决定更依赖哪一路。
        if self.fusion_mode == "gated":
            context = out_y.mean(dim=(1, 3))
            gate = torch.softmax(self.route_gate(context), dim=-1)
            fused = 0.0
            for idx, route in enumerate(routes):
                fused = fused + route * gate[:, idx].view(route.size(0), 1, 1, 1, 1)
            return fused
        # concat:把多路输出在通道维拼起来，再过一个线性层投回 d_inner。
        if self.fusion_mode == "concat":
            return self.route_concat_proj(torch.cat(routes, dim=-1))
        raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

    # merge_scans：把每条扫描结果重新配对、还原、融合
    # 把“正向”和“反向”扫描的结果对齐后相加
    # 用对应的 reshape_* 恢复成 [B, D, H, W, C]
    # 得到若干路 routes
    # 用 _fuse_routes 融合
    def merge_scans(self, out_y, B, D, H, W):
        if self.group_type != "Cube":
            raise ValueError(f"Unsupported group_type: {self.group_type}")

        routes = []

        if self.scan_type == "Axis-parallel":
            yx = self.reshape_x_priority(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            yy = self.reshape_y_priority(out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1]), B, D, H, W)
            yz = self.reshape_spectral_spatial(out_y[:, 4] + torch.flip(out_y[:, 5], dims=[-1]), B, D, H, W)
            routes = [yx, yy, yz]
        elif self.scan_type == "Axis-z":
            yz = self.reshape_spectral_spatial(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            routes = [yz]
        elif self.scan_type == "Axis-xz":
            yx = self.reshape_x_priority(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            yz = self.reshape_spectral_spatial(out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1]), B, D, H, W)
            routes = [yx, yz]
        elif self.scan_type == "Axis-yz":
            yy = self.reshape_y_priority(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            yz = self.reshape_spectral_spatial(out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1]), B, D, H, W)
            routes = [yy, yz]
        elif self.scan_type in {"Order-xy-z", "Spectral-priority"}:
            y = self.reshape_spectral_spatial(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            routes = [y]
        elif self.scan_type in {"Order-z-xy", "Spatial-priority"}:
            y = self.reshape_spatial_spectral(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            routes = [y]
        elif self.scan_type == "Cross spectral-spatial":
            y_fwd = self.reshape_spectral_spatial(out_y[:, 0], B, D, H, W)
            y_rvs = self.reshape_spatial_spectral(torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            routes = [y_fwd, y_rvs]
        elif self.scan_type == "Cross spatial-spectral":
            y_fwd = self.reshape_spatial_spectral(out_y[:, 0], B, D, H, W)
            y_rvs = self.reshape_spectral_spatial(torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            routes = [y_fwd, y_rvs]
        elif self.scan_type == "Parallel spectral-spatial":
            ye = self.reshape_spectral_spatial(out_y[:, 0] + torch.flip(out_y[:, 1], dims=[-1]), B, D, H, W)
            ya = self.reshape_spatial_spectral(out_y[:, 2] + torch.flip(out_y[:, 3], dims=[-1]), B, D, H, W)
            routes = [ye, ya]
        else:
            raise ValueError(f"Unsupported scan_type: {self.scan_type}")

        return self._fuse_routes(routes, out_y)

    # forward_core：真正调用 Mamba selective scan
    def forward_core(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        L = D * H * W
        xs = self.scan(x)       # xs.shape = [B, K, d_inner, L]

        K = self.k_group

        x_dbl = torch.einsum("b k c l, k d c -> b k d l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k c r -> b k c l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # xs.shape = [B, K * d_inner, L]
        dts = dts.contiguous().float().view(B, -1, L)   # delta.shape = [B, K * d_inner, L]
        Bs = Bs.float().view(B, K, -1, L)  # Bs.shape = [B, K, d_state, L]
        Cs = Cs.float().view(B, K, -1, L)  # Cs.shape = [B, K, d_state, L]
        Ds = self.Ds.float().view(-1)      # Ds.shape = [K * d_inner]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # As.shape = [K * d_inner, d_state]
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # dt_projs_bias.shape = [K * d_inner]
        out_y = selective_scan_fn(
            xs,     # 输入序列本身
            dts,    # 每个位置的“更新强度”
            As,     # 状态衰减矩阵，控制旧状态怎么保留/衰减
            Bs,     # 输入到状态的投影矩阵，控制输入怎么影响状态更新
            Cs,     # 状态到输出的投影矩阵，控制状态怎么转成输出
            Ds,     # 跳跃连接/残差项，控制输出里有多少直接来自输入
            z=None, # None表示不在 selective scan 里面做门控，门控放到外面 forward() 里做了
            delta_bias=dt_projs_bias,   # 每个位置的步长偏置，控制不同位置的更新强度
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        y = self.merge_scans(out_y, B, D, H, W)
        return y

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x))

        y = self.forward_core(x)
        y = self.out_norm(y)
        if self.use_gate:
            y = y * F.silu(z)

        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)

        return out
