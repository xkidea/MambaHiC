from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba(nn.Module):
    """Pure PyTorch compatibility implementation of mamba_ssm.Mamba.

    This matches the Mamba v1 parameter names used by the saved checkpoint.
    It is intended for CPU inference when the CUDA extension is unavailable.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, _ = hidden_states.shape

        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(x.transpose(1, 2).reshape(batch * seqlen, self.d_inner))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = dt.reshape(self.d_inner, batch, seqlen).permute(1, 0, 2).contiguous()
        B = B.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()
        C = C.reshape(batch, seqlen, self.d_state).transpose(1, 2).contiguous()

        y = self._selective_scan(x, dt, B, C, z)
        return self.out_proj(y.transpose(1, 2))

    def _selective_scan(self, u, delta, B, C, z):
        dtype_in = u.dtype
        u_f = u.float()
        delta_f = F.softplus(delta.float() + self.dt_proj.bias.float().view(1, -1, 1))
        A = -torch.exp(self.A_log.float())
        B_f = B.float()
        C_f = C.float()

        state = torch.zeros(u.shape[0], self.d_inner, self.d_state, device=u.device, dtype=torch.float32)
        ys = []
        for i in range(u.shape[-1]):
            delta_i = delta_f[:, :, i]
            deltaA = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))
            deltaB_u = delta_i.unsqueeze(-1) * B_f[:, :, i].unsqueeze(1) * u_f[:, :, i].unsqueeze(-1)
            state = state * deltaA + deltaB_u
            y_i = torch.sum(state * C_f[:, :, i].unsqueeze(1), dim=-1)
            ys.append(y_i)

        y = torch.stack(ys, dim=-1)
        y = y + u_f * self.D.float().view(1, -1, 1)
        y = y * F.silu(z.float())
        return y.to(dtype=dtype_in)
