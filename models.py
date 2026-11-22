"""Forecasting models: a small patch transformer with switchable parts, plus linear baselines.

All models map x [B, L, C] -> y_hat [B, H, C].
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al., 2022): per-window, per-channel."""

    def __init__(self, c: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        return (x - self.mean) / self.std * self.weight + self.bias

    def denorm(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.bias) / (self.weight + self.eps * self.eps) * self.std + self.mean


class Block(nn.Module):
    """Pre-LayerNorm transformer encoder block using fused SDPA attention."""

    def __init__(self, d: int, heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d))
        self.drop = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


@dataclass(frozen=True)
class PatchConfig:
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    heads: int = 4
    layers: int = 2
    d_ff: int = 128
    dropout: float = 0.2
    revin: bool = True
    pos: bool = True
    channel_independent: bool = True


def n_patches(seq_len: int, patch_len: int, stride: int) -> int:
    """Patch count after padding the end with `stride` copies of the last step."""
    return (seq_len + stride - patch_len) // stride + 1


class PatchTransformer(nn.Module):
    def __init__(self, c: int, seq_len: int, pred_len: int, cfg: PatchConfig):
        super().__init__()
        self.c, self.pred_len, self.cfg = c, pred_len, cfg
        self.revin = RevIN(c) if cfg.revin else None
        self.n = n_patches(seq_len, cfg.patch_len, cfg.stride)
        width = 1 if cfg.channel_independent else c
        self.embed = nn.Linear(cfg.patch_len * width, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n, cfg.d_model) * 0.02) if cfg.pos else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.heads, cfg.d_ff, cfg.dropout) for _ in range(cfg.layers))
        self.ln = nn.LayerNorm(cfg.d_model) if cfg.layers else nn.Identity()
        self.head = nn.Linear(self.n * cfg.d_model, pred_len * width)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, W] -> [B, N, P*W]"""
        cfg = self.cfg
        x = torch.cat([x, x[:, -1:].expand(-1, cfg.stride, -1)], dim=1)
        p = x.unfold(1, cfg.patch_len, cfg.stride)  # [B, N, W, P]
        return p.transpose(2, 3).reshape(x.shape[0], self.n, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        if self.revin is not None:
            x = self.revin.norm(x)
        if self.cfg.channel_independent:
            x = x.permute(0, 2, 1).reshape(b * self.c, -1, 1)
        z = self.embed(self.patchify(x))
        if self.pos is not None:
            z = z + self.pos
        z = self.drop(z)
        for blk in self.blocks:
            z = blk(z)
        y = self.head(self.ln(z).flatten(1))
        if self.cfg.channel_independent:
            y = y.view(b, self.c, self.pred_len).permute(0, 2, 1)
        else:
            y = y.view(b, self.pred_len, self.c)
        if self.revin is not None:
            y = self.revin.denorm(y)
        return y


class NLinear(nn.Module):
    """Zeng et al. (2023): subtract last value, one shared linear map L -> H, add back."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.lin = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:].detach()
        return self.lin((x - last).transpose(1, 2)).transpose(1, 2) + last


class DLinear(nn.Module):
    """Zeng et al. (2023): moving-average trend + remainder, one shared linear each."""

    def __init__(self, seq_len: int, pred_len: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend, self.season = nn.Linear(seq_len, pred_len), nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel - 1) // 2
        xp = torch.cat([x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        trend = F.avg_pool1d(xp.transpose(1, 2), self.kernel, stride=1)  # [B, C, L]
        season = x.transpose(1, 2) - trend
        return (self.trend(trend) + self.season(season)).transpose(1, 2)


def naive_last(x: torch.Tensor, pred_len: int) -> torch.Tensor:
    return x[:, -1:].expand(-1, pred_len, -1)


def seasonal_naive(x: torch.Tensor, pred_len: int, period: int) -> torch.Tensor:
    reps = -(-pred_len // period)
    return x[:, -period:].repeat(1, reps, 1)[:, :pred_len]


# Transformer variants studied. Each ablation changes exactly one switch from FULL.
FULL = PatchConfig()
VARIANTS: dict[str, PatchConfig] = {
    "patchtst": FULL,
    "no_patch": PatchConfig(patch_len=1, stride=1),
    "no_revin": PatchConfig(revin=False),
    "channel_mix": PatchConfig(channel_independent=False),
    "no_pos": PatchConfig(pos=False),
    "no_attn": PatchConfig(layers=0),
}
TRAINED = [*VARIANTS, "nlinear", "dlinear"]


def build(name: str, c: int, seq_len: int, pred_len: int) -> nn.Module:
    if name in VARIANTS:
        return PatchTransformer(c, seq_len, pred_len, VARIANTS[name])
    if name == "nlinear":
        return NLinear(seq_len, pred_len)
    if name == "dlinear":
        return DLinear(seq_len, pred_len)
    raise ValueError(f"unknown model {name}")
