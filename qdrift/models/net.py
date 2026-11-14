"""Small neural decoders: every model maps (B, r+1, d-1) 0/1 floats to logits (B,)."""
from __future__ import annotations

import torch
from torch import nn

MODEL_KINDS = ("mlp", "conv", "gru")


class MLP(nn.Module):
    def __init__(self, d: int, r: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        n_in = (r + 1) * (d - 1)
        for _ in range(depth):
            layers += [nn.Linear(n_in, width), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            n_in = width
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = 2.0 * x - 1.0
        return self.net(x.flatten(1)).squeeze(-1)


class Conv(nn.Module):
    def __init__(self, d: int, r: int, width: int, depth: int, dropout: float):
        super().__init__()
        ch = max(16, width // 4)
        layers: list[nn.Module] = []
        c_in = 1
        for _ in range(max(1, min(depth, 4))):
            layers += [nn.Conv2d(c_in, ch, 3, padding=1), nn.GELU()]
            c_in = ch
        self.conv = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(ch * (r + 1) * (d - 1), 1)

    def forward(self, x):
        x = (2.0 * x - 1.0).unsqueeze(1)
        h = self.conv(x).flatten(1)
        return self.head(self.drop(h)).squeeze(-1)


class GRU(nn.Module):
    def __init__(self, d: int, r: int, width: int, depth: int, dropout: float):
        super().__init__()
        layers = max(1, min(depth, 2))
        self.rnn = nn.GRU(d - 1, width, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        out, _ = self.rnn(2.0 * x - 1.0)
        return self.head(out[:, -1]).squeeze(-1)


def build_model(kind: str, d: int, r: int, width: int = 256, depth: int = 3,
                dropout: float = 0.0) -> nn.Module:
    if kind == "mlp":
        return MLP(d, r, width, depth, dropout)
    if kind == "conv":
        return Conv(d, r, width, depth, dropout)
    if kind == "gru":
        return GRU(d, r, width, depth, dropout)
    raise ValueError(f"unknown model kind {kind!r}; choose from {MODEL_KINDS}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
