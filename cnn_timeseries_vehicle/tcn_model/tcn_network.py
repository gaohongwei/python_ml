"""TCN (Temporal Convolutional Network) definition.

Input  : (batch, num_channels, seq_len) - channels-first, one channel per feature
Output : (batch, horizon)               - the next `horizon` target values

Why a TCN rather than an RNN:
- dilated convolutions reach far back with few layers (field grows 2^levels)
- every time step is computed in parallel, so training is fast
- convolutions are *causal*: output at step t only sees steps <= t
"""

from typing import Dict

import torch.nn as nn

try:
    # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - older torch
    from torch.nn.utils import weight_norm

INIT_WEIGHT_STD = 0.01


class Chomp1d(nn.Module):
    """Cut the extra right-side padding so the convolution stays causal.

    Conv1d pads both ends; keeping the right pad would let step t read t+1.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size <= 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


def build_causal_conv(in_channels: int, out_channels: int, kernel_size: int, dilation: int):
    """One weight-normalized dilated conv, plus the pad size to chomp off."""
    padding = (kernel_size - 1) * dilation
    conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
    )
    # Initialize before wrapping: weight_norm re-parameterizes `weight`.
    nn.init.normal_(conv.weight, mean=0.0, std=INIT_WEIGHT_STD)
    nn.init.zeros_(conv.bias)
    return weight_norm(conv), padding


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions plus a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        first_conv, padding = build_causal_conv(
            in_channels, out_channels, kernel_size, dilation
        )
        second_conv, _ = build_causal_conv(
            out_channels, out_channels, kernel_size, dilation
        )
        self.net = nn.Sequential(
            first_conv,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            second_conv,
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 1x1 conv only when the channel count changes, so shapes can be added.
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, mean=0.0, std=INIT_WEIGHT_STD)
            nn.init.zeros_(self.downsample.bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)


class TcnNetwork(nn.Module):
    """Stack of temporal blocks with doubling dilation, then a linear head."""

    def __init__(
        self,
        num_channels: int,
        out_dim: int,
        num_levels: int,
        num_hidden_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        blocks = []
        for level in range(num_levels):
            blocks.append(
                TemporalBlock(
                    in_channels=num_channels if level == 0 else num_hidden_channels,
                    out_channels=num_hidden_channels,
                    kernel_size=kernel_size,
                    dilation=2**level,  # 1, 2, 4, 8, ... reaches far back cheaply
                    dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(num_hidden_channels, out_dim)

    def forward(self, x):
        """(batch, num_channels, seq_len) -> (batch, out_dim)"""
        features = self.tcn(x)
        # Causality means the last step already summarizes the whole window.
        last_step = features[:, :, -1]
        return self.head(last_step)


def build_tcn_network(num_channels: int, out_dim: int, arch: Dict) -> TcnNetwork:
    """Create a network from a plain dict, the same dict stored in the artifact."""
    return TcnNetwork(
        num_channels=num_channels,
        out_dim=out_dim,
        num_levels=arch["num_levels"],
        num_hidden_channels=arch["num_hidden_channels"],
        kernel_size=arch["kernel_size"],
        dropout=arch["dropout"],
    )


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, handy for spotting an oversized model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
