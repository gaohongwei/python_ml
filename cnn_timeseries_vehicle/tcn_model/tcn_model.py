"""TCN (Temporal Convolutional Network) definition.

Input  : (batch, num_channels, seq_len) - channels-first, one channel per feature
Output : (batch, horizon)               - the next `horizon` target values

Concretely, training on speed + throttle + coolant_temp with seq_len=64 and
horizon=5 means every sample is (3, 64) going in and (5,) coming out: 64 past
time steps of 3 signals, 5 future values of the one target.

Note the naming seam: the config calls it `horizon`, the network calls it
`out_dim`, because the network does not care that its outputs are future steps -
it only knows how many numbers to produce. `build_tcn_model` is where the two
meet.

Why a TCN rather than an RNN:
- dilated convolutions reach far back with few layers (field grows 2^levels)
- every time step is computed in parallel, so training is fast
- convolutions are *causal*: output at step t only sees steps <= t

How far back one output actually sees is `get_receptive_field(num_levels,
kernel_size)` in train_config.py, printed at the start of every run. It must
cover `seq_len`, or the early steps of each window are wasted.

Reading order here is bottom-up: a padding trim, one convolution, one residual
block, then the stack.
"""

from typing import Dict

import torch.nn as nn

try:
    # torch >= 2.1
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - older torch
    from torch.nn.utils import weight_norm

# Small init keeps the first forward pass from saturating a deep dilated stack;
# 0.01 is the value the reference TCN implementation uses.
INIT_WEIGHT_STD = 0.01


class TrimRightPadding(nn.Module):
    """Cut the extra right-side padding so the convolution stays causal.

    Conv1d pads *both* ends to keep the length, but a right-side pad means step t
    convolves over t+1, i.e. the model peeks at the future it is asked to predict.
    Trimming that pad is what makes the stack causal; it is not an optimization.

    Published TCN code calls this `Chomp1d`, which is the same thing.
    """

    def __init__(self, trim_size: int):
        super().__init__()
        self.trim_size = trim_size

    def forward(self, x):
        if self.trim_size <= 0:
            return x
        return x[:, :, : -self.trim_size].contiguous()


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
            TrimRightPadding(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            second_conv,
            TrimRightPadding(padding),
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


class TcnModel(nn.Module):
    """Stack of temporal blocks with doubling dilation, then a linear head.

    - `num_channels` : input signals per time step, set by the data, not the config
    - `out_dim`      : numbers to predict per sample, i.e. the config's `horizon`
    - `num_levels`   : how many blocks; each one doubles how far back the model sees
    - `num_hidden_channels` : width inside the stack, same for every block

    The only stateful thing here is the weights: no scaler, no column names, no
    idea which signal is the target. That context lives in the artifact.
    """

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
        self.num_channels = num_channels
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
        self.check_input_shape(x)
        features = self.tcn(x)
        # Causal convolutions mean the last position has seen the whole window and
        # nothing after it, so one step carries the summary - no pooling needed.
        last_step = features[:, :, -1]
        return self.head(last_step)

    def check_input_shape(self, x) -> None:
        """Fail with the expected layout rather than a deep Conv1d error.

        Passing (batch, seq_len, channels) - the layout a DataFrame suggests - is
        the easy mistake here; Conv1d would either crash far from the cause or,
        when seq_len happens to equal the channel count, train on nonsense.
        """
        if x.dim() != 3:
            raise ValueError(
                f"expected 3 dims (batch, channels={self.num_channels}, seq_len), "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} channels at dim 1, got shape "
                f"{tuple(x.shape)}: this layout is channels-first, so transpose "
                f"(batch, seq_len, channels) inputs"
            )


def build_tcn_model(num_channels: int, out_dim: int, arch: Dict) -> TcnModel:
    """Create a network from a plain dict, the same dict stored in the artifact."""
    return TcnModel(
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
