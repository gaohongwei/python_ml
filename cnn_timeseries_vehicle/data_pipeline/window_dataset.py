"""Sliding windows, chronological split, and leak-free scaling.

Two rules drive every function here:

1. Time order is never shuffled across the split. Rows are cut into
   train / val / test by position, oldest first.
2. Statistics (mean, std) are computed on training rows only. A scaler fitted
   on all rows would hand the model information about the future.

A window is `seq_len` past steps of every channel; its label is the next
`horizon` values of the target. Both always sit inside the same split, so no
sample ever straddles the train/val boundary.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

TRowRange = Tuple[int, int]  # [start, end) row positions


class WindowBuildError(Exception):
    """Raised when the data is too short to build the requested windows."""


@dataclass(frozen=True)
class TWindowSpec:
    """Look back `seq_len` steps, predict `horizon` steps, step by `stride`."""

    seq_len: int
    horizon: int
    stride: int = 1

    @property
    def span(self) -> int:
        """Rows consumed by one training sample: inputs plus labels."""
        return self.seq_len + self.horizon


@dataclass
class TDataBundle:
    """Everything training needs, plus the scalers prediction will need."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: Optional[DataLoader]
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    num_channels: int
    num_train_windows: int
    num_val_windows: int
    num_test_windows: int


def split_row_ranges(
    num_rows: int, val_ratio: float, tst_ratio: float
) -> Tuple[TRowRange, TRowRange, TRowRange]:
    """Cut rows into train / val / test by time order (oldest rows train)."""
    train_end = int(num_rows * (1.0 - val_ratio - tst_ratio))
    val_end = int(num_rows * (1.0 - tst_ratio))
    return (0, train_end), (train_end, val_end), (val_end, num_rows)


def get_window_starts(row_range: TRowRange, spec: TWindowSpec) -> List[int]:
    """Legal window starts inside one range; inputs and labels stay in range."""
    start_row, end_row = row_range
    last_start = end_row - spec.span
    if last_start < start_row:
        return []
    return list(range(start_row, last_start + 1, spec.stride))


def slice_window(x_values: np.ndarray, start: int, seq_len: int) -> np.ndarray:
    """One window as channels-first (channels, seq_len), which is what Conv1d wants.

    Training and prediction share this function so the layout can never diverge.
    """
    window = x_values[start : start + seq_len].T
    return np.ascontiguousarray(window)


def build_window_batch(
    x_values: np.ndarray, start_indices: List[int], seq_len: int
) -> np.ndarray:
    """Stack several windows into (batch, channels, seq_len) for inference."""
    return np.stack(
        [slice_window(x_values, start, seq_len) for start in start_indices], axis=0
    )


def get_window_label(y_values: np.ndarray, start: int, spec: TWindowSpec) -> np.ndarray:
    """The `horizon` target values that follow a window."""
    label_start = start + spec.seq_len
    return np.ascontiguousarray(y_values[label_start : label_start + spec.horizon])


class WindowDataset(Dataset):
    """Stores the full arrays plus window starts; slices lazily in __getitem__.

    Memory stays O(rows) instead of O(windows * seq_len).
    """

    def __init__(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        start_indices: List[int],
        spec: TWindowSpec,
    ):
        self.x_values = x_values
        self.y_values = y_values
        self.start_indices = start_indices
        self.spec = spec

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int):
        start = self.start_indices[index]
        x_window = slice_window(self.x_values, start, self.spec.seq_len)
        y_label = get_window_label(self.y_values, start, self.spec)
        return torch.from_numpy(x_window), torch.from_numpy(y_label)


def fit_feature_scaler(x_values: np.ndarray, train_range: TRowRange) -> StandardScaler:
    """Fit the channel scaler on training rows only."""
    start, end = train_range
    if end <= start:
        raise WindowBuildError("training range is empty: check val_ratio / tst_ratio")
    return StandardScaler().fit(x_values[start:end])


def fit_target_scaler(y_values: np.ndarray, train_range: TRowRange) -> StandardScaler:
    """Fit the target scaler on training rows only.

    Scaling the target keeps the loss in a sane range whatever the unit is; the
    scaler is saved so predictions can be converted back to real units.
    """
    start, end = train_range
    return StandardScaler().fit(y_values[start:end].reshape(-1, 1))


def scale_features(x_values: np.ndarray, scaler_x: StandardScaler) -> np.ndarray:
    """Apply the training-fitted channel scaler to all rows."""
    return scaler_x.transform(x_values).astype(np.float32)


def scale_target(y_values: np.ndarray, scaler_y: StandardScaler) -> np.ndarray:
    """Apply the training-fitted target scaler to all rows."""
    return scaler_y.transform(y_values.reshape(-1, 1)).reshape(-1).astype(np.float32)


def inverse_scale_target(values: np.ndarray, scaler_y: StandardScaler) -> np.ndarray:
    """Convert scaled predictions back to the original unit (km/h, rpm, ...)."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return scaler_y.inverse_transform(flat).reshape(np.asarray(values).shape)


def create_window_loader(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    row_range: TRowRange,
    spec: TWindowSpec,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> Optional[DataLoader]:
    """One DataLoader for one split; None when the range holds no full window."""
    start_indices = get_window_starts(row_range, spec)
    if not start_indices:
        return None
    dataset = WindowDataset(x_scaled, y_scaled, start_indices, spec)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def create_data_bundle(
    x_values: np.ndarray,
    y_values: np.ndarray,
    spec: TWindowSpec,
    batch_size: int,
    val_ratio: float,
    tst_ratio: float,
    random_seed: int = 0,
) -> TDataBundle:
    """Split rows, fit scalers on train, then build the three loaders."""
    num_rows = x_values.shape[0]
    train_range, val_range, test_range = split_row_ranges(
        num_rows, val_ratio, tst_ratio
    )

    scaler_x = fit_feature_scaler(x_values, train_range)
    scaler_y = fit_target_scaler(y_values, train_range)
    x_scaled = scale_features(x_values, scaler_x)
    y_scaled = scale_target(y_values, scaler_y)

    generator = torch.Generator()
    generator.manual_seed(random_seed)

    # Windows may be shuffled: the time order *inside* each window is untouched.
    train_loader = create_window_loader(
        x_scaled, y_scaled, train_range, spec, batch_size, True, generator
    )
    val_loader = create_window_loader(
        x_scaled, y_scaled, val_range, spec, batch_size, False
    )
    test_loader = create_window_loader(
        x_scaled, y_scaled, test_range, spec, batch_size, False
    )

    if train_loader is None or val_loader is None:
        raise WindowBuildError(
            f"seq_len={spec.seq_len} + horizon={spec.horizon} does not fit in the "
            f"{num_rows} available rows (train rows={train_range[1] - train_range[0]}, "
            f"val rows={val_range[1] - val_range[0]}): lower seq_len or add data"
        )

    bundle = TDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        num_channels=x_values.shape[1],
        num_train_windows=len(train_loader.dataset),
        num_val_windows=len(val_loader.dataset),
        num_test_windows=len(test_loader.dataset) if test_loader else 0,
    )
    logger.info(
        f"windows: train={bundle.num_train_windows}, val={bundle.num_val_windows}, "
        f"test={bundle.num_test_windows}, channels={bundle.num_channels}"
    )
    return bundle


def get_min_required_rows(spec: TWindowSpec, val_ratio: float, tst_ratio: float) -> int:
    """Rows needed so that every non-empty split can hold at least one window.

    The smallest split is the binding constraint: with val_ratio=0.15 one window
    of span 65 already needs ~434 rows.
    """
    smallest_ratio = min([ratio for ratio in (val_ratio, tst_ratio) if ratio > 0] or [1.0])
    return int(np.ceil(spec.span / smallest_ratio))


def get_inference_window_starts(num_rows: int, seq_len: int, stride: int = 1) -> List[int]:
    """Window starts for prediction: only inputs are needed, no labels."""
    last_start = num_rows - seq_len
    if last_start < 0:
        raise WindowBuildError(
            f"need at least seq_len={seq_len} rows to predict, got {num_rows}"
        )
    return list(range(0, last_start + 1, stride))
