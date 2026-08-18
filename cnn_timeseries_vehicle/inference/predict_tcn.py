"""Prediction: load an artifact, rebuild the exact training-time preprocessing.

Everything here mirrors training on purpose. If the two ever drift apart the
model still returns numbers, they are just quietly wrong - so both sides call
the same loader, the same slicing helper and the same saved scalers.
"""

import logging
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from data_pipeline.combined_frame import load_any_feature_frame
from data_pipeline.csv_feature_loader import DataLoadingError
from tcn_model import TLoadedModel
from data_pipeline.window_dataset import (
    build_window_batch,
    get_inference_window_starts,
    inverse_scale_target,
    scale_features,
)

logger = logging.getLogger(__name__)

DEFAULT_INFERENCE_BATCH_SIZE = 256


def load_frame_for_model(
    loaded: TLoadedModel,
    csv_paths: Sequence[str] = (),
    frame: Optional[pd.DataFrame] = None,
    combined_csv: Optional[str] = None,
    data_dir: Optional[str] = None,
    file_type: str = "csv",
    min_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Read new data with the preprocessing settings stored in the artifact.

    The source shape is free - a combined DataFrame, a combined CSV, a directory
    or a file list - because only the preprocessing has to match training.
    """
    preprocess = loaded.preprocess
    return load_any_feature_frame(
        frame=frame,
        combined_csv=combined_csv,
        data_dir=data_dir,
        csv_paths=csv_paths,
        file_type=file_type,
        time_col=preprocess.get("time_col", "timestamp"),
        value_col=preprocess.get("value_col", "value"),
        resample_rule=preprocess.get("resample_rule"),
        fill_method=preprocess.get("fill_method", "ffill"),
        min_rows=min_rows or loaded.spec.seq_len,
    )


def align_frame_columns(frame: pd.DataFrame, channel_cols: List[str]) -> pd.DataFrame:
    """Reorder to the training column order; a missing channel is a hard error.

    Column order is part of the contract: channel 2 of the network is whatever
    column 2 was during training, nothing re-checks that at runtime.
    """
    missing = [column for column in channel_cols if column not in frame.columns]
    if missing:
        raise DataLoadingError(
            f"missing channels required by the model: {missing} "
            f"(provided: {list(frame.columns)})"
        )
    return frame[channel_cols]


def prepare_input_array(loaded: TLoadedModel, frame: pd.DataFrame) -> np.ndarray:
    """Aligned, scaled float32 array ready to be cut into windows."""
    aligned = align_frame_columns(frame, loaded.channel_cols)
    return scale_features(aligned.to_numpy(dtype=np.float64), loaded.scaler_x)


@torch.no_grad()
def predict_windows(
    loaded: TLoadedModel,
    x_scaled: np.ndarray,
    start_indices: List[int],
    device: Optional[torch.device] = None,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> np.ndarray:
    """Predict for the given window starts; returns (num_windows, horizon)."""
    device = device or torch.device("cpu")
    loaded.model.to(device).eval()

    outputs = []
    for batch_start in range(0, len(start_indices), batch_size):
        batch_starts = start_indices[batch_start : batch_start + batch_size]
        batch_array = build_window_batch(x_scaled, batch_starts, loaded.spec.seq_len)
        batch_tensor = torch.from_numpy(batch_array.astype(np.float32)).to(device)
        outputs.append(loaded.model(batch_tensor).cpu().numpy())

    scaled_predictions = np.concatenate(outputs, axis=0)
    return inverse_scale_target(scaled_predictions, loaded.scaler_y)


def build_prediction_frame(
    frame: pd.DataFrame,
    start_indices: List[int],
    predictions: np.ndarray,
    loaded: TLoadedModel,
) -> pd.DataFrame:
    """Label each prediction with the timestamp of its last input row.

    One row per window; columns `step_1 .. step_horizon` hold the forecast for
    1 .. horizon steps after `window_end_time`.
    """
    end_positions = [start + loaded.spec.seq_len - 1 for start in start_indices]
    result = pd.DataFrame(
        predictions,
        columns=[f"step_{step + 1}" for step in range(loaded.spec.horizon)],
    )
    result.insert(0, "window_end_time", frame.index[end_positions])
    return result


def predict_from_prepared_frame(
    loaded: TLoadedModel,
    frame: pd.DataFrame,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Prepared table in, prediction table out - the core of every predict path.

    `frame` must already be aligned / resampled / filled, i.e. it comes out of
    `load_frame_for_model`. Use `predict_from_source` for a raw DataFrame.
    """
    x_scaled = prepare_input_array(loaded, frame)
    start_indices = get_inference_window_starts(
        num_rows=x_scaled.shape[0], seq_len=loaded.spec.seq_len, stride=stride
    )
    predictions = predict_windows(loaded, x_scaled, start_indices, device)
    logger.info(f"predicted {len(start_indices)} windows from {len(frame)} rows")
    return build_prediction_frame(frame, start_indices, predictions, loaded)


def predict_latest_from_prepared_frame(
    loaded: TLoadedModel,
    frame: pd.DataFrame,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Forecast from the most recent `seq_len` rows of a prepared table."""
    x_scaled = prepare_input_array(loaded, frame)
    last_start = x_scaled.shape[0] - loaded.spec.seq_len
    predictions = predict_windows(loaded, x_scaled, [last_start], device)
    return predictions[0]


def predict_from_source(
    loaded: TLoadedModel,
    frame: Optional[pd.DataFrame] = None,
    combined_csv: Optional[str] = None,
    data_dir: Optional[str] = None,
    csv_paths: Sequence[str] = (),
    file_type: str = "csv",
    stride: int = 1,
    latest: bool = False,
    device: Optional[torch.device] = None,
):
    """Any source in, forecast out; `latest=True` returns only (horizon,) values.

    Source precedence is the same as training's, since both go through
    `load_any_feature_frame`.
    """
    prepared = load_frame_for_model(
        loaded,
        csv_paths=csv_paths,
        frame=frame,
        combined_csv=combined_csv,
        data_dir=data_dir,
        file_type=file_type,
    )
    if latest:
        return predict_latest_from_prepared_frame(loaded, prepared, device)
    return predict_from_prepared_frame(loaded, prepared, stride, device)


def predict_from_frame(
    loaded: TLoadedModel,
    frame: pd.DataFrame,
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Combined DataFrame in, prediction table out (preprocessing runs here)."""
    return predict_from_source(loaded, frame=frame, stride=stride, device=device)


def predict_from_csv_files(
    loaded: TLoadedModel,
    csv_paths: Sequence[str],
    stride: int = 1,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Per-feature CSV files in, prediction table out."""
    return predict_from_source(
        loaded, csv_paths=csv_paths, stride=stride, device=device
    )


def predict_latest(
    loaded: TLoadedModel,
    csv_paths: Sequence[str],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Forecast from the most recent `seq_len` rows only: shape (horizon,)."""
    return predict_from_source(
        loaded, csv_paths=csv_paths, latest=True, device=device
    )
