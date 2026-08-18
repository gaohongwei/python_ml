"""Turn many one-feature CSV files into one aligned wide DataFrame.

Input  : speed.csv, engine_rpm.csv, ... each holding `timestamp,value`
Output : DataFrame indexed by time, one column per feature (column name = file stem)

The files may have different sampling rates and different timestamps; alignment
happens here so the rest of the code only ever sees a clean rectangular table.
"""

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from train_config import (
    DEFAULT_FILL_METHOD,
    DEFAULT_TIME_COL,
    DEFAULT_VALUE_COL,
    SUPPORTED_FILE_TYPES,
)

logger = logging.getLogger(__name__)

# Epoch numbers are ambiguous; pick the unit by magnitude of a sample value.
EPOCH_UNIT_THRESHOLDS = (
    (1e17, "ns"),
    (1e14, "us"),
    (1e11, "ms"),
    (0.0, "s"),
)


class DataLoadingError(Exception):
    """Raised when the CSV files cannot be turned into a usable dataset."""


def list_data_files(data_dir: str, file_type: str = "csv") -> List[Path]:
    """Every *.<file_type> in a directory, sorted by name for stable column order.

    `file_type` exists so the combined-frame tool can grow other formats later;
    today only "csv" is supported and anything else fails loudly.
    """
    normalized = file_type.lower().lstrip(".")
    if normalized not in SUPPORTED_FILE_TYPES:
        raise DataLoadingError(
            f"file_type={file_type!r} not supported, expected one of {SUPPORTED_FILE_TYPES}"
        )
    directory = Path(data_dir)
    if not directory.is_dir():
        raise DataLoadingError(f"not a directory: {data_dir}")
    files = sorted(directory.glob(f"*.{normalized}"))
    if not files:
        raise DataLoadingError(f"no *.{normalized} found in {data_dir}")
    return files


def list_csv_files(data_dir: str) -> List[Path]:
    """Every *.csv in a directory, sorted by name for reproducible column order."""
    return list_data_files(data_dir, "csv")


def get_feature_name(csv_path: Path) -> str:
    """File name without extension is the feature name: speed.csv -> "speed"."""
    return Path(csv_path).stem


def resolve_time_and_value_cols(
    frame: pd.DataFrame, time_col: str, value_col: str, csv_path: Path
) -> pd.DataFrame:
    """Keep exactly two columns and rename them to (time_col, value_col).

    Falls back to positional columns for headerless / differently named files,
    but only when the file has exactly two columns - guessing beyond that would
    silently pick the wrong signal.
    """
    if time_col in frame.columns and value_col in frame.columns:
        return frame[[time_col, value_col]]
    if frame.shape[1] == 2:
        logger.warning(
            f"{csv_path.name}: no {time_col!r}/{value_col!r} header, "
            f"using the two columns positionally: {list(frame.columns)}"
        )
        renamed = frame.copy()
        renamed.columns = [time_col, value_col]
        return renamed
    raise DataLoadingError(
        f"{csv_path}: expected columns {time_col!r} and {value_col!r}, "
        f"found {list(frame.columns)}"
    )


def infer_epoch_unit(values: pd.Series) -> str:
    """Guess whether numeric timestamps are in s / ms / us / ns."""
    sample = float(values.dropna().abs().max() or 0.0)
    for threshold, unit in EPOCH_UNIT_THRESHOLDS:
        if sample >= threshold:
            return unit
    return "s"


def parse_time_values(raw_values: pd.Series) -> pd.DatetimeIndex:
    """Parse ISO strings or numeric epochs into a DatetimeIndex.

    A real time index is what makes `resample()` and time-based joins possible.
    """
    numeric = pd.to_numeric(raw_values, errors="coerce")
    if numeric.notna().all():
        unit = infer_epoch_unit(numeric)
        return pd.DatetimeIndex(pd.to_datetime(numeric, unit=unit))
    parsed = pd.to_datetime(raw_values, errors="coerce")
    if parsed.isna().any():
        raise DataLoadingError("some timestamps could not be parsed")
    return pd.DatetimeIndex(parsed)


def drop_duplicate_timestamps(series: pd.Series) -> pd.Series:
    """Keep the last value per timestamp, so repeated samples cannot double-count."""
    if series.index.has_duplicates:
        before = len(series)
        series = series[~series.index.duplicated(keep="last")]
        logger.warning(
            f"{series.name}: dropped {before - len(series)} duplicate timestamps"
        )
    return series


def build_feature_series(
    raw_frame: pd.DataFrame,
    csv_path: Path,
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> pd.Series:
    """Turn an already-read `timestamp,value` frame into a time-indexed Series.

    Split out of `read_feature_csv` so the combined-frame tool can inspect a file
    once and then reuse this without reading it a second time.
    """
    csv_path = Path(csv_path)
    two_cols = resolve_time_and_value_cols(raw_frame, time_col, value_col, csv_path)
    series = pd.Series(
        pd.to_numeric(two_cols[value_col], errors="coerce").to_numpy(dtype=np.float64),
        index=parse_time_values(two_cols[time_col]),
        name=get_feature_name(csv_path),
    )
    series = drop_duplicate_timestamps(series.sort_index())
    logger.info(
        f"read {csv_path.name}: {len(series)} rows, "
        f"{series.index.min()} .. {series.index.max()}"
    )
    return series


def read_feature_csv(
    csv_path: Path,
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> pd.Series:
    """Read one feature file into a time-indexed float Series named after the file."""
    csv_path = Path(csv_path)
    return build_feature_series(pd.read_csv(csv_path), csv_path, time_col, value_col)


def read_all_feature_csvs(
    csv_paths: Sequence[Path],
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> List[pd.Series]:
    """Read every file; duplicate feature names are a hard error, not a merge."""
    series_list = [read_feature_csv(path, time_col, value_col) for path in csv_paths]
    names = [series.name for series in series_list]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise DataLoadingError(f"duplicate feature names: {sorted(duplicates)}")
    return series_list


def merge_feature_series(series_list: List[pd.Series]) -> pd.DataFrame:
    """Outer-join features on the union of their timestamps, sorted by time."""
    if not series_list:
        raise DataLoadingError("no feature series to merge")
    merged = pd.concat(series_list, axis=1, join="outer").sort_index()
    logger.info(
        f"merged {len(series_list)} features -> {merged.shape[0]} rows "
        f"({int(merged.isna().sum().sum())} holes to fill)"
    )
    return merged


def resample_frame(frame: pd.DataFrame, resample_rule: Optional[str]) -> pd.DataFrame:
    """Put every feature on one fixed grid (mean within each bucket).

    Strongly recommended when the CSVs have different sampling rates: without it
    the union of timestamps is mostly holes, and one row no longer means one
    fixed time step - which is exactly what a fixed-length window assumes.
    """
    if not resample_rule:
        return frame
    resampled = frame.resample(resample_rule).mean()
    logger.info(
        f"resampled to {resample_rule}: {frame.shape[0]} -> {resampled.shape[0]} rows"
    )
    return resampled


def fill_missing_values(
    frame: pd.DataFrame, fill_method: str = DEFAULT_FILL_METHOD
) -> pd.DataFrame:
    """Fill holes forward in time only - never backwards, that would leak."""
    if fill_method == "ffill":
        return frame.ffill()
    if fill_method == "interpolate":
        return frame.interpolate(method="time", limit_direction="forward")
    return frame


def drop_incomplete_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that still hold NaN (typically the head, before every feature started).

    Filling those with 0 would invent measurements that never happened.
    """
    cleaned = frame.dropna(axis=0, how="any")
    dropped = len(frame) - len(cleaned)
    if dropped:
        logger.info(f"dropped {dropped} rows that were still incomplete after filling")
    return cleaned


def check_frame_is_usable(frame: pd.DataFrame, min_rows: int) -> None:
    """Fail early with a message that says what to change."""
    if frame.empty:
        raise DataLoadingError(
            "no complete rows left: the files may not overlap in time"
        )
    if len(frame) < min_rows:
        raise DataLoadingError(
            f"only {len(frame)} usable rows, need at least {min_rows}: "
            f"lower seq_len/horizon, or provide more data"
        )


def prepare_feature_frame(
    frame: pd.DataFrame,
    resample_rule: Optional[str] = None,
    fill_method: str = DEFAULT_FILL_METHOD,
    min_rows: int = 1,
) -> pd.DataFrame:
    """resample -> fill -> drop incomplete -> check, on an already-merged table.

    Every data source (many CSVs, one combined CSV, an in-memory DataFrame) ends
    here, so the table the model sees is prepared exactly one way.
    """
    frame = resample_frame(frame, resample_rule)
    frame = fill_missing_values(frame, fill_method)
    frame = drop_incomplete_rows(frame)
    check_frame_is_usable(frame, min_rows)
    logger.info(f"feature frame ready: {frame.shape[0]} rows x {frame.shape[1]} features")
    return frame


def load_feature_frame(
    csv_paths: Sequence[Path],
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
    resample_rule: Optional[str] = None,
    fill_method: str = DEFAULT_FILL_METHOD,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Full read -> align -> resample -> fill -> clean pipeline.

    Training and prediction both call this one function, so a model never sees
    data prepared in a different way than it was trained on.
    """
    series_list = read_all_feature_csvs(csv_paths, time_col, value_col)
    frame = merge_feature_series(series_list)
    return prepare_feature_frame(frame, resample_rule, fill_method, min_rows)


def select_channel_cols(
    frame: pd.DataFrame, target_feature: str, include_target_as_channel: bool
) -> List[str]:
    """Decide which columns feed the network, in a fixed, saved order."""
    if target_feature not in frame.columns:
        raise DataLoadingError(
            f"target {target_feature!r} not among features: {list(frame.columns)}"
        )
    channel_cols = [
        column
        for column in frame.columns
        if include_target_as_channel or column != target_feature
    ]
    if not channel_cols:
        raise DataLoadingError(
            "no input channels left: add more CSV files, or keep the target as a channel"
        )
    return channel_cols


def frame_to_arrays(
    frame: pd.DataFrame, channel_cols: List[str], target_feature: str
):
    """Split the table into the two float32 arrays the dataset works on.

    Returns (x_values (rows, channels), y_values (rows,)).
    """
    x_values = frame[channel_cols].to_numpy(dtype=np.float32)
    y_values = frame[target_feature].to_numpy(dtype=np.float32)
    return x_values, y_values
