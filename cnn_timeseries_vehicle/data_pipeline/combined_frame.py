"""Build one combined (wide) DataFrame from whatever the caller happens to have.

Three shapes of input all end as the same table - time index, one column per
feature - and are then handed to the same `prepare_feature_frame` step:

- a directory of files          -> `build_combined_frame(data_dir, "csv")`
- one already-combined CSV      -> `read_combined_csv(path)`
- an in-memory DataFrame        -> `coerce_time_index(frame)`

A file inside the directory may itself be either layout, and both are detected
per file, so a directory of `timestamp,value` feature files, a directory holding
one wide export, or a mix of the two all work:

    long / per-feature (column name comes from the file name)
        timestamp,value
        1704067200000,0.0

    wide / combined (column names come from the header)
        timestamp,speed,throttle
        1704067200000,0.0,12.5

`load_any_feature_frame` is the single door the CLI and inference use: give it
one source, get back the cleaned table.
"""

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from data_pipeline.csv_feature_loader import (
    DataLoadingError,
    build_feature_series,
    list_data_files,
    load_feature_frame,
    merge_feature_series,
    parse_time_values,
    prepare_feature_frame,
)
from train_config import DEFAULT_FILL_METHOD, DEFAULT_TIME_COL, DEFAULT_VALUE_COL

logger = logging.getLogger(__name__)

# What a file's columns say about its layout.
LAYOUT_LONG = "long"  # timestamp,value -> one feature, named after the file
LAYOUT_WIDE = "wide"  # timestamp,a,b,c -> several features, named by the header


def classify_csv_layout(
    frame: pd.DataFrame, time_col: str, value_col: str, csv_path: Path
) -> str:
    """Decide whether a read file holds one feature or several.

    The header decides: a lone `value` column next to the timestamp is the
    per-feature layout, anything else is a combined export. Headerless files are
    only accepted with exactly two columns, where the order is unambiguous.
    """
    columns = list(frame.columns)
    if time_col not in columns:
        if len(columns) == 2:
            return LAYOUT_LONG  # build_feature_series warns about the fallback
        raise DataLoadingError(
            f"{csv_path}: no {time_col!r} column and {len(columns)} columns "
            f"({columns}): add a header, or pass --time-col"
        )
    feature_columns = [column for column in columns if column != time_col]
    if not feature_columns:
        raise DataLoadingError(f"{csv_path}: only a {time_col!r} column, no values")
    if feature_columns == [value_col]:
        return LAYOUT_LONG
    return LAYOUT_WIDE


def normalize_time_index(frame: pd.DataFrame, csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Sort by time and keep the last row per timestamp.

    Same rule as the per-feature path: a repeated timestamp is one measurement
    reported twice, not two steps, so it must not become two rows.
    """
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        before = len(frame)
        frame = frame[~frame.index.duplicated(keep="last")]
        label = csv_path.name if csv_path else "combined frame"
        logger.warning(f"{label}: dropped {before - len(frame)} duplicate timestamps")
    return frame


def build_wide_frame(
    raw_frame: pd.DataFrame,
    time_col: str,
    csv_path: Path,
) -> pd.DataFrame:
    """Turn a read `timestamp,a,b,...` frame into a time-indexed float table."""
    feature_columns = [column for column in raw_frame.columns if column != time_col]
    wide = pd.DataFrame(
        {
            column: pd.to_numeric(raw_frame[column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            for column in feature_columns
        },
        index=parse_time_values(raw_frame[time_col]),
    )
    wide = normalize_time_index(wide, csv_path)
    logger.info(
        f"read {csv_path.name}: {len(wide)} rows x {wide.shape[1]} features "
        f"{list(wide.columns)}, {wide.index.min()} .. {wide.index.max()}"
    )
    return wide


def read_frame_from_file(
    csv_path: Union[str, Path],
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> pd.DataFrame:
    """Read one file into a time-indexed frame, whatever layout it uses."""
    csv_path = Path(csv_path)
    raw_frame = pd.read_csv(csv_path)
    layout = classify_csv_layout(raw_frame, time_col, value_col, csv_path)
    if layout == LAYOUT_LONG:
        series = build_feature_series(raw_frame, csv_path, time_col, value_col)
        return series.to_frame()
    return build_wide_frame(raw_frame, time_col, csv_path)


def check_no_duplicate_features(frames: Sequence[pd.DataFrame]) -> None:
    """A feature must come from exactly one place, otherwise the join is a guess."""
    names = [str(column) for frame in frames for column in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise DataLoadingError(
            f"duplicate feature names across files: {duplicates} "
            f"(a feature must be defined once)"
        )


def combine_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Outer-join several time-indexed frames on the union of their timestamps."""
    if not frames:
        raise DataLoadingError("no frames to combine")
    check_no_duplicate_features(frames)
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames, axis=1, join="outer").sort_index()
    logger.info(
        f"combined {len(frames)} sources -> {combined.shape[0]} rows x "
        f"{combined.shape[1]} features ({int(combined.isna().sum().sum())} holes to fill)"
    )
    return combined


def build_combined_frame(
    data_dir: str,
    file_type: str = "csv",
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> pd.DataFrame:
    """Directory + file type -> one combined DataFrame, ready for the next step.

    Reads every *.<file_type> in `data_dir` (sorted, so column order is stable),
    detects each file's layout, and outer-joins everything on time. No resampling
    or filling happens here - that is `prepare_feature_frame`'s job.
    """
    paths = list_data_files(data_dir, file_type)
    logger.info(f"combining {len(paths)} {file_type} file(s) from {data_dir}")
    frames = [read_frame_from_file(path, time_col, value_col) for path in paths]
    return combine_frames(frames)


def read_combined_csv(
    csv_path: Union[str, Path],
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
) -> pd.DataFrame:
    """One already-combined CSV -> time-indexed frame, one column per feature."""
    return read_frame_from_file(csv_path, time_col, value_col)


def coerce_time_index(
    frame: pd.DataFrame, time_col: str = DEFAULT_TIME_COL
) -> pd.DataFrame:
    """Accept a caller's DataFrame: index it by time and make the values float.

    Either the frame is already time-indexed, or it carries a `time_col` column
    that gets parsed the same way a CSV's timestamps would be.
    """
    if frame.empty:
        raise DataLoadingError("the given DataFrame is empty")
    working = frame.copy()
    if time_col in working.columns:
        index = parse_time_values(working[time_col])
        working = working.drop(columns=[time_col])
        working.index = index
    elif isinstance(working.index, pd.DatetimeIndex):
        pass
    else:
        index = parse_time_values(pd.Series(working.index))
        working.index = index
    if not len(working.columns):
        raise DataLoadingError("the given DataFrame holds no feature columns")
    working = working.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return normalize_time_index(working)


def load_any_feature_frame(
    frame: Optional[pd.DataFrame] = None,
    combined_csv: Optional[str] = None,
    data_dir: Optional[str] = None,
    csv_paths: Sequence[Union[str, Path]] = (),
    file_type: str = "csv",
    time_col: str = DEFAULT_TIME_COL,
    value_col: str = DEFAULT_VALUE_COL,
    resample_rule: Optional[str] = None,
    fill_method: str = DEFAULT_FILL_METHOD,
    min_rows: int = 1,
) -> pd.DataFrame:
    """One source in, one cleaned table out; sources are tried most specific first.

    1. `frame`        - a combined DataFrame the caller already has
    2. `combined_csv` - one wide CSV holding every feature
    3. `csv_paths`    - an explicit per-feature file list
    4. `data_dir`     - every *.<file_type> in a directory, combined here

    Preprocessing (resample / fill / drop) is identical whichever source wins, so
    a model trained from one shape can predict from another.
    """
    if frame is not None:
        return prepare_feature_frame(
            coerce_time_index(frame, time_col), resample_rule, fill_method, min_rows
        )
    if combined_csv:
        return prepare_feature_frame(
            read_combined_csv(combined_csv, time_col, value_col),
            resample_rule,
            fill_method,
            min_rows,
        )
    if csv_paths:
        return load_feature_frame(
            csv_paths=list(csv_paths),
            time_col=time_col,
            value_col=value_col,
            resample_rule=resample_rule,
            fill_method=fill_method,
            min_rows=min_rows,
        )
    if data_dir:
        return prepare_feature_frame(
            build_combined_frame(data_dir, file_type, time_col, value_col),
            resample_rule,
            fill_method,
            min_rows,
        )
    raise DataLoadingError(
        "no data source: pass a DataFrame, a combined csv, a data dir, or csv paths"
    )


def write_combined_csv(
    frame: pd.DataFrame, out_path: Union[str, Path], time_col: str = DEFAULT_TIME_COL
) -> Path:
    """Save a combined frame in the layout `read_combined_csv` reads back.

    Holes stay empty rather than being filled here: filling belongs to the one
    place that owns it, and an empty cell is honest about what was measured.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_frame = frame.copy()
    out_frame.index.name = time_col
    out_frame.to_csv(path)
    logger.info(f"combined csv written to {path}")
    return path
