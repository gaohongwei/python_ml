"""All tunable settings in one place, plus validation.

No hyper-parameter search here: one config = one training run.
Change a value, run again, compare the printed scores.

The block right below is the part you edit for a new dataset: where the data is,
what kind of files it is, and which signal to predict. With those set,
`python run_train.py` needs no flags; every flag just overrides one of them.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

# ─── what to train on (edit these first) ───
DEFAULT_DATA_DIR = "sample_data"  # folder of data files, combined into one table
DEFAULT_FILE_TYPE = "csv"  # which files inside that folder to read
DEFAULT_TARGET_FEATURE = "speed"  # the signal to forecast
DEFAULT_COMBINED_CSV = None  # set instead of DATA_DIR for one wide csv
DEFAULT_RESAMPLE_RULE = "100ms"  # one row per 100 ms; None keeps raw timestamps

SUPPORTED_FILE_TYPES = ("csv",)

# ─── CSV layout ───
DEFAULT_TIME_COL = "timestamp"
DEFAULT_VALUE_COL = "value"

# ─── preprocessing ───
DEFAULT_FILL_METHOD = "ffill"  # "ffill" | "interpolate" | "none"
SUPPORTED_FILL_METHODS = ("ffill", "interpolate", "none")

# ─── windowing ───
DEFAULT_SEQ_LEN = 64  # how many past steps the model looks at
DEFAULT_HORIZON = 1  # how many future steps it predicts
DEFAULT_STRIDE = 1  # gap between two consecutive window starts

# ─── chronological split ───
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TST_RATIO = 0.15

# ─── network shape ───
# 5 levels x kernel 3 reach 125 steps, enough to cover the default seq_len of 64.
DEFAULT_NUM_LEVELS = 5
DEFAULT_NUM_HIDDEN_CHANNELS = 32
DEFAULT_KERNEL_SIZE = 3
DEFAULT_DROPOUT = 0.1

# ─── what "wrong" costs during training ───
# mse   : squares the error, so one big miss outweighs many small ones
# mae   : counts the error as it is, so rare spikes pull the fit around less
# huber : mse near zero, mae beyond `huber_delta` - the usual compromise
DEFAULT_LOSS_FN = "mse"
SUPPORTED_LOSS_FNS = ("mse", "mae", "huber")
DEFAULT_HUBER_DELTA = 1.0  # in scaled units (~1 standard deviation of the target)

# ─── optimization ───
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_GRAD_CLIP_MAX_NORM = 1.0  # time-series models blow up easily
DEFAULT_RANDOM_SEED = 42

# ─── output ───
DEFAULT_OUT_DIR = "artifacts"
DEFAULT_MODEL_FILE_NAME = "tcn_model.pth"
DEFAULT_MODEL_PATH = f"{DEFAULT_OUT_DIR}/{DEFAULT_MODEL_FILE_NAME}"


@dataclass
class TcnTrainConfig:
    """One training run, end to end.

    `target_feature` is the name of the column to predict - the CSV file name
    (without .csv) in the per-feature layout, or the column header in a combined
    table. Every other feature becomes an input channel.

    The data source is whichever of these is set first: `combined_csv`,
    `csv_paths`, `data_dir`. An in-memory DataFrame is not a config field; pass
    it straight to `train_from_config(config, frame=...)`.
    """

    target_feature: str = DEFAULT_TARGET_FEATURE
    combined_csv: Optional[str] = DEFAULT_COMBINED_CSV  # one wide csv, every feature
    csv_paths: List[str] = field(default_factory=list)  # explicit file list
    data_dir: Optional[str] = DEFAULT_DATA_DIR  # combine *.file_type from here

    # CSV layout
    file_type: str = DEFAULT_FILE_TYPE
    time_col: str = DEFAULT_TIME_COL
    value_col: str = DEFAULT_VALUE_COL

    # preprocessing
    resample_rule: Optional[str] = DEFAULT_RESAMPLE_RULE  # offset alias: "100ms", "1s"
    fill_method: str = DEFAULT_FILL_METHOD
    include_target_as_channel: bool = True  # target history is a strong predictor

    # windowing
    seq_len: int = DEFAULT_SEQ_LEN
    horizon: int = DEFAULT_HORIZON
    stride: int = DEFAULT_STRIDE

    # split
    val_ratio: float = DEFAULT_VAL_RATIO
    tst_ratio: float = DEFAULT_TST_RATIO

    # network
    num_levels: int = DEFAULT_NUM_LEVELS
    num_hidden_channels: int = DEFAULT_NUM_HIDDEN_CHANNELS
    kernel_size: int = DEFAULT_KERNEL_SIZE
    dropout: float = DEFAULT_DROPOUT

    # optimization
    loss_fn: str = DEFAULT_LOSS_FN  # "mse" | "mae" | "huber"
    huber_delta: float = DEFAULT_HUBER_DELTA  # only used by loss_fn="huber"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM
    random_seed: int = DEFAULT_RANDOM_SEED

    # output
    out_dir: str = DEFAULT_OUT_DIR
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"


class ConfigFileError(Exception):
    """Raised when a --config file cannot be turned into a TcnTrainConfig."""


def get_config_field_names() -> List[str]:
    """Every settable field name, i.e. every key a config file may use."""
    return [config_field.name for config_field in fields(TcnTrainConfig)]


def read_config_file(config_path: str) -> dict:
    """Read a .json / .yaml / .yml settings file into a plain dict.

    A file keeps one experiment's settings next to its results, which a shell
    command in someone's history does not.
    """
    path = Path(config_path)
    if not path.is_file():
        raise ConfigFileError(f"config file not found: {config_path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional: only needed for YAML configs
        except ImportError as error:  # pragma: no cover - depends on the install
            raise ConfigFileError(
                f"{config_path}: reading YAML needs pyyaml (pip install pyyaml), "
                f"or use a .json file"
            ) from error
        settings = yaml.safe_load(text) or {}
    elif path.suffix.lower() == ".json":
        settings = json.loads(text or "{}")
    else:
        raise ConfigFileError(
            f"{config_path}: unsupported config format {path.suffix!r}, "
            f"expected .json, .yaml or .yml"
        )
    if not isinstance(settings, dict):
        raise ConfigFileError(f"{config_path}: expected a mapping of setting -> value")
    return settings


def build_train_config(settings: dict) -> TcnTrainConfig:
    """Turn a settings dict into a config; an unknown key is an error.

    Silently ignoring a misspelled key is how a run ends up not doing what its
    config file says it does.
    """
    known = get_config_field_names()
    unknown = [key for key in settings if key not in known]
    if unknown:
        raise ConfigFileError(
            f"unknown setting(s): {sorted(unknown)}; known settings are {known}"
        )
    return TcnTrainConfig(**settings)


def merge_settings(*layers: dict) -> dict:
    """Later layers win, key by key: the config file first, typed flags last.

    A `None` in a layer is kept, not skipped: `resample_rule: null` in a config
    file is a decision ("use the raw timestamps"), not a missing value. Layers
    only carry keys that were actually set, which is what makes that safe.
    """
    merged: dict = {}
    for layer in layers:
        merged.update(layer)
    return merged


def get_network_arch(config: TcnTrainConfig) -> dict:
    """The subset of the config that defines the network shape.

    Saved next to the weights so prediction can rebuild the exact same model.
    """
    return {
        "num_levels": config.num_levels,
        "num_hidden_channels": config.num_hidden_channels,
        "kernel_size": config.kernel_size,
        "dropout": config.dropout,
    }


def get_receptive_field(num_levels: int, kernel_size: int) -> int:
    """How many past steps the last output actually sees.

    Dilations double per level (1, 2, 4, ...) and each level has 2 conv layers,
    so the field is 1 + 2 * (k - 1) * (2^levels - 1).
    """
    return 1 + 2 * (kernel_size - 1) * (2**num_levels - 1)


def has_data_source(config: TcnTrainConfig) -> bool:
    """True when the config names something to read (a DataFrame is passed separately)."""
    return bool(config.combined_csv or config.data_dir or config.csv_paths)


def describe_data_source(config: TcnTrainConfig) -> str:
    """Name the source that will win, in the order `load_any_feature_frame` tries."""
    if config.combined_csv:
        return f"combined csv {config.combined_csv}"
    if config.csv_paths:
        return f"{len(config.csv_paths)} explicit csv file(s)"
    if config.data_dir:
        return f"{config.file_type} files in {config.data_dir}"
    return "in-memory combined DataFrame"


def validate_config(config: TcnTrainConfig, has_frame: bool = False) -> List[str]:
    """Return a list of problems; empty list means the config is usable.

    Returning instead of raising lets the caller show every problem at once.
    `has_frame` says a DataFrame is being passed in, so no path is needed.
    """
    problems = []
    if not has_frame and not has_data_source(config):
        problems.append(
            "no data source: set combined_csv, data_dir or csv_paths, or pass a DataFrame"
        )
    if not config.target_feature:
        problems.append("target_feature is required")
    if config.file_type.lower().lstrip(".") not in SUPPORTED_FILE_TYPES:
        problems.append(f"file_type={config.file_type!r} not in {SUPPORTED_FILE_TYPES}")
    if config.fill_method not in SUPPORTED_FILL_METHODS:
        problems.append(
            f"fill_method={config.fill_method!r} not in {SUPPORTED_FILL_METHODS}"
        )
    if config.loss_fn not in SUPPORTED_LOSS_FNS:
        problems.append(f"loss_fn={config.loss_fn!r} not in {SUPPORTED_LOSS_FNS}")
    if config.huber_delta <= 0.0:
        problems.append("huber_delta must be > 0")
    if config.seq_len < 2:
        problems.append("seq_len must be >= 2")
    if config.horizon < 1:
        problems.append("horizon must be >= 1")
    if config.stride < 1:
        problems.append("stride must be >= 1")
    if not 0.0 < config.val_ratio < 1.0:
        problems.append("val_ratio must be in (0, 1)")
    if not 0.0 <= config.tst_ratio < 1.0:
        problems.append("tst_ratio must be in [0, 1)")
    if config.val_ratio + config.tst_ratio >= 0.9:
        problems.append("val_ratio + tst_ratio leaves almost no training data")
    if config.kernel_size < 2:
        problems.append("kernel_size must be >= 2 (kernel 1 cannot see the past)")
    if not 0.0 <= config.dropout < 1.0:
        problems.append("dropout must be in [0, 1)")
    return problems


def describe_config(config: TcnTrainConfig) -> str:
    """Human-readable one-block summary, printed at the start of a run."""
    receptive_field = get_receptive_field(config.num_levels, config.kernel_size)
    lines = [
        f"target            : {config.target_feature}",
        f"data source       : {describe_data_source(config)}",
        f"resample rule     : {config.resample_rule or '(none, use raw timestamps)'}",
        f"window            : seq_len={config.seq_len}, horizon={config.horizon}, "
        f"stride={config.stride}",
        f"split             : val={config.val_ratio}, test={config.tst_ratio}",
        f"network           : levels={config.num_levels}, "
        f"hidden={config.num_hidden_channels}, kernel={config.kernel_size}, "
        f"dropout={config.dropout}",
        f"receptive field   : {receptive_field} steps "
        f"({'covers' if receptive_field >= config.seq_len else 'SHORTER THAN'} seq_len)",
        f"loss              : {config.loss_fn}"
        + (f" (delta={config.huber_delta})" if config.loss_fn == "huber" else ""),
        f"optimization      : epochs={config.max_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patience={config.early_stop_patience}",
    ]
    return "\n".join(lines)
