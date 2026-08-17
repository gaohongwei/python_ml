"""All tunable settings in one place, plus validation.

No hyper-parameter search here: one config = one training run.
Change a value, run again, compare the printed scores.
"""

from dataclasses import dataclass, field
from typing import List, Optional

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

# ─── optimization ───
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_EARLY_STOP_PATIENCE = 8
DEFAULT_GRAD_CLIP_MAX_NORM = 1.0  # time-series models blow up easily
DEFAULT_RANDOM_SEED = 42

# ─── output ───
DEFAULT_MODEL_FILE_NAME = "tcn_model.pth"


@dataclass
class TcnTrainConfig:
    """One training run, end to end.

    `target_feature` is the name of the CSV file (without .csv) whose future
    value we want to predict. Every other CSV becomes an input channel.
    """

    target_feature: str
    csv_paths: List[str] = field(default_factory=list)

    # CSV layout
    time_col: str = DEFAULT_TIME_COL
    value_col: str = DEFAULT_VALUE_COL

    # preprocessing
    resample_rule: Optional[str] = None  # pandas offset alias, e.g. "100ms", "1s"
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
    batch_size: int = DEFAULT_BATCH_SIZE
    max_epochs: int = DEFAULT_MAX_EPOCHS
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    early_stop_patience: int = DEFAULT_EARLY_STOP_PATIENCE
    grad_clip_max_norm: float = DEFAULT_GRAD_CLIP_MAX_NORM
    random_seed: int = DEFAULT_RANDOM_SEED

    # output
    out_dir: str = "artifacts"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"


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


def validate_config(config: TcnTrainConfig) -> List[str]:
    """Return a list of problems; empty list means the config is usable.

    Returning instead of raising lets the caller show every problem at once.
    """
    problems = []
    if not config.csv_paths:
        problems.append("csv_paths is empty: nothing to read")
    if not config.target_feature:
        problems.append("target_feature is required")
    if config.fill_method not in SUPPORTED_FILL_METHODS:
        problems.append(
            f"fill_method={config.fill_method!r} not in {SUPPORTED_FILL_METHODS}"
        )
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
        f"csv files         : {len(config.csv_paths)}",
        f"resample rule     : {config.resample_rule or '(none, use raw timestamps)'}",
        f"window            : seq_len={config.seq_len}, horizon={config.horizon}, "
        f"stride={config.stride}",
        f"split             : val={config.val_ratio}, test={config.tst_ratio}",
        f"network           : levels={config.num_levels}, "
        f"hidden={config.num_hidden_channels}, kernel={config.kernel_size}, "
        f"dropout={config.dropout}",
        f"receptive field   : {receptive_field} steps "
        f"({'covers' if receptive_field >= config.seq_len else 'SHORTER THAN'} seq_len)",
        f"optimization      : epochs={config.max_epochs}, batch={config.batch_size}, "
        f"lr={config.learning_rate}, patience={config.early_stop_patience}",
    ]
    return "\n".join(lines)
