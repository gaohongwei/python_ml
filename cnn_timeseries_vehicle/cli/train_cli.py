"""The train CLI: parse argv, call the library, print the result.

Entry point is `run_train(argv)`, which run_train.py at the root calls.

No logic of its own - everything it does lives in the modules it imports.

    python run_train.py --data-dir sample_data --target speed --resample 100ms
    python run_train.py --combined-csv combined.csv --target speed --resample 100ms

`train_from_config(config, frame=...)` is the same run without argparse, for training
straight from a combined DataFrame.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from data_pipeline.combined_frame import load_any_feature_frame
from data_pipeline.csv_feature_loader import (
    DataLoadingError,
    frame_to_arrays,
    select_channel_cols,
)
from data_pipeline.window_dataset import (
    TWindowSpec,
    WindowBuildError,
    create_data_bundle,
    get_min_required_rows,
)
from cli.runtime_setup import configure_logging, resolve_device, set_random_seed
from tcn_model import build_preprocess_dict, save_model_artifact
from train_config import (
    DEFAULT_MODEL_FILE_NAME,
    ConfigFileError,
    TcnTrainConfig,
    build_train_config,
    describe_config,
    merge_settings,
    read_config_file,
    validate_config,
)
from training.metric_scores import format_scores
from training.train_tcn import TTrainOutcome, train_tcn_model

logger = logging.getLogger(__name__)

HISTORY_FILE_NAME = "train_history.csv"
SCORES_FILE_NAME = "train_scores.json"


# Flag name (argparse dest) -> config field. One line per setting, so a new
# setting needs no new mapping code, and a flag can never write a typo'd field.
FLAG_TO_CONFIG_FIELD = {
    "data_dir": "data_dir",
    "combined_csv": "combined_csv",
    "csv": "csv_paths",
    "target": "target_feature",
    "file_type": "file_type",
    "time_col": "time_col",
    "value_col": "value_col",
    "resample": "resample_rule",
    "fill_method": "fill_method",
    "seq_len": "seq_len",
    "horizon": "horizon",
    "stride": "stride",
    "val_ratio": "val_ratio",
    "tst_ratio": "tst_ratio",
    "num_levels": "num_levels",
    "hidden": "num_hidden_channels",
    "kernel_size": "kernel_size",
    "dropout": "dropout",
    "loss": "loss_fn",
    "huber_delta": "huber_delta",
    "batch_size": "batch_size",
    "epochs": "max_epochs",
    "lr": "learning_rate",
    "weight_decay": "weight_decay",
    "patience": "early_stop_patience",
    "seed": "random_seed",
    "out_dir": "out_dir",
    "device": "device",
}


def parse_args(argv=None) -> argparse.Namespace:
    """Command line surface; every setting has a default in train_config.py.

    Nothing is required: with train_config's data dir, file type and target left
    as they are, `python run_train.py` runs. Settings are applied in three
    layers, each overriding the one before:

        train_config.py defaults  ->  --config file  ->  flags typed here

    To make that last layer work, the settings flags use argparse.SUPPRESS: a
    flag you did not type is simply absent, instead of arriving as a default that
    would silently overwrite the config file.
    """
    defaults = TcnTrainConfig()
    parser = argparse.ArgumentParser(
        prog="run_train.py",
        description="Train a TCN on vehicle signals; defaults live in train_config.py.",
        epilog=f"defaults: --data-dir {defaults.data_dir} --target "
        f"{defaults.target_feature} --resample {defaults.resample_rule} "
        f"--file-type {defaults.file_type} (see train_config.py for the rest)",
    )

    def add(*names, **kwargs) -> None:
        """Add a settings flag: absent means "leave this setting alone"."""
        parser.add_argument(*names, default=argparse.SUPPRESS, **kwargs)

    parser.add_argument(
        "--config",
        help="settings file (.json / .yaml) applied on top of train_config.py",
    )
    add(
        "--data-dir",
        help="directory of files to combine: one CSV per feature, or wide CSVs, or both",
    )
    add(
        "--combined-csv",
        help="one already-combined CSV: timestamp column plus one column per feature",
    )
    add("--csv", nargs="+", help="explicit CSV file list")
    add("--target", help="feature to forecast, e.g. speed")

    add("--file-type", help="which files --data-dir is combined from (csv)")
    add("--time-col")
    add("--value-col")
    add(
        "--resample",
        help="pandas offset alias to put all features on one grid, e.g. 100ms, 1s",
    )
    add("--fill-method")
    parser.add_argument(
        "--drop-target-channel",
        action="store_true",
        default=argparse.SUPPRESS,
        help="do not feed the target's own history to the model",
    )

    add("--seq-len", type=int)
    add("--horizon", type=int)
    add("--stride", type=int)
    add("--val-ratio", type=float)
    add("--tst-ratio", type=float)

    add("--num-levels", type=int)
    add("--hidden", type=int)
    add("--kernel-size", type=int)
    add("--dropout", type=float)

    add("--loss", help="mse | mae | huber - what one unit of error costs")
    add("--huber-delta", type=float, help="error above which huber switches to mae")
    add("--batch-size", type=int)
    add("--epochs", type=int)
    add("--lr", type=float)
    add("--weight-decay", type=float)
    add("--patience", type=int)
    add("--seed", type=int)

    add("--out-dir")
    add("--device", help="auto | cpu | cuda | mps")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> dict:
    """Only the settings actually typed on the command line."""
    given = vars(args)
    settings = {
        field_name: given[flag]
        for flag, field_name in FLAG_TO_CONFIG_FIELD.items()
        if flag in given
    }
    if "csv_paths" in settings:
        settings["csv_paths"] = [str(Path(path)) for path in settings["csv_paths"]]
    if "drop_target_channel" in given:
        settings["include_target_as_channel"] = not given["drop_target_channel"]
    return settings


def build_config(args: argparse.Namespace) -> TcnTrainConfig:
    """train_config.py defaults, then the --config file, then the typed flags.

    Source flags are carried over as given; `load_any_feature_frame` owns the
    precedence between them, so training and prediction cannot disagree on it.
    """
    file_settings = read_config_file(args.config) if args.config else {}
    settings = merge_settings(file_settings, settings_from_args(args))
    return build_train_config(settings)


def build_data_bundle(
    config: TcnTrainConfig,
    spec: TWindowSpec,
    frame: Optional[pd.DataFrame] = None,
):
    """Any source -> aligned frame -> arrays -> loaders. Returns (bundle, channel_cols).

    Pass `frame` to train from a combined DataFrame you already hold in memory;
    leave it out and the config's source (combined csv / data dir / csv list) is
    read instead. Either way the same preprocessing runs.
    """
    frame = load_any_feature_frame(
        frame=frame,
        combined_csv=config.combined_csv,
        data_dir=config.data_dir,
        csv_paths=config.csv_paths,
        file_type=config.file_type,
        time_col=config.time_col,
        value_col=config.value_col,
        resample_rule=config.resample_rule,
        fill_method=config.fill_method,
        min_rows=get_min_required_rows(spec, config.val_ratio, config.tst_ratio),
    )
    channel_cols = select_channel_cols(
        frame, config.target_feature, config.include_target_as_channel
    )
    x_values, y_values = frame_to_arrays(frame, channel_cols, config.target_feature)
    bundle = create_data_bundle(
        x_values=x_values,
        y_values=y_values,
        spec=spec,
        batch_size=config.batch_size,
        val_ratio=config.val_ratio,
        tst_ratio=config.tst_ratio,
        random_seed=config.random_seed,
    )
    return bundle, channel_cols


def write_history_csv(out_dir: Path, outcome: TTrainOutcome) -> Path:
    """Per-epoch losses, for plotting the learning curve later."""
    path = out_dir / HISTORY_FILE_NAME
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "is_best"])
        for record in outcome.history:
            writer.writerow(
                [record.epoch, record.train_loss, record.val_loss, int(record.is_best)]
            )
    return path


def write_scores_json(out_dir: Path, config: TcnTrainConfig, outcome: TTrainOutcome) -> Path:
    """Scores plus the settings that produced them, for run-to-run comparison."""
    path = out_dir / SCORES_FILE_NAME
    payload = {
        "target_feature": config.target_feature,
        "seq_len": config.seq_len,
        "horizon": config.horizon,
        "loss_fn": config.loss_fn,
        "best_epoch": outcome.best_epoch,
        "best_val_loss": outcome.best_val_loss,
        "val_scores": outcome.val_scores,
        "test_scores": outcome.test_scores,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def print_summary(outcome: TTrainOutcome, model_path: str) -> None:
    """Final block the user actually reads."""
    print("\n=== training done ===")
    print(f"best epoch : {outcome.best_epoch}")
    print(f"val        : {format_scores(outcome.val_scores)}")
    if outcome.test_scores:
        print(f"test       : {format_scores(outcome.test_scores)}")
    print(f"model      : {model_path}")


def train_from_config(config: TcnTrainConfig, frame: Optional[pd.DataFrame] = None) -> int:
    """Train, save and report from a config - the whole run, without argparse.

    This is also the library entry point for a combined DataFrame:

        train_from_config(TcnTrainConfig(target_feature="speed", resample_rule="100ms"),
                     frame=my_combined_df)
    """
    problems = validate_config(config, has_frame=frame is not None)
    if problems:
        for problem in problems:
            logger.error(f"config: {problem}")
        return 2

    # flush: logs go to stderr, so an unflushed stdout block would show up last
    print(describe_config(config) + "\n", flush=True)
    set_random_seed(config.random_seed)
    device = resolve_device(config.device)
    logger.info(f"device: {device}")

    spec = TWindowSpec(config.seq_len, config.horizon, config.stride)
    try:
        bundle, channel_cols = build_data_bundle(config, spec, frame)
    except (DataLoadingError, WindowBuildError) as error:
        # Expected user-facing problems (bad files, too little data): no traceback.
        logger.error(f"data: {error}")
        return 1
    outcome = train_tcn_model(bundle, config, device)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_model_artifact(
        model_path=str(out_dir / DEFAULT_MODEL_FILE_NAME),
        model=outcome.model,
        arch=outcome.arch,
        channel_cols=channel_cols,
        target_feature=config.target_feature,
        spec=spec,
        scaler_x=bundle.scaler_x,
        scaler_y=bundle.scaler_y,
        preprocess=build_preprocess_dict(config),
        scores={"val": outcome.val_scores, "test": outcome.test_scores},
    )
    write_history_csv(out_dir, outcome)
    write_scores_json(out_dir, config, outcome)
    print_summary(outcome, model_path)
    return 0


def run_train(argv=None) -> int:
    """Parse argv, build the config, train; returns a process exit code."""
    args = parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = build_config(args)
    except ConfigFileError as error:
        # A bad --config file is a user problem, not a crash.
        logger.error(f"config: {error}")
        return 2
    return train_from_config(config)

