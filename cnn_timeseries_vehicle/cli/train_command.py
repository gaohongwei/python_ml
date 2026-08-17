"""The `train` command: parse arguments, call the library, print the result.

No logic of its own - everything it does lives in the modules it imports.

    python run.py train --data-dir sample_data --target speed --resample 100ms
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import List

from data_pipeline.csv_feature_loader import (
    DataLoadingError,
    frame_to_arrays,
    list_csv_files,
    load_feature_frame,
    select_channel_cols,
)
from data_pipeline.window_dataset import (
    TWindowSpec,
    WindowBuildError,
    create_data_bundle,
    get_min_required_rows,
)
from cli.runtime_setup import configure_logging, resolve_device, set_random_seed
from tcn_model.model_artifact import build_preprocess_dict, save_model_artifact
from train_config import (
    DEFAULT_MODEL_FILE_NAME,
    TcnTrainConfig,
    describe_config,
    validate_config,
)
from training.metric_scores import format_scores
from training.train_tcn import TTrainOutcome, train_tcn_model

logger = logging.getLogger(__name__)

HISTORY_FILE_NAME = "train_history.csv"
SCORES_FILE_NAME = "train_scores.json"


def parse_args(argv=None) -> argparse.Namespace:
    """Command line surface; every default comes from train_config."""
    defaults = TcnTrainConfig(target_feature="")
    parser = argparse.ArgumentParser(
        prog="run.py train",
        description="Train a TCN on per-feature CSV files (timestamp,value).",
    )

    parser.add_argument("--data-dir", help="directory holding one CSV per feature")
    parser.add_argument("--csv", nargs="+", default=[], help="explicit CSV file list")
    parser.add_argument("--target", required=True, help="feature to forecast, e.g. speed")

    parser.add_argument("--time-col", default=defaults.time_col)
    parser.add_argument("--value-col", default=defaults.value_col)
    parser.add_argument(
        "--resample",
        default=defaults.resample_rule,
        help="pandas offset alias to put all features on one grid, e.g. 100ms, 1s",
    )
    parser.add_argument("--fill-method", default=defaults.fill_method)
    parser.add_argument(
        "--drop-target-channel",
        action="store_true",
        help="do not feed the target's own history to the model",
    )

    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--horizon", type=int, default=defaults.horizon)
    parser.add_argument("--stride", type=int, default=defaults.stride)
    parser.add_argument("--val-ratio", type=float, default=defaults.val_ratio)
    parser.add_argument("--tst-ratio", type=float, default=defaults.tst_ratio)

    parser.add_argument("--num-levels", type=int, default=defaults.num_levels)
    parser.add_argument("--hidden", type=int, default=defaults.num_hidden_channels)
    parser.add_argument("--kernel-size", type=int, default=defaults.kernel_size)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)

    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--patience", type=int, default=defaults.early_stop_patience)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)

    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument("--device", default=defaults.device, help="auto | cpu | cuda | mps")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def collect_csv_paths(args: argparse.Namespace) -> List[str]:
    """Explicit --csv wins; otherwise take every CSV in --data-dir."""
    if args.csv:
        return [str(Path(path)) for path in args.csv]
    if args.data_dir:
        return [str(path) for path in list_csv_files(args.data_dir)]
    raise SystemExit("provide --data-dir or --csv")


def build_config(args: argparse.Namespace) -> TcnTrainConfig:
    """Map parsed arguments onto the config dataclass."""
    return TcnTrainConfig(
        target_feature=args.target,
        csv_paths=collect_csv_paths(args),
        time_col=args.time_col,
        value_col=args.value_col,
        resample_rule=args.resample,
        fill_method=args.fill_method,
        include_target_as_channel=not args.drop_target_channel,
        seq_len=args.seq_len,
        horizon=args.horizon,
        stride=args.stride,
        val_ratio=args.val_ratio,
        tst_ratio=args.tst_ratio,
        num_levels=args.num_levels,
        num_hidden_channels=args.hidden,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.patience,
        random_seed=args.seed,
        out_dir=args.out_dir,
        device=args.device,
    )


def build_data_bundle(config: TcnTrainConfig, spec: TWindowSpec):
    """CSV files -> aligned frame -> arrays -> loaders. Returns (bundle, channel_cols)."""
    frame = load_feature_frame(
        csv_paths=config.csv_paths,
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


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config = build_config(args)

    problems = validate_config(config)
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
        bundle, channel_cols = build_data_bundle(config, spec)
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

