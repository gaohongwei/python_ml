"""The predict CLI: parse argv, load an artifact, forecast, print or save the table.

Entry point is `run_predict(argv)`, which run_predict.py at the root calls.

    python run_predict.py --latest
    python run_predict.py --combined-csv combined.csv --out predictions.csv
    python run_predict.py --model artifacts/tcn_model.pth --data-dir my_logs

Deliberately no --config here: a model file plus data is all prediction needs.
Which columns, in which order, at which sampling rate, scaled by which scaler -
all of that was decided during training and travels inside the artifact. Reading
a training config here would invite editing seq_len or loss_fn and expecting the
forecast to change, when the artifact would silently keep winning.
"""

import argparse
import logging
from pathlib import Path

from data_pipeline.csv_feature_loader import DataLoadingError
from data_pipeline.window_dataset import WindowBuildError
from inference.predict_tcn import predict_from_source
from cli.runtime_setup import configure_logging, resolve_device
from tcn_model import load_model_artifact
from train_config import (
    DEFAULT_COMBINED_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_FILE_TYPE,
    DEFAULT_MODEL_PATH,
)

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    """Three things only: which model, which data, and how to run.

    The data flags mirror training's, and default to the same values in
    train_config.py, so `python run_predict.py --latest` works after a plain
    `python run_train.py`.
    """
    parser = argparse.ArgumentParser(
        prog="run_predict.py",
        description="Forecast with a trained model file; the artifact carries the rest.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH, help="path to tcn_model.pth"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="directory of files to combine: one CSV per feature, or wide CSVs, or both",
    )
    parser.add_argument(
        "--combined-csv",
        default=DEFAULT_COMBINED_CSV,
        help="one already-combined CSV: timestamp column plus one column per feature",
    )
    parser.add_argument("--csv", nargs="+", default=[], help="explicit CSV file list")
    parser.add_argument(
        "--file-type",
        default=DEFAULT_FILE_TYPE,
        help="which files --data-dir is combined from (csv)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="only forecast from the most recent window",
    )
    parser.add_argument("--stride", type=int, default=1, help="gap between windows")
    parser.add_argument("--out", help="write the prediction table to this CSV")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def print_latest_forecast(target_feature: str, predictions) -> None:
    """One line per forecast step, in the target's original unit."""
    print(f"\n=== latest forecast for {target_feature} ===")
    for step, value in enumerate(predictions, start=1):
        print(f"step {step:3d}: {value:.4f}")


def run_predict(argv=None) -> int:
    """Parse argv, load the model, forecast; returns a process exit code."""
    args = parse_args(argv)
    configure_logging(args.verbose)

    if not (args.combined_csv or args.data_dir or args.csv):
        raise SystemExit("provide --combined-csv, --data-dir or --csv")

    device = resolve_device(args.device)
    loaded = load_model_artifact(args.model, device)

    try:
        result = predict_from_source(
            loaded,
            combined_csv=args.combined_csv,
            data_dir=args.data_dir,
            csv_paths=[str(Path(path)) for path in args.csv],
            file_type=args.file_type,
            stride=args.stride,
            latest=args.latest,
            device=device,
        )
        if args.latest:
            print_latest_forecast(loaded.target_feature, result)
            return 0
    except (DataLoadingError, WindowBuildError) as error:
        # Missing channel or fewer rows than seq_len: report, don't dump a stack.
        logger.error(f"data: {error}")
        return 1
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out, index=False)
        print(f"{len(result)} predictions written to {args.out}")
    else:
        print(result.head(20).to_string(index=False))
        print(f"... {len(result)} rows total (use --out to save)")
    return 0

