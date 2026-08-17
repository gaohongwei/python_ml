"""The `predict` command: load an artifact, forecast, print or save the table.

    python run.py predict --model artifacts/tcn_model.pth --data-dir sample_data
    python run.py predict --model artifacts/tcn_model.pth --data-dir sample_data --latest
"""

import argparse
import logging
from pathlib import Path
from typing import List

from data_pipeline.csv_feature_loader import DataLoadingError, list_csv_files
from data_pipeline.window_dataset import WindowBuildError
from inference.predict_tcn import predict_from_csv_files, predict_latest
from cli.runtime_setup import configure_logging, resolve_device
from tcn_model.model_artifact import load_model_artifact

logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py predict",
        description="Predict with a trained TCN artifact.",
    )
    parser.add_argument("--model", required=True, help="path to tcn_model.pth")
    parser.add_argument("--data-dir", help="directory holding one CSV per feature")
    parser.add_argument("--csv", nargs="+", default=[], help="explicit CSV file list")
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


def collect_csv_paths(args: argparse.Namespace) -> List[str]:
    """Same resolution rule as training: --csv wins over --data-dir."""
    if args.csv:
        return [str(Path(path)) for path in args.csv]
    if args.data_dir:
        return [str(path) for path in list_csv_files(args.data_dir)]
    raise SystemExit("provide --data-dir or --csv")


def print_latest_forecast(target_feature: str, predictions) -> None:
    """One line per forecast step, in the target's original unit."""
    print(f"\n=== latest forecast for {target_feature} ===")
    for step, value in enumerate(predictions, start=1):
        print(f"step {step:3d}: {value:.4f}")


def main(argv=None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)

    device = resolve_device(args.device)
    loaded = load_model_artifact(args.model, device)
    csv_paths = collect_csv_paths(args)

    try:
        if args.latest:
            predictions = predict_latest(loaded, csv_paths, device)
            print_latest_forecast(loaded.target_feature, predictions)
            return 0
        result = predict_from_csv_files(loaded, csv_paths, args.stride, device)
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

