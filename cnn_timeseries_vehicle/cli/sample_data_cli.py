"""The sample-data CLI: make fake vehicle data, to run the tool before real logs exist.

Entry point is `run_sample_data(argv)`, which run_sample_data.py at the root calls.

It writes the same files a real recording would: one CSV per signal (speed,
throttle, coolant temperature), each with its own sampling rate. Nothing here is
measured - it is generated from a formula, which is the point:

The relation is deliberately known:

    speed[t] = SPEED_MEMORY * speed[t-1] + THROTTLE_GAIN * throttle[t - LAG_STEPS]

so a correct implementation must reach r2 > 0.9 when forecasting speed one step
ahead. A low score on this data means the windowing, alignment or scaling is
wrong - not that the model is too small.

The three files are written at *different* sampling rates on purpose, which is
what `--resample` in the train command exists for.

    python run_sample_data.py
    python run_sample_data.py --data-dir sample_data --combined-out combined.csv
"""

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

from data_pipeline.combined_frame import (
    combine_frames,
    read_frame_from_file,
    write_combined_csv,
)
from train_config import DEFAULT_DATA_DIR

# ─── generator settings ───
BASE_HZ = 20  # clock of the underlying simulation
START_EPOCH_MS = 1_704_067_200_000  # 2024-01-01 00:00:00 UTC
DEFAULT_NUM_STEPS = 12_000  # 10 minutes at 20 Hz

# ─── the known relation ───
SPEED_MEMORY = 0.97  # how much of the previous speed carries over
THROTTLE_GAIN = 0.06  # how strongly throttle pushes the speed
LAG_STEPS = 10  # throttle acts 0.5 s later (10 steps at 20 Hz)
NOISE_STD = 0.05  # small measurement noise, keeps the task non-trivial

# ─── per-feature sampling rates (keep every Nth base step) ───
FEATURE_KEEP_EVERY = {
    "speed": 1,  # 20 Hz
    "throttle": 2,  # 10 Hz
    "coolant_temp": 20,  # 1 Hz
}

# ─── coolant model ───
AMBIENT_TEMP = 25.0
TEMP_HEATING_GAIN = 0.0004
TEMP_COOLING_RATE = 0.0008


def build_timestamps_ms(num_steps: int) -> np.ndarray:
    """Epoch milliseconds on the base clock."""
    step_ms = 1000 // BASE_HZ
    return START_EPOCH_MS + np.arange(num_steps, dtype=np.int64) * step_ms


def generate_throttle(num_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth 0..100 pedal signal: a few slow sinusoids plus light noise."""
    time_steps = np.arange(num_steps) / BASE_HZ
    signal = np.zeros(num_steps)
    for period_s, weight in ((120.0, 1.0), (37.0, 0.5), (11.0, 0.25)):
        phase = rng.uniform(0, 2 * np.pi)
        signal += weight * np.sin(2 * np.pi * time_steps / period_s + phase)
    signal += rng.normal(0.0, 0.05, num_steps)
    normalized = (signal - signal.min()) / (signal.max() - signal.min())
    return normalized * 100.0


def generate_speed(throttle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Leaky integrator of a lagged throttle - the relation the model must learn."""
    speed = np.zeros_like(throttle)
    for step in range(1, len(throttle)):
        lagged_throttle = throttle[max(step - LAG_STEPS, 0)]
        speed[step] = (
            SPEED_MEMORY * speed[step - 1]
            + THROTTLE_GAIN * lagged_throttle
            + rng.normal(0.0, NOISE_STD)
        )
    return speed


def generate_coolant_temp(speed: np.ndarray) -> np.ndarray:
    """Slow thermal signal: heats with speed, cools toward ambient."""
    temperature = np.full_like(speed, AMBIENT_TEMP)
    for step in range(1, len(speed)):
        heating = TEMP_HEATING_GAIN * speed[step]
        cooling = TEMP_COOLING_RATE * (temperature[step - 1] - AMBIENT_TEMP)
        temperature[step] = temperature[step - 1] + heating - cooling
    return temperature


def write_feature_csv(
    out_dir: Path,
    feature_name: str,
    timestamps_ms: np.ndarray,
    values: np.ndarray,
    keep_every: int,
) -> Path:
    """Write one `timestamp,value` file, subsampled to its own rate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{feature_name}.csv"
    kept_times = timestamps_ms[::keep_every]
    kept_values = values[::keep_every]
    lines = ["timestamp,value"]
    lines += [f"{time},{value:.6f}" for time, value in zip(kept_times, kept_values)]
    path.write_text("\n".join(lines) + "\n")
    return path


def generate_sample_files(out_dir: str, num_steps: int, seed: int) -> List[Path]:
    """Build all features and write them out; returns the written paths."""
    rng = np.random.default_rng(seed)
    timestamps_ms = build_timestamps_ms(num_steps)

    throttle = generate_throttle(num_steps, rng)
    speed = generate_speed(throttle, rng)
    coolant_temp = generate_coolant_temp(speed)
    feature_values = {
        "speed": speed,
        "throttle": throttle,
        "coolant_temp": coolant_temp,
    }

    return [
        write_feature_csv(
            Path(out_dir), name, timestamps_ms, values, FEATURE_KEEP_EVERY[name]
        )
        for name, values in feature_values.items()
    ]


def print_written_files(paths: Sequence[Path]) -> None:
    for path in paths:
        rate_hz = BASE_HZ / FEATURE_KEEP_EVERY[path.stem]
        print(f"{path}  ({rate_hz:g} Hz)")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_sample_data.py",
        description="Write fake vehicle signal CSVs with a known, learnable relation.",
    )
    # --data-dir means the same thing in all three commands: the folder the data
    # lives in. --out-dir stays as an alias, it is what this command used to take.
    parser.add_argument(
        "--data-dir",
        "--out-dir",
        dest="data_dir",
        default=DEFAULT_DATA_DIR,
        help="folder to write the per-feature CSVs into",
    )
    parser.add_argument(
        "--combined-out",
        help="also write one wide csv here, to pass to `train --combined-csv`",
    )
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def write_combined_sample(paths: Sequence[Path], combined_out: str) -> Path:
    """Join the per-feature files into one wide csv, to exercise that input shape.

    Cells stay empty where a slower signal had no sample at that timestamp: the
    combined layout does not hide the different rates, `--resample` handles them.
    """
    combined = combine_frames([read_frame_from_file(path) for path in paths])
    return write_combined_csv(combined, combined_out)


def run_sample_data(argv=None) -> int:
    """Parse argv, write the fake data files; returns a process exit code."""
    args = parse_args(argv)
    paths = generate_sample_files(args.data_dir, args.num_steps, args.seed)
    print_written_files(paths)
    if args.combined_out:
        print(f"{write_combined_sample(paths, args.combined_out)}  (combined, all features)")
    print(
        f"\nknown relation: speed[t] = {SPEED_MEMORY} * speed[t-1] + "
        f"{THROTTLE_GAIN} * throttle[t-{LAG_STEPS}]"
    )
    return 0

