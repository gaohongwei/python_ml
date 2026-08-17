# cnn_vehicle_data — standalone TCN trainer for per-feature CSV files

Trains a Temporal Convolutional Network on vehicle signals that arrive as one
CSV per feature, and forecasts the next value(s) of a chosen signal.

- **Self-contained** — no imports from any surrounding project; copy or move the
  whole directory anywhere and it still runs
- **No hyper-parameter search** — no Optuna, no sweeps; one config = one run
- **Small functions** — every step (read, align, window, split, scale, train,
  score, save, predict) is its own named function, so it can be read and reused
  in isolation

## Input format

- One CSV per feature, file name = feature name
  - `speed.csv` → column `speed`
  - `engine_rpm.csv` → column `engine_rpm`
- Each file holds two columns: `timestamp,value`

```csv
timestamp,value
1704067200000,0.000000
1704067200050,0.857571
```

- `timestamp` may be an epoch number (s / ms / us / ns, auto-detected) or an ISO
  string (`2024-01-01 00:00:00.050`)
- Headerless files work if they have exactly two columns (a warning is logged)
- Files may have **different sampling rates** and different timestamps; use
  `--resample` to put them on one grid

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

`run.py` is the only entry point; run it from inside this directory.

```bash
# 1. write synthetic sample data with a known answer (3 features, 3 rates)
python run.py sample-data --out-dir sample_data

# 2. train: forecast speed 1 step ahead on a 100 ms grid
python run.py train --data-dir sample_data --target speed --resample 100ms

# 3. predict every window, save the table
python run.py predict --model artifacts/tcn_model.pth --data-dir sample_data \
    --stride 100 --out predictions.csv

# 3b. or just the forecast from the most recent window
python run.py predict --model artifacts/tcn_model.pth --data-dir sample_data --latest
```

On the sample data this reaches `r2 > 0.99` for 1-step-ahead speed. That is the
smoke test: the sample relation is deterministic, so a low score means the
windowing / alignment / scaling broke, not that the model is too small.

Outputs land in `--out-dir` (default `artifacts/`):

- `tcn_model.pth` — the model plus everything needed to predict with it
- `train_history.csv` — per-epoch train/val loss, for a learning curve
- `train_scores.json` — final scores and the settings that produced them

## Useful flags

- `--target speed` — which feature to forecast (required)
- `--resample 100ms` — put all features on one fixed grid; use whenever the
  input rates differ
- `--seq-len 64` — how many past steps the model sees
- `--horizon 5` — forecast 5 steps ahead instead of 1 (output gains `step_1..5`)
- `--stride 1` — gap between window starts; raise it to train on less overlapping data
- `--drop-target-channel` — do not feed the target's own history to the model
- `--num-levels / --hidden / --kernel-size / --dropout` — network size
- `--epochs / --batch-size / --lr / --patience` — optimization
- `--device auto|cpu|cuda|mps`, `--seed 42`, `--verbose`

Run `python run.py train --help` for the full list, or `python run.py` for the
list of commands.

## Layout

Only two files sit at the root: the one command you run, and the one file you
edit. Everything else is grouped by pipeline stage.

```
cnn_vehicle_data/
├── run.py                     # the only entry point: train | predict | sample-data
├── train_config.py            # every setting, in one place
├── cli/                       # argument parsing and orchestration, no logic
│   ├── train_command.py
│   ├── predict_command.py
│   ├── sample_data_command.py
│   └── runtime_setup.py       # logging, seeding, device choice
├── data_pipeline/             # CSV files -> aligned table -> scaled windows
│   ├── csv_feature_loader.py
│   └── window_dataset.py
├── tcn_model/                 # the network, and the artifact that stores it
│   ├── tcn_network.py
│   └── model_artifact.py
├── training/                  # epoch loop, early stopping, metrics
│   ├── train_tcn.py
│   └── metric_scores.py
└── inference/                 # prediction with a saved artifact
    └── predict_tcn.py
```

## Reading order

1. [train_config.py](train_config.py) — every setting, its default, and validation
2. [data_pipeline/csv_feature_loader.py](data_pipeline/csv_feature_loader.py) — many CSVs → one aligned table
3. [data_pipeline/window_dataset.py](data_pipeline/window_dataset.py) — windows, chronological split, scaling
4. [tcn_model/tcn_network.py](tcn_model/tcn_network.py) — the model itself
5. [training/train_tcn.py](training/train_tcn.py) — epoch loop, early stopping, scoring
6. [tcn_model/model_artifact.py](tcn_model/model_artifact.py) — what gets saved, and why
7. [inference/predict_tcn.py](inference/predict_tcn.py) — inference, mirroring training exactly
8. [run.py](run.py) + [cli/](cli/) — dispatch, argument parsing, calling the above
9. [cli/sample_data_command.py](cli/sample_data_command.py) — synthetic data with a known answer
10. [training/metric_scores.py](training/metric_scores.py), [cli/runtime_setup.py](cli/runtime_setup.py) — metrics; logging / seed / device

## How the data becomes a training sample

1. Read each CSV into a time-indexed series named after the file
2. Outer-join all features on the union of timestamps
3. Optionally resample to a fixed grid (`--resample`), averaging within a bucket
4. Forward-fill holes, then drop rows that are still incomplete (typically the
   head, before every feature had started)
5. Cut rows into train / val / test **by time order**, oldest rows first
6. Fit `StandardScaler` on **training rows only**, apply to all rows
7. Inside each split, take every window of `seq_len` rows whose `horizon` labels
   also fall in that split
8. Feed windows as `(batch, channels, seq_len)`; the model outputs `horizon` values

## Design rules worth keeping

- **No shuffled split.** Rows are cut by position, oldest first. Shuffling rows
  would put the future in the training set and inflate every score.
- **Scalers fit on training rows only.** `create_data_bundle` fits, then
  transforms; nothing looks at val/test statistics.
- **No window crosses a split boundary.** Inputs and labels of one sample always
  sit in the same split.
- **Fill forward only.** Backward filling would copy a future measurement into
  an earlier row. Rows that cannot be filled are dropped, not zero-filled.
- **The artifact is the contract.** `tcn_model.pth` carries the arch, weights,
  channel names *in training order*, window spec, both scalers, and the
  preprocessing settings — so prediction reproduces training without the config.
- **Metrics are reported in the original unit.** Predictions are inverse-scaled
  before scoring, so `mean_absolute_error` is in km/h, not in standard deviations.
- **Best epoch wins.** The weights of the lowest-validation-loss epoch are
  restored before saving, so the saved model matches the reported score.

## Limitations

- Regression / forecasting only — no classification head
- One target feature per model
- One continuous recording per run; there is no group column for
  multiple vehicles, so concatenating trips into one CSV would create windows
  that span a discontinuity
- Every feature must overlap in time, otherwise no complete row survives step 4
