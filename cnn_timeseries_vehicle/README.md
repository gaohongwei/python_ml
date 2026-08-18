# cnn_timeseries_vehicle — forecast a vehicle signal from recorded signals

## What it does, in one paragraph

A car records many signals over time: speed, throttle position, coolant
temperature, engine rpm. This tool learns how they move together and then
answers one question: **given the last few seconds of every signal, what will
this one signal be a moment from now?** You point it at a folder of recordings,
name the signal to forecast, and it writes back one model file plus a score that
says how close its forecasts were on data it never trained on.

- Input: vehicle log files (CSV)
- Output: one model file, a score sheet, and a table of forecasts
- The model is a TCN (Temporal Convolutional Network) — a small neural network
  that reads a stretch of time the way image models read a picture

## Why it might matter

- Predict a signal a sensor reports late or rarely, from signals reported often
- Flag a measurement that disagrees with what the other signals imply
- Compress the question "does this behave normally?" into one number per moment

## A worked use case

*"Our test fleet logs speed at 20 Hz, throttle at 10 Hz and coolant temperature
at 1 Hz. Can we predict speed half a second ahead from the other signals?"*

```bash
# 0. one-time setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. make fake data with a known answer, so you can see the tool work first
python run_sample_data.py

# 2. train: forecast speed, 5 steps ahead, on a 100 ms grid
python run_train.py --target speed --resample 100ms --horizon 5

# 3. forecast from the most recent window
python run_predict.py --latest
```

Step 1 is not optional on a fresh clone — `sample_data/` is generated, not
committed — and it prints what it wrote:

```
sample_data/speed.csv  (20 Hz)
sample_data/throttle.csv  (10 Hz)
sample_data/coolant_temp.csv  (1 Hz)

known relation: speed[t] = 0.97 * speed[t-1] + 0.06 * throttle[t-10]
```

Three signals, three different rates, exactly the awkward input `--resample`
exists for. Add `--combined-out drive.csv` to get the same data as one wide file
too, or `--data-dir somewhere_else` to write it elsewhere.

What step 2 prints:

```
target            : speed
data source       : csv files in sample_data
resample rule     : 100ms
window            : seq_len=64, horizon=5, stride=1
...
=== training done ===
best epoch : 5
val        : r2=0.9996 mean_absolute_error=0.7072 ...
test       : r2=0.9992 mean_absolute_error=0.8115 ...
model      : artifacts/tcn_model.pth
```

How to read that:

- `r2 = 0.9992` — the forecast explains 99.9 % of how much speed actually varies
  (1.0 is perfect, 0.0 is no better than guessing the average)
- `mean_absolute_error = 0.81` — on average the forecast is off by 0.81 km/h,
  reported in the signal's own unit, not in some internal scale
- `test` is scored on the last stretch of the recording, which the model never
  saw during training — that is the honest number
- ~0.99 is the *expected* score on this data, because step 1 generated it from
  that fixed relation. It is the smoke test that the tool is wired up correctly,
  not a claim about real logs: a low score here means the loading, alignment or
  windowing broke, not that the model is too small.

Then step 3 prints the actual forecast:

```
=== latest forecast for speed ===
step   1: 86.4869
step   2: 86.9119
step   3: 86.7213
step   4: 87.3586
step   5: 88.0600
```

Five steps on a 100 ms grid, i.e. speed over the next half second, in km/h.

Point the same three commands at real logs by changing one line in
[train_config.py](train_config.py), or by passing `--data-dir my_logs`.

## What you have to decide

Only three things, and all three have a default in
[train_config.py](train_config.py):

- **Where the data is** — `DEFAULT_DATA_DIR`, or `--data-dir my_logs`
- **What to forecast** — `DEFAULT_TARGET_FEATURE`, or `--target speed`
- **How fine a time grid** — `DEFAULT_RESAMPLE_RULE`, or `--resample 100ms`

Edit those three values once and `python run_train.py` needs no flags at all.
Everything else already has a sane default.

### The target variable

The **target** is the one signal you want predicted — `speed` in every example
here, `target_feature` in the config. It plays two roles at once:

- It is the **answer** the model is graded on: at each moment, the target's value
  `horizon` steps in the future is the number the model must reproduce, and every
  score in the output is about that number only
- It is also an **input** by default, because a vehicle signal's own recent
  history is usually its best predictor — pass `--drop-target-channel` to forbid
  that and force the model to work from the other signals alone

Every other signal in the data becomes an input automatically. One model predicts
one target; to forecast two signals, train twice.

## Settings in a file (training only)

A pile of flags is not a record of anything, so a training run can be written
down instead:

```bash
cp example_train_config.yaml my_run.yaml     # then edit my_run.yaml
python run_train.py --config my_run.yaml
```

- [example_train_config.yaml](example_train_config.yaml) is a template to copy,
  not a file that is read on its own — nothing loads it unless you pass `--config`
- Any setting in [train_config.py](train_config.py) is a valid key; the example
  lists them all with comments
- `.json` works the same as `.yaml`
- Settings apply in three layers, each overriding the one before:
  **train_config.py defaults → `--config` file → flags you type**
- A misspelled key is an error listing the valid names, not a silently ignored line

So: one file per experiment, plus a flag when you want to try a single change
without editing the file.

```bash
python run_train.py --config example_train_config.yaml --horizon 20 --out-dir artifacts/h20
```

**Prediction takes no config file**, on purpose. It needs exactly two things:

```bash
python run_predict.py --model artifacts/tcn_model.pth --data-dir new_logs --latest
```

Which columns in which order, the sampling rate, the fitted scalers, the window
length — all of it was decided during training and travels inside the model file.
Accepting a training config here would let someone edit `seq_len`, run predict,
and see nothing change, because the artifact would keep winning. Both flags have
defaults, so `python run_predict.py --latest` works right after a training run.

## Accepted input shapes

Four ways to hand over the same data; the tool prepares all of them identically,
so a model trained from one shape can forecast from another.

1. **A folder of one-signal files** (`--data-dir my_logs`) — file name is the
   signal name

   ```csv
   # my_logs/speed.csv
   timestamp,value
   1704067200000,0.000000
   1704067200050,0.857571
   ```

2. **One combined file** (`--combined-csv drive.csv`) — a timestamp column plus
   one column per signal, i.e. what a database export or a `df.to_csv()` looks
   like

   ```csv
   timestamp,speed,throttle,coolant_temp
   1704067200000,0.000000,14.805,25.0
   1704067200050,0.857571,,
   ```

3. **A folder of either, or both mixed** (`--data-dir my_logs`) — each file is
   inspected on its own; a `timestamp,value` file becomes one signal named after
   the file, a wider file contributes every column it has

4. **A DataFrame you already hold in Python** — no files involved:

   ```python
   from cli.train_cli import train_from_config
   from data_pipeline.combined_frame import build_combined_frame
   from train_config import TcnTrainConfig

   combined = build_combined_frame("my_logs", "csv")   # folder + file type -> one table
   train_from_config(TcnTrainConfig(target_feature="speed", data_dir=None), frame=combined)
   ```

Details that hold for every shape:

- `timestamp` may be an epoch number (s / ms / µs / ns, auto-detected) or an ISO
  string (`2024-01-01 00:00:00.050`)
- Signals may have **different sampling rates**; `--resample` puts them on one
  grid, and empty cells are expected until then
- A signal must be defined once — the same name from two files is an error, not
  a silent merge
- When several sources are given, the first of these wins: a DataFrame, then
  `--combined-csv`, then `--csv`, then `--data-dir`

## Scoring: how "wrong" is measured

Two different things use the word error here, and mixing them up is the most
common way to misread a run.

- **The loss** is what the model is *trained* to make small. It is computed on
  the internally scaled target, thousands of times per epoch, and its only job is
  to point the model in a direction.
- **The metrics** are what the run is *reported* with. They are computed once per
  epoch on data held back from training, in the signal's own unit (km/h, rpm, °C).

### The four reported metrics

Every run prints all four for the validation and test splits:

- **r2** — share of the target's real variation the forecast explains. 1.0 is
  perfect; 0.0 is no better than always guessing the average; negative is worse
  than that. Best single "did this work at all?" number.
- **mean_absolute_error (MAE)** — average miss, in the signal's unit. "Off by
  1.2 km/h on average." The number to quote to someone who has to live with the
  forecast.
- **root_mean_squared_error (RMSE)** — like MAE but squares each miss first, so a
  few large misses dominate. RMSE ≫ MAE means the errors are spiky rather than
  spread evenly — worth knowing when one bad forecast is much worse than several
  small ones.
- **mean_absolute_percentage_error (MAPE)** — average miss as a percentage of the
  true value, for comparing across signals with different units. Meaningless near
  zero, so values below `1e-6` are skipped.

### Choosing the training loss

`--loss mse|mae|huber` (or `loss_fn` in the config) decides what a miss costs
while training:

- **mse** (default) — squares the error, so one 10-off miss weighs as much as a
  hundred 1-off misses. The model works hardest on the worst moments. Use when
  big misses are genuinely the expensive ones; it optimizes what RMSE reports.
- **mae** — every unit of error costs the same. A handful of corrupt sensor
  samples cannot drag the whole fit toward themselves. Use on noisy real logs
  with occasional junk values; it optimizes what MAE reports.
- **huber** — squared error near zero, absolute error beyond `--huber-delta`
  (in scaled units, so `1.0` ≈ one standard deviation of the target). The usual
  compromise: smooth gradients where the model is close, outlier-resistant where
  it is not. Reach for it when MSE overreacts to spikes but MAE trains sluggishly.

Rules of thumb:

- Clean data, big misses matter → **mse**
- Dirty data, occasional wild values → **mae** or **huber**
- Changing the loss changes what the model optimizes, not what is reported; all
  four metrics are printed either way, so the comparison stays honest
- The loss used is recorded in `train_scores.json`, so two runs can be compared
  later without guessing

## Useful flags

- `--config run.yaml` — training only: read every setting from a file (flags still override)
- `--target speed` — which signal to forecast
- `--resample 100ms` — put all signals on one fixed grid; use whenever rates differ
- `--seq-len 64` — how many past steps the model sees
- `--horizon 5` — forecast 5 steps ahead instead of 1 (output gains `step_1..5`)
- `--stride 1` — gap between window starts; raise it to train on less overlapping data
- `--drop-target-channel` — do not feed the target's own history to the model
- `--file-type csv` — which files a `--data-dir` is combined from
- `--loss mse|mae|huber`, `--huber-delta 1.0` — what a miss costs while training
- `--num-levels / --hidden / --kernel-size / --dropout` — network size
- `--epochs / --batch-size / --lr / --patience` — optimization
- `--device auto|cpu|cuda|mps`, `--seed 42`, `--verbose`

Run `python run_train.py --help` or `python run_predict.py --help` for the full
flag list of either script.

## What lands on disk

Outputs go to `--out-dir` (default `artifacts/`):

- `tcn_model.pth` — the model plus everything needed to forecast with it
- `train_history.csv` — per-epoch train/val loss, for a learning curve
- `train_scores.json` — final scores, the loss used, and the window settings that
  produced them

## Engineering notes

Written to be read: no hyper-parameter search (no Optuna, no Ray, no sweeps —
one config, one run), no imports from any surrounding project, and every step
(read, combine, window, split, scale, train, score, save, predict) is its own
named function.

### Layout

```
cnn_timeseries_vehicle/
├── run_train.py                # entry point: train and save a model
├── run_predict.py              # entry point: forecast with a saved model
├── run_sample_data.py          # entry point: write fake data to try it on
├── train_config.py             # every setting, in one place
├── example_train_config.yaml   # a run written down; copy it, then --config it
├── cli/                        # argument parsing and orchestration, no logic
│   ├── train_cli.py            # run_train(argv), called by run_train.py
│   ├── predict_cli.py          # run_predict(argv), called by run_predict.py
│   ├── sample_data_cli.py      # run_sample_data(argv)
│   └── runtime_setup.py        # logging, seeding, device choice
├── data_pipeline/              # files or DataFrame -> aligned table -> scaled windows
│   ├── csv_feature_loader.py   # one-signal-per-file reading, align / resample / fill
│   ├── combined_frame.py       # folder or wide csv or DataFrame -> one combined table
│   └── window_dataset.py
├── tcn_model/                  # the network, and the artifact that stores it
│   ├── __init__.py             # `from tcn_model import TcnModel, load_model_artifact`
│   ├── tcn_model.py            # TcnModel: dilated conv stack + linear head
│   └── artifact.py             # save / load the one .pth that carries everything
├── training/                   # epoch loop, early stopping, metrics
│   ├── train_tcn.py
│   └── metric_scores.py
└── inference/                  # prediction with a saved artifact
    └── predict_tcn.py
```

### Reading order

1. [train_config.py](train_config.py) — every setting, its default, validation, and
   config-file loading
2. [data_pipeline/combined_frame.py](data_pipeline/combined_frame.py) — any source → one combined table
3. [data_pipeline/csv_feature_loader.py](data_pipeline/csv_feature_loader.py) — per-signal reading, align / resample / fill
4. [data_pipeline/window_dataset.py](data_pipeline/window_dataset.py) — windows, chronological split, scaling
5. [tcn_model/tcn_model.py](tcn_model/tcn_model.py) — the model itself
6. [training/train_tcn.py](training/train_tcn.py) — epoch loop, early stopping, scoring
7. [tcn_model/artifact.py](tcn_model/artifact.py) — what gets saved, and why
8. [inference/predict_tcn.py](inference/predict_tcn.py) — inference, mirroring training exactly
9. the `run_*.py` scripts + [cli/](cli/) — argument parsing, config layering, and
   `run_train` / `run_predict` / `run_sample_data`, which the scripts call
10. [cli/sample_data_cli.py](cli/sample_data_cli.py) — synthetic data with a known answer
11. [training/metric_scores.py](training/metric_scores.py), [cli/runtime_setup.py](cli/runtime_setup.py) — metrics; logging / seed / device

### How the data becomes a training sample

1. Read each file into a time-indexed table (one column per signal)
2. Outer-join everything on the union of timestamps
3. Optionally resample to a fixed grid (`--resample`), averaging within a bucket
4. Forward-fill holes, then drop rows that are still incomplete (typically the
   head, before every signal had started)
5. Cut rows into train / val / test **by time order**, oldest rows first
6. Fit `StandardScaler` on **training rows only**, apply to all rows
7. Inside each split, take every window of `seq_len` rows whose `horizon` labels
   also fall in that split
8. Feed windows as `(batch, channels, seq_len)`; the model outputs `horizon` values

### Design rules worth keeping

- **One preparation path.** Folder, combined CSV, file list and DataFrame all
  meet in `prepare_feature_frame`, so no source can be prepared differently from
  the one the model was trained on.
- **A setting exists once.** Defaults live in `train_config.py`; the config file
  and the flags only override fields of the same dataclass, so a flag cannot
  write a setting the config file has never heard of. Flags use
  `argparse.SUPPRESS`, which is what lets an untyped flag stay silent instead of
  overwriting the config file with a default.
- **Unknown settings are errors.** A misspelled config key names the valid ones
  and stops. Ignoring it would produce a run that quietly did not do what its own
  file says.
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

### Limitations

- Regression / forecasting only — no classification head
- One target signal per model
- One continuous recording per run; there is no group column for multiple
  vehicles, so concatenating trips into one file would create windows that span a
  discontinuity
- Every signal must overlap in time, otherwise no complete row survives step 4
- `--file-type` accepts `csv` today; the combining step is where another format
  would be added
