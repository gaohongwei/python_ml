"""Forecast with a saved model. Needs only a model file and some data.

    python run_predict.py --latest                       # forecast from the newest window
    python run_predict.py --stride 100 --out out.csv     # one row per window
    python run_predict.py --model artifacts/tcn_model.pth --combined-csv drive.csv
    python run_predict.py --data-dir new_logs --latest

There is no --config: how to read and preprocess the data was decided during
training and is stored inside the model file, so it cannot drift here.

`python run_predict.py --help` lists every flag.
"""

import sys

# The real work lives in cli/predict_cli.py; this file only exists to be the
# thing you type. Run it from inside this directory, so `cli` and the other
# packages are importable.
from cli.predict_cli import run_predict

if __name__ == "__main__":
    sys.exit(run_predict())
