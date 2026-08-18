"""Train a model and save it. Every setting has a default in train_config.py.

    python run_train.py                                  # uses train_config.py as-is
    python run_train.py --config example_train_config.yaml      # one experiment, written down
    python run_train.py --data-dir my_logs --target engine_rpm --resample 1s
    python run_train.py --combined-csv my_logs/drive.csv --target speed

`python run_train.py --help` lists every flag.
"""

import sys

# The real work lives in cli/train_cli.py; this file only exists to be the
# thing you type. Run it from inside this directory, so `cli` and the other
# packages are importable.
from cli.train_cli import run_train

if __name__ == "__main__":
    sys.exit(run_train())
