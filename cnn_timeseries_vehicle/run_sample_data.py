"""Make fake vehicle data, so the other two scripts can be run before real logs exist.

    python run_sample_data.py
    python run_sample_data.py --data-dir sample_data --combined-out drive.csv

The signals follow a fixed formula, so training on them must reach r2 > 0.99;
that is the smoke test that this tool is wired up correctly.
"""

import sys

# The real work lives in cli/sample_data_cli.py; this file only exists to be
# the thing you type. Run it from inside this directory, so `cli` and the other
# packages are importable.
from cli.sample_data_cli import run_sample_data

if __name__ == "__main__":
    sys.exit(run_sample_data())
