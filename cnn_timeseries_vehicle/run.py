"""Single entry point for everything in this project.

    python run.py sample-data --out-dir sample_data
    python run.py train       --data-dir sample_data --target speed --resample 100ms
    python run.py predict     --model artifacts/tcn_model.pth --data-dir sample_data

Each command parses its own arguments, so `python run.py train --help` shows the
full flag list for training only.
"""

import sys

from cli.predict_command import main as run_predict
from cli.sample_data_command import main as run_sample_data
from cli.train_command import main as run_train

# command name -> handler(argv) -> exit code
COMMAND_HANDLERS = {
    "sample-data": run_sample_data,
    "train": run_train,
    "predict": run_predict,
}

COMMAND_SUMMARIES = {
    "sample-data": "write synthetic per-feature CSVs with a known answer",
    "train": "train a TCN and save the model artifact",
    "predict": "forecast with a saved model artifact",
}


def print_usage() -> None:
    """List the commands; each one has its own --help."""
    print(__doc__.strip())
    print("\ncommands:")
    for name, summary in COMMAND_SUMMARIES.items():
        print(f"  {name:<12} {summary}")
    print("\nuse `python run.py <command> --help` for the flags of one command")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print_usage()
        return 0 if argv else 2

    command, command_argv = argv[0], argv[1:]
    handler = COMMAND_HANDLERS.get(command)
    if handler is None:
        print(f"unknown command: {command}\n")
        print_usage()
        return 2
    return handler(command_argv)


if __name__ == "__main__":
    sys.exit(main())
