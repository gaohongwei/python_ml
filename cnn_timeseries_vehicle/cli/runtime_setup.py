"""Small process-level helpers: logging, seeding, device choice."""

import logging
import random

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def configure_logging(verbose: bool = False) -> None:
    """Send readable logs to stdout; call once from a CLI entry point."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )


def set_random_seed(seed: int) -> None:
    """Seed python / numpy / torch so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Pick the device: explicit name, or best available when "auto"."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
