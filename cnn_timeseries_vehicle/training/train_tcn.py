"""The training loop: one epoch at a time, early stopping on validation loss.

No hyper-parameter search. The config decides the model; this module only
answers "how good is it, and when should we stop".

Selection rule: the epoch with the lowest validation loss wins, and its weights
are restored before the model is returned - so the saved model is the same one
the reported validation score describes.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metric_scores import calculate_regression_scores, format_scores
from data_pipeline.window_dataset import TDataBundle, inverse_scale_target
from tcn_model.tcn_network import TcnNetwork, build_tcn_network, count_parameters
from train_config import get_network_arch

logger = logging.getLogger(__name__)


@dataclass
class TEpochRecord:
    """One row of the training history."""

    epoch: int
    train_loss: float
    val_loss: float
    is_best: bool


@dataclass
class TTrainOutcome:
    """Result of a full training run."""

    model: TcnNetwork
    arch: Dict
    best_epoch: int
    best_val_loss: float
    val_scores: Dict[str, float]
    test_scores: Optional[Dict[str, float]] = None
    history: List[TEpochRecord] = field(default_factory=list)


def create_model(config, num_channels: int) -> TcnNetwork:
    """Build the network described by the config. out_dim = horizon steps."""
    model = build_tcn_network(
        num_channels=num_channels,
        out_dim=config.horizon,
        arch=get_network_arch(config),
    )
    logger.info(f"model built: {count_parameters(model)} trainable parameters")
    return model


def create_optimizer(model: nn.Module, config) -> torch.optim.Optimizer:
    """Adam is a safe default for this size of model."""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_loss_fn() -> nn.Module:
    """MSE on the scaled target: forecasting is a plain regression problem."""
    return nn.MSELoss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip_max_norm: float,
) -> float:
    """One pass over the training windows; returns the mean batch loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        # Clip before stepping: dilated stacks can spike the gradient norm.
        if grad_clip_max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; returns (y_true, y_pred) in scaled space."""
    model.eval()
    true_batches = []
    pred_batches = []
    for x_batch, y_batch in loader:
        predictions = model(x_batch.to(device))
        true_batches.append(y_batch.numpy())
        pred_batches.append(predictions.cpu().numpy())
    return np.concatenate(true_batches, axis=0), np.concatenate(pred_batches, axis=0)


def calculate_mean_squared_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Validation loss, in the same scaled space the model trains on.

    Kept separate from the reported metrics: this one only has to be comparable
    across epochs, while the metrics have to be readable by a human.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def score_in_original_units(
    y_true_scaled: np.ndarray, y_pred_scaled: np.ndarray, scaler_y
) -> Dict[str, float]:
    """Undo target scaling, then score - so MAE is in km/h, not in sigmas."""
    y_true = inverse_scale_target(y_true_scaled, scaler_y)
    y_pred = inverse_scale_target(y_pred_scaled, scaler_y)
    return calculate_regression_scores(y_true, y_pred)


def evaluate_loader(
    model: nn.Module, loader: Optional[DataLoader], device: torch.device, scaler_y
) -> Optional[Dict[str, float]]:
    """Metrics for one split in original units; None when the split is empty."""
    if loader is None:
        return None
    y_true, y_pred = collect_predictions(model, loader, device)
    return score_in_original_units(y_true, y_pred, scaler_y)


def is_improved(val_loss: float, best_val_loss: float) -> bool:
    """Lower validation loss is better; NaN never counts as an improvement."""
    return bool(np.isfinite(val_loss)) and val_loss < best_val_loss


def train_tcn_model(
    bundle: TDataBundle, config, device: torch.device
) -> TTrainOutcome:
    """Train until `max_epochs` or until validation stops improving."""
    model = create_model(config, bundle.num_channels).to(device)
    optimizer = create_optimizer(model, config)
    loss_fn = create_loss_fn()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_gain = 0
    history: List[TEpochRecord] = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model, bundle.train_loader, optimizer, loss_fn, device,
            config.grad_clip_max_norm,
        )
        y_true, y_pred = collect_predictions(model, bundle.val_loader, device)
        val_loss = calculate_mean_squared_loss(y_true, y_pred)

        improved = is_improved(val_loss, best_val_loss)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1

        history.append(TEpochRecord(epoch, train_loss, val_loss, improved))
        logger.info(
            f"epoch {epoch:3d}/{config.max_epochs} | train_loss={train_loss:.5f} "
            f"| val_loss={val_loss:.5f}{'  <- best' if improved else ''}"
        )

        if epochs_without_gain >= config.early_stop_patience:
            logger.info(
                f"early stop at epoch {epoch}: no gain for "
                f"{config.early_stop_patience} epochs"
            )
            break

    # Restore the best epoch so saved weights match the reported scores.
    model.load_state_dict(best_state)
    val_scores = evaluate_loader(model, bundle.val_loader, device, bundle.scaler_y)
    test_scores = evaluate_loader(model, bundle.test_loader, device, bundle.scaler_y)

    logger.info(f"best epoch {best_epoch} | val   {format_scores(val_scores or {})}")
    if test_scores:
        logger.info(f"best epoch {best_epoch} | test  {format_scores(test_scores)}")

    return TTrainOutcome(
        model=model,
        arch=get_network_arch(config),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        val_scores=val_scores or {},
        test_scores=test_scores,
        history=history,
    )
