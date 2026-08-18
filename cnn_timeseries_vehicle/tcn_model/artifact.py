"""Save / load one self-contained model file.

The .pth holds *everything* inference needs, so a prediction run needs no config
and no training code path:

- network shape (`arch`) and weights (`state_dict`)
- channel names in their exact training order
- window definition (seq_len / horizon / stride)
- fitted scalers for channels and target
- the preprocessing settings used to build the training table
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from tcn_model.tcn_model import TcnModel, build_tcn_model
from data_pipeline.window_dataset import TWindowSpec

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1
MODEL_TYPE = "tcn"


@dataclass
class TLoadedModel:
    """A ready-to-use model plus the contracts it was trained under."""

    model: TcnModel
    arch: Dict
    channel_cols: List[str]
    target_feature: str
    spec: TWindowSpec
    scaler_x: object
    scaler_y: object
    preprocess: Dict

    @property
    def num_channels(self) -> int:
        return len(self.channel_cols)


def build_preprocess_dict(config) -> Dict:
    """The reading settings prediction must reuse verbatim."""
    return {
        "time_col": config.time_col,
        "value_col": config.value_col,
        "resample_rule": config.resample_rule,
        "fill_method": config.fill_method,
        "include_target_as_channel": config.include_target_as_channel,
    }


def save_model_artifact(
    model_path: str,
    model: TcnModel,
    arch: Dict,
    channel_cols: List[str],
    target_feature: str,
    spec: TWindowSpec,
    scaler_x,
    scaler_y,
    preprocess: Dict,
    scores: Optional[Dict] = None,
) -> str:
    """Write the artifact; parent directories are created as needed."""
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "arch": arch,
        "state_dict": model.state_dict(),
        "channel_cols": channel_cols,
        "target_feature": target_feature,
        "seq_len": spec.seq_len,
        "horizon": spec.horizon,
        "stride": spec.stride,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "preprocess": preprocess,
        "scores": scores or {},
    }
    torch.save(payload, path)
    logger.info(f"model saved to {path}")
    return str(path)


def load_model_artifact(model_path: str, device: Optional[torch.device] = None) -> TLoadedModel:
    """Rebuild the network and restore weights, scalers and contracts."""
    device = device or torch.device("cpu")
    # weights_only=False: the payload contains pickled sklearn scalers.
    payload = torch.load(model_path, map_location=device, weights_only=False)

    if payload.get("model_type") != MODEL_TYPE:
        raise ValueError(f"{model_path}: not a {MODEL_TYPE} artifact")

    spec = TWindowSpec(
        seq_len=payload["seq_len"],
        horizon=payload["horizon"],
        stride=payload.get("stride", 1),
    )
    model = build_tcn_model(
        num_channels=len(payload["channel_cols"]),
        out_dim=spec.horizon,
        arch=payload["arch"],
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    logger.info(
        f"model loaded from {model_path}: target={payload['target_feature']}, "
        f"channels={payload['channel_cols']}, seq_len={spec.seq_len}, "
        f"horizon={spec.horizon}"
    )
    return TLoadedModel(
        model=model,
        arch=payload["arch"],
        channel_cols=payload["channel_cols"],
        target_feature=payload["target_feature"],
        spec=spec,
        scaler_x=payload["scaler_x"],
        scaler_y=payload["scaler_y"],
        preprocess=payload.get("preprocess", {}),
    )
