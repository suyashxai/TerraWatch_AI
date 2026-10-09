```python
"""
inference.py
------------
Loads the Prithvi-EO-2.0-300M-BurnScars model
and performs burn-scar segmentation.
"""

from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
LOCAL_WEIGHTS = MODELS_DIR / "Prithvi_EO_V2_300M_BurnScars.pt"
HF_REPO_ID = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
WEIGHTS_FILENAME = "Prithvi_EO_V2_300M_BurnScars.pt"


def get_device() -> torch.device:
    """Choose GPU when available; otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_available() -> bool:
    """Check whether model weights exist locally."""
    return LOCAL_WEIGHTS.is_file()


def _ensure_weights() -> Path:
    """Find existing weights or download them from Hugging Face."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_WEIGHTS.is_file():
        logger.info("Using local model weights: %s", LOCAL_WEIGHTS)
        return LOCAL_WEIGHTS

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(
            repo_id=HF_REPO_ID,
            filename=WEIGHTS_FILENAME,
        )

        if cached and Path(cached).is_file():
            logger.info("Using cached model weights: %s", cached)
            return Path(cached)

    except Exception as exc:
        logger.debug("Could not use cached weights: %s", exc)

    logger.info("Downloading Prithvi weights from Hugging Face...")

    try:
        from huggingface_hub import hf_hub_download

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=WEIGHTS_FILENAME,
            local_dir=str(MODELS_DIR),
        )

        path = Path(downloaded_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Downloaded model weights were not found: {path}"
            )

        return path

    except Exception as exc:
        raise RuntimeError(
            "Could not obtain the Prithvi model weights. "
            "Check the Hugging Face repository, internet access, "
            "and available disk space."
        ) from exc


def _build_model_terratorch(
    device: torch.device,
    weights_path: Path,
):
    """Build the official TerraTorch segmentation architecture."""

    from terratorch.models import EncoderDecoderFactory

    factory = EncoderDecoderFactory()

    model = factory.build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=False,
        backbone_bands=[
            "BLUE",
            "GREEN",
            "RED",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ],
        necks=[
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        num_classes=2,
    )

    logger.info("Loading checkpoint: %s", weights_path)

    try:
        try:
            checkpoint = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except (TypeError, RuntimeError, ValueError):
            # Compatibility fallback for unsupported mmap/checkpoint formats.
            checkpoint = torch.load(
                str(weights_path),
                map_location="cpu",
                weights_only=False,
            )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError("The checkpoint does not contain a valid state dictionary.")

        # Remove the common PyTorch Lightning model prefix.
        cleaned = {
            (key[len("model."):] if key.startswith("model.") else key): value
            for key, value in state_dict.items()
        }

        incompatible = model.load_state_dict(cleaned, strict=False)
```
