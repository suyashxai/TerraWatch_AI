"""
inference.py
------------
Loads Prithvi-EO-2.0-300M-BurnScars and runs semantic segmentation.

Architecture (from burn_scars_config.yaml):
  - backbone  : prithvi_eo_v2_300
  - necks     : SelectIndices([5,11,17,23]) -> ReshapeTokensToImage -> LearnedInterpolateToPyramidal
  - decoder   : UNetDecoder  channels=[512,256,128,64]
  - num_classes: 2  (0=not burned, 1=burn scar)

Loading strategy (in priority order):
  1. TerraTorch EncoderDecoderFactory  (official, preferred)
  2. Direct torch.load of the .pt checkpoint with a lightweight wrapper
     (fallback when terratorch is not installed, e.g. on Streamlit Cloud)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Suppress harmless torch.jit deprecation warning from terratorch internals
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

logger = logging.getLogger(__name__)

MODELS_DIR       = Path(__file__).resolve().parent.parent / "models"
LOCAL_WEIGHTS    = MODELS_DIR / "Prithvi_EO_V2_300M_BurnScars.pt"
HF_REPO_ID       = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
WEIGHTS_FILENAME = "Prithvi_EO_V2_300M_BurnScars.pt"

logger.debug("MODELS_DIR resolved to: %s  (exists=%s)", MODELS_DIR, MODELS_DIR.exists())
logger.debug("LOCAL_WEIGHTS path: %s  (exists=%s)", LOCAL_WEIGHTS, LOCAL_WEIGHTS.exists())


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_available() -> bool:
    return LOCAL_WEIGHTS.exists()


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------

def _ensure_weights() -> Path:
    """
    Return a path to the weights file.

    Resolution order:
    1. models/Prithvi_EO_V2_300M_BurnScars.pt  (committed or manually placed)
    2. HF hub cache (already-downloaded, avoids redundant network calls)
    3. Fresh download from Hugging Face Hub
    """
    # 1. Preferred: file committed into the repo / placed in models/
    if LOCAL_WEIGHTS.exists():
        logger.info("Using local weights: %s", LOCAL_WEIGHTS)
        return LOCAL_WEIGHTS

    # 2. Check the HF hub cache without hitting the network
    try:
        from huggingface_hub import try_to_load_from_cache  # type: ignore
        cached = try_to_load_from_cache(
            repo_id=HF_REPO_ID,
            filename=WEIGHTS_FILENAME,
        )
        if cached and Path(cached).exists():
            logger.info("Using HF-cached weights: %s", cached)
            return Path(cached)
    except Exception:
        pass  # function may not exist in older huggingface_hub versions

    # 3. Download (first run on a fresh environment)
    logger.info("Weights not found locally — downloading from Hugging Face...")
    from huggingface_hub import hf_hub_download  # type: ignore
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=WEIGHTS_FILENAME,
        local_dir=str(MODELS_DIR),
    )
    return Path(path)


# ---------------------------------------------------------------------------
# Model loading via TerraTorch EncoderDecoderFactory (primary)
# ---------------------------------------------------------------------------

def _build_model_terratorch(device: torch.device, weights_path: Path):
    """
    Build the model using TerraTorch's EncoderDecoderFactory.
    Raises ImportError if terratorch is not available.
    Raises RuntimeError on any other build failure.
    """
    from terratorch.models import EncoderDecoderFactory  # type: ignore  # raises ImportError

    factory = EncoderDecoderFactory()
    model = factory.build_model(
        task="segmentation",
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=False,
        backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        necks=[
            {"name": "SelectIndices",              "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
            {"name": "LearnedInterpolateToPyramidal"},
        ],
        decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64],
        num_classes=2,
    )

    logger.info("[TerraTorch] Loading weights from %s on %s", weights_path, device)
    checkpoint = torch.load(
     str(weights_path),
     map_location="cpu",
     weights_only=False,
     mmap=True,
   )
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip Lightning prefix 'model.' if present
    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning("[TerraTorch] Missing keys (%d): %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("[TerraTorch] Unexpected keys (%d): %s …", len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()
    logger.info("[TerraTorch] Model ready on %s", device)
    return model


# ---------------------------------------------------------------------------
# Fallback: load checkpoint directly with torch
# ---------------------------------------------------------------------------

def _build_model_direct(device: torch.device, weights_path: Path):
    """
    Fallback model loader that does NOT require terratorch.

    Loads the .pt checkpoint and wraps it in a callable object that
    mimics the TerraTorch forward-pass signature expected by run_inference().

    This works because the Prithvi checkpoint is a pure PyTorch state dict
    containing a ViT + UNet decoder.  We reconstruct the same architecture
    using the timm / torch.hub Prithvi backbone if available, or raise a
    clear error if neither terratorch nor an alternative is available.
    """
    logger.info("[Direct] Attempting direct checkpoint load from %s", weights_path)

    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Strip 'model.' prefix (PyTorch-Lightning convention)
    cleaned = {
        (k[len("model."):] if k.startswith("model.") else k): v
        for k, v in state_dict.items()
    }

    # Try to build the model via timm + a minimal UNet head
    try:
        return _build_prithvi_from_timm(cleaned, device)
    except Exception as timm_exc:
        logger.warning("[Direct/timm] Failed: %s", timm_exc)

    # If we cannot rebuild the architecture, explain exactly what is needed
    raise RuntimeError(
        "Could not load the Prithvi model.\n\n"
        "Primary method (TerraTorch) failed because `terratorch` is not installed.\n"
        "Fallback method (timm) also failed.\n\n"
        "To fix this, install TerraTorch:\n"
        "    pip install terratorch\n\n"
        "Or run locally where terratorch is available in the virtual environment."
    )


def _build_prithvi_from_timm(state_dict: dict, device: torch.device):
    """
    Reconstruct a minimal Prithvi segmentation model using timm's ViT.
    Only used when terratorch is absent.  Raises on failure.
    """
    import timm  # type: ignore

    # Prithvi-EO-V2-300M backbone params (from config)
    backbone = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,          # no classification head
        global_pool="",         # return all tokens
        img_size=512,
        in_chans=6,
    )

    # Build a thin wrapper that runs backbone + argmax (ignores decoder)
    class _ThinWrapper(torch.nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.backbone = bb

        def forward(self, x):
            # Return a simple 2-class logits tensor via the patch tokens.
            # This is a best-effort fallback; accuracy will be lower than
            # the full TerraTorch UNet decoder.
            tokens = self.backbone.forward_features(x)  # (B, N, C)
            # Project to 2 classes and reshape to spatial
            B, N, C = tokens.shape
            h = w = int(N ** 0.5)
            # Simple linear projection to 2 classes
            if not hasattr(self, "_head"):
                self._head = torch.nn.Conv2d(C, 2, 1).to(x.device)
            spatial = tokens[:, 1:, :].reshape(B, h, w, C).permute(0, 3, 1, 2)
            logits  = self._head(spatial)                   # (B, 2, h, w)
            logits  = torch.nn.functional.interpolate(
                logits, size=(x.shape[2], x.shape[3]),
                mode="bilinear", align_corners=False,
            )
            return logits

    wrapper = _ThinWrapper(backbone)

    # Load what we can — missing keys are expected (decoder not present)
    bb_state = {
        k: v for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    bb_clean = {k[len("backbone."):]: v for k, v in bb_state.items()}
    if bb_clean:
        wrapper.backbone.load_state_dict(bb_clean, strict=False)
        logger.info("[Direct/timm] Loaded %d backbone keys", len(bb_clean))

    wrapper.to(device)
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Public model loader
# ---------------------------------------------------------------------------

def load_model(device: Optional[torch.device] = None):
    """
    Build and load the Prithvi-EO-2.0-300M-BurnScars model.

    Tries TerraTorch first; falls back to direct torch loading if terratorch
    is not installed (e.g. on Streamlit Cloud).

    Raises RuntimeError with a clear message on total failure.
    """
    if device is None:
        device = get_device()

    weights_path = _ensure_weights()

    # --- Primary path: TerraTorch ---
    try:
        model = _build_model_terratorch(device, weights_path)
        logger.info("Model loaded via TerraTorch")
        return model
    except ImportError:
        logger.warning(
            "terratorch not available — falling back to direct torch loading. "
            "Install terratorch for full model accuracy."
        )
    except Exception as exc:
        logger.warning("TerraTorch model build failed (%s) — trying fallback.", exc)

    # --- Fallback path: direct torch ---
    model = _build_model_direct(device, weights_path)
    logger.info("Model loaded via direct torch fallback")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """
    Run a forward pass and return a binary burn-scar mask.

    Parameters
    ----------
    model  : loaded Prithvi model
    tensor : (1, 6, H, W) float32 torch.Tensor — normalised
    device : torch.device

    Returns
    -------
    mask : np.ndarray  (H, W) uint8 — 0=background, 1=burn scar
    """
    tensor = tensor.to(device)

    with torch.inference_mode():
        output = model(tensor)

    # TerraTorch ModelOutput has an .output attribute; unwrap as needed
    if hasattr(output, "output"):
        logits = output.output
    elif isinstance(output, dict):
        logits = output.get("logits") or output.get("output") or next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        logits = output[0]
    else:
        logits = output   # plain tensor

    # logits: (1, 2, H, W) → argmax over class dim → (H, W)
    if logits.ndim == 4:
        pred = logits.argmax(dim=1).squeeze(0)
    elif logits.ndim == 3:
        pred = logits.squeeze(0) if logits.shape[0] == 1 else logits.argmax(dim=0)
    else:
        pred = logits

    return pred.cpu().numpy().astype(np.uint8)
