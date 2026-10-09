"""
preprocessing.py
----------------
Handles reading and preprocessing of 6-band HLS GeoTIFF images for
Prithvi-EO-2.0-300M-BurnScars inference.

Expected band order (1-indexed, matching HLS convention):
  1: Blue
  2: Green
  3: Red
  4: Narrow NIR
  5: SWIR1
  6: SWIR2
"""

import numpy as np
import rasterio
from rasterio.transform import Affine
from typing import Tuple, Dict, Any
import torch


# Per-band normalization statistics taken directly from burn_scars_config.yaml
# (ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars).
# The model was trained on float32 reflectance in the range ~0–1
# (NOT raw DN × 10000).
BAND_MEANS = [
    0.033349706741586264,
    0.05701185520536176,
    0.05889748132001316,
    0.2323245113436119,
    0.1972854853760658,
    0.11944914225186566,
]
BAND_STDS = [
    0.02269135568823774,
    0.026807560223070237,
    0.04004109844362779,
    0.07791732423672691,
    0.08708738838140137,
    0.07241979477437814,
]

EXPECTED_BANDS = 6


def read_geotiff(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a multi-band GeoTIFF and return the array plus metadata.

    Returns
    -------
    data : np.ndarray  shape (bands, H, W), dtype float32
    meta : dict with keys:
        profile     – full rasterio profile
        width       – image width in pixels
        height      – image height in pixels
        count       – number of bands
        crs         – coordinate reference system (or None)
        transform   – affine transform (or None)
        pixel_width_m  – pixel size in metres (x-direction) or None
        pixel_height_m – pixel size in metres (y-direction) or None
    """
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)   # (bands, H, W)
        profile = src.profile
        crs = src.crs
        transform = src.transform

        pixel_width_m = None
        pixel_height_m = None

        if transform is not None:
            # transform.a = pixel width  (x),  transform.e = pixel height (y, negative)
            pw = abs(transform.a)
            ph = abs(transform.e)
            # If the CRS is geographic (degrees), skip – we cannot trivially convert
            if crs is not None and crs.is_projected:
                pixel_width_m = pw
                pixel_height_m = ph
            elif pw > 1:
                # Heuristic: if values look metric (> 1 unit) assume metres
                pixel_width_m = pw
                pixel_height_m = ph

    meta = {
        "profile": profile,
        "width": data.shape[2],
        "height": data.shape[1],
        "count": data.shape[0],
        "crs": str(crs) if crs else None,
        "transform": transform,
        "pixel_width_m": pixel_width_m,
        "pixel_height_m": pixel_height_m,
    }
    return data, meta


def validate_bands(data: np.ndarray) -> None:
    """Raise ValueError if the image does not have exactly 6 bands."""
    if data.shape[0] != EXPECTED_BANDS:
        raise ValueError(
            f"This model expects a 6-band HLS satellite image "
            f"(Blue, Green, Red, NIR, SWIR1, SWIR2). "
            f"Received {data.shape[0]} band(s)."
        )


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Apply per-band z-score normalization using the statistics from
    burn_scars_config.yaml (float32 reflectance scale, range ~0–1).

    If the input appears to be raw HLS DN (values >> 1), it is divided by
    10000 first so the normalization is always applied in the correct domain.

    RAM note: normalisation is done in-place on a single working copy so only
    one extra (6, H, W) float32 array (~6 MB for 512×512) is allocated.

    Parameters
    ----------
    data : np.ndarray  shape (6, H, W)

    Returns
    -------
    normalized : np.ndarray  shape (6, H, W), float32
    """
    # One copy — we mutate this in-place below
    working = data.astype(np.float32)

    # Detect raw DN: 99th-percentile of non-zero pixels > 2.0
    nonzero = working[working > 0]
    if nonzero.size > 0 and float(np.percentile(nonzero, 99)) > 2.0:
        working /= 10000.0          # in-place scale

    # Per-band z-score in-place (avoids allocating a second full array)
    means = np.array(BAND_MEANS, dtype=np.float32).reshape(-1, 1, 1)
    stds  = np.array(BAND_STDS,  dtype=np.float32).reshape(-1, 1, 1)
    working -= means
    working /= (stds + 1e-8)
    return working


def preprocess_for_inference(data: np.ndarray) -> torch.Tensor:
    """
    Normalize and convert a (6, H, W) array into a (1, 6, H, W) float32 tensor
    ready for Prithvi inference.
    """
    validate_bands(data)
    normed = normalize(data)
    tensor = torch.from_numpy(normed).unsqueeze(0)   # (1, 6, H, W)
    return tensor


def get_rgb_preview(data: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """
    Build a uint8 RGB preview image from bands 3, 2, 1 (Red, Green, Blue).

    Returns
    -------
    rgb : np.ndarray  shape (H, W, 3), uint8
    """
    red   = data[2].astype(np.float32)
    green = data[1].astype(np.float32)
    blue  = data[0].astype(np.float32)

    def stretch(band: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(band, [percentile, 100 - percentile])
        clipped = np.clip(band, lo, hi)
        if hi > lo:
            scaled = (clipped - lo) / (hi - lo) * 255.0
        else:
            scaled = np.zeros_like(clipped)
        return scaled.astype(np.uint8)

    rgb = np.stack([stretch(red), stretch(green), stretch(blue)], axis=-1)
    return rgb
