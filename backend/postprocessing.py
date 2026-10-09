"""
postprocessing.py
-----------------
Converts the raw binary burn-scar mask into:
  • a georeferenced output GeoTIFF
  • an RGBA overlay suitable for display
  • area/pixel statistics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeoTIFF output
# ---------------------------------------------------------------------------

def save_mask_geotiff(
    mask: np.ndarray,
    meta: Dict[str, Any],
    out_path: str,
) -> None:
    """
    Write the binary burn-scar mask to a single-band GeoTIFF, preserving the
    CRS and affine transform of the source image where available.

    Parameters
    ----------
    mask     : np.ndarray  (H, W) uint8  – 0 = background, 1 = burn scar
    meta     : dict returned by preprocessing.read_geotiff
    out_path : destination file path
    """
    profile = meta["profile"].copy()
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="lzw",
        nodata=255,
    )
    # Ensure spatial metadata is preserved
    if meta.get("crs"):
        profile["crs"] = meta["crs"]
    if meta.get("transform") is not None:
        profile["transform"] = meta["transform"]

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

    logger.info("Mask saved to %s", out_path)


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def build_overlay_image(
    rgb: np.ndarray,
    mask: np.ndarray,
    alpha: int = 128,
    burn_color: tuple = (220, 50, 50),
) -> np.ndarray:
    """
    Composite a transparent red overlay of the burn-scar mask over an RGB image.

    Parameters
    ----------
    rgb        : (H, W, 3) uint8 — true-colour preview
    mask       : (H, W) uint8 — 0/1
    alpha      : transparency of the burn overlay (0 = invisible, 255 = opaque)
    burn_color : RGB colour for burned pixels

    Returns
    -------
    overlay : (H, W, 4) uint8 RGBA
    """
    h, w = rgb.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    # Base layer: original RGB with full opacity
    overlay[..., :3] = rgb
    overlay[..., 3] = 255

    # Burn layer: apply colour where mask == 1
    burn_mask = mask == 1
    overlay[burn_mask, 0] = np.clip(
        rgb[burn_mask, 0].astype(np.int32) // 2 + burn_color[0] // 2, 0, 255
    ).astype(np.uint8)
    overlay[burn_mask, 1] = np.clip(
        rgb[burn_mask, 1].astype(np.int32) // 4, 0, 255
    ).astype(np.uint8)
    overlay[burn_mask, 2] = np.clip(
        rgb[burn_mask, 2].astype(np.int32) // 4, 0, 255
    ).astype(np.uint8)
    overlay[burn_mask, 3] = alpha

    return overlay


def apply_opacity(rgb: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    """
    Re-build the overlay with a user-specified opacity (0.0–1.0).
    Returns an (H, W, 3) uint8 array suitable for st.image().
    """
    alpha_int = int(np.clip(opacity * 255, 0, 255))
    rgba = build_overlay_image(rgb, mask, alpha=alpha_int)

    # Composite RGBA onto a white background → plain RGB for display
    bg = np.ones_like(rgb, dtype=np.float32) * 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    fg = rgba[..., :3].astype(np.float32)
    composed = (fg * a + bg * (1.0 - a)).astype(np.uint8)
    return composed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_statistics(
    mask: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute burned-area statistics from the prediction mask.

    Returns a dict with:
        burned_pixels      – int
        total_valid_pixels – int
        percentage_affected – float or None
        pixel_width_m      – float or None
        pixel_height_m     – float or None
        burned_area_km2    – float or None
        resolution_available – bool
    """
    burned_pixels = int((mask == 1).sum())
    total_pixels = int(mask.size)

    pw = meta.get("pixel_width_m")
    ph = meta.get("pixel_height_m")

    if pw is not None and ph is not None:
        pixel_area_m2 = pw * ph
        burned_area_m2 = burned_pixels * pixel_area_m2
        burned_area_km2 = burned_area_m2 / 1_000_000.0
        resolution_available = True
    else:
        burned_area_km2 = None
        resolution_available = False

    percentage_affected = (
        (burned_pixels / total_pixels * 100.0) if total_pixels > 0 else None
    )

    return {
        "burned_pixels": burned_pixels,
        "total_valid_pixels": total_pixels,
        "percentage_affected": round(percentage_affected, 4) if percentage_affected is not None else None,
        "pixel_width_m": pw,
        "pixel_height_m": ph,
        "burned_area_km2": round(burned_area_km2, 4) if burned_area_km2 is not None else None,
        "resolution_available": resolution_available,
    }
