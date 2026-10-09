"""
utils.py
--------
Shared utility helpers for TerraWatch AI.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV export  (includes optional evaluation metrics when available)
# ---------------------------------------------------------------------------

def build_stats_csv(
    filename: str,
    meta: Dict[str, Any],
    stats: Dict[str, Any],
    eval_metrics: Optional[Dict[str, Any]] = None,
    inference_time_s: Optional[float] = None,
    device_name: Optional[str] = None,
    epsg: Optional[int] = None,
) -> bytes:
    """
    Return a UTF-8 encoded CSV as bytes containing key statistics.
    Evaluation metrics are only included when eval_metrics is provided.

    Parameters
    ----------
    filename         : original uploaded filename
    meta             : dict from preprocessing.read_geotiff
    stats            : dict from postprocessing.calculate_statistics
    eval_metrics     : dict from evaluation.compute_metrics, or None
    inference_time_s : measured inference wall-clock time, or None
    device_name      : "CPU" or "GPU (CUDA)", or None
    epsg             : EPSG code from CRS, or None
    """
    rows = [
        ["field", "value"],
        ["filename",             filename],
        ["image_width",          meta.get("width", "")],
        ["image_height",         meta.get("height", "")],
        ["num_bands",            meta.get("count", "")],
        ["crs",                  meta.get("crs") or "unavailable"],
        ["epsg",                 epsg if epsg is not None else "unavailable"],
        ["pixel_width_m",        stats.get("pixel_width_m")
                                 if stats.get("resolution_available") else "unavailable"],
        ["pixel_height_m",       stats.get("pixel_height_m")
                                 if stats.get("resolution_available") else "unavailable"],
        ["total_valid_pixels",   stats.get("total_valid_pixels", "")],
        ["burned_pixels",        stats.get("burned_pixels", "")],
        ["burned_area_km2",      stats.get("burned_area_km2")
                                 if stats.get("resolution_available") else "unavailable"],
        ["percentage_affected",  stats.get("percentage_affected")
                                 if stats.get("percentage_affected") is not None else "unavailable"],
        ["model",                "Prithvi-EO-2.0-300M-BurnScars"],
        ["source",               "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"],
        ["confidence",           "not_available"],
        ["inference_time_s",     inference_time_s if inference_time_s is not None else "unavailable"],
        ["device",               device_name or "unavailable"],
    ]

    if eval_metrics:
        rows += [
            ["ground_truth_evaluation", "yes"],
            ["iou",       eval_metrics.get("iou",       "")],
            ["miou",      eval_metrics.get("miou",      "")],
            ["dice",      eval_metrics.get("dice",      "")],
            ["f1",        eval_metrics.get("f1",        "")],
            ["precision", eval_metrics.get("precision", "")],
            ["recall",    eval_metrics.get("recall",    "")],
            ["true_positives",  eval_metrics.get("true_positives",  "")],
            ["false_positives", eval_metrics.get("false_positives", "")],
            ["false_negatives", eval_metrics.get("false_negatives", "")],
            ["true_negatives",  eval_metrics.get("true_negatives",  "")],
        ]
    else:
        rows.append(["ground_truth_evaluation", "no_ground_truth_provided"])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# File-size formatting
# ---------------------------------------------------------------------------

def human_file_size(n_bytes: int) -> str:
    """Return a human-readable file-size string (e.g. '4.2 MB')."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(level: int = logging.INFO) -> None:
    """Configure a simple console logger for the application."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=level,
    )


# ---------------------------------------------------------------------------
# Image bytes helpers (for Streamlit download buttons)
# ---------------------------------------------------------------------------

def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    """
    Encode an (H, W, 3) or (H, W, 4) uint8 array as PNG bytes.
    Supports both imageio v2 (imwrite with format kwarg) and v3 (imwrite to BytesIO).
    Falls back to PIL if imageio is unavailable or fails.
    """
    buf = io.BytesIO()
    try:
        import imageio  # type: ignore
        if hasattr(imageio, "v3"):
            # imageio >= 3.x
            imageio.v3.imwrite(buf, arr, extension=".png")
        else:
            # imageio 2.x
            imageio.imwrite(buf, arr, format="png")
        return buf.getvalue()
    except Exception:
        pass
    # PIL fallback
    from PIL import Image as _PILImage  # type: ignore
    _PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def geotiff_bytes(path: str) -> bytes:
    """Read a file from disk and return its raw bytes (for download buttons)."""
    with open(path, "rb") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# CRS / geographic helpers
# ---------------------------------------------------------------------------

def parse_crs_info(crs) -> Dict[str, Any]:
    """
    Extract a clean, beginner-friendly CRS summary from a rasterio CRS object.

    Returns dict with keys:
        epsg        – int or None
        authority   – str or None
        name        – str
        is_projected – bool
        utm_zone    – str or None  (e.g. "10N")
        wkt         – str (full WKT, for advanced tab)
    """
    if crs is None:
        return {
            "epsg": None, "authority": None, "name": "Not available",
            "is_projected": False, "utm_zone": None, "wkt": "",
        }

    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass

    # Human-readable name
    name = "Unknown"
    try:
        name = crs.name
    except Exception:
        pass

    # UTM zone detection
    utm_zone = None
    try:
        if epsg and 32601 <= epsg <= 32660:
            utm_zone = f"{epsg - 32600}N"
        elif epsg and 32701 <= epsg <= 32760:
            utm_zone = f"{epsg - 32700}S"
        elif epsg and 27701 <= epsg <= 27760:
            utm_zone = f"{epsg - 27700}N"
    except Exception:
        pass

    wkt = ""
    try:
        wkt = crs.to_wkt()
    except Exception:
        pass

    return {
        "epsg":         epsg,
        "authority":    f"EPSG:{epsg}" if epsg else None,
        "name":         name,
        "is_projected": bool(crs.is_projected),
        "utm_zone":     utm_zone,
        "wkt":          wkt,
    }


def bounds_to_latlon(bounds, crs) -> Optional[Dict[str, float]]:
    """
    Convert rasterio BoundingBox to approximate lat/lon corners.
    Returns None if conversion fails or CRS is already geographic.
    """
    if crs is None:
        return None
    try:
        from pyproj import Transformer  # type: ignore
        if crs.is_geographic:
            return {
                "min_lat": bounds.bottom, "max_lat": bounds.top,
                "min_lon": bounds.left,   "max_lon": bounds.right,
            }
        transformer = Transformer.from_crs(crs.to_epsg() or crs.to_wkt(),
                                           "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bounds.left,  bounds.bottom)
        max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
        # Sanity check
        if not (-90 <= min_lat <= 90 and -180 <= min_lon <= 180):
            return None
        return {
            "min_lat": round(min_lat, 5), "max_lat": round(max_lat, 5),
            "min_lon": round(min_lon, 5), "max_lon": round(max_lon, 5),
        }
    except Exception:
        return None
