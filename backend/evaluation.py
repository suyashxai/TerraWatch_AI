"""
evaluation.py
-------------
Computes binary segmentation metrics (IoU, mIoU, Dice/F1, Precision, Recall,
Confusion Matrix) when a ground-truth burn-scar mask is available.

IMPORTANT:
- Metrics are NEVER fabricated or approximated.
- If ground truth is absent, callers display "Evaluation unavailable."
- Class convention:  0 = non-burned/background,  1 = burned/burn scar.

IoU  = Intersection over Union (per-class overlap measure).
mIoU = Mean IoU averaged over both classes (background + burn scar).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: str, reference_shape: Tuple[int, int]) -> np.ndarray:
    """
    Load a single-band ground-truth mask GeoTIFF or PNG and validate
    it against the reference (predicted mask) shape.

    Parameters
    ----------
    path            : file path to the ground-truth mask
    reference_shape : (H, W) of the predicted mask

    Returns
    -------
    gt : np.ndarray  (H, W) uint8 — 0=background, 1=burned

    Raises
    ------
    ValueError  if the file cannot be read or shapes are incompatible
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            gt = src.read(1).astype(np.uint8)
    except Exception:
        try:
            import imageio  # type: ignore
            raw = np.array(imageio.imread(path))
            gt = raw[..., 0] if raw.ndim == 3 else raw
            gt = gt.astype(np.uint8)
        except Exception as exc:
            raise ValueError(
                f"Could not read ground-truth mask: {exc}. "
                "Please upload a single-band GeoTIFF or PNG."
            ) from exc

    # Binarise: any value > 0 → 1
    gt = (gt > 0).astype(np.uint8)

    if gt.shape != reference_shape:
        raise ValueError(
            f"Shape mismatch: predicted mask is {reference_shape[0]}\u00d7{reference_shape[1]} px "
            f"but ground-truth mask is {gt.shape[0]}\u00d7{gt.shape[1]} px. "
            "Both masks must have identical dimensions."
        )

    return gt


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _iou_for_class(pred_b: np.ndarray, gt_b: np.ndarray) -> float:
    """IoU for a single binary class."""
    tp = int(( pred_b &  gt_b).sum())
    fp = int(( pred_b & ~gt_b).sum())
    fn = int((~pred_b &  gt_b).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute binary segmentation metrics.

    Parameters
    ----------
    pred : (H, W) uint8 — predicted mask (0/1)
    gt   : (H, W) uint8 — ground-truth mask (0/1)

    Returns
    -------
    dict with keys:
        iou          – IoU for burned class (class 1)
        iou_bg       – IoU for background class (class 0)
        miou         – mean IoU across both classes
        dice         – Dice/F1 for burned class
        f1           – same as dice
        precision    – precision for burned class
        recall       – recall for burned class
        true_positives, false_positives, false_negatives, true_negatives
        confusion_matrix  (2x2 np.ndarray, [[TN,FP],[FN,TP]])
    """
    pred_burn = (pred == 1)
    gt_burn   = (gt   == 1)
    pred_bg   = ~pred_burn
    gt_bg     = ~gt_burn

    tp = int(( pred_burn &  gt_burn).sum())
    fp = int(( pred_burn & ~gt_burn).sum())
    fn = int((~pred_burn &  gt_burn).sum())
    tn = int((~pred_burn & ~gt_burn).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    iou_burn = _iou_for_class(pred_burn, gt_burn)
    iou_bg   = _iou_for_class(pred_bg,   gt_bg)
    miou     = (iou_burn + iou_bg) / 2.0

    confusion = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    return {
        "iou":               round(iou_burn, 4),
        "iou_bg":            round(iou_bg,   4),
        "miou":              round(miou,     4),
        "dice":              round(f1,       4),
        "f1":                round(f1,       4),
        "precision":         round(precision, 4),
        "recall":            round(recall,    4),
        "true_positives":    tp,
        "false_positives":   fp,
        "false_negatives":   fn,
        "true_negatives":    tn,
        "confusion_matrix":  confusion,
    }


# ---------------------------------------------------------------------------
# Published benchmark — clearly labelled, never shown as user-image results
# ---------------------------------------------------------------------------

# Source: ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars model card
# (https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars)
# These are aggregate results on the held-out HLS BurnScars benchmark test set.
PUBLISHED_BENCHMARK: Dict[str, Any] = {
    "source": (
        "IBM-NASA Prithvi-EO-2.0-300M-BurnScars — published benchmark "
        "(HLS BurnScars dataset, held-out test split)"
    ),
    # Test set
    "test_iou_burned":  87.52,   # %
    "test_miou":        93.00,   # %
    # Validation set
    "val_iou_burned":   84.28,   # %
    "val_miou":         90.95,   # %
    "note": (
        "Published by ibm-nasa-geospatial on Hugging Face. "
        "These are aggregate results on the held-out benchmark dataset "
        "and DO NOT reflect performance on any individual uploaded image."
    ),
}
