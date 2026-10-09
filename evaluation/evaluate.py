"""
evaluate.py
-----------
Evaluation script for TerraWatch AI burn-scar predictions.

Computes standard binary segmentation metrics when a ground-truth mask is
available.  Metrics are NEVER fabricated — if ground truth is absent the
script exits cleanly with a clear message.

Usage
-----
    python evaluation/evaluate.py \
        --pred  outputs/my_image_mask.tif \
        --gt    data/ground_truth/my_image_gt.tif

Optional: evaluate a whole directory:
    python evaluation/evaluate.py \
        --pred_dir  outputs/ \
        --gt_dir    data/ground_truth/ \
        --output    evaluation/results.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute binary segmentation metrics.

    Parameters
    ----------
    pred : np.ndarray uint8  (H, W)  — predicted mask (0/1)
    gt   : np.ndarray uint8  (H, W)  — ground-truth mask (0/1)

    Returns
    -------
    dict with keys: iou, dice, precision, recall, f1
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred.shape} vs ground truth {gt.shape}."
        )

    pred_bool = (pred == 1)
    gt_bool   = (gt   == 1)

    tp = int((pred_bool & gt_bool).sum())
    fp = int((pred_bool & ~gt_bool).sum())
    fn = int((~pred_bool & gt_bool).sum())

    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    f1         = 2 * precision * recall / (precision + recall + 1e-8)
    iou        = tp / (tp + fp + fn + 1e-8)
    dice       = f1  # Dice == F1 for binary segmentation

    return {
        "true_positives":  tp,
        "false_positives": fp,
        "false_negatives": fn,
        "iou":             round(iou, 6),
        "dice":            round(dice, 6),
        "precision":       round(precision, 6),
        "recall":          round(recall, 6),
        "f1":              round(f1, 6),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_mask(path: str) -> np.ndarray:
    """Load a single-band GeoTIFF or PNG mask and binarise it."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(1)
    except Exception:
        import imageio  # type: ignore
        data = np.array(imageio.imread(path))
        if data.ndim == 3:
            data = data[..., 0]

    # Binarise: anything > 0 → 1
    return (data > 0).astype(np.uint8)


def save_csv(results: list, out_path: str) -> None:
    """Write a list of result dicts to a CSV file."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Evaluation results saved to %s", out_path)


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(pred_path: str, gt_path: str) -> Dict:
    """Load a prediction/ground-truth pair and return metrics."""
    if not os.path.exists(gt_path):
        logger.warning(
            "Ground-truth file not found: %s — skipping. "
            "Ground-truth evaluation not available for this prediction.",
            gt_path,
        )
        return {}

    pred = load_mask(pred_path)
    gt   = load_mask(gt_path)
    metrics = compute_metrics(pred, gt)
    metrics["pred_file"] = os.path.basename(pred_path)
    metrics["gt_file"]   = os.path.basename(gt_path)
    return metrics


# ---------------------------------------------------------------------------
# Directory evaluation
# ---------------------------------------------------------------------------

def evaluate_directory(
    pred_dir: str,
    gt_dir: str,
    output_csv: Optional[str] = None,
) -> None:
    """
    Match prediction files with ground-truth files by stem and evaluate each pair.
    Files are matched by filename stem (e.g. 'scene001_mask.tif' ↔ 'scene001.tif').
    """
    pred_files = sorted(Path(pred_dir).glob("*_mask.tif"))
    if not pred_files:
        logger.error("No *_mask.tif files found in %s", pred_dir)
        return

    all_results = []
    for pred_path in pred_files:
        # Try to find GT with matching stem (strip trailing _mask)
        stem = pred_path.stem.replace("_mask", "")
        gt_candidates = list(Path(gt_dir).glob(f"{stem}*.tif"))
        if not gt_candidates:
            logger.warning(
                "No ground-truth file found for %s — "
                "Ground-truth evaluation not available for this prediction.",
                pred_path.name,
            )
            continue

        metrics = evaluate_pair(str(pred_path), str(gt_candidates[0]))
        if metrics:
            all_results.append(metrics)
            logger.info(
                "%s → IoU=%.4f  Dice=%.4f  Precision=%.4f  Recall=%.4f",
                pred_path.name,
                metrics["iou"],
                metrics["dice"],
                metrics["precision"],
                metrics["recall"],
            )

    if all_results:
        # Print aggregate summary
        keys = ["iou", "dice", "precision", "recall", "f1"]
        print("\n=== Aggregate Evaluation (mean across all scenes) ===")
        for k in keys:
            vals = [r[k] for r in all_results]
            print(f"  {k:12s}: {np.mean(vals):.4f}")

        if output_csv:
            save_csv(all_results, output_csv)
    else:
        print(
            "\nGround-truth evaluation not available — "
            "no valid prediction/ground-truth pairs found."
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate TerraWatch AI burn-scar predictions against ground truth.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred",      help="Path to a single prediction mask (GeoTIFF or PNG).")
    mode.add_argument("--pred_dir",  help="Directory of prediction masks (*_mask.tif).")

    parser.add_argument("--gt",      help="Path to a single ground-truth mask.")
    parser.add_argument("--gt_dir",  help="Directory of ground-truth masks.")
    parser.add_argument("--output",  help="Output CSV path for batch evaluation results.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.pred:
        # Single-pair mode
        if not args.gt:
            print(
                "Ground-truth evaluation not available for this prediction.\n"
                "Please supply --gt <path> to enable metric calculation."
            )
            sys.exit(0)

        metrics = evaluate_pair(args.pred, args.gt)
        if metrics:
            print("\n=== Evaluation Metrics ===")
            for k, v in metrics.items():
                print(f"  {k:16s}: {v}")
        else:
            print("Ground-truth evaluation not available for this prediction.")

    elif args.pred_dir:
        if not args.gt_dir:
            print(
                "Ground-truth evaluation not available.\n"
                "Please supply --gt_dir <path> to enable batch evaluation."
            )
            sys.exit(0)
        evaluate_directory(args.pred_dir, args.gt_dir, args.output)


if __name__ == "__main__":
    main()
