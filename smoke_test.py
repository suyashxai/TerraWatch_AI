"""
smoke_test.py
-------------
Validates all new features without starting Streamlit.
"""
import sys, logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

TIFF = "data/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
ERRORS = []

# ── 1. Backend imports ──────────────────────────────────────────────────────
print("=== 1. Import check ===")
try:
    from backend.preprocessing import read_geotiff, preprocess_for_inference, get_rgb_preview
    from backend.inference import load_model, run_inference, get_device
    from backend.postprocessing import calculate_statistics, build_overlay_image, save_mask_geotiff
    from backend.evaluation import load_ground_truth, compute_metrics, PUBLISHED_BENCHMARK
    from backend.utils import build_stats_csv, parse_crs_info, bounds_to_latlon, human_file_size
    from backend.model import run_pipeline
    print("  All imports OK")
except Exception as e:
    ERRORS.append(f"Import failed: {e}")
    print(f"  FAIL: {e}"); sys.exit(1)

# ── 2. Pipeline ─────────────────────────────────────────────────────────────
print("=== 2. Pipeline ===")
try:
    result = run_pipeline(TIFF, output_dir="outputs")
    mask  = result["mask"]
    stats = result["stats"]
    meta  = result["meta"]
    assert set(mask.flatten().tolist()).issubset({0, 1}), "Mask has unexpected values"
    assert stats["burned_pixels"] > 0
    assert stats["burned_area_km2"] is not None
    print(f"  Burned pixels: {stats['burned_pixels']:,}")
    print(f"  Burned area  : {stats['burned_area_km2']} km2")
    print(f"  % affected   : {stats['percentage_affected']} %")
    print("  Pipeline OK")
except Exception as e:
    ERRORS.append(f"Pipeline: {e}"); print(f"  FAIL: {e}")

# ── 3. CRS helpers ──────────────────────────────────────────────────────────
print("=== 3. CRS / utils ===")
try:
    import rasterio
    with rasterio.open(TIFF) as src:
        crs    = src.crs
        bounds = src.bounds
    crs_info = parse_crs_info(crs)
    assert crs_info["epsg"] is not None or crs_info["name"] != "Not available"
    print(f"  EPSG   : {crs_info['epsg']}")
    print(f"  UTM    : {crs_info['utm_zone']}")
    latlon = bounds_to_latlon(bounds, crs)
    if latlon:
        print(f"  LatLon : {latlon['min_lat']:.4f} – {latlon['max_lat']:.4f} N, "
              f"{latlon['min_lon']:.4f} – {latlon['max_lon']:.4f} E")
    else:
        print("  LatLon : conversion returned None (acceptable)")
    print("  CRS helpers OK")
except Exception as e:
    ERRORS.append(f"CRS: {e}"); print(f"  FAIL: {e}")

# ── 4. Published benchmark keys ─────────────────────────────────────────────
print("=== 4. Published benchmark keys ===")
try:
    assert PUBLISHED_BENCHMARK["test_iou_burned"] > 0
    assert PUBLISHED_BENCHMARK["test_miou"] > 0
    assert PUBLISHED_BENCHMARK["val_iou_burned"] > 0
    assert PUBLISHED_BENCHMARK["val_miou"] > 0
    print(f"  Test  IoU Burned : {PUBLISHED_BENCHMARK['test_iou_burned']} %")
    print(f"  Test  mIoU       : {PUBLISHED_BENCHMARK['test_miou']} %")
    print(f"  Val   IoU Burned : {PUBLISHED_BENCHMARK['val_iou_burned']} %")
    print(f"  Val   mIoU       : {PUBLISHED_BENCHMARK['val_miou']} %")
    print("  Published benchmark OK")
except Exception as e:
    ERRORS.append(f"Benchmark: {e}"); print(f"  FAIL: {e}")

# ── 5. Evaluation — self-comparison (pred vs itself = perfect score) ────────
print("=== 5. Evaluation (pred vs itself — expect IoU=1.0) ===")
try:
    import numpy as np
    import tempfile, os
    import imageio
    gt_path = "outputs/_smoke_gt.png"
    imageio.imwrite(gt_path, (mask * 255).astype(np.uint8))
    gt = load_ground_truth(gt_path, reference_shape=mask.shape)
    metrics = compute_metrics(mask, gt)
    assert metrics["iou"]  == 1.0, f"Expected IoU=1.0, got {metrics['iou']}"
    assert metrics["miou"] == 1.0, f"Expected mIoU=1.0, got {metrics['miou']}"
    assert metrics["precision"] == 1.0
    assert metrics["recall"]    == 1.0
    os.remove(gt_path)
    print(f"  IoU={metrics['iou']}  mIoU={metrics['miou']}  Dice={metrics['dice']}  P={metrics['precision']}  R={metrics['recall']}")
    print("  Evaluation OK")
except Exception as e:
    ERRORS.append(f"Evaluation: {e}"); print(f"  FAIL: {e}")

# ── 6. Evaluation — shape mismatch error ────────────────────────────────────
print("=== 6. Evaluation shape mismatch -> clear error ===")
try:
    import numpy as np, imageio, tempfile
    bad_mask = np.zeros((100, 100), dtype=np.uint8)
    bad_path = "outputs/_smoke_bad.png"
    imageio.imwrite(bad_path, bad_mask)
    try:
        load_ground_truth(bad_path, reference_shape=mask.shape)
        ERRORS.append("Shape mismatch: should have raised ValueError")
    except ValueError as ve:
        print(f"  Correctly raised ValueError: {ve}")
    import os; os.remove(bad_path)
except Exception as e:
    ERRORS.append(f"Shape mismatch test: {e}"); print(f"  FAIL: {e}")

# ── 7. CSV with and without eval metrics ────────────────────────────────────
print("=== 7. CSV export ===")
try:
    csv_no_eval = build_stats_csv("test.tif", meta, stats, None)
    assert b"no_ground_truth_provided" in csv_no_eval
    csv_with_eval = build_stats_csv("test.tif", meta, stats, metrics)
    assert b"iou" in csv_with_eval
    assert b"precision" in csv_with_eval
    print("  CSV without eval : OK")
    print("  CSV with eval    : OK")
except Exception as e:
    ERRORS.append(f"CSV: {e}"); print(f"  FAIL: {e}")

# ── 8. Input validation — wrong band count ───────────────────────────────────
print("=== 8. Band-count validation ===")
try:
    from backend.preprocessing import validate_bands
    import numpy as np
    try:
        validate_bands(np.zeros((3, 512, 512)))
        ERRORS.append("Band validation: should have raised ValueError for 3-band input")
    except ValueError as ve:
        print(f"  3-band input correctly rejected: {str(ve)[:60]}")
    validate_bands(np.zeros((6, 512, 512)))
    print("  6-band input accepted: OK")
except Exception as e:
    ERRORS.append(f"Validation: {e}"); print(f"  FAIL: {e}")

# ── Summary ─────────────────────────────────────────────────────────────────
print()
if ERRORS:
    print(f"SMOKE TEST FAILED — {len(ERRORS)} error(s):")
    for e in ERRORS:
        print(f"  • {e}")
    sys.exit(1)
else:
    print("ALL SMOKE TESTS PASSED")
