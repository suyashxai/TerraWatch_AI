import logging
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

from backend.preprocessing import read_geotiff, preprocess_for_inference, get_rgb_preview
from backend.inference import load_model, run_inference, get_device
from backend.postprocessing import calculate_statistics, build_overlay_image, save_mask_geotiff
import imageio, numpy as np

TIFF = "data/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"

print("=== Step 1: Read GeoTIFF ===")
data, meta = read_geotiff(TIFF)
print("  Shape:", data.shape, " dtype:", data.dtype)
print("  Pixel size:", meta["pixel_width_m"], "x", meta["pixel_height_m"], "m")

print("=== Step 2: Preprocess ===")
tensor = preprocess_for_inference(data)
print("  Tensor:", tensor.shape, " mean={:.4f}  std={:.4f}".format(float(tensor.mean()), float(tensor.std())))

print("=== Step 3: Load model ===")
device = get_device()
print("  Device:", device)
model = load_model(device)
print("  Model loaded OK")

print("=== Step 4: Inference ===")
mask = run_inference(model, tensor, device)
unique = sorted(set(mask.flatten().tolist()))
print("  Mask shape:", mask.shape, " unique values:", unique)

print("=== Step 5: Statistics ===")
stats = calculate_statistics(mask, meta)
print("  Burned pixels     :", stats["burned_pixels"])
print("  Total pixels      :", stats["total_valid_pixels"])
print("  Percentage affected:", stats["percentage_affected"], "%")
print("  Burned area km2   :", stats["burned_area_km2"], "km2")

print("=== Step 6: Save outputs ===")
save_mask_geotiff(mask, meta, "outputs/test_mask.tif")
rgb = get_rgb_preview(data)
overlay = build_overlay_image(rgb, mask, alpha=128)
imageio.imwrite("outputs/test_overlay.png", overlay)
print("  Saved: outputs/test_mask.tif")
print("  Saved: outputs/test_overlay.png")

print("=== PIPELINE COMPLETE ===")
