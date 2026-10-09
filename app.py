"""
app.py
------
TerraWatch AI — Post-Wildfire Burn Scar Detection Dashboard
IBM-NASA Prithvi-EO-2.0-300M-BurnScars · Semantic Segmentation

Run:
    streamlit run app.py
"""

from __future__ import annotations

import datetime
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

import rasterio
from rasterio.io import MemoryFile

from backend.utils import (
    build_stats_csv,
    bounds_to_latlon,
    configure_logging,
    geotiff_bytes,
    human_file_size,
    numpy_to_png_bytes,
    parse_crs_info,
)
from backend.postprocessing import apply_opacity

configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TerraWatch AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 16px;
}
h1 { color: #1f2328; }
h2, h3 { color: #3b82d4; }
.caption-muted { color: #57606a; font-size: 0.85rem; }
.info-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.eval-unavailable {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 6px;
    padding: 12px 16px;
    color: #5d4037;
    font-size: 0.9rem;
}
.benchmark-box {
    background: #e8f4fd;
    border: 1px solid #90caf9;
    border-radius: 6px;
    padding: 14px 18px;
    color: #0d47a1;
    font-size: 0.88rem;
}
.interpretation-box {
    background: #fff3e0;
    border: 1px solid #ffcc80;
    border-radius: 6px;
    padding: 14px 18px;
    color: #4e342e;
    font-size: 0.95rem;
}
.limitation-box {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    border-left: 4px solid #ef9a9a;
    border-radius: 4px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #424242;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌍 TerraWatch AI")
    st.markdown(
        "<span style='color:#57606a;font-size:0.85rem;'>Post-Wildfire Burn Scar Detection</span>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.expander("🤖 Model", expanded=True):
        st.markdown("""
**Prithvi-EO-2.0-300M-BurnScars**  
IBM × NASA foundation model

**Task:** Semantic Segmentation  
**Input:** 6-band HLS GeoTIFF  
**Resolution:** ~30 m/pixel  
**Classes:** 0 = Background · 1 = Burn scar  
**Confidence:** Not available

[🔗 Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars)
""")

    with st.expander("📋 Input Requirements"):
        st.markdown("""
- Format: **GeoTIFF** (.tif / .tiff)
- Bands: exactly **6 HLS bands**  
  1. Blue · 2. Green · 3. Red  
  4. Narrow NIR · 5. SWIR1 · 6. SWIR2
- Recommended: **512 × 512 px**
- Source: **HLS (Harmonised Landsat Sentinel-2)**
""")

    with st.expander("🌍 About"):
        st.markdown("""
TerraWatch AI performs **pixel-level burn-scar mapping**
from satellite imagery acquired after a wildfire event.

It uses IBM-NASA Prithvi semantic segmentation to label
every pixel as burned or non-burned.

**Use cases:**
- 🔥 Post-wildfire damage assessment
- 🌲 Forest management & replanting
- 🚨 Emergency response support
- 📊 Environmental monitoring
- 🌡️ Climate & ecosystem research

> *Detects observed burn scars from past satellite imagery.  
> Does not predict future wildfires.*
""")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🌍 TerraWatch AI")
st.markdown(
    "<p style='color:#57606a;font-size:1.05rem;margin-top:-0.4rem;'>"
    "Post-Wildfire Burn Scar Detection &nbsp;·&nbsp; "
    "IBM-NASA Prithvi-EO-2.0-300M-BurnScars &nbsp;·&nbsp; "
    "Semantic Segmentation"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for _k in (
    "result", "upload_meta", "tmp_path",
    "crs_obj", "bounds_obj", "crs_info_cache",
    "eval_metrics", "gt_tmp_path",
    "file_bytes",
):
    if _k not in st.session_state:
        st.session_state[_k] = None

# ===========================================================================
# 📡 Upload Satellite Image
# ===========================================================================
st.markdown("## 📡 Upload Satellite Image")
st.markdown(
    "Upload a **6-band HLS GeoTIFF** (.tif / .tiff) containing "
    "Blue, Green, Red, Narrow NIR, SWIR1 and SWIR2 spectral bands.  \n"
    "JPG, PNG and regular RGB images are **not supported** — "
    "they lack the six spectral bands required by the Prithvi model."
)

# Sample image helper — shown before any upload
with st.expander("📥 Don't have a compatible HLS image? Download the official sample."):
    st.markdown(
        "The Prithvi model repository includes an official sample HLS GeoTIFF "
        "you can use to test TerraWatch AI immediately.  \n"
        "It is a genuine 6-band HLS Sentinel-2 scene over Northern California."
    )
    SAMPLE_HF_URL = (
        "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"
        "/resolve/main/subsetted_512x512_HLS.S30.T10SEH.2018212.v1.4.tif"
    )
    local_sample = Path("data/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif")
    if local_sample.exists():
        with open(local_sample, "rb") as _f:
            _sample_bytes = _f.read()
        st.download_button(
            "📥 Download Sample HLS Image (local copy)",
            data=_sample_bytes,
            file_name=local_sample.name,
            mime="image/tiff",
            use_container_width=True,
        )
        st.caption(
            "This is the test image already present in your `data/` folder. "
            "Upload it above to test the full pipeline."
        )
    else:
        st.link_button(
            "🔗 Download Official Sample from Hugging Face",
            SAMPLE_HF_URL,
            use_container_width=True,
        )
        st.caption(
            "Downloads `subsetted_512x512_HLS.S30.T10SEH.2018212.v1.4.tif` "
            "— an official 6-band HLS GeoTIFF from the Prithvi model repository."
        )

uploaded_file = st.file_uploader(
    "Select GeoTIFF file",
    type=["tif", "tiff"],
    help="Multi-band HLS GeoTIFF with exactly 6 spectral bands.",
)

_valid_upload = False

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    try:
        with MemoryFile(file_bytes) as mem:
            with mem.open() as src:
                width     = src.width
                height    = src.height
                count     = src.count
                crs       = src.crs
                transform = src.transform
                bounds    = src.bounds
                nodata    = src.nodata

        pw = abs(transform.a) if transform else None
        ph = abs(transform.e) if transform else None
        if crs is not None and crs.is_geographic:
            pw = ph = None

        # ---- Band-count validation ----
        if count == 3:
            st.error(
                "❌ **RGB image detected — not supported.**  \n"
                "TerraWatch AI requires a **6-band HLS GeoTIFF** containing "
                "Blue, Green, Red, Narrow NIR, SWIR1 and SWIR2 bands.  \n"
                "Regular RGB images (3 bands) do not contain the required spectral information."
            )
        elif count == 1:
            st.error(
                "❌ **Single-band image detected — not supported.**  \n"
                "TerraWatch AI requires a compatible **6-band HLS GeoTIFF**."
            )
        elif count != 6:
            st.error(
                f"❌ **Unsupported input — {count} band(s) detected.**  \n"
                "TerraWatch AI requires a compatible **6-band HLS GeoTIFF** "
                "(Blue, Green, Red, NIR, SWIR1, SWIR2).  \n"
                "Please re-export your imagery with all six HLS bands."
            )
        else:
            st.success("✅ Valid 6-band GeoTIFF detected — ready for inference.")
            _valid_upload = True

            # Only write a new temp file when the uploaded filename has changed.
            _prev_meta = st.session_state.get("upload_meta") or {}
            _is_new_file = (_prev_meta.get("filename") != uploaded_file.name)

            if _is_new_file:
                # Write to system temp dir (survives across reruns on same session)
                with tempfile.NamedTemporaryFile(
                    suffix=".tif",
                    delete=False,
                    dir=tempfile.gettempdir()
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp.flush()
                    temp_path = tmp.name

                logger.info("Uploaded file saved to temp path: %s", temp_path)

                st.session_state.tmp_path    = temp_path
                st.session_state.file_bytes  = file_bytes
                st.session_state.upload_meta = {
                    "filename": uploaded_file.name,
                    "width": width, "height": height, "count": count,
                    "pw": pw, "ph": ph, "nodata": nodata,
                }
                st.session_state.crs_obj         = crs
                st.session_state.bounds_obj      = bounds
                st.session_state.crs_info_cache  = parse_crs_info(crs)
                # Reset results when a genuinely new file is uploaded
                st.session_state.result          = None
                st.session_state.eval_metrics    = None
            else:
                # Same file — reuse what is already in session state
                file_bytes = st.session_state.file_bytes

    except Exception as exc:
        st.error(
            "❌ **Could not open this file as a GeoTIFF.**  \n"
            "Please upload a valid .tif or .tiff file.  \n"
            f"*({type(exc).__name__})*"
        )
        logger.exception("File read error: %s", exc)

# ===========================================================================
# 🛰️ Satellite Image Information
# ===========================================================================
if _valid_upload and st.session_state.upload_meta:
    um       = st.session_state.upload_meta
    crs_obj  = st.session_state.crs_obj
    bounds   = st.session_state.bounds_obj
    crs_info = st.session_state.crs_info_cache or parse_crs_info(crs_obj)
    latlon   = bounds_to_latlon(bounds, crs_obj)

    st.divider()
    st.markdown("## 🛰️ Satellite Image Information")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dimensions",       f"{um['width']} × {um['height']} px")
    c2.metric("Spectral Bands",   str(um["count"]))
    c3.metric("Resolution",
              f"{um['pw']:.0f} × {um['ph']:.0f} m"
              if (um["pw"] and um["ph"]) else "Not available")
    _fb = st.session_state.file_bytes
    c4.metric("File Size", human_file_size(len(_fb)) if _fb else "—")

    st.markdown("<div class='info-card'>", unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("**CRS / Projection**")
        auth = crs_info.get("authority")
        st.markdown(f"`{auth}`" if auth else "Not available")
    with cc2:
        st.markdown("**UTM Zone**")
        st.markdown(crs_info.get("utm_zone") or "Not identified")
    with cc3:
        st.markdown("**Projection type**")
        st.markdown("Projected" if crs_info.get("is_projected") else "Geographic / Unknown")
    st.markdown("</div>", unsafe_allow_html=True)

    if latlon:
        st.markdown(
            f"**Approximate scene location (WGS 84):** "
            f"Lat {latlon['min_lat']:.4f}° – {latlon['max_lat']:.4f}° &nbsp;·&nbsp; "
            f"Lon {latlon['min_lon']:.4f}° – {latlon['max_lon']:.4f}°"
        )
    else:
        st.caption("Scene bounds in WGS 84 could not be determined from this file's CRS.")

# ===========================================================================
# 🔥 Burn-Scar Detection
# ===========================================================================
if _valid_upload and st.session_state.get("tmp_path"):
    st.divider()
    st.markdown("## 🔥 Burn-Scar Detection")

    if st.button("🔥 Detect Burn Scars", type="primary", use_container_width=True):
        with st.spinner(
            "Running Prithvi-EO-2.0-300M-BurnScars segmentation…  "
            "(first run loads model weights — approx. 30–120 s on CPU)"
        ):
            try:
                from backend.model import run_pipeline
                result = run_pipeline(
                    tiff_path=st.session_state.tmp_path,
                    output_dir="outputs",
                )
                st.session_state.result       = result
                st.session_state.eval_metrics = None
                st.success("✅ Burn-scar segmentation complete.")
            except FileNotFoundError as exc:
                st.error(
                    "❌ **Model weights not found.**  \n"
                    "Place `Prithvi_EO_V2_300M_BurnScars.pt` in the `models/` directory.  \n"
                    "See `models/README.md` for download instructions."
                )
                logger.error("Weights not found: %s", exc)
            except ValueError as exc:
                st.error(f"❌ **Input error:** {exc}")
                logger.error("Input validation error: %s", exc)
            except RuntimeError as exc:
                st.error(
                    f"❌ **Inference error:**  \n{exc}  \n\n"
                    "If you are on Streamlit Cloud, ensure the model weights file "
                    "`Prithvi_EO_V2_300M_BurnScars.pt` is present in the `models/` directory."
                )
                logger.exception("RuntimeError during pipeline: %s", exc)
            except MemoryError:
                st.error(
                    "❌ **Out of memory.**  \n"
                    "The server ran out of RAM. Try a smaller image (512×512 px recommended)."
                )
                logger.error("MemoryError during pipeline")
            except Exception as exc:
                st.error(
                    "❌ **Burn-scar detection failed.**  \n"
                    f"Error: `{type(exc).__name__}: {exc}`  \n"
                    "Please check the input image and model configuration."
                )
                logger.exception("Pipeline error: %s", exc)

# ===========================================================================
# Results block — only when inference has completed
# ===========================================================================
result: Optional[Dict[str, Any]] = st.session_state.get("result")

if result is not None:
    # Guard: ensure all required keys are present in result
    _required_keys = ("rgb_preview", "mask", "stats", "meta")
    _missing = [k for k in _required_keys if k not in result]
    if _missing:
        st.error(
            f"❌ **Result data is incomplete** (missing keys: {_missing}).  \n"
            "Please re-run detection."
        )
        st.stop()

    rgb              = result["rgb_preview"]
    mask             = result["mask"]
    stats            = result["stats"]
    meta             = result["meta"]

    # Safe access to upload_meta — may be None if session was reset
    _upload_meta = st.session_state.get("upload_meta") or {}
    fname            = _upload_meta.get("filename", "result")
    inference_time_s = result.get("inference_time_s")
    device_name      = result.get("device_name")
    crs_info         = st.session_state.crs_info_cache or {}

    # Ensure mask is a proper 2-D uint8 array before display
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        st.error("❌ **Prediction mask has unexpected shape.** Please re-run detection.")
        st.stop()
    mask = mask.astype(np.uint8)

    # Ensure rgb is a proper (H, W, 3) uint8 array
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        st.error("❌ **RGB preview has unexpected shape.** Please re-run detection.")
        st.stop()
    rgb = rgb.astype(np.uint8)

    # -----------------------------------------------------------------------
    # Visual results (opacity slider + 3 images)
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## 🖼️ Burn-Scar Detection Results")

    opacity_pct = st.slider(
        "🎨 Overlay Opacity",
        min_value=0, max_value=100, value=50, step=5,
        help="Blend strength of the red burn-scar overlay on the original image.",
    )
    try:
        overlay_rgb = apply_opacity(rgb, mask, opacity_pct / 100.0)
    except Exception as _oe:
        logger.warning("apply_opacity failed: %s — falling back to plain RGB", _oe)
        overlay_rgb = rgb.copy()

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        st.markdown("**Original Satellite Image**")
        st.markdown(
            "<p class='caption-muted'>True-colour composite (R/G/B bands)</p>",
            unsafe_allow_html=True,
        )
        st.image(rgb, use_container_width=True)
    with ic2:
        st.markdown("**Predicted Burn-Scar Mask**")
        st.markdown(
            "<p class='caption-muted'>White = burned &nbsp;·&nbsp; Black = background</p>",
            unsafe_allow_html=True,
        )
        # Explicit uint8 conversion prevents st.image crash on non-standard dtypes
        _mask_display = (mask.astype(np.uint8) * 255)
        st.image(_mask_display, use_container_width=True, clamp=True)
    with ic3:
        st.markdown("**Burn-Scar Overlay**")
        st.markdown(
            f"<p class='caption-muted'>Red overlay at {opacity_pct}% opacity</p>",
            unsafe_allow_html=True,
        )
        st.image(overlay_rgb, use_container_width=True)

    # -----------------------------------------------------------------------
    # 📊 Area Statistics
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## 📊 Area Statistics")

    burned_km2 = stats.get("burned_area_km2")
    pct        = stats.get("percentage_affected")
    pw         = stats.get("pixel_width_m")
    ph         = stats.get("pixel_height_m")
    burned_px  = stats.get("burned_pixels", 0)
    total_px   = stats.get("total_valid_pixels", 0)

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric(
        "🔥 Burned Area",
        f"{burned_km2:.2f} km²" if burned_km2 is not None else "Not available",
        help="burned_pixels × pixel_area_m² / 1,000,000",
    )
    sc2.metric(
        "📊 Area Affected",
        f"{pct:.2f} %" if pct is not None else "Not available",
    )
    sc3.metric("🟫 Burned Pixels",  f"{burned_px:,}")
    sc4.metric(
        "🛰️ Spatial Resolution",
        f"{pw:.0f} × {ph:.0f} m" if (pw and ph) else "Not available",
    )

    if not stats.get("resolution_available"):
        st.warning(
            "⚠️ Spatial resolution unavailable — burned area in km² cannot be calculated. "
            "The GeoTIFF does not contain a projected CRS with metric units."
        )

    st.caption(
        "**Confidence:** Not available — "
        "the model performs pixel-level semantic segmentation and does not produce "
        "a per-image confidence score."
    )

    # ---- Plain-language interpretation ----
    if burned_km2 is not None and pct is not None:
        st.markdown(
            "<div class='interpretation-box'>"
            f"🔥 <strong>Burn scar detected.</strong> Approximately "
            f"<strong>{burned_km2:.2f} km²</strong> of the analysed scene was classified "
            f"as burned, representing <strong>{pct:.2f}%</strong> of the valid image area "
            f"({burned_px:,} out of {total_px:,} pixels)."
            "</div>",
            unsafe_allow_html=True,
        )
    elif burned_px == 0:
        st.markdown(
            "<div class='interpretation-box'>"
            "✅ No burn scar detected in this scene."
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 🎯 Model Evaluation
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## 🎯 Model Evaluation")

    # ---- A. Published Benchmark ----
    from backend.evaluation import PUBLISHED_BENCHMARK
    st.markdown("### 📚 Prithvi Published Benchmark")
    st.markdown(
        "<div class='benchmark-box'>"
        "ℹ️ <strong>Published benchmark results — reported by the IBM-NASA model authors.</strong><br>"
        "These metrics are computed on the held-out HLS BurnScars test/validation dataset "
        "and are <strong>NOT</strong> calculated from your currently uploaded image. "
        "Do not interpret these as the accuracy for your specific scene."
        "</div>",
        unsafe_allow_html=True,
    )

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric(
        "IoU — Burned Class (Test)",
        f"{PUBLISHED_BENCHMARK['test_iou_burned']:.2f} %",
        help="Intersection over Union for the burned class on the held-out test set.",
    )
    bm2.metric(
        "mIoU (Test)",
        f"{PUBLISHED_BENCHMARK['test_miou']:.2f} %",
        help="Mean IoU across background + burned classes on the test set.",
    )
    bm3.metric(
        "IoU — Burned Class (Val)",
        f"{PUBLISHED_BENCHMARK['val_iou_burned']:.2f} %",
    )
    bm4.metric(
        "mIoU (Val)",
        f"{PUBLISHED_BENCHMARK['val_miou']:.2f} %",
    )

    with st.expander("ℹ️ What do IoU and mIoU mean?"):
        st.markdown("""
**IoU (Intersection over Union)**  
Measures the overlap between the predicted burn-scar region and the actual (ground-truth) burn-scar region.  
- IoU = 1.0 → perfect match  
- IoU = 0.0 → no overlap  
- Formula: `TP / (TP + FP + FN)`

**mIoU (Mean Intersection over Union)**  
The average IoU computed over *all* segmentation classes (background + burned).  
It gives a balanced view of how well the model performs across every class, not just the burned class.  
- Formula: `(IoU_background + IoU_burned) / 2`

For burn-scar mapping, the **IoU (Burned class)** is the most important metric because  
it directly measures how accurately the model identifies burned pixels.
""")

    # ---- B. Evaluate on Your Data ----
    st.markdown("### 🎯 Evaluate on Your Data")
    st.markdown(
        "To compute evaluation metrics for **your specific uploaded image**, "
        "provide a matching ground-truth burn-scar mask.  \n"
        "The mask must be a single-band GeoTIFF or PNG with the **same pixel dimensions** "
        "as the prediction, using values: `0` = background · `1` = burned."
    )

    gt_file = st.file_uploader(
        "Upload ground-truth mask (optional)",
        type=["tif", "tiff", "png"],
        help="Single-band binary mask: 0=background, 1=burn scar. Same dimensions as prediction.",
        key="gt_uploader",
    )

    if gt_file is not None:
        gt_bytes = gt_file.read()
        try:
            gt_tmp = tempfile.NamedTemporaryFile(
                suffix=Path(gt_file.name).suffix, delete=False
            )
            gt_tmp.write(gt_bytes)
            gt_tmp.flush()
            gt_tmp.close()
            logger.info("Ground-truth file saved to temp path: %s", gt_tmp.name)
            st.session_state.gt_tmp_path = gt_tmp.name
        except Exception as exc:
            st.error(f"❌ Could not save ground-truth file: {exc}")
            logger.exception("GT temp file error: %s", exc)

        if st.session_state.gt_tmp_path and st.button("▶️ Run Evaluation", key="run_eval"):
            from backend.evaluation import load_ground_truth, compute_metrics
            try:
                gt_mask = load_ground_truth(
                    st.session_state.gt_tmp_path,
                    reference_shape=mask.shape,
                )
                ev = compute_metrics(mask, gt_mask)
                st.session_state.eval_metrics = ev
                st.success("✅ Evaluation complete.")
            except ValueError as exc:
                st.error(f"❌ {exc}")
            except Exception as exc:
                st.error(f"❌ Evaluation failed: {type(exc).__name__}: {exc}")
                logger.exception("Evaluation error: %s", exc)

    eval_metrics: Optional[Dict] = st.session_state.get("eval_metrics")

    if eval_metrics:
        st.markdown("#### Results on Your Data")
        st.caption(
            "Calculated by comparing the Prithvi prediction against your uploaded ground-truth mask."
        )
        ev1, ev2, ev3, ev4, ev5 = st.columns(5)
        ev1.metric("IoU (Burned)",  f"{eval_metrics['iou']  * 100:.2f} %")
        ev2.metric("mIoU",          f"{eval_metrics['miou'] * 100:.2f} %")
        ev3.metric("Dice / F1",     f"{eval_metrics['dice'] * 100:.2f} %")
        ev4.metric("Precision",     f"{eval_metrics['precision'] * 100:.2f} %")
        ev5.metric("Recall",        f"{eval_metrics['recall']    * 100:.2f} %")

        with st.expander("🔢 Confusion Matrix"):
            cm = eval_metrics["confusion_matrix"]
            st.markdown(f"""
| | **Predicted: Background** | **Predicted: Burned** |
|---|---|---|
| **Actual: Background** | {cm[0,0]:,} (TN) | {cm[0,1]:,} (FP) |
| **Actual: Burned** | {cm[1,0]:,} (FN) | {cm[1,1]:,} (TP) |
""")
    else:
        st.markdown(
            "<div class='eval-unavailable'>"
            "⚠️ <strong>Evaluation unavailable for this image because no ground-truth mask "
            "was provided.</strong><br>"
            "Upload a matching ground-truth mask above and click "
            "<em>Run Evaluation</em> to calculate IoU, mIoU, Dice/F1, Precision and Recall."
            "</div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # 📥 Download Results
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## 📥 Download Results")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        # Safe mask GeoTIFF download — re-generate from in-memory data if file gone
        _mask_path = result.get("mask_path", "")
        try:
            if _mask_path and os.path.exists(_mask_path):
                _mask_bytes_dl = geotiff_bytes(_mask_path)
            else:
                # Regenerate from memory
                from backend.postprocessing import save_mask_geotiff as _smg
                _tmp_mask = tempfile.NamedTemporaryFile(suffix="_mask.tif", delete=False)
                _tmp_mask.close()
                _smg(mask, meta, _tmp_mask.name)
                _mask_bytes_dl = geotiff_bytes(_tmp_mask.name)
                try:
                    os.remove(_tmp_mask.name)
                except OSError:
                    pass
        except Exception as _mde:
            logger.warning("Mask download prep failed: %s", _mde)
            _mask_bytes_dl = b""

        st.download_button(
            "📥 Burn-Scar Mask (GeoTIFF)",
            data=_mask_bytes_dl,
            file_name=f"{Path(fname).stem}_mask.tif",
            mime="image/tiff",
            use_container_width=True,
            help="Binary burn-scar mask with original CRS and affine transform preserved.",
            disabled=len(_mask_bytes_dl) == 0,
        )

    with dl2:
        try:
            _overlay_png_bytes = numpy_to_png_bytes(overlay_rgb)
        except Exception as _oe2:
            logger.warning("Overlay PNG encode failed: %s", _oe2)
            _overlay_png_bytes = b""
        st.download_button(
            "🖼️ Burn-Scar Overlay (PNG)",
            data=_overlay_png_bytes,
            file_name=f"{Path(fname).stem}_overlay.png",
            mime="image/png",
            use_container_width=True,
            disabled=len(_overlay_png_bytes) == 0,
        )

    with dl3:
        epsg_val = crs_info.get("epsg") if crs_info else None
        csv_bytes = build_stats_csv(
            fname, meta, stats,
            eval_metrics=eval_metrics,
            inference_time_s=inference_time_s,
            device_name=device_name,
            epsg=epsg_val,
        )
        st.download_button(
            "📊 Statistics (CSV)",
            data=csv_bytes,
            file_name=f"{Path(fname).stem}_stats.csv",
            mime="text/csv",
            use_container_width=True,
            help="Includes IoU/mIoU/Dice/Precision/Recall if ground-truth was provided.",
        )

    st.caption(
        "The prediction GeoTIFF preserves the original CRS and affine transform. "
        "The CSV includes evaluation metrics only when a ground-truth mask was provided."
    )

    # -----------------------------------------------------------------------
    # ⚙️ System Performance
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## ⚙️ System Performance")
    sp1, sp2 = st.columns(2)
    sp1.metric(
        "Inference Time",
        f"{inference_time_s:.1f} s" if inference_time_s is not None else "Not available",
        help="Wall-clock time for the Prithvi forward pass only (excludes model loading).",
    )
    sp2.metric(
        "Device",
        device_name or "Not available",
    )
    st.caption(
        "Inference time covers only the forward pass. "
        "First-run model loading (weights into memory) is not included."
    )

    # -----------------------------------------------------------------------
    # 📄 Analysis Report
    # -----------------------------------------------------------------------
    st.divider()
    st.markdown("## 📄 Analysis Report")

    _now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _epsg_str = (
        f"EPSG:{crs_info['epsg']}" if (crs_info and crs_info.get("epsg"))
        else (crs_info.get("authority") or "Not available") if crs_info else "Not available"
    )
    _res_str = (
        f"{pw:.0f} × {ph:.0f} m" if (pw and ph) else "Not available"
    )

    _um = st.session_state.get("upload_meta") or {}
    report_lines = [
        "# TerraWatch AI — Analysis Report",
        f"Generated: {_now}",
        "",
        "## Input",
        f"- Filename: {fname}",
        f"- Dimensions: {_um.get('width', '?')} × {_um.get('height', '?')} px",
        f"- Bands: {_um.get('count', '?')}",
        f"- Spatial resolution: {_res_str}",
        f"- CRS: {_epsg_str}",
        "",
        "## Burned Area Statistics",
        f"- Burned pixels: {burned_px:,}",
        f"- Total valid pixels: {total_px:,}",
        f"- Burned area: {f'{burned_km2:.2f} km²' if burned_km2 is not None else 'Not available'}",
        f"- Percentage affected: {f'{pct:.2f} %' if pct is not None else 'Not available'}",
        "",
        "## Model",
        "- Model: IBM-NASA Prithvi-EO-2.0-300M-BurnScars",
        "- Task: Semantic Segmentation",
        "- Confidence: Not available",
        f"- Inference time: {f'{inference_time_s:.1f} s' if inference_time_s else 'Not available'}",
        f"- Device: {device_name or 'Not available'}",
        "",
    ]

    if eval_metrics:
        report_lines += [
            "## Evaluation (against uploaded ground-truth mask)",
            f"- IoU (Burned): {eval_metrics['iou'] * 100:.2f} %",
            f"- mIoU: {eval_metrics['miou'] * 100:.2f} %",
            f"- Dice / F1: {eval_metrics['dice'] * 100:.2f} %",
            f"- Precision: {eval_metrics['precision'] * 100:.2f} %",
            f"- Recall: {eval_metrics['recall'] * 100:.2f} %",
            "",
        ]
    else:
        report_lines.append(
            "## Evaluation\n"
            "- No ground-truth mask provided — evaluation metrics not available.\n"
        )

    report_lines += [
        "## Important Limitations",
        "- TerraWatch AI performs post-wildfire burn-scar mapping.",
        "- It is NOT a real-time wildfire detection system.",
        "- It does NOT predict future wildfire spread.",
        "- Results depend on input image quality, date, and spectral characteristics.",
        "- Evaluation metrics for an individual image require a matching ground-truth mask.",
    ]

    report_text = "\n".join(report_lines)

    with st.expander("📋 Preview Report"):
        st.markdown(report_text)

    st.download_button(
        "⬇️ Download Report (.md)",
        data=report_text.encode("utf-8"),
        file_name=f"{Path(fname).stem}_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # -----------------------------------------------------------------------
    # 🔧 Advanced / Technical Details  (expandable)
    # -----------------------------------------------------------------------
    st.divider()
    with st.expander("🔧 Advanced / Technical Details"):
        st.markdown("#### Raw Statistics")
        # Convert numpy arrays to plain Python types for st.json
        _safe_stats = {
            k: (int(v) if isinstance(v, (np.integer,)) else
                float(v) if isinstance(v, (np.floating,)) else v)
            for k, v in stats.items()
        }
        _safe_meta = {
            k: str(v) for k, v in meta.items()
            if k not in ("profile", "transform")
        }
        _json_payload: Dict[str, Any] = {
            "burned_area_stats": _safe_stats,
            "image_metadata": _safe_meta,
            "system": {
                "inference_time_s": inference_time_s,
                "device": device_name,
            },
        }
        if eval_metrics:
            _json_payload["evaluation_metrics"] = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else
                    v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in eval_metrics.items()
                if k != "confusion_matrix"
            }
        st.json(_json_payload)

        _crs_obj = st.session_state.crs_obj
        if _crs_obj:
            _ci = parse_crs_info(_crs_obj)
            if _ci.get("wkt"):
                st.markdown("#### Full WKT / PROJCS String")
                st.code(_ci["wkt"], language="text")

# ===========================================================================
# 📍 How to Get HLS Satellite Imagery
# ===========================================================================
st.divider()
st.markdown("## 📍 How to Get HLS Satellite Imagery")

with st.expander("Step-by-step guide — click to expand", expanded=False):
    st.markdown("""
TerraWatch AI requires **Harmonised Landsat Sentinel-2 (HLS)** imagery containing
the six required spectral bands. Regular photographs, screenshots, JPG or PNG files
cannot be used — they do not contain the spectral information needed by the Prithvi model.

---

### Why can't I use a JPG or PNG?

Standard RGB images contain only 3 colour channels (Red, Green, Blue).
TerraWatch AI requires **6 scientific spectral bands**:
- **Narrow NIR** — sensitive to healthy vegetation (distinguishes it from ash/soil)
- **SWIR1 & SWIR2** — penetrate smoke and distinguish burned soil from healthy land

These bands are captured only by multispectral satellite sensors, not by standard cameras.

---

### Step-by-step: NASA Earthdata Search

**Step 1** — Open [NASA Earthdata Search](https://search.earthdata.nasa.gov/)

**Step 2** — Create a free NASA Earthdata account, or sign in if you already have one.

**Step 3** — In the search box, search for:
```
HLSS30   (Sentinel-2 based HLS)
HLSL30   (Landsat based HLS)
```

**Step 4** — Draw your region of interest on the map, or enter a place name.

**Step 5** — Select a date range covering the post-fire period you want to analyse.

**Step 6** — Browse results and select an HLS tile that covers your area.

**Step 7** — Download the six individual band files (.tif):

| Band needed | Sentinel-2 HLS suffix | Landsat HLS suffix |
|------------|----------------------|-------------------|
| Blue       | B02 | B02 |
| Green      | B03 | B03 |
| Red        | B04 | B04 |
| Narrow NIR | B8A | B05 |
| SWIR1      | B11 | B06 |
| SWIR2      | B12 | B07 |

**Step 8** — Stack the 6 band files into a single GeoTIFF (using GDAL or QGIS):
```bash
gdal_merge.py -separate -o scene_6band.tif B02.tif B03.tif B04.tif B8A.tif B11.tif B12.tif
```

**Step 9** — Upload the 6-band GeoTIFF to TerraWatch AI.

---

### Important notes

- Not every HLS download is automatically compatible. You must stack the **correct six bands in the correct order**.
- Recommended image size is **512 × 512 pixels** (larger scenes can be subsetted in QGIS or GDAL).
- HLS has **30 m spatial resolution** — suitable for landscape-scale burn mapping.
""")
    st.link_button(
        "🔗 Open NASA Earthdata Search",
        "https://search.earthdata.nasa.gov/",
        use_container_width=True,
    )

# ===========================================================================
# ⚠️ Important Limitations
# ===========================================================================
st.divider()
st.markdown("## ⚠️ Important Limitations")
st.markdown(
    "<div class='limitation-box'>"
    "<ul style='margin:0;padding-left:18px;'>"
    "<li>TerraWatch AI performs <strong>post-wildfire burn-scar mapping</strong> from satellite imagery acquired <em>after</em> a fire event.</li>"
    "<li>It is <strong>NOT</strong> a real-time wildfire detection system and does <strong>NOT</strong> detect active flames.</li>"
    "<li>It does <strong>NOT</strong> predict future wildfire spread or fire risk.</li>"
    "<li>Results depend on the quality, acquisition date, spatial resolution and spectral characteristics of the input imagery.</li>"
    "<li>The application requires compatible <strong>six-band HLS GeoTIFF</strong> input — standard RGB images are not supported.</li>"
    "<li>Area calculations require a projected CRS with metric units. Geographic CRS (degrees) will not produce accurate km² values.</li>"
    "<li>Evaluation metrics for an individual image require a matching pixel-accurate ground-truth mask.</li>"
    "</ul>"
    "</div>",
    unsafe_allow_html=True,
)

# ===========================================================================
# Footer
# ===========================================================================
st.divider()
st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.78rem;padding:6px 0;'>"
    "TerraWatch AI &nbsp;·&nbsp; IBM-NASA Prithvi-EO-2.0-300M-BurnScars &nbsp;·&nbsp; "
    "Post-Wildfire Burn Scar Mapping &nbsp;·&nbsp; Environmental Earth Observation"
    "</div>",
    unsafe_allow_html=True,
)
