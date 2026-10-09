# 🌍 TerraWatch AI

**Post-Wildfire Burn Scar Detection from Satellite Imagery**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-red)](https://streamlit.io)
[![Model](https://img.shields.io/badge/Model-Prithvi--EO--2.0--300M--BurnScars-green)](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars)

> **TerraWatch AI is a post-wildfire burn scar mapping system.**
> It detects observed burn scars in satellite imagery acquired after a fire event.
> It does **not** predict future wildfires or detect active flames.

---

## Problem Statement

Wildfires are increasing in frequency and scale globally. After a fire, rapid and accurate
mapping of burn scars is critical for:

- Allocating emergency response resources
- Assessing ecological damage
- Informing forest replanting and restoration
- Quantifying carbon emissions for climate science
- Supporting insurance and land-use policy

Traditional manual mapping from satellite imagery is slow and does not scale.
TerraWatch AI automates pixel-level burn-scar detection using a purpose-built
AI foundation model developed by IBM and NASA.

---

## AI / ML Approach: Semantic Segmentation

TerraWatch AI uses **semantic segmentation** — not image classification.

| Approach | Output |
|----------|--------|
| Image Classification | "This entire scene is fire-damaged." |
| **Semantic Segmentation** | "These specific pixels are burned; those are not." |

Every pixel in the satellite image receives an individual label:
- `0` — Background / non-burned
- `1` — Burn scar

This enables accurate burned-area measurements at 30 m spatial resolution.

---

## AI Model: Prithvi-EO-2.0-300M-BurnScars

| Property | Value |
|----------|-------|
| Name | Prithvi-EO-2.0-300M-BurnScars |
| Provider | IBM × NASA |
| Architecture | Vision Transformer backbone + UperNet/UNet decoder |
| Parameters | ~300 million |
| Task | Semantic segmentation — burn scar mapping |
| Training data | HLS wildfire scenes (NASA FIRMS-validated) |
| Published IoU | 0.8752 (HLS BurnScars benchmark test set) |
| Published mIoU | 0.9300 |
| Hugging Face | [ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) |

> Published benchmark metrics reflect performance on the held-out HLS BurnScars
> test dataset. They are not per-image results.

---

## HLS Satellite Imagery

TerraWatch AI requires **Harmonised Landsat Sentinel-2 (HLS)** imagery —
a NASA product that combines data from the Landsat 8/9 and Sentinel-2 satellites
into a consistent surface reflectance dataset at 30 m resolution.

### Why HLS specifically?

The Prithvi model was fine-tuned exclusively on HLS data. Using other
satellite products (e.g. MODIS, PlanetScope, Sentinel-1 SAR) without
re-training will produce unreliable results.

### Six Required Bands

| Band # | Name | HLS Sentinel-2 | HLS Landsat |
|--------|------|---------------|-------------|
| 1 | Blue | B02 | B02 |
| 2 | Green | B03 | B03 |
| 3 | Red | B04 | B04 |
| 4 | Narrow NIR | B8A | B05 |
| 5 | SWIR1 | B11 | B06 |
| 6 | SWIR2 | B12 | B07 |

### Why not JPG or PNG?

RGB images contain only 3 colour channels. Narrow NIR, SWIR1 and SWIR2
are invisible to standard cameras and are only available in multispectral
satellite products. Without them, burn scars cannot be reliably distinguished
from dark soil, shadows or water.

---

## Burned Area Calculation

```
pixel_area_m2     = pixel_width_m × pixel_height_m
burned_area_m2    = burned_pixels × pixel_area_m2
burned_area_km2   = burned_area_m2 / 1,000,000
percentage_affected = burned_pixels / total_valid_pixels × 100
```

- Pixel size is read directly from the GeoTIFF spatial metadata.
- If the CRS is not projected (no metric units), area cannot be calculated and is displayed as "Not available".
- No-data pixels flagged in the GeoTIFF are not counted as valid pixels.
- Results are derived entirely from the actual model prediction — nothing is fabricated or hard-coded.

---

## Evaluation Methodology

Evaluation metrics are computed **only when a ground-truth mask is provided**.

| Metric | Formula |
|--------|---------|
| IoU | TP / (TP + FP + FN) |
| Dice / F1 | 2·TP / (2·TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |

Ground-truth masks must be single-band binary GeoTIFFs or PNGs with the same
spatial dimensions as the prediction (0 = background, 1 = burn scar).

**If no ground truth is provided**, the application clearly states:
> "Evaluation unavailable — no ground-truth mask provided."

No evaluation metrics are ever fabricated or approximated.

---

## Supported Input Format

| Property | Required value |
|----------|---------------|
| Format | GeoTIFF (.tif / .tiff) |
| Bands | Exactly 6 (Blue, Green, Red, NIR, SWIR1, SWIR2) |
| Data type | float32 or uint16 |
| Spatial resolution | ~30 m (HLS standard) |
| CRS | Any projected CRS (UTM recommended) |
| Recommended size | 512 × 512 pixels |

---

## How to Obtain HLS Data

1. Open [NASA Earthdata Search](https://search.earthdata.nasa.gov/)
2. Create a free Earthdata account
3. Search for `HLSS30` (Sentinel-2) or `HLSL30` (Landsat)
4. Select your region and post-fire date range
5. Download the 6 required band files
6. Stack into a 6-band GeoTIFF:
   ```bash
   gdal_merge.py -separate -o scene_6band.tif B02.tif B03.tif B04.tif B8A.tif B11.tif B12.tif
   ```
7. Upload to TerraWatch AI

---

## Installation (Local)

### Prerequisites

- Python 3.11 (not 3.14 — ML libraries do not yet support it)
- ~1.5 GB free disk space for model weights

### Setup

```bash
# 1. Clone
git clone https://github.com/your-username/terrawatch-ai.git
cd terrawatch-ai

# 2. Create virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Install PyTorch (CPU)
pip install torch torchvision

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. (Recommended) Install TerraTorch for full model accuracy
pip install terratorch

# 6. Download model weights
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars',
                'Prithvi_EO_V2_300M_BurnScars.pt', local_dir='models/')
hf_hub_download('ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars',
                'burn_scars_config.yaml', local_dir='models/')
"
```

---

## How to Run Locally

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Deploy on Streamlit Cloud

> Full accuracy requires `terratorch`. On Streamlit Cloud (CPU-only environment),
> the app will run with the timm-based fallback backbone if terratorch cannot be installed.
> For production use, run locally with terratorch installed.

### Steps

1. **Push to GitHub**

   ```bash
   git init
   git remote add origin https://github.com/your-username/terrawatch-ai.git

   # The model weights file (≈1.2 GB) must be committed or available at runtime.
   # Option A — commit with Git LFS:
   git lfs install
   git lfs track "models/*.pt"
   git add .gitattributes models/Prithvi_EO_V2_300M_BurnScars.pt
   git add .
   git commit -m "Initial commit"
   git push -u origin main
   ```

   > **Important:** `models/Prithvi_EO_V2_300M_BurnScars.pt` (≈1.2 GB) must be
   > reachable at runtime. Options:
   > - **Git LFS** (GitHub Free tier: 1 GB LFS storage — file is borderline; upgrade storage if needed)
   > - **Hugging Face download on first run** — weights are auto-downloaded via
   >   `hf_hub_download` in `backend/inference.py` if the file is absent.
   >   Streamlit Cloud will download them on first cold start (~2–5 min).

2. **Connect to Streamlit Cloud**
   - Open [share.streamlit.io](https://share.streamlit.io)
   - Click **New app** → select your GitHub repository
   - Set **Main file path** to `app.py`
   - Click **Deploy**

3. **Set Secrets (optional — only if you use a private HF model)**
   - In the Streamlit Cloud app settings → **Secrets**
   - Add: `HF_TOKEN = "hf_xxxx"` if your HF repo requires authentication

4. **Packages**
   - `packages.txt` in the repo root installs system-level GDAL libraries
     (required by `rasterio`) automatically on Streamlit Cloud.
   - `requirements.txt` installs all Python dependencies.

### File structure required on GitHub

```
terrawatch-ai/
├── app.py                   ← main Streamlit app
├── requirements.txt         ← Python dependencies
├── packages.txt             ← system packages (GDAL for rasterio)
├── .streamlit/
│   └── config.toml          ← Streamlit settings (upload size, theme)
├── backend/
│   ├── __init__.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── model.py
│   ├── postprocessing.py
│   ├── preprocessing.py
│   └── utils.py
├── models/
│   ├── README.md
│   ├── burn_scars_config.yaml
│   └── Prithvi_EO_V2_300M_BurnScars.pt   ← weights (via Git LFS or auto-download)
├── data/
│   └── subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif  ← sample image
└── outputs/
    └── .gitkeep
```

---

## Application Workflow

```
Upload 6-band HLS GeoTIFF
        ↓
Input validation (band count, file format, CRS)
        ↓
Display image information (dimensions, resolution, CRS, scene bounds)
        ↓
Click "🔥 Detect Burn Scars"
        ↓
Preprocessing (normalisation using config statistics)
        ↓
Prithvi-EO-2.0-300M-BurnScars inference
        ↓
Binary burn-scar mask (0/1 per pixel)
        ↓
┌─────────────────────────────────────────────────┐
│  Visual Results  │  Area Statistics  │  Downloads│
│  Original image  │  Burned km²       │  Mask TIFF│
│  Mask overlay    │  % affected       │  Overlay  │
│  Opacity slider  │  Burned pixels    │  CSV      │
└─────────────────────────────────────────────────┘
        ↓ (optional)
Upload ground-truth mask → Run Evaluation
        ↓
IoU · Dice · Precision · Recall · Confusion Matrix
```

---

## Output Explanation

| Output | Description |
|--------|-------------|
| Prediction mask (GeoTIFF) | Binary mask, same CRS/resolution as input |
| Burn-scar overlay (PNG) | True-colour image with red burn overlay |
| Statistics (CSV) | Pixel counts, burned area, %, eval metrics if available |

---

## Crash Prevention — What Was Fixed

The following crash-prone issues were resolved in the current version:

| Issue | Fix |
|-------|-----|
| `terratorch` not on PyPI / uninstallable on Streamlit Cloud | Added `timm`-based fallback backbone in `inference.py` |
| `imageio.imwrite(buf, arr, format="png")` breaks on imageio v3 | Dual v2/v3 API support + PIL fallback in `utils.py` and `model.py` |
| `st.session_state.upload_meta["filename"]` crash when session reset | Safe `.get()` access with default throughout `app.py` |
| `geotiff_bytes(result["mask_path"])` crash when temp file deleted by OS | In-memory regeneration fallback in `app.py` |
| `st.json(…)` crash on numpy scalars in stats dict | Explicit numpy → Python type conversion before `st.json` |
| Mask array shape/dtype not validated before `st.image` | Shape and dtype guards added in `app.py` |
| `pyproj` missing from `requirements.txt` | Added `pyproj>=3.3.0` |

---

## Limitations

- Requires exactly 6 HLS bands in the correct order
- Optimal for 512 × 512 px scenes (the fine-tuning resolution)
- Area calculation requires a projected CRS with metric units
- CPU inference is significantly slower than GPU (~2–5 min vs ~15 s)
- No per-image confidence score is available from the model
- Results depend on input image quality, cloud cover, and shadow

---

## Technologies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.0+ | Model inference |
| TerraTorch | 0.1+ | IBM-NASA model factory (optional, recommended) |
| timm | 0.9+ | Fallback ViT backbone |
| Hugging Face Hub | 0.22+ | Weight distribution |
| Rasterio | 1.3+ | GeoTIFF I/O |
| pyproj | 3.3+ | CRS / coordinate transformation |
| NumPy | 1.24+ | Array processing |
| Streamlit | 1.33+ | Web dashboard |
| imageio | 2.31+ | Image encoding (v2 and v3 supported) |
| Pillow | 10.0+ | Image fallback encoding |

---

## Acknowledgements

- **IBM Research** and **NASA** for the Prithvi-EO foundation model
- **NASA Earthdata** and the HLS programme for open satellite data
- The open-source geospatial Python ecosystem (rasterio, GDAL, TerraTorch, pyproj)
