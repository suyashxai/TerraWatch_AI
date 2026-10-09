# Data

This directory is reserved for input satellite imagery and optional ground-truth masks.

**Do NOT commit large GeoTIFF files to Git.**  
The `.gitignore` excludes `*.tif` and `*.tiff` files inside this directory.

---

## Expected input format

| Property | Value |
|----------|-------|
| Format | GeoTIFF (`.tif` / `.tiff`) |
| Bands | **6** (must be exactly 6) |
| Band order | Blue · Green · Red · Narrow NIR · SWIR1 · SWIR2 |
| Dataset | Harmonised Landsat Sentinel-2 (HLS) |
| Spatial resolution | ~30 metres/pixel |
| Recommended size | 512 × 512 pixels |
| CRS | Any projected CRS (EPSG:32601–32660 UTM zones work well) |

---

## Suggested directory layout

```
data/
├── input/
│   └── hls_scene_20230812.tif     ← your upload / test image
│
├── ground_truth/                  ← optional, for evaluation only
│   └── hls_scene_20230812_gt.tif  ← binary mask (0=background, 1=burn)
│
└── README.md                      ← this file
```

---

## Where to obtain HLS data

### NASA Earthdata

1. Create a free account at <https://urs.earthdata.nasa.gov/>
2. Search the **LP DAAC** portal: <https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/>
3. Download HLS tiles (`.hdf` or GeoTIFF) for a region of interest

### Microsoft Planetary Computer

Provides HLS data via STAC API:
<https://planetarycomputer.microsoft.com/dataset/hls>

### Kaggle datasets

Several open wildfire / burn-scar datasets exist on Kaggle and include
pre-stacked 6-band GeoTIFFs ready for Prithvi inference.

---

## Ground-truth masks (for evaluation)

Ground-truth masks should be **single-band GeoTIFF** files with:

- `0` = background / non-burned
- `1` = burn scar

If you have a ground-truth mask, run the evaluation script:

```bash
python evaluation/evaluate.py \
    --pred outputs/hls_scene_20230812_mask.tif \
    --gt   data/ground_truth/hls_scene_20230812_gt.tif
```

---

## Important

TerraWatch AI does **not** fabricate or hard-code statistics.  
All burn-scar area measurements are derived from actual model predictions.  
Do not supply synthetic or pre-processed masks as ground truth.
