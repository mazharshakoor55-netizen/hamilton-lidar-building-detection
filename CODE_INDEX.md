# Code index

Sixteen scripts, run in order. Each is standalone and re-runnable; every
network call retries with exponential backoff and downloads resume.

## Dependencies

```
pip install requests pillow pyproj rasterio scipy torch segmentation-models-pytorch
```

A free LINZ API key is required for imagery and building outlines:
https://data.linz.govt.nz/my/api/

Elevation rasters come from a public S3 bucket and need no credentials.

## Pipeline

| Script | Purpose | Key output |
|---|---|---|
| `step1b_check_change.py` | Fetch before/after aerial imagery for a candidate area | Visual confirmation the project has a subject |
| `step1e_dedupe_buildings.py` | Count new buildings from LINZ outlines, deduplicated by `building_id` | 87,124 unique buildings, ~2,064/year |
| `step1f_find_pilot_square.py` | Grid the study area by new-building density | Optimal pilot square, data-selected |
| `step2c_download_tiles.py` | Download DEM/DSM from `s3://nz-elevation` | Four tiles, resumable |
| `gridding.py` | Point cloud to raster, pure numpy (no PDAL) | Grid reducers, return-ratio band |
| `step3a_ndsm_difference.py` | nDSM per epoch, difference, DEM drift check | 2 cm vendor agreement |
| `step3c_three_epoch.py` | Three-epoch trajectory classification | Buildings vs vegetation, no reference data |
| `step3e_segment_first.py` | Segment buildings, then classify by height history | 1,016 new buildings |
| `step4a_export_polygons.py` | Vectorise to GeoJSON with full height history | Auditable per-building records |
| `step4c_render_map_v2.py` | Cartographic output | Overview and detail maps |
| `step5a_build_review_pack.py` | Four-panel validation sheets, stratified sample | 10 sheets, 100 detections |
| `step5b_validation_stats.py` | Precision with Wilson intervals, stratified | 54% (95% CI 44–63%) |
| `step6a_build_training_data.py` | Training patches with spatial block splits | 429 patches, 12 blocks |
| `step6b_train_unet.py` | U-Net, Dice + boundary-weighted BCE | Test IoU 0.78 |
| `step6c_apply_model.py` | Sliding-window inference, difference epochs | 1,162 new buildings |
| `step6d_export_cnn_polygons.py` | Export CNN output, schema-matched | Directly comparable to rule-based |

## Design notes

**No PDAL.** Its Python bindings require system libraries unavailable via pip.
Since LINZ ships pre-classified point clouds, the hard part — ground filtering —
is already done, and the remaining gridding is a few numpy operations.

**Geometry written by hand.** Ramer–Douglas–Peucker simplification is implemented
directly rather than importing Shapely, keeping the dependency list to six
packages so the pipeline reproduces without a conda environment.

**Spatial block splits throughout.** Random splits leak: adjacent patches share
suburb, roof material and flightline.

**Fixed normalisation divisors, never per-patch statistics.** Per-patch
normalisation destroys the absolute height that separates a house from a garden
wall.
