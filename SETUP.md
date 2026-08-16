# Setup

## 1. Install dependencies

```bash
pip install requests pillow pyproj rasterio scipy
```

For the deep-learning steps (6a–6d) additionally:

```bash
pip install torch segmentation-models-pytorch
```

## 2. Add your LINZ API key

Get a free key at https://data.linz.govt.nz/my/api/ — choose **Data access only**.

Create a file named `linz_key.txt` in the `src/` folder and paste the key into
it. Nothing else, no quotes.

```
src/linz_key.txt
```

Every script reads the key from this one file. It is listed in `.gitignore`, so
it never leaves your machine.

Elevation rasters come from a public AWS bucket and need no credentials at all —
`step2c_download_tiles.py` works without a key.

## 3. Run

```bash
cd src
python step1b_check_change.py          # confirm change is visible
python step1e_dedupe_buildings.py      # count new buildings from LINZ outlines
python step1f_find_pilot_square.py     # locate the densest change cluster
python step2c_download_tiles.py        # download DEM/DSM, 2019 + 2023
python step3e_segment_first.py         # segment and classify
python step4a_export_polygons.py       # export GeoJSON
python step4c_render_map_v2.py         # render maps
```

Add the 2025 epoch and the neural pipeline:

```bash
python step3c_three_epoch.py           # downloads 2025, trajectory validation
python step6a_build_training_data.py   # training patches from LINZ outlines
python step6b_train_unet.py            # train (GPU recommended)
python step6c_apply_model.py           # apply to both epochs, difference
python step6d_export_cnn_polygons.py   # export
```

Validation:

```bash
python step5a_build_review_pack.py     # build review sheets
python step5b_validation_stats.py      # precision with Wilson intervals
```

## Notes

Every network call retries with exponential backoff, and downloads resume — a
dropped connection mid-run is recoverable by re-running the same command.

Elevation rasters total roughly 190 MB for a single tile across three epochs.
They are excluded from this repository and re-downloadable at any time.

Close Excel before running anything that writes a CSV; Windows locks the file
exclusively and the script will fail with a permission error.
