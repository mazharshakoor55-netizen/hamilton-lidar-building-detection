"""
STEP 6C - Apply the model to both epochs and difference
=======================================================

The trained U-Net segments buildings from a single epoch's elevation surface.
Run it on 2019 and on 2023, then difference the two masks:

    present in 2023, absent in 2019   ->  NEW
    present in both                   ->  EXISTING
    present in 2019, absent in 2023    ->  DEMOLISHED

This is the payoff of framing the task as segmentation rather than change
detection. Each epoch is judged on its own, so errors do not compound the way
they did when thresholds were applied to a difference raster.

Two things this does that a naive apply-and-subtract would not
--------------------------------------------------------------
OVERLAPPING TILES  Predictions are made on overlapping windows and averaged.
                   A building sitting on a tile seam would otherwise be cut in
                   half, and seam artefacts were a real problem in the earlier
                   rule-based version.

2025 CONFIRMATION  Every NEW candidate is checked against the 2025 surface.
                   A real building holds its height; vegetation keeps growing.
                   This carries over the validation trick that worked before,
                   now applied to much cleaner candidates.

RUN
---
    cd /d D:\\hamilton
    python step6c_apply_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

try:
    import torch
except ImportError:
    sys.exit("Needs PyTorch:  pip install torch")

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
MODELS = OUT / "models"
TILE = "BD33_10000_0305"

PATCH = 256
OVERLAP = 64               # prediction window overlap, pixels
THRESH = 0.5

MIN_AREA = 40.0            # m2
MAX_AREA = 5000.0
MAX_GROWTH = 0.9           # 2023 -> 2025, vegetation discriminator
MIN_HEIGHT = 2.5


def features(year):
    """Same three bands and normalisation used in training."""
    def rd(prod):
        p = DATA / f"{year}_{prod}_{TILE}.tiff"
        if not p.exists():
            sys.exit(f"Missing {p.name}")
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            return a, s.transform, s.crs

    dsm, tr, crs = rd("dsm_1m")
    dem, _, _ = rd("dem_1m")
    h = np.clip(np.nan_to_num(dsm) - np.nan_to_num(dem), 0, 60).astype(np.float32)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    dzdx = ndimage.convolve(h, kx, mode="nearest") / 8.0
    dzdy = ndimage.convolve(h, ky, mode="nearest") / 8.0
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)

    m = ndimage.uniform_filter(h, 3)
    m2 = ndimage.uniform_filter(h * h, 3)
    rough = np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)

    x = np.stack([np.clip(h / 30.0, 0, 1),
                  np.clip(slope / 90.0, 0, 1),
                  np.clip(rough / 5.0, 0, 1)]).astype(np.float32)
    return x, h, tr, crs


def load_model(device):
    ck = MODELS / "unet_buildings.pt"
    if not ck.exists():
        sys.exit("unet_buildings.pt missing - run step6b_train_unet.py first.")
    state = torch.load(ck, map_location=device)

    try:
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                         in_channels=state.get("in_ch", 3), classes=1)
    except ImportError:
        sys.path.insert(0, str(HERE))
        from step6b_train_unet import SimpleUNet
        model = SimpleUNet(state.get("in_ch", 3))

    model.load_state_dict(state["model"])
    model.to(device).eval()
    print(f"  loaded epoch {state['epoch']}, val IoU {state['val_iou']:.4f}")
    return model


@torch.no_grad()
def predict(model, x, device):
    """
    Sliding-window inference with overlap-averaging.

    Without overlap, buildings falling on a window seam get split. Averaging
    the overlaps removes the seam entirely.
    """
    C, H, W = x.shape
    step = PATCH - OVERLAP
    acc = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)

    rows = list(range(0, max(H - PATCH, 0) + 1, step))
    cols = list(range(0, max(W - PATCH, 0) + 1, step))
    if rows[-1] + PATCH < H:
        rows.append(H - PATCH)
    if cols[-1] + PATCH < W:
        cols.append(W - PATCH)

    total = len(rows) * len(cols)
    done = 0
    for r in rows:
        batch, pos = [], []
        for c in cols:
            batch.append(x[:, r:r+PATCH, c:c+PATCH])
            pos.append((r, c))
            if len(batch) == 8:
                t = torch.from_numpy(np.stack(batch)).to(device)
                p = torch.sigmoid(model(t).squeeze(1)).cpu().numpy()
                for (rr, cc), pp in zip(pos, p):
                    acc[rr:rr+PATCH, cc:cc+PATCH] += pp
                    cnt[rr:rr+PATCH, cc:cc+PATCH] += 1
                done += len(batch)
                batch, pos = [], []
                print(f"    {done}/{total} windows", end="\r")
        if batch:
            t = torch.from_numpy(np.stack(batch)).to(device)
            p = torch.sigmoid(model(t).squeeze(1)).cpu().numpy()
            for (rr, cc), pp in zip(pos, p):
                acc[rr:rr+PATCH, cc:cc+PATCH] += pp
                cnt[rr:rr+PATCH, cc:cc+PATCH] += 1
            done += len(batch)
            print(f"    {done}/{total} windows", end="\r")

    print(f"    {done}/{total} windows        ")
    return acc / np.maximum(cnt, 1)


def clean(prob, thr=THRESH):
    m = prob > thr
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_opening(m, np.ones((3, 3)))
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = load_model(device)

    probs, heights = {}, {}
    for year in (2019, 2023, 2025):
        print(f"\nPredicting {year}...")
        x, h, tr, crs = features(year)
        probs[year] = predict(model, x, device)
        heights[year] = h
        print(f"    {100*(probs[year] > THRESH).mean():.1f}% building pixels")

    m19 = clean(probs[2019])
    m23 = clean(probs[2023])

    px = abs(tr.a)
    print(f"\n  2019 building area: {m19.sum()*px*px/1e6:.2f} km2")
    print(f"  2023 building area: {m23.sum()*px*px/1e6:.2f} km2")

    print("\nDifferencing...")
    new_mask = m23 & ~m19
    dem_mask = m19 & ~m23

    lab, n = ndimage.label(new_mask)
    print(f"  {n} raw new-building components")

    keep, recs = [], []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = float(m.sum()) * px * px
        if not (MIN_AREA <= area <= MAX_AREA):
            continue

        h23 = float(np.median(heights[2023][sl][m]))
        if h23 < MIN_HEIGHT:
            continue

        h19 = float(np.median(heights[2019][sl][m]))
        h25 = float(np.median(heights[2025][sl][m]))
        growth = h25 - h23
        if growth >= MAX_GROWTH:          # still growing = vegetation
            continue

        keep.append(i)
        recs.append({"area": area, "h19": h19, "h23": h23,
                     "h25": h25, "growth": growth,
                     "conf": float(np.mean(probs[2023][sl][m]))})

    print(f"  {len(recs)} after area, height and 2025-stability filters")

    if recs:
        a = np.array([r["area"] for r in recs])
        print(f"\n  median footprint : {np.median(a):.0f} m2")
        print(f"  median 2019 h    : {np.median([r['h19'] for r in recs]):.2f} m")
        print(f"  median 2023 h    : {np.median([r['h23'] for r in recs]):.2f} m")
        print(f"  median growth    : {np.median([r['growth'] for r in recs]):+.2f} m")
        print(f"  median confidence: {np.median([r['conf'] for r in recs]):.3f}")
        print(f"  total roof area  : {a.sum()/1e4:.2f} ha")

    prof = {"driver": "GTiff", "height": m23.shape[0], "width": m23.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": tr,
            "nodata": -9999, "compress": "deflate", "tiled": True}

    for name, arr in [("cnn_buildings_2019", m19.astype(np.float32)),
                      ("cnn_buildings_2023", m23.astype(np.float32)),
                      ("cnn_new_buildings", np.isin(lab, keep).astype(np.float32)),
                      ("cnn_demolished", dem_mask.astype(np.float32))]:
        with rasterio.open(OUT / f"{name}_{TILE}.tif", "w", **prof) as d:
            d.write(arr, 1)
        print(f"  wrote {name}_{TILE}.tif")

    summary = {
        "new_buildings": len(recs),
        "total_roof_area_ha": round(float(sum(r["area"] for r in recs)) / 1e4, 2)
        if recs else 0,
        "buildings_2019_km2": round(float(m19.sum()) * px * px / 1e6, 3),
        "buildings_2023_km2": round(float(m23.sum()) * px * px / 1e6, 3),
        "median_footprint_m2": round(float(np.median([r["area"] for r in recs])), 1)
        if recs else 0,
    }
    (OUT / "cnn_change_summary.json").write_text(json.dumps(summary, indent=1))

    print(f"\n{'='*58}")
    print("CNN vs RULE-BASED")
    print(f"{'='*58}")
    print(f"  rule-based new buildings : 1016  (54% precision, manual review)")
    print(f"  CNN new buildings        : {len(recs)}")
    print(f"\n  Rerun step5a on the CNN output to get a comparable precision")
    print(f"  figure - that comparison is the headline result of the project.")


if __name__ == "__main__":
    main()
