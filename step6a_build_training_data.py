"""
STEP 6A - Build training data for a segmentation CNN
====================================================

Why we are NOT training on the change detections
------------------------------------------------
Manual review put the new-building labels at 53.5% precision. Training on those
would teach a model to reproduce a coin flip.

But notice WHERE the errors were. Almost none were "there is no building here".
They were "this building is not NEW" - 27 existing buildings, plus vegetation
that a better segmenter would reject anyway. The *buildingness* signal is far
cleaner than the *newness* signal.

So we change the task. Instead of learning change directly:

    TRAIN   segment buildings from a single epoch's elevation surface
    LABELS  LINZ Building Outlines - 25,151 human-drawn polygons in this tile
    APPLY   run the trained model on 2019 and 2023 separately, then difference
            the two building masks

This is better on three counts:

  1. Real labels. Human-digitised outlines, not our own geometric guesses.
  2. It attacks the actual bottleneck. Segmentation recall was 67%; a CNN that
     learns what roofs look like should beat fixed thresholds comfortably.
  3. Errors stop compounding. Segment well in each epoch and the change
     detection becomes a simple set difference.

Inputs are three physical bands, same as the rule-based version:
    nDSM, slope, roughness

RUN
---
    cd /d D:\\hamilton
    python step6a_build_training_data.py
"""

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from scipy import ndimage

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
TRAIN = HERE / "data" / "training"
TRAIN.mkdir(parents=True, exist_ok=True)

TILE = "BD33_10000_0305"
LAYER = 101292
PAGE = 10000

PATCH = 256
STRIDE = 256
BLOCK_TILES = 8            # ~2 km spatial blocks
MIN_VALID = 0.90
KEEP_EMPTY = 0.20          # fraction of building-free patches to retain

GEOM_CACHE = HERE / "linz_outlines_tile.json"


def get_key():
    """
    Read the LINZ API key from linz_key.txt in this folder.

    Keeping the key in its own file means no script contains a secret, so the
    repository can be published without redaction. linz_key.txt is listed in
    .gitignore and stays local.
    """
    f = HERE / "linz_key.txt"
    if not f.exists():
        raise SystemExit(
            "Create " + str(f) + " and paste your LINZ API key into it. "
            "Get one free at https://data.linz.govt.nz/my/api/")
    k = f.read_text().strip()
    if not k:
        raise SystemExit(str(f.name) + " is empty.")
    return k

def ndsm(year):
    def rd(prod):
        p = DATA / f"{year}_{prod}_{TILE}.tiff"
        if not p.exists():
            sys.exit(f"Missing {p.name}")
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            return a, s.transform, s.crs, s.bounds
    dsm, tr, crs, b = rd("dsm_1m")
    dem, _, _, _ = rd("dem_1m")
    valid = ~(np.isnan(dsm) | np.isnan(dem))
    h = np.clip(np.nan_to_num(dsm) - np.nan_to_num(dem), 0, 60)
    return h.astype(np.float32), tr, crs, b, valid


def fetch_outlines(bounds):
    """
    Building outline geometry for the tile, cached.

    The key is only read when a download is actually needed, so a cached run
    works with no credentials at all.
    """
    if GEOM_CACHE.exists():
        print(f"  using cached {GEOM_CACHE.name}")
        return json.loads(GEOM_CACHE.read_text())

    url = f"https://data.linz.govt.nz/services;key={get_key()}/wfs"
    geoms, start = [], 0
    while True:
        for attempt in range(6):
            try:
                r = requests.get(url, params={
                    "service": "WFS", "version": "2.0.0", "request": "GetFeature",
                    "typeNames": f"layer-{LAYER}", "outputFormat": "json",
                    # northing first - EPSG:2193 declared axis order
                    "bbox": f"{bounds.bottom},{bounds.left},"
                            f"{bounds.top},{bounds.right},"
                            f"urn:ogc:def:crs:EPSG::2193",
                    "count": PAGE, "startIndex": start,
                }, timeout=300)
                r.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                print(f"    retry {attempt+1}/6 ({type(e).__name__})")
                time.sleep(2 ** attempt)
        else:
            sys.exit("WFS failed repeatedly.")

        feats = r.json().get("features", [])
        for f in feats:
            g = f.get("geometry")
            if g and g.get("type") in ("Polygon", "MultiPolygon"):
                geoms.append(g)
        print(f"    {len(geoms)} outlines...", end="\r")
        if len(feats) < PAGE:
            break
        start += PAGE
        if start > 400000:
            break

    print(f"    {len(geoms)} outlines fetched      ")
    GEOM_CACHE.write_text(json.dumps(geoms))
    return geoms


def features_for(year):
    h, tr, crs, bounds, valid = ndsm(year)

    # Slope of the surface, Horn's method
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    dzdx = ndimage.convolve(h, kx, mode="nearest") / 8.0
    dzdy = ndimage.convolve(h, ky, mode="nearest") / 8.0
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)

    # Local height standard deviation - roofs planar, canopy chaotic
    m = ndimage.uniform_filter(h, 3)
    m2 = ndimage.uniform_filter(h * h, 3)
    rough = np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)

    # Fixed divisors, NOT per-patch statistics. Per-patch normalisation would
    # destroy the absolute height that separates a house from a garden wall.
    stack = np.stack([
        np.clip(h / 30.0, 0, 1),
        np.clip(slope / 90.0, 0, 1),
        np.clip(rough / 5.0, 0, 1),
    ]).astype(np.float32)

    return stack, tr, crs, bounds, valid


def main():
    print("Building features for 2023...")
    X, tr, crs, bounds, valid = features_for(2023)
    C, H, W = X.shape
    print(f"  {C} bands, {W} x {H} px")

    print("\nFetching LINZ building outlines...")
    geoms = fetch_outlines(bounds)

    print("\nRasterising labels...")
    mask = rasterize(((g, 1) for g in geoms), out_shape=(H, W),
                     transform=tr, fill=0, dtype=np.uint8, all_touched=False)
    print(f"  {100*mask.mean():.1f}% of pixels are building")

    # Boundary class - most object-level error lives at roof edges, so weight
    # them during training
    er = ndimage.binary_erosion(mask > 0, np.ones((3, 3)))
    y = np.where((mask > 0) & ~er, 2, mask).astype(np.uint8)
    print(f"  {100*(y==1).mean():.1f}% interior, {100*(y==2).mean():.1f}% boundary")

    print("\nTiling...")
    rows = list(range(0, H - PATCH + 1, STRIDE))
    cols = list(range(0, W - PATCH + 1, STRIDE))
    print(f"  {len(rows)} x {len(cols)} = {len(rows)*len(cols)} candidate patches")

    # Spatial block split. Random patch splits leak: neighbouring patches share
    # suburb, roof material and flightline.
    rng = np.random.default_rng(42)
    blocks = {}
    for r in rows:
        for c in cols:
            blocks.setdefault((r // (PATCH * BLOCK_TILES),
                               c // (PATCH * BLOCK_TILES)), []).append((r, c))
    keys = list(blocks)
    rng.shuffle(keys)
    n_tr = int(round(len(keys) * 0.70))
    n_va = int(round(len(keys) * 0.15))
    split_of = {}
    for i, k in enumerate(keys):
        s = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for rc in blocks[k]:
            split_of[rc] = s
    print(f"  {len(keys)} spatial blocks -> "
          f"{n_tr} train / {n_va} val / {len(keys)-n_tr-n_va} test")

    counts = {"train": 0, "val": 0, "test": 0}
    empty_kept = empty_seen = 0
    manifest = []

    for (r, c), split in split_of.items():
        v = valid[r:r+PATCH, c:c+PATCH]
        if v.mean() < MIN_VALID:
            continue
        yy = y[r:r+PATCH, c:c+PATCH]
        frac = float((yy > 0).mean())

        if frac < 0.002:
            empty_seen += 1
            # Keep some negatives - a model trained only on positives never
            # learns what a building is not - but not all, or background wins.
            if rng.random() > KEEP_EMPTY:
                continue
            empty_kept += 1

        xx = X[:, r:r+PATCH, c:c+PATCH]
        name = f"{split}_{r:05d}_{c:05d}.npz"
        np.savez_compressed(TRAIN / name, x=xx, y=yy)
        counts[split] += 1
        manifest.append({"file": name, "split": split, "row": r, "col": c,
                         "building_frac": round(frac, 4)})
        if sum(counts.values()) % 50 == 0:
            print(f"  wrote {sum(counts.values())} patches...", end="\r")

    (TRAIN / "manifest.json").write_text(json.dumps(manifest, indent=1))

    print(f"  wrote {sum(counts.values())} patches            ")
    print(f"\n{'='*58}")
    print("TRAINING SET")
    print(f"{'='*58}")
    for k in ("train", "val", "test"):
        sub = [m for m in manifest if m["split"] == k]
        bf = np.mean([m["building_frac"] for m in sub]) if sub else 0
        print(f"  {k:<6}{len(sub):>5} patches   mean building fraction {bf:.3f}")
    print(f"\n  empty patches: {empty_kept} kept of {empty_seen} seen")
    mb = sum(f.stat().st_size for f in TRAIN.glob('*.npz')) / 1024**2
    print(f"  on disk: {mb:.0f} MB in {TRAIN}")

    print(f"\n{'='*58}")
    print("  Next: pip install torch segmentation-models-pytorch")
    print("  then  python step6b_train_unet.py")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
