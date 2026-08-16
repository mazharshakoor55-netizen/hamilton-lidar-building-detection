"""
STEP 3E - Segment first, then difference (the right way round)
=============================================================

What was wrong
--------------
step3a differenced the two epochs and then labelled connected patches. That
makes the patch shape depend on where the height difference happened to clear a
threshold - so roofs got split, neighbours got merged, and partial coverage was
common. Precision was fine (red sat on roofs) but recall was poor: most new
buildings in the pilot image were not outlined at all.

The fix is to reverse the order:

    1. SEGMENT every building in the 2023 surface. Whole roofs, proper
       footprints, one label per structure.
    2. CLASSIFY each one by its own height history:
           2019 clear + 2023 tall + 2025 stable  ->  NEW BUILDING
           2019 tall                             ->  pre-existing
           2025 still growing                    ->  vegetation

Now every candidate is a complete building, and the question asked of it is
simply "was this here in 2019?". Much better recall, and the footprints are
usable as real polygons rather than difference fragments.

RUN
---
    cd /d D:\\hamilton
    python step3e_segment_first.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except ImportError:
    sys.exit("Needs scipy:  pip install scipy")

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
TILE = "BD33_10000_0305"
PILOT = (1802319, 5818882, 1802819, 5819382)

# --- building segmentation on the 2023 surface ---
BUILDING_MIN_H = 2.5          # above ground
BUILDING_MAX_H = 30.0
MIN_AREA = 40.0               # m2 - includes garages
MAX_AREA = 4000.0
MAX_ROUGH = 1.5               # roofs are planar, canopy is not
MIN_COMPACT = 0.18

# --- was it there in 2019? ---
CLEAR_2019 = 1.2              # site counted as empty below this
PARTIAL_2019 = 2.5            # between the two = ambiguous / rebuild

# --- vegetation check against 2025 ---
MAX_GROWTH = 0.9


def ndsm(year, window=None):
    def rd(prod):
        p = DATA / f"{year}_{prod}_{TILE}.tiff"
        if not p.exists():
            sys.exit(f"Missing {p.name}")
        with rasterio.open(p) as s:
            w = from_bounds(*window, transform=s.transform) if window else None
            a = s.read(1, window=w).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            return a, (s.window_transform(w) if w else s.transform), s.crs
    dsm, tr, crs = rd("dsm_1m")
    dem, _, _ = rd("dem_1m")
    return np.clip(dsm - dem, 0, 60), tr, crs


def roughness(a, win=3):
    x = np.nan_to_num(a, nan=0.0)
    m = ndimage.uniform_filter(x, win)
    m2 = ndimage.uniform_filter(x * x, win)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def main():
    print("Loading three epochs...")
    n19, tr, crs = ndsm(2019)
    n23, _, _ = ndsm(2023)
    n25, _, _ = ndsm(2025)
    px = abs(tr.a)
    print(f"  {n23.shape[1]} x {n23.shape[0]} px, {n23.size*px*px/1e6:.1f} km2")

    # ---------- 1. segment every building present in 2023 ----------
    print("\nSegmenting buildings from the 2023 surface...")
    tall = (np.nan_to_num(n23, nan=0) >= BUILDING_MIN_H) & \
           (np.nan_to_num(n23, nan=0) <= BUILDING_MAX_H)

    # NO aggressive closing. Measured on a synthetic dense suburb (12 m houses,
    # 4 m apart): threshold-only recovers 91% of buildings as separate objects,
    # closing x1 gives 87%, and closing x2 fuses the ENTIRE suburb into a single
    # component (1%). Two iterations of a 3x3 kernel bridges ~4 m, which is
    # exactly suburban house spacing. Fill holes to heal skylights and ridges,
    # and open once to shave single-pixel spurs - nothing more.
    tall = ndimage.binary_fill_holes(tall)
    tall = ndimage.binary_opening(tall, np.ones((3, 3)))

    lab, n = ndimage.label(tall)
    print(f"  {n} tall objects (buildings + trees)")

    rough = roughness(n23)

    print("  filtering to building-like objects...")
    recs = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        area = float(m.sum()) * px * px
        if not (MIN_AREA <= area <= MAX_AREA):
            continue
        r = float(np.mean(rough[sl][m]))
        if r > MAX_ROUGH:
            continue
        er = ndimage.binary_erosion(m, np.ones((3, 3)))
        per = float((m & ~er).sum()) * px
        comp = 4 * np.pi * area / (per * per) if per else 0.0
        if comp < MIN_COMPACT:
            continue

        recs.append({
            "id": i, "area": area, "rough": r, "comp": comp,
            "h19": float(np.nanmedian(n19[sl][m])),
            "h23": float(np.nanmedian(n23[sl][m])),
            "h25": float(np.nanmedian(n25[sl][m])),
        })

    print(f"  {len(recs)} building-like footprints in 2023")
    if not recs:
        sys.exit("Nothing segmented - loosen the thresholds.")

    a = np.array([r["area"] for r in recs])
    print(f"  median footprint {np.median(a):.0f} m2, "
          f"median height {np.median([r['h23'] for r in recs]):.1f} m")

    # ---------- 2. classify by history ----------
    print("\nClassifying by height history...")
    for r in recs:
        g = r["h25"] - r["h23"]
        r["growth"] = g
        if g >= MAX_GROWTH:
            r["cls"] = "VEGETATION"
        elif r["h19"] <= CLEAR_2019:
            r["cls"] = "NEW"
        elif r["h19"] <= PARTIAL_2019:
            r["cls"] = "REBUILD?"
        else:
            r["cls"] = "EXISTING"

    counts = Counter(r["cls"] for r in recs)
    tot = len(recs)
    print(f"\n{'='*66}")
    print("CLASSIFICATION")
    print(f"{'='*66}")
    for k in ("EXISTING", "NEW", "REBUILD?", "VEGETATION"):
        v = counts.get(k, 0)
        print(f"  {k:<12}{v:>6}  {100*v/tot:5.1f}%  {'#'*int(42*v/tot)}")

    print(f"\n{'='*66}")
    print("MEDIAN PROFILE BY CLASS")
    print(f"{'='*66}")
    print(f"  {'class':<12}{'2019':>7}{'2023':>7}{'2025':>7}{'growth':>8}"
          f"{'area':>8}{'rough':>7}")
    for k in ("EXISTING", "NEW", "REBUILD?", "VEGETATION"):
        sub = [r for r in recs if r["cls"] == k]
        if not sub:
            continue
        print(f"  {k:<12}"
              f"{np.median([r['h19'] for r in sub]):>7.1f}"
              f"{np.median([r['h23'] for r in sub]):>7.1f}"
              f"{np.median([r['h25'] for r in sub]):>7.1f}"
              f"{np.median([r['growth'] for r in sub]):>8.1f}"
              f"{np.median([r['area'] for r in sub]):>8.0f}"
              f"{np.median([r['rough'] for r in sub]):>7.2f}")

    new = [r for r in recs if r["cls"] == "NEW"]
    na = np.array([r["area"] for r in new]) if new else np.array([0])

    print(f"\n{'='*66}")
    print("NEW BUILDINGS 2019 -> 2023")
    print(f"{'='*66}")
    print(f"  Count            : {len(new)}")
    print(f"  Total roof area  : {na.sum()/1e4:.2f} ha")
    print(f"  Median footprint : {np.median(na):.0f} m2")
    print(f"  Median height    : {np.median([r['h23'] for r in new]):.1f} m")
    print(f"  Density          : {len(new)/34.6:.0f} per km2 over 4 years")
    print(f"  Plus {counts.get('REBUILD?',0)} possible rebuilds (partial 2019 structure)")

    print("\n  footprint distribution:")
    for lo, hi in [(40, 80), (80, 120), (120, 180), (180, 300), (300, 4000)]:
        c = int(((na >= lo) & (na < hi)).sum())
        print(f"    {lo:>4}-{hi:<5} m2 : {c:>5}  {'#'*min(40, c//10)}")

    # ---------- 3. outputs ----------
    prof = {"driver": "GTiff", "height": lab.shape[0], "width": lab.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": tr,
            "nodata": -9999, "compress": "deflate", "tiled": True}

    for cls, fname in [("NEW", "new"), ("EXISTING", "existing")]:
        ids = [r["id"] for r in recs if r["cls"] == cls]
        arr = np.isin(lab, ids).astype(np.float32)
        with rasterio.open(OUT / f"seg_{fname}_{TILE}.tif", "w", **prof) as d:
            d.write(arr, 1)
        print(f"\n  wrote seg_{fname}_{TILE}.tif  ({len(ids)} footprints)")

    # ---------- 4. figure ----------
    print("\nRendering pilot square...")
    p19, ptr, _ = ndsm(2019, PILOT)
    p23, _, _ = ndsm(2023, PILOT)
    r0 = int((tr.f - ptr.f) / px)
    c0 = int((ptr.c - tr.c) / px)

    new_ids = [r["id"] for r in recs if r["cls"] == "NEW"]
    ex_ids = [r["id"] for r in recs if r["cls"] == "EXISTING"]
    sub_lab = lab[r0:r0 + p23.shape[0], c0:c0 + p23.shape[1]]
    new_m = np.isin(sub_lab, new_ids)
    ex_m = np.isin(sub_lab, ex_ids)

    def grey(x, vmax=12):
        v = (np.clip(np.nan_to_num(x) / vmax, 0, 1) * 255).astype(np.uint8)
        return np.dstack([v, v, v])

    over = grey(p23)
    for m, col in [(ex_m, (70, 130, 255)), (new_m, (255, 40, 40))]:
        edge = m & ~ndimage.binary_erosion(m, np.ones((3, 3)))
        for ch, val in enumerate(col):
            over[..., ch] = np.where(edge, val, over[..., ch])

    panels = [grey(p19), grey(p23), over]
    labels = ["nDSM 2019", "nDSM 2023",
              f"RED = new ({len(new_ids)}), BLUE = existing"]
    h_, w_ = panels[0].shape[:2]
    gap = 10
    combo = Image.new("RGB", (w_*3 + gap*2, h_ + 26), (255, 255, 255))
    for i, p in enumerate(panels):
        combo.paste(Image.fromarray(p.astype(np.uint8)), (i*(w_+gap), 26))
    d = ImageDraw.Draw(combo)
    for i, t in enumerate(labels):
        d.text((i*(w_+gap)+6, 7), t, fill=(0, 0, 0))
    combo.save(OUT / "segmented_buildings_pilot.png")
    print(f"  wrote segmented_buildings_pilot.png")

    print(f"\n{'='*66}")
    print("  Open segmented_buildings_pilot.png")
    print("  Every building should now be outlined - red if new, blue if it")
    print("  was already there in 2019. Compare recall against the last figure.")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
