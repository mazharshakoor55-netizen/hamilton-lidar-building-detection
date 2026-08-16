"""
STEP 6D - Export CNN detections as polygons
===========================================

step6c wrote a raster mask. The review pack needs polygons with the same
attribute schema the rule-based export produced, so the two can be reviewed
identically and their precision figures compared directly.

Output:
    outputs/cnn_buildings_new.geojson   <- what step5a will read
    outputs/cnn_buildings_new.csv

Identical fields to buildings_new.geojson: class, area_m2, h2019, h2023,
h2025, height_gain, growth_23_25, compactness, easting, northing - plus a
`confidence` field, which the rule-based method had no equivalent of.

RUN
---
    cd /d D:\\hamilton
    python step6d_export_cnn_polygons.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio import features
from scipy import ndimage

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
TILE = "BD33_10000_0305"

SIMPLIFY_M = 0.5
MIN_AREA = 40.0

_to_wgs = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)


def ndsm(year):
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
    return np.clip(np.nan_to_num(dsm) - np.nan_to_num(dem), 0, 60), tr, crs


def simplify_ring(ring, tol):
    """Ramer-Douglas-Peucker, iterative to avoid recursion limits."""
    pts = [tuple(p) for p in ring]
    if len(pts) < 4:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5
        best, bi = -1.0, -1
        for i in range(i0 + 1, i1):
            x, y = pts[i]
            d = (((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 if norm == 0
                 else abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm)
            if d > best:
                best, bi = d, i
        if best > tol:
            keep[bi] = True
            stack.append((i0, bi))
            stack.append((bi, i1))
    out = [p for p, k in zip(pts, keep) if k]
    return out if len(out) >= 4 else pts


def main():
    mask_path = OUT / f"cnn_new_buildings_{TILE}.tif"
    if not mask_path.exists():
        sys.exit(f"{mask_path.name} missing - run step6c_apply_model.py first.")

    print("Loading CNN mask and elevation...")
    with rasterio.open(mask_path) as s:
        mask = s.read(1) > 0.5
        tr = s.transform
        crs = s.crs
    n19, _, _ = ndsm(2019)
    n23, _, _ = ndsm(2023)
    n25, _, _ = ndsm(2025)
    px = abs(tr.a)
    print(f"  {int(mask.sum()):,} building pixels")

    # Confidence, if the probability raster was kept
    prob_path = OUT / f"cnn_prob_2023_{TILE}.tif"
    prob = None
    if prob_path.exists():
        with rasterio.open(prob_path) as s:
            prob = s.read(1)

    print("Vectorising...")
    feats = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask,
                                     transform=tr):
        if val != 1:
            continue
        rings = geom["coordinates"]
        if not rings:
            continue

        outer = rings[0]
        xs = [p[0] for p in outer]
        ys = [p[1] for p in outer]
        area = abs(sum(xs[i] * ys[i-1] - xs[i-1] * ys[i]
                       for i in range(len(xs)))) / 2.0
        if area < MIN_AREA:
            continue

        perim = sum(((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2) ** 0.5
                    for i in range(1, len(xs)))
        comp = 4 * np.pi * area / (perim * perim) if perim else 0.0

        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        col = int((cx - tr.c) / px)
        row = int((tr.f - cy) / px)
        r0, r1 = max(0, row-3), min(n19.shape[0], row+4)
        c0, c1 = max(0, col-3), min(n19.shape[1], col+4)
        if r1 <= r0 or c1 <= c0:
            continue

        h19 = float(np.nanmedian(n19[r0:r1, c0:c1]))
        h23 = float(np.nanmedian(n23[r0:r1, c0:c1]))
        h25 = float(np.nanmedian(n25[r0:r1, c0:c1]))
        conf = (float(np.mean(prob[r0:r1, c0:c1])) if prob is not None else None)

        simp = [simplify_ring(r, SIMPLIFY_M) for r in rings]
        wgs = [[list(_to_wgs.transform(x, y)) for x, y in r] for r in simp]

        props = {
            "class": "NEW",
            "area_m2": round(area, 1),
            "h2019": round(h19, 2),
            "h2023": round(h23, 2),
            "h2025": round(h25, 2),
            "height_gain": round(h23 - h19, 2),
            "growth_23_25": round(h25 - h23, 2),
            "compactness": round(comp, 3),
            "easting": round(cx, 1),
            "northing": round(cy, 1),
            "method": "cnn",
        }
        if conf is not None:
            props["confidence"] = round(conf, 3)

        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": wgs},
                      "properties": props})

        if len(feats) % 200 == 0:
            print(f"  {len(feats)} polygons...", end="\r")

    print(f"  {len(feats)} polygons              ")

    gj = OUT / "cnn_buildings_new.geojson"
    gj.write_text(json.dumps({"type": "FeatureCollection",
                              "name": "hamilton_cnn_new_buildings",
                              "features": feats}))
    print(f"  wrote {gj.name}  ({gj.stat().st_size/1024**2:.1f} MB)")

    cols = ["area_m2", "h2019", "h2023", "h2025", "height_gain",
            "growth_23_25", "compactness", "easting", "northing"]
    csv = OUT / "cnn_buildings_new.csv"
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("id," + ",".join(cols) + "\n")
        for i, f in enumerate(feats, 1):
            p = f["properties"]
            fh.write(f"{i}," + ",".join(str(p[c]) for c in cols) + "\n")
    print(f"  wrote {csv.name}")

    if feats:
        a = np.array([f["properties"]["area_m2"] for f in feats])
        g = np.array([f["properties"]["height_gain"] for f in feats])
        print(f"\n{'='*56}")
        print("CNN DETECTIONS")
        print(f"{'='*56}")
        print(f"  count            : {len(feats)}")
        print(f"  total roof area  : {a.sum()/1e4:.2f} ha")
        print(f"  median footprint : {np.median(a):.0f} m2")
        print(f"  median gain      : {np.median(g):.1f} m")
        print("\n  footprint distribution:")
        for lo, hi in [(40, 80), (80, 120), (120, 180), (180, 300), (300, 5000)]:
            c = int(((a >= lo) & (a < hi)).sum())
            print(f"    {lo:>4}-{hi:<5} m2 : {c:>5}  {'#'*min(40, c//12)}")

    print(f"\n{'='*56}")
    print("  Next: rebuild the review pack against these detections.")
    print("  step5a should already point at cnn_buildings_new.geojson.")
    print("  Move the old validation/ folder aside first so you do not")
    print("  overwrite the rule-based review you already completed:")
    print("\n    move validation validation_rulebased")
    print("    python step5a_build_review_pack.py")
    print(f"{'='*56}")


if __name__ == "__main__":
    main()
