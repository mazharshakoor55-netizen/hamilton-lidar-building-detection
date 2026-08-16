"""
STEP 4A - Export buildings as polygons
======================================

Rasters are fine for computing but wrong for delivering. A council does not
want a GeoTIFF mask - they want a table of buildings with attributes they can
sort, filter and join to their own records.

This vectorises the classified footprints and writes GeoJSON with one feature
per building, carrying:

    class        NEW / EXISTING / REBUILD? / VEGETATION
    area_m2      footprint area
    h2019/23/25  median height above ground at each survey
    height_gain  2023 minus 2019
    growth       2025 minus 2023 (the vegetation discriminator)
    compactness  Polsby-Popper shape measure

Two files are written:

    buildings_all.geojson    every classified footprint
    buildings_new.geojson    just the new ones - the headline result

Coordinates are written in WGS84 because that is what the GeoJSON spec expects
and what every tool reads without argument. All measurements stay in metres,
computed in NZTM before reprojection.

Only needs rasterio, scipy, pyproj - all already installed.

RUN
---
    cd /d D:\\hamilton
    python step4a_export_polygons.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from pyproj import Transformer

try:
    from scipy import ndimage
except ImportError:
    sys.exit("Needs scipy:  pip install scipy")

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
TILE = "BD33_10000_0305"

SIMPLIFY_M = 0.5      # removes the 1 m pixel staircase without eating corners

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
    return np.clip(dsm - dem, 0, 60), tr, crs


def simplify_ring(ring, tol):
    """
    Ramer-Douglas-Peucker, iterative so deep rings do not blow the stack.

    Written out rather than imported because shapely is not installed and this
    is the only geometry operation needed.
    """
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
            if norm == 0:
                d = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
            else:
                d = abs(dy * x - dx * y + x1 * y0 - y1 * x0) / norm
            if d > best:
                best, bi = d, i

        if best > tol:
            keep[bi] = True
            stack.append((i0, bi))
            stack.append((bi, i1))

    out = [p for p, k in zip(pts, keep) if k]
    return out if len(out) >= 4 else pts


def main():
    print("Loading elevation...")
    n19, tr, crs = ndsm(2019)
    n23, _, _ = ndsm(2023)
    n25, _, _ = ndsm(2025)
    px = abs(tr.a)

    layers = {}
    for cls, fname in [("NEW", "new"), ("EXISTING", "existing")]:
        p = OUT / f"seg_{fname}_{TILE}.tif"
        if not p.exists():
            sys.exit(f"Missing {p.name} - run step3e_segment_first.py first")
        with rasterio.open(p) as s:
            layers[cls] = s.read(1) > 0.5
        print(f"  {cls}: {int(layers[cls].sum()):,} pixels")

    feats_all, feats_new = [], []
    counts = Counter()

    for cls, mask in layers.items():
        print(f"\nVectorising {cls}...")
        lab, n = ndimage.label(mask)

        for geom, val in features.shapes(mask.astype(np.uint8), mask=mask,
                                         transform=tr):
            if val != 1:
                continue
            rings = geom["coordinates"]
            if not rings:
                continue

            outer = rings[0]
            # Area by the shoelace formula, in NZTM metres
            xs = [p[0] for p in outer]
            ys = [p[1] for p in outer]
            area = abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i]
                           for i in range(len(xs)))) / 2.0
            if area < 20:
                continue

            perim = sum(((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5
                        for i in range(1, len(xs)))
            comp = 4 * np.pi * area / (perim * perim) if perim else 0.0

            # Sample heights inside the footprint
            cx, cy = float(np.mean(xs)), float(np.mean(ys))
            col = int((cx - tr.c) / px)
            row = int((tr.f - cy) / px)
            r0, r1 = max(0, row - 3), min(n19.shape[0], row + 4)
            c0, c1 = max(0, col - 3), min(n19.shape[1], col + 4)
            if r1 <= r0 or c1 <= c0:
                continue

            h19 = float(np.nanmedian(n19[r0:r1, c0:c1]))
            h23 = float(np.nanmedian(n23[r0:r1, c0:c1]))
            h25 = float(np.nanmedian(n25[r0:r1, c0:c1]))

            simp = [simplify_ring(r, SIMPLIFY_M) for r in rings]
            wgs = [[list(_to_wgs.transform(x, y)) for x, y in r] for r in simp]

            f = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": wgs},
                "properties": {
                    "class": cls,
                    "area_m2": round(area, 1),
                    "h2019": round(h19, 2),
                    "h2023": round(h23, 2),
                    "h2025": round(h25, 2),
                    "height_gain": round(h23 - h19, 2),
                    "growth_23_25": round(h25 - h23, 2),
                    "compactness": round(comp, 3),
                    "easting": round(cx, 1),
                    "northing": round(cy, 1),
                },
            }
            feats_all.append(f)
            if cls == "NEW":
                feats_new.append(f)
            counts[cls] += 1

        print(f"  {counts[cls]} polygons")

    def write(path, feats, name):
        fc = {
            "type": "FeatureCollection",
            "name": name,
            "features": feats,
        }
        path.write_text(json.dumps(fc))
        mb = path.stat().st_size / 1024 ** 2
        print(f"  wrote {path.name}  ({len(feats)} features, {mb:.1f} MB)")

    print("\nWriting...")
    write(OUT / "buildings_all.geojson", feats_all, "hamilton_buildings")
    write(OUT / "buildings_new.geojson", feats_new, "hamilton_new_buildings")

    # A plain CSV too - opens in Excel, no GIS needed
    csv = OUT / "buildings_new.csv"
    cols = ["area_m2", "h2019", "h2023", "h2025", "height_gain",
            "growth_23_25", "compactness", "easting", "northing"]
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("id," + ",".join(cols) + "\n")
        for i, f in enumerate(feats_new, 1):
            p = f["properties"]
            fh.write(f"{i}," + ",".join(str(p[c]) for c in cols) + "\n")
    print(f"  wrote {csv.name}  ({len(feats_new)} rows)")

    if feats_new:
        a = np.array([f["properties"]["area_m2"] for f in feats_new])
        g = np.array([f["properties"]["height_gain"] for f in feats_new])
        print(f"\n{'='*58}")
        print("NEW BUILDINGS SUMMARY")
        print(f"{'='*58}")
        print(f"  count           : {len(feats_new)}")
        print(f"  total roof area : {a.sum()/1e4:.2f} ha")
        print(f"  median footprint: {np.median(a):.0f} m2")
        print(f"  median gain     : {np.median(g):.1f} m")

    print(f"\n{'='*58}")
    print("  Drag buildings_new.geojson into QGIS - it will land in the")
    print("  right place with all attributes. Style by height_gain or")
    print("  area_m2 for a finished map.")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
