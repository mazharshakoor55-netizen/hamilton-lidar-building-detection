"""
STEP 3A - nDSM difference: find buildings that appeared between 2019 and 2023
=============================================================================

This is the core result of the project.

    nDSM = DSM - DEM              height above ground
    change = nDSM_2023 - nDSM_2019

A new building is a compact patch of ground that gained several metres of
height. Everything else - roads, lawns, paddocks - stays flat.

What it guards against
----------------------
  GRID ALIGNMENT. If the two epochs sit on even slightly different grids, every
  building edge produces a false ring of change. Checked explicitly and refused
  if wrong, rather than silently producing a beautiful wrong answer.

  TREE GROWTH. Vegetation gains height too - a poplar can add 3 m in four
  years. Buildings are flat on top and compact in plan; canopy is neither. The
  roughness and shape filters separate them.

  DEM DRIFT. The two surveys were flown by different vendors. If their bare
  earth disagrees by a constant offset, that offset propagates into every
  height. Measured over stable ground and reported.

REQUIRES
--------
    pip install rasterio scipy

RUN
---
    cd /d D:\\hamilton
    python step3a_ndsm_difference.py
"""

import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from PIL import Image

try:
    from scipy import ndimage
except ImportError:
    sys.exit("Needs scipy:  pip install scipy")

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TILE = "BD33_10000_0305"
PILOT = (1802319, 5818882, 1802819, 5819382)     # 500 m visualisation window

# A new building must gain at least this much height. 2.5 m clears hedges,
# parked trucks, shipping containers and retaining walls.
MIN_HEIGHT_GAIN = 2.5

# Plan area limits. LINZ maps buildings from 10 m2 up; below ~25 m2 the
# false-positive rate climbs steeply.
MIN_AREA_M2 = 25
MAX_AREA_M2 = 5000

# Roofs are planar, tree crowns are chaotic. Measured as local height std.
MAX_ROUGHNESS = 1.6

# Polsby-Popper compactness. Buildings are compact; the ragged shapes that
# leak out of thresholding are not.
MIN_COMPACTNESS = 0.20


def path(year, prod):
    p = DATA / f"{year}_{prod}_{TILE}.tiff"
    if not p.exists():
        sys.exit(f"Missing {p.name} - run step2c_download_tiles.py first.")
    return p


def read(year, prod, window=None):
    with rasterio.open(path(year, prod)) as src:
        w = None
        if window:
            w = from_bounds(*window, transform=src.transform)
        arr = src.read(1, window=w).astype(np.float32)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
        tr = src.window_transform(w) if w else src.transform
        return arr, tr, src.crs


def check_alignment():
    """
    Confirm both epochs sit on the identical grid.

    A sub-pixel offset makes every building edge look like change. It is the
    single most likely way to produce a convincing but wrong result, so verify
    rather than assume.
    """
    print("Checking grid alignment...")
    meta = {}
    for year in (2019, 2023):
        for prod in ("dem_1m", "dsm_1m"):
            with rasterio.open(path(year, prod)) as s:
                meta[(year, prod)] = (s.transform, s.width, s.height, s.crs)

    ref = meta[(2019, "dem_1m")]
    for k, v in meta.items():
        same = (v[0] == ref[0]) and (v[1], v[2]) == (ref[1], ref[2])
        print(f"  {k[0]} {k[1]:8s} {v[1]}x{v[2]}  {v[3]}  "
              f"{'aligned' if same else 'MISMATCH'}")
        if not same:
            sys.exit("Grids differ - would need resampling before differencing.")

    print(f"  origin ({ref[0].c:.1f}, {ref[0].f:.1f})  pixel {ref[0].a:.2f} m")
    print("  All four rasters share one grid.\n")
    return ref


def compute_ndsm(year, window=None):
    dsm, tr, crs = read(year, "dsm_1m", window)
    dem, _, _ = read(year, "dem_1m", window)
    ndsm = dsm - dem
    # Negative height above ground is meaningless; clip. Cap guards against
    # powerline and bird returns that survived noise classification.
    ndsm = np.clip(ndsm, 0, 60)
    return ndsm, tr, crs, dem


def roughness(arr, win=3):
    a = np.nan_to_num(arr, nan=0.0)
    m = ndimage.uniform_filter(a, win)
    m2 = ndimage.uniform_filter(a * a, win)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def colourise_height(a, vmax=12):
    """Greyscale height ramp for the nDSM panels."""
    v = np.clip(np.nan_to_num(a, nan=0) / vmax, 0, 1)
    g = (v * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def colourise_change(d, vmax=8):
    """Red = height gained, blue = height lost, dark = unchanged."""
    h, w = d.shape
    img = np.zeros((h, w, 3), np.uint8)
    d = np.nan_to_num(d, nan=0)
    pos = np.clip(d / vmax, 0, 1)
    neg = np.clip(-d / vmax, 0, 1)
    img[..., 0] = (pos * 255).astype(np.uint8)
    img[..., 2] = (neg * 255).astype(np.uint8)
    img[..., 1] = (np.clip((np.abs(d) < 0.5).astype(float) * 0.18, 0, 1) * 255).astype(np.uint8)
    return img


def main():
    ref = check_alignment()

    # ---------------- whole tile: the headline numbers ----------------
    print("Computing nDSM for the whole tile...")
    n19, tr, crs, dem19 = compute_ndsm(2019)
    n23, _, _, dem23 = compute_ndsm(2023)
    px = abs(tr.a)
    print(f"  raster {n19.shape[1]} x {n19.shape[0]} px at {px:.1f} m = "
          f"{n19.size * px * px / 1e6:.1f} km2")

    # DEM drift between vendors, measured on genuinely flat stable ground
    flat = (n19 < 0.3) & (n23 < 0.3)
    drift = np.nanmedian((dem23 - dem19)[flat])
    spread = np.nanpercentile(np.abs((dem23 - dem19)[flat] - drift), 68)
    print(f"\n  DEM drift on stable ground: median {drift:+.3f} m, "
          f"scatter {spread:.3f} m")
    if abs(drift) > 0.25:
        print(f"  Correcting for a {drift:+.2f} m systematic offset.")
    else:
        print("  Negligible - the two surveys agree on bare earth.")

    change = (n23 - n19) - (drift if abs(drift) > 0.25 else 0.0)

    # ---------------- candidate new buildings ----------------
    print(f"\nFinding patches that gained >= {MIN_HEIGHT_GAIN} m...")
    grew = np.nan_to_num(change, nan=0) >= MIN_HEIGHT_GAIN
    grew = ndimage.binary_closing(grew, np.ones((3, 3)), iterations=2)
    grew = ndimage.binary_fill_holes(grew)
    grew = ndimage.binary_opening(grew, np.ones((3, 3)))

    lab, n = ndimage.label(grew)
    print(f"  {n} raw patches")

    rough23 = roughness(n23)
    keep, stats = [], []

    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        area = m.sum() * px * px
        if not (MIN_AREA_M2 <= area <= MAX_AREA_M2):
            continue

        r = float(np.mean(rough23[sl][m]))
        if r > MAX_ROUGHNESS:
            continue

        # Compactness from the patch perimeter
        er = ndimage.binary_erosion(m, np.ones((3, 3)))
        per = (m & ~er).sum() * px
        comp = 4 * np.pi * area / (per * per) if per else 0
        if comp < MIN_COMPACTNESS:
            continue

        keep.append(i)
        stats.append({"area": area,
                      "height": float(np.median(change[sl][m])),
                      "rough": r, "comp": comp})

    print(f"  {len(keep)} survive area, roughness and compactness filters")

    if stats:
        a = np.array([s["area"] for s in stats])
        hgt = np.array([s["height"] for s in stats])
        print(f"\n  median footprint : {np.median(a):6.0f} m2")
        print(f"  median height gain: {np.median(hgt):5.1f} m")
        print(f"  total new roof area: {a.sum()/1e4:6.2f} ha")
        print("\n  size distribution:")
        for lo, hi in [(25, 60), (60, 120), (120, 250), (250, 600), (600, 5000)]:
            c = ((a >= lo) & (a < hi)).sum()
            print(f"    {lo:>4}-{hi:<4} m2 : {c:>4}  {'#' * min(40, c // 3)}")

    # ---------------- write rasters ----------------
    prof = {"driver": "GTiff", "height": n19.shape[0], "width": n19.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": tr,
            "nodata": -9999, "compress": "deflate", "tiled": True}

    for name, arr in [("ndsm_2019", n19), ("ndsm_2023", n23),
                      ("ndsm_change", change)]:
        with rasterio.open(OUT / f"{name}_{TILE}.tif", "w", **prof) as d:
            d.write(np.nan_to_num(arr, nan=-9999).astype(np.float32), 1)
        print(f"\n  wrote {name}_{TILE}.tif")

    mask = np.isin(lab, keep).astype(np.float32)
    with rasterio.open(OUT / f"new_buildings_{TILE}.tif", "w", **prof) as d:
        d.write(mask, 1)
    print(f"  wrote new_buildings_{TILE}.tif")

    # ---------------- visualisation over the pilot square ----------------
    print("\nRendering the pilot square...")
    p19, ptr, _, _ = compute_ndsm(2019, PILOT)
    p23, _, _, _ = compute_ndsm(2023, PILOT)
    pchg = (p23 - p19) - (drift if abs(drift) > 0.25 else 0.0)

    panels = [colourise_height(p19), colourise_height(p23), colourise_change(pchg)]
    labels = ["nDSM 2019", "nDSM 2023", "CHANGE (red = new height)"]

    h, w = panels[0].shape[:2]
    gap = 10
    combo = Image.new("RGB", (w * 3 + gap * 2, h + 26), (255, 255, 255))
    for i, (p, lab_txt) in enumerate(zip(panels, labels)):
        im = Image.fromarray(p)
        combo.paste(im, (i * (w + gap), 26))
    from PIL import ImageDraw
    d = ImageDraw.Draw(combo)
    for i, lab_txt in enumerate(labels):
        d.text((i * (w + gap) + 6, 7), lab_txt, fill=(0, 0, 0))

    combo.save(OUT / "ndsm_change_pilot.png")
    print(f"  wrote ndsm_change_pilot.png  ({combo.width} x {combo.height})")

    print("\n" + "=" * 60)
    print("  Open outputs/ndsm_change_pilot.png")
    print("  Red blobs = new buildings. Compare against chartwell_COMPARE.png")
    print("  and they should line up with the houses that appeared.")
    print("=" * 60)


if __name__ == "__main__":
    main()
