"""
STEP 3C - Three-epoch check: buildings stay put, vegetation keeps growing
========================================================================

The problem step3b could not solve
----------------------------------
93.8% of detections were UNMAPPED, which is ambiguous. LINZ's newest capture is
2020, so an unmapped detection is either:

    a genuine building constructed 2021-2023, or
    vegetation that gained height

A reference layer that stops in 2020 cannot tell these apart. But physics can.

    A BUILDING   jumps once, then holds.   2019 low -> 2023 high -> 2025 same
    VEGETATION   grows continuously.       2019 low -> 2023 mid  -> 2025 higher

Hamilton was flown three times - 2019, 2023 and 2025 - so the trajectory is
measurable. A building that appeared before 2023 should show near-zero further
growth by 2025. Anything still climbing at 1 m+ over two years is a plant.

This is a genuinely strong validation: it needs no reference layer at all, so
it works in the post-2020 window where LINZ has no data. Worth writing up
properly - most change-detection projects have nothing equivalent.

RUN
---
    cd /d D:\\hamilton
    python step3c_three_epoch.py
"""

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except ImportError:
    sys.exit("Needs scipy:  pip install scipy")

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
DATA.mkdir(parents=True, exist_ok=True)

BUCKET = "https://nz-elevation.s3.ap-southeast-2.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
TILE = "BD33_10000_0305"
PILOT = (1802319, 5818882, 1802819, 5819382)

# A building that existed by 2023 should not gain height by 2025.
# Allow a little for roof furniture and survey noise.
STABLE_TOL = 0.8

# Vegetation growth over two years. Poplars and eucalypts manage well over this.
GROWTH_MIN = 1.0


def retry(fn, what, tries=8):
    delay = 2.0
    for i in range(tries):
        try:
            return fn()
        except requests.exceptions.RequestException as e:
            kind = "DNS/network" if "getaddrinfo" in str(e) else type(e).__name__
            print(f"    {what}: {kind}, retry {i+1}/{tries} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def fetch_2025():
    """Download 2025 DEM and DSM for the same tile."""
    for prod in ("dem_1m", "dsm_1m"):
        dest = DATA / f"2025_{prod}_{TILE}.tiff"
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"  have {dest.name} ({dest.stat().st_size/1024**2:.1f} MB)")
            continue

        prefix = f"waikato/hamilton_2025/{prod}/"

        def _find():
            keys, token = [], None
            while True:
                p = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
                if token:
                    p["continuation-token"] = token
                r = requests.get(BUCKET, params=p, timeout=60)
                r.raise_for_status()
                root = ET.fromstring(r.content)
                for c in root.findall("s3:Contents", NS):
                    keys.append((c.find("s3:Key", NS).text,
                                 int(c.find("s3:Size", NS).text)))
                nt = root.find("s3:NextContinuationToken", NS)
                if nt is None:
                    return keys
                token = nt.text

        keys = retry(_find, f"list {prod}")
        if not keys:
            sys.exit("Could not list the 2025 survey.")

        match = [(k, s) for k, s in keys if TILE in k and k.endswith((".tif", ".tiff"))]
        if not match:
            sys.exit(f"Tile {TILE} not in the 2025 {prod} survey.")

        key, size = match[0]
        print(f"  downloading {prod} ({size/1024**2:.1f} MB)")

        def _get():
            r = requests.get(f"{BUCKET}/{key}", timeout=(30, 300), stream=True)
            r.raise_for_status()
            got = 0
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
                    got += len(chunk)
                    print(f"    {got/1024**2:6.1f} / {size/1024**2:.1f} MB", end="\r")
            return got

        if retry(_get, dest.name) is None:
            sys.exit(f"Failed to download {prod}.")
        print(f"    got {dest.name}                    ")


def ndsm(year, window=None):
    def rd(prod):
        p = DATA / f"{year}_{prod}_{TILE}.tiff"
        if not p.exists():
            sys.exit(f"Missing {p.name}")
        with rasterio.open(p) as s:
            w = None
            if window:
                from rasterio.windows import from_bounds
                w = from_bounds(*window, transform=s.transform)
            a = s.read(1, window=w).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
            return a, (s.window_transform(w) if w else s.transform), s.crs

    dsm, tr, crs = rd("dsm_1m")
    dem, _, _ = rd("dem_1m")
    return np.clip(dsm - dem, 0, 60), tr, crs


def main():
    print("Fetching the 2025 survey...")
    fetch_2025()

    print("\nComputing nDSM for all three epochs...")
    n19, tr, crs = ndsm(2019)
    n23, _, _ = ndsm(2023)
    n25, _, _ = ndsm(2025)
    print(f"  {n19.shape[1]} x {n19.shape[0]} px")

    mask_path = OUT / f"new_buildings_{TILE}.tif"
    if not mask_path.exists():
        sys.exit("Run step3a_ndsm_difference.py first.")
    with rasterio.open(mask_path) as s:
        mask = s.read(1) > 0.5

    lab, n = ndimage.label(mask)
    print(f"  {n} detections to classify")

    px = abs(tr.a)
    recs = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        m = lab[sl] == i
        h19 = float(np.nanmedian(n19[sl][m]))
        h23 = float(np.nanmedian(n23[sl][m]))
        h25 = float(np.nanmedian(n25[sl][m]))
        recs.append({
            "id": i, "area": float(m.sum()) * px * px,
            "h19": h19, "h23": h23, "h25": h25,
            "gain_19_23": h23 - h19, "growth_23_25": h25 - h23,
        })

    def classify(r):
        g = r["growth_23_25"]
        if g >= GROWTH_MIN:
            return "VEGETATION"        # still climbing
        if abs(g) <= STABLE_TOL:
            return "BUILDING"          # jumped, then held
        if g <= -GROWTH_MIN:
            return "REMOVED"           # gone by 2025 - cleared, or was a crane
        return "UNCERTAIN"

    for r in recs:
        r["cls"] = classify(r)

    from collections import Counter
    counts = Counter(r["cls"] for r in recs)
    total = len(recs)

    print(f"\n{'='*64}")
    print("TRAJECTORY CLASSIFICATION  (2019 -> 2023 -> 2025)")
    print(f"{'='*64}")
    for k in ("BUILDING", "VEGETATION", "REMOVED", "UNCERTAIN"):
        v = counts.get(k, 0)
        print(f"  {k:<12}{v:>6}  {100*v/total:5.1f}%  {'#'*int(44*v/total)}")

    print(f"\n{'='*64}")
    print("MEDIAN HEIGHT TRAJECTORY BY CLASS")
    print(f"{'='*64}")
    print(f"  {'class':<12}{'2019':>7}{'2023':>7}{'2025':>7}"
          f"{'gain':>8}{'growth':>9}{'area':>8}")
    for k in ("BUILDING", "VEGETATION", "REMOVED", "UNCERTAIN"):
        sub = [r for r in recs if r["cls"] == k]
        if not sub:
            continue
        print(f"  {k:<12}"
              f"{np.median([r['h19'] for r in sub]):>7.1f}"
              f"{np.median([r['h23'] for r in sub]):>7.1f}"
              f"{np.median([r['h25'] for r in sub]):>7.1f}"
              f"{np.median([r['gain_19_23'] for r in sub]):>8.1f}"
              f"{np.median([r['growth_23_25'] for r in sub]):>9.1f}"
              f"{np.median([r['area'] for r in sub]):>8.0f}")

    print(f"\n{'='*64}")
    print("BUILDING SHARE BY PATCH SIZE")
    print(f"{'='*64}")
    for lo, hi in [(25, 50), (50, 80), (80, 120), (120, 200), (200, 400), (400, 5000)]:
        sub = [r for r in recs if lo <= r["area"] < hi]
        if not sub:
            continue
        b = sum(1 for r in sub if r["cls"] == "BUILDING")
        v = sum(1 for r in sub if r["cls"] == "VEGETATION")
        print(f"  {lo:>5}-{hi:<6}{len(sub):>6}   building {100*b/len(sub):>3.0f}%"
              f"   vegetation {100*v/len(sub):>3.0f}%")

    # Write a mask containing only the confirmed buildings
    keep = {r["id"] for r in recs if r["cls"] == "BUILDING"}
    confirmed = np.isin(lab, list(keep)).astype(np.float32)
    prof = {"driver": "GTiff", "height": lab.shape[0], "width": lab.shape[1],
            "count": 1, "dtype": "float32", "crs": crs, "transform": tr,
            "nodata": -9999, "compress": "deflate", "tiled": True}
    with rasterio.open(OUT / f"confirmed_buildings_{TILE}.tif", "w", **prof) as d:
        d.write(confirmed, 1)
    print(f"\n  wrote confirmed_buildings_{TILE}.tif  ({len(keep)} patches)")

    area = sum(r["area"] for r in recs if r["cls"] == "BUILDING")
    print(f"\n{'='*64}")
    print("RESULT")
    print(f"{'='*64}")
    print(f"  Confirmed new buildings : {len(keep)}")
    print(f"  Total new roof area     : {area/1e4:.2f} ha")
    print(f"  Rejected as vegetation  : {counts.get('VEGETATION',0)}")
    print(f"\n  The 2025 survey resolves what LINZ cannot: it separates real")
    print(f"  construction from plant growth in the post-2020 window where no")
    print(f"  reference layer exists.")


if __name__ == "__main__":
    main()
