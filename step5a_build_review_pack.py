"""
STEP 5A - Manual validation: build the review pack
==================================================

Every number so far rests on internal consistency - the method agreeing with
itself across three epochs. That is good evidence but it is not independent
verification. This produces the pack you need to check a sample by eye against
data the detector never saw.

What it makes
-------------
A stratified random sample of N detections, and for each one a four-panel strip:

    nDSM 2019 | nDSM 2023 | aerial 2020-21 | aerial 2022-23

The first two are the data the detection was made from. The last two are
independent imagery. If a building is real, panel 3 shows bare ground or an
older structure and panel 4 shows a roof.

Panels are grouped into contact sheets of 10, so you review 100 buildings in
ten images rather than clicking 100 links.

Stratified by footprint size, because small detections are where the errors
concentrate and a purely random sample under-represents the size bands that
matter for tuning.

Output
------
    validation/sheet_01.png ... sheet_10.png
    validation/review.csv          <- fill in the verdict column
    validation/README.txt          <- how to score

Then run step5b_validation_stats.py to turn the filled CSV into a precision
figure with a confidence interval.

RUN
---
    cd /d D:\\hamilton
    python step5a_build_review_pack.py
"""

import io
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.windows import from_bounds

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
OUT = HERE / "outputs"
VAL = HERE / "validation"
VAL.mkdir(exist_ok=True)

TILE = "BD33_10000_0305"
N_SAMPLE = 100
PER_SHEET = 10
CHIP_M = 70                # ground window per panel, metres
PANEL_PX = 300
ZOOM = 18

AERIAL = {"2020-21": 110193, "2022-23": 114136}
TILE_URL = ("https://tiles-a.data-cdn.linz.govt.nz/services;key={key}"
            "/tiles/v4/layer={layer}/EPSG:3857/{z}/{x}/{y}.png")

_to_wgs = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
_tile_cache = {}


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

def font(sz, bold=False):
    for n in (["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
              else ["arial.ttf", "DejaVuSans.ttf"]):
        try:
            return ImageFont.truetype(n, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def lonlat_to_px(lon, lat, z):
    n = 256 * 2 ** z
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def get_tile(session, key, layer, z, x, y):
    """Cached tile fetch - neighbouring buildings share tiles, so this matters."""
    ck = (layer, z, x, y)
    if ck in _tile_cache:
        return _tile_cache[ck]

    url = TILE_URL.format(key=key, layer=layer, z=z, x=x, y=y)
    img = None
    delay = 0.3
    for _ in range(4):
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                break
            if r.status_code == 404:
                break
        except Exception:
            pass
        time.sleep(delay)
        delay *= 2

    _tile_cache[ck] = img
    time.sleep(0.08)
    return img


def aerial_chip(session, key, layer, e, n, size_m=CHIP_M, z=ZOOM):
    """Crop a square of aerial imagery centred on an NZTM point."""
    half = size_m / 2
    lon0, lat0 = _to_wgs.transform(e - half, n - half)
    lon1, lat1 = _to_wgs.transform(e + half, n + half)
    x0, y0 = lonlat_to_px(lon0, lat1, z)
    x1, y1 = lonlat_to_px(lon1, lat0, z)

    tx0, ty0 = int(x0 // 256), int(y0 // 256)
    tx1, ty1 = int(x1 // 256), int(y1 // 256)
    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256),
                       (35, 35, 35))
    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            t = get_tile(session, key, layer, z, tx, ty)
            if t:
                canvas.paste(t, (i * 256, j * 256))

    crop = canvas.crop((int(x0 - tx0 * 256), int(y0 - ty0 * 256),
                        int(x1 - tx0 * 256), int(y1 - ty0 * 256)))
    return crop.resize((PANEL_PX, PANEL_PX), Image.LANCZOS)


def ndsm_chip(year, e, n, size_m=CHIP_M, vmax=9.0):
    """Crop the local nDSM as a greyscale panel."""
    half = size_m / 2
    box = (e - half, n - half, e + half, n + half)

    def rd(prod):
        with rasterio.open(DATA / f"{year}_{prod}_{TILE}.tiff") as s:
            w = from_bounds(*box, transform=s.transform)
            a = s.read(1, window=w, boundless=True, fill_value=0).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = 0
            return a

    h = np.clip(rd("dsm_1m") - rd("dem_1m"), 0, 60)
    if h.size == 0:
        h = np.zeros((10, 10), np.float32)
    v = (np.clip(h / vmax, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([v, v, v]))
    return img.resize((PANEL_PX, PANEL_PX), Image.NEAREST)


def crosshair(img, colour=(255, 60, 60)):
    """Mark the detection centre so the reviewer knows what to judge."""
    d = ImageDraw.Draw(img)
    c = PANEL_PX // 2
    r = int(PANEL_PX * 0.13)
    d.ellipse([c - r, c - r, c + r, c + r], outline=colour, width=3)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        d.line([(c + dx * r, c + dy * r),
                (c + dx * (r + 12), c + dy * (r + 12))], fill=colour, width=3)
    return img


def main():
    gj = OUT / "buildings_new.geojson"
    if not gj.exists():
        sys.exit("buildings_new.geojson missing - run step4a first.")
    feats = json.loads(gj.read_text())["features"]
    print(f"{len(feats)} detections available")

    # Stratified by area - small detections carry most of the error, and a
    # plain random sample would under-sample the bands that matter
    bands = [(40, 80), (80, 120), (120, 180), (180, 300), (300, 100000)]
    rng = random.Random(42)
    sample = []
    per = max(1, N_SAMPLE // len(bands))
    for lo, hi in bands:
        pool = [f for f in feats if lo <= f["properties"]["area_m2"] < hi]
        take = min(per, len(pool))
        sample += rng.sample(pool, take)
        print(f"  {lo:>4}-{hi:<6} m2 : {len(pool):>4} available, {take} sampled")

    rng.shuffle(sample)          # so the reviewer cannot infer size from order
    print(f"\n{len(sample)} buildings sampled")

    key = get_key()
    session = requests.Session()
    session.headers.update({"User-Agent": "hamilton-lidar-validation/1.0"})

    lab_f = font(15, bold=True)
    hdr_f = font(19, bold=True)
    sub_f = font(13)

    rows = []
    sheets = (len(sample) + PER_SHEET - 1) // PER_SHEET

    for si in range(sheets):
        chunk = sample[si * PER_SHEET:(si + 1) * PER_SHEET]
        rowh = PANEL_PX + 56
        sheet = Image.new("RGB", (PANEL_PX * 4 + 5 * 10 + 60,
                                  rowh * len(chunk) + 70), (250, 250, 248))
        d = ImageDraw.Draw(sheet)
        d.text((16, 14), f"Validation sheet {si+1} of {sheets}",
               fill=(20, 20, 20), font=hdr_f)
        d.text((16, 40),
               "red circle marks the detection  |  panels 3-4 are independent imagery",
               fill=(110, 110, 110), font=sub_f)

        for ri, f in enumerate(chunk):
            p = f["properties"]
            idx = si * PER_SHEET + ri + 1
            e, n = p["easting"], p["northing"]
            y = 70 + ri * rowh

            panels = [
                ("nDSM 2019", ndsm_chip(2019, e, n)),
                ("nDSM 2023", ndsm_chip(2023, e, n)),
                ("aerial 2020-21", aerial_chip(session, key, AERIAL["2020-21"], e, n)),
                ("aerial 2022-23", aerial_chip(session, key, AERIAL["2022-23"], e, n)),
            ]
            for pi, (nm, im) in enumerate(panels):
                im = crosshair(im)
                x = 50 + pi * (PANEL_PX + 10)
                sheet.paste(im, (x, y))
                d.rectangle([x, y, x + PANEL_PX, y + PANEL_PX],
                            outline=(70, 70, 70))
                d.text((x + 4, y + PANEL_PX + 4), nm,
                       fill=(90, 90, 90), font=sub_f)

            d.text((10, y + PANEL_PX // 2 - 10), f"{idx}",
                   fill=(20, 20, 20), font=hdr_f)
            d.text((50, y + PANEL_PX + 18),
                   f"#{idx}   {p['area_m2']:.0f} m2   "
                   f"gain {p['height_gain']:.1f} m   "
                   f"2025 growth {p['growth_23_25']:+.1f} m",
                   fill=(60, 60, 60), font=lab_f)

            rows.append({
                "id": idx, "area_m2": p["area_m2"],
                "height_gain": p["height_gain"],
                "growth_23_25": p["growth_23_25"],
                "easting": e, "northing": n,
            })

            print(f"  sheet {si+1}: {ri+1}/{len(chunk)}", end="\r")

        out = VAL / f"sheet_{si+1:02d}.png"
        sheet.save(out)
        print(f"  wrote {out.name}                 ")

    csv = VAL / "review.csv"
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write("id,area_m2,height_gain,growth_23_25,easting,northing,verdict,note\n")
        for r in rows:
            fh.write(f"{r['id']},{r['area_m2']},{r['height_gain']},"
                     f"{r['growth_23_25']},{r['easting']},{r['northing']},,\n")
    print(f"\n  wrote {csv.name}  ({len(rows)} rows to score)")

    (VAL / "README.txt").write_text(
        "MANUAL VALIDATION\n"
        "=================\n\n"
        "Open sheet_01.png through sheet_10.png. Each row is one detection.\n"
        "The red circle marks it. Judge ONLY the circled structure.\n\n"
        "Panels 1-2 are the LiDAR the detection came from.\n"
        "Panels 3-4 are aerial imagery - independent evidence.\n\n"
        "Put one of these in the verdict column of review.csv:\n\n"
        "  correct    a building is present in 2023 and absent in 2019\n"
        "  wrong      no building - vegetation, shadow, vehicle, artefact\n"
        "  existing   a building is there but it was ALSO there in 2019\n"
        "  partial    real building but the footprint is badly wrong\n"
        "  unclear    cannot tell from these panels\n\n"
        "Score every row before moving on. Do not skip the hard ones -\n"
        "they are the informative ones. Note anything odd in the note column.\n\n"
        "A caution: the aerial panels were flown at different dates from the\n"
        "LiDAR. 2020-21 imagery is roughly a year AFTER the 2019 LiDAR, so a\n"
        "building can legitimately appear in panel 3 while being genuinely\n"
        "absent from the 2019 point cloud. Weight panels 1-2 for the 'was it\n"
        "there in 2019' question and panels 3-4 for 'is it a building at all'.\n\n"
        "Then run:  python step5b_validation_stats.py\n"
    )
    print(f"  wrote README.txt")

    print(f"\n{'='*58}")
    print(f"  {sheets} sheets in {VAL}")
    print("  Read README.txt, score review.csv, then run step5b.")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()
