"""
STEP 1b - Did anything actually get built in Peacocke?
======================================================

Downloads LINZ aerial photos of the same patch of Hamilton from two different
years and saves them side by side so you can see what changed.

SETUP
-----
  1. In this same folder (D:\\hamilton), create a plain text file named:
         linz_key.txt
     Paste your LINZ API key into it. Nothing else. Save.

  2. In Anaconda Prompt:
         cd /d D:\\hamilton
         python step1b_check_change.py

Keeping the key in its own file means this script never contains a secret,
so you can share it or put it on GitHub without leaking anything.

OUTPUT
------
  outputs\\peacocke_COMPARE.png     <- open this one
  outputs\\peacocke_before.png
  outputs\\peacocke_after.png
"""

import io
import math
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from pyproj import Transformer


# =========================================================================
# Settings
# =========================================================================

HERE = Path(__file__).parent
KEY_FILE = HERE / "linz_key.txt"
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# LINZ layer IDs, looked up from data.linz.govt.nz. The catalog search API
# needs broader permission than a "Data access only" key has, so these are
# hard-coded - the tile service works fine with a data-access key.
BEFORE_LAYER = 110193      # Hamilton 0.05m Urban Aerial Photos (2020-2021)
AFTER_LAYER = 114136       # Hamilton 0.05m Urban Aerial Photos (2023)

BEFORE_NAME = "2020-2021"
AFTER_NAME = "2022-2023"

# A 2024-2025 capture also exists (layer 123014), but 114136 is the right one
# here: it was flown at the same time as the 2023 LiDAR, so imagery and
# elevation show the same world on the same day.

# Peacocke pilot square in NZTM2000: 1 km x 1 km
AREA_NZTM = (1798384, 5809807, 1799384, 5810807)

ZOOM = 17          # roughly 1 m per pixel

TILE_URL = ("https://tiles-a.data-cdn.linz.govt.nz/services;key={key}"
            "/tiles/v4/layer={layer}/EPSG:3857/{z}/{x}/{y}.png")


# =========================================================================
# Tile maths
# =========================================================================

_to_wgs84 = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)


def lonlat_to_tile(lon, lat, z):
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return int(x), int(y)


def area_to_tile_range(area_nztm, z):
    xmin, ymin, xmax, ymax = area_nztm
    lon0, lat0 = _to_wgs84.transform(xmin, ymin)
    lon1, lat1 = _to_wgs84.transform(xmax, ymax)
    tx0, ty0 = lonlat_to_tile(lon0, lat1, z)
    tx1, ty1 = lonlat_to_tile(lon1, lat0, z)
    return tx0, ty0, tx1, ty1


# =========================================================================
# Download
# =========================================================================

def fetch_mosaic(api_key, layer_id, label, z=ZOOM):
    tx0, ty0, tx1, ty1 = area_to_tile_range(AREA_NZTM, z)
    ncols, nrows = tx1 - tx0 + 1, ty1 - ty0 + 1
    total = ncols * nrows

    print(f"\n  {label}: downloading {total} tiles (layer {layer_id})")

    canvas = Image.new("RGB", (ncols * 256, nrows * 256), (35, 35, 35))
    ok = 0
    first_error = None

    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            url = TILE_URL.format(key=api_key, layer=layer_id, z=z, x=tx, y=ty)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200 and len(resp.content) > 100:
                    canvas.paste(
                        Image.open(io.BytesIO(resp.content)).convert("RGB"),
                        (i * 256, j * 256),
                    )
                    ok += 1
                elif first_error is None:
                    first_error = f"HTTP {resp.status_code}"
            except Exception as e:
                if first_error is None:
                    first_error = str(e)
        print(f"    {ok} retrieved...", end="\r")

    print(f"    {ok} of {total} tiles retrieved        ")

    if ok == 0:
        raise RuntimeError(
            f"No tiles returned for layer {layer_id} ({first_error}).\n"
            "  HTTP 401 / 403 -> API key wrong, or lacks tile access\n"
            "  HTTP 404       -> layer ID no longer valid"
        )
    if ok < total * 0.5:
        print("    NOTE: many tiles missing - imagery may not cover this square")

    return canvas


def caption(img, text):
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 40], fill=(0, 0, 0))
    d.text((14, 14), text, fill=(255, 235, 0))
    return out


# =========================================================================
# Main
# =========================================================================

def main():
    if not KEY_FILE.exists():
        print("STOP: no linz_key.txt found.")
        print(f"      Create a text file here: {KEY_FILE}")
        print("      Paste your LINZ API key into it and save.")
        return

    api_key = KEY_FILE.read_text().strip()

    if not api_key:
        print(f"STOP: {KEY_FILE.name} is empty. Paste your API key into it.")
        return

    print(f"Key loaded from {KEY_FILE.name} ({len(api_key)} characters)")
    print("Area: Peacocke, Hamilton - 1 km x 1 km")

    before = fetch_mosaic(api_key, BEFORE_LAYER, f"BEFORE {BEFORE_NAME}")
    after = fetch_mosaic(api_key, AFTER_LAYER, f"AFTER  {AFTER_NAME}")

    before.save(OUT_DIR / "peacocke_before.png")
    after.save(OUT_DIR / "peacocke_after.png")

    b = caption(before, f"BEFORE - flown {BEFORE_NAME}")
    a = caption(after, f"AFTER - flown {AFTER_NAME}")
    gap = 16
    combo = Image.new("RGB", (b.width + a.width + gap, max(b.height, a.height)),
                      (255, 255, 255))
    combo.paste(b, (0, 0))
    combo.paste(a, (b.width + gap, 0))
    combo.save(OUT_DIR / "peacocke_COMPARE.png")

    print(f"\nSaved into {OUT_DIR}")
    print("  peacocke_COMPARE.png   <- OPEN THIS")
    print("  peacocke_before.png")
    print("  peacocke_after.png")
    print("\n>> Look for bare paddocks on the LEFT that are streets and houses")
    print(">> on the RIGHT. If you see that, the project works.")


if __name__ == "__main__":
    main()
