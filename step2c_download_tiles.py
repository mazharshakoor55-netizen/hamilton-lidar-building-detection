"""
STEP 2C - Download the four pilot tiles
=======================================

step2b identified the tile covering Chartwell: BD33_10000_0305

So there is nothing left to search for. This just fetches four files:

    2019 DEM    2019 DSM    2023 DEM    2023 DSM

all for that one tile. Roughly 150-180 MB.

Every network call is wrapped in retries, including DNS failures - the
getaddrinfo error that killed step2b is a transient name-resolution drop, and
it will happen again on a long download if nothing catches it.

Resumable: already-downloaded files are skipped, so if it dies halfway just
run it again.

RUN
---
    cd /d D:\\hamilton
    python step2c_download_tiles.py
"""

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

HERE = Path(__file__).parent
DATA = HERE / "data" / "elevation"
DATA.mkdir(parents=True, exist_ok=True)

BUCKET = "https://nz-elevation.s3.ap-southeast-2.amazonaws.com"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

TILE = "BD33_10000_0305"          # found by step2b - covers the pilot square

SURVEYS = {2019: "waikato/hamilton_2019/", 2023: "waikato/hamilton_2023/"}
PRODUCTS = ["dem_1m", "dsm_1m"]

MAX_TRIES = 8


def retry(fn, what):
    """
    Run fn with exponential backoff.

    Catches DNS failures too. On a home connection a name-resolution drop
    mid-download is common, and a bare requests call turns it into a crash.
    """
    delay = 2.0
    for attempt in range(MAX_TRIES):
        try:
            return fn()
        except requests.exceptions.RequestException as e:
            kind = "DNS/network" if "getaddrinfo" in str(e) else type(e).__name__
            print(f"    {what}: {kind}, retry {attempt+1}/{MAX_TRIES} in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    print(f"    {what}: gave up after {MAX_TRIES} attempts")
    return None


def find_key(prefix, tile):
    """Locate the full S3 key for a tile - the folder layout may include a CRS level."""
    def _list():
        keys, token = [], None
        while True:
            params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                params["continuation-token"] = token
            r = requests.get(BUCKET, params=params, timeout=60)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for c in root.findall("s3:Contents", NS):
                keys.append((c.find("s3:Key", NS).text,
                             int(c.find("s3:Size", NS).text)))
            nt = root.find("s3:NextContinuationToken", NS)
            if nt is None:
                return keys
            token = nt.text

    keys = retry(_list, f"list {prefix}")
    if not keys:
        return None, 0

    for k, s in keys:
        if tile in k and k.lower().endswith((".tif", ".tiff")):
            return k, s
    return None, 0


def download(key, dest, expect):
    if dest.exists() and abs(dest.stat().st_size - expect) < 4096:
        print(f"    have {dest.name}  ({dest.stat().st_size/1024**2:.1f} MB)")
        return True

    part = dest.with_suffix(dest.suffix + ".part")

    def _get():
        # Resume a partial download rather than starting over
        start = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        r = requests.get(f"{BUCKET}/{key}", headers=headers,
                         timeout=(30, 300), stream=True)
        r.raise_for_status()

        mode = "ab" if start and r.status_code == 206 else "wb"
        got = start if mode == "ab" else 0

        with open(part, mode) as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                got += len(chunk)
                pct = 100 * got / expect if expect else 0
                print(f"    {dest.name}: {got/1024**2:6.1f} MB ({pct:4.1f}%)", end="\r")
        return got

    got = retry(_get, dest.name)
    if got is None:
        return False

    part.replace(dest)
    print(f"    got  {dest.name}  ({got/1024**2:.1f} MB)              ")
    return True


def main():
    print(f"Tile: {TILE}  (covers the Chartwell pilot square)")
    print(f"Saving to: {DATA}\n")

    jobs = []
    for year, prefix in SURVEYS.items():
        for prod in PRODUCTS:
            key, size = find_key(f"{prefix}{prod}/", TILE)
            if key:
                print(f"  {year} {prod:8s} -> {key}  ({size/1024**2:.1f} MB)")
                jobs.append((year, prod, key, size))
            else:
                print(f"  {year} {prod:8s} -> NOT FOUND")

    if not jobs:
        sys.exit("\nNothing found. Check your internet connection and rerun.")

    total = sum(s for *_, s in jobs) / 1024 ** 2
    print(f"\n{'='*56}")
    print(f"  {len(jobs)} files, {total:.0f} MB total")
    print(f"{'='*56}\n")

    ok = 0
    for year, prod, key, size in jobs:
        if download(key, DATA / f"{year}_{prod}_{TILE}.tiff", size):
            ok += 1

    print(f"\n{ok}/{len(jobs)} downloaded to {DATA}")

    if ok == len(jobs):
        print("\nAll four tiles present. Next: compute nDSM for each epoch")
        print("(DSM minus DEM), then difference 2023 against 2019. New")
        print("buildings show up as blocks of positive height change.")
        print("\nYou will need rasterio for that step:")
        print("    pip install rasterio")
    else:
        print("\nSome files missing - just run this again, it resumes.")


if __name__ == "__main__":
    main()
