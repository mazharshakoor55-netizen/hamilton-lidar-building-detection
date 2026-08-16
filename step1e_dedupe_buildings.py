"""
STEP 1E - Count buildings PROPERLY (deduplicated)
=================================================

What went wrong in step1d
-------------------------
The "All Sources" layer stores one row per building PER CAPTURE CAMPAIGN. A
house traced from 2012, 2016 and 2020 imagery appears three times. Counting
rows by capture year therefore counts the same city over and over - which is
why 112 km2 appeared to contain 199,872 buildings when Hamilton has roughly
60,000.

The fix: group by building_id and take the EARLIEST capture_source_from. That
is the first time the building was seen, i.e. when it appeared.

What this can and cannot tell you
---------------------------------
The newest capture in this layer is 2020. Your LiDAR runs Nov 2019 -> Nov 2023.
So this measures 2016 -> 2020 construction, which overlaps only the first year
of your window. It is a proxy for whether Hamilton was building at all, not a
count of your actual target.

Buildings first seen in the 2020 capture were built somewhere between the 2016
and 2020 flights. Only the portion built after Nov 2019 is missing from your
2019 point cloud - so treat this number as a generous UPPER BOUND.

RUN
---
    cd /d D:\\hamilton
    python step1e_dedupe_buildings.py
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).parent
CACHE = HERE / "buildings_cache.json"

LAYER = 101292
AOI = (1795069, 5809132, 1803069, 5823132)
PAGE = 10000

# Only the fields needed. Dropping geometry and the unused attributes cuts the
# download from hundreds of MB to a few MB.
FIELDS = "building_id,capture_source_from,capture_source_to,suburb_locality"


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

def fetch_all(key):
    if CACHE.exists():
        print(f"Using cached download ({CACHE.name}). Delete it to re-fetch.")
        return json.loads(CACHE.read_text())

    x0, y0, x1, y1 = AOI
    url = f"https://data.linz.govt.nz/services;key={key}/wfs"
    rows = []
    start = 0

    while True:
        r = requests.get(url, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": f"layer-{LAYER}", "outputFormat": "json",
            # northing first - EPSG:2193's declared axis order
            "bbox": f"{y0},{x0},{y1},{x1},urn:ogc:def:crs:EPSG::2193",
            "propertyName": FIELDS,
            "count": PAGE, "startIndex": start,
        }, timeout=300)

        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code}\n{r.text[:300]}")

        got = r.json().get("features", [])
        rows.extend(f.get("properties", {}) for f in got)
        print(f"  fetched {len(rows)} rows...", end="\r")

        if len(got) < PAGE:
            break
        start += PAGE
        if start > 400000:
            break

    print(f"  fetched {len(rows)} rows total          ")
    CACHE.write_text(json.dumps(rows))
    print(f"  cached to {CACHE.name}")
    return rows


def main():
    key = get_api_key()
    rows = fetch_all(key)

    if not rows:
        sys.exit("Nothing returned.")

    # Earliest capture year per building_id = when it first appeared
    first_seen = {}
    last_seen = {}
    no_id = 0

    for p in rows:
        bid = p.get("building_id")
        raw = p.get("capture_source_from")
        if bid is None or not raw:
            no_id += 1
            continue
        try:
            yr = int(str(raw)[:4])
        except ValueError:
            no_id += 1
            continue
        if bid not in first_seen or yr < first_seen[bid]:
            first_seen[bid] = yr
        if bid not in last_seen or yr > last_seen[bid]:
            last_seen[bid] = yr

    n_rows = len(rows)
    n_bldg = len(first_seen)

    print(f"\n  rows downloaded      : {n_rows:>8}")
    print(f"  unique buildings     : {n_bldg:>8}")
    print(f"  duplication factor   : {n_rows/max(n_bldg,1):>8.2f}x")
    if no_id:
        print(f"  unusable rows        : {no_id:>8}")

    appear = Counter(first_seen.values())
    captures = sorted({y for y in first_seen.values()} | {y for y in last_seen.values()})

    print(f"\n  capture campaigns present: {captures}")

    print("\nFIRST APPEARANCE (deduplicated - this is the real number)")
    print("-" * 58)
    peak = max(appear.values()) if appear else 1
    for y in sorted(appear):
        bar = "#" * int(40 * appear[y] / peak)
        note = "  <- existing stock at first mapping" if y == min(appear) else "  = NEW since previous capture"
        print(f"  {y}  {appear[y]:>7}  {bar}{note}")

    newest = max(captures)
    new_in_last = appear.get(newest, 0)
    prev = sorted(captures)[-2] if len(captures) > 1 else None

    print("\n" + "=" * 58)
    print("VERDICT")
    print("=" * 58)
    print(f"  Newest capture in this dataset : {newest}")
    print(f"  Buildings new in that capture  : {new_in_last}")
    if prev:
        print(f"  i.e. built between {prev} and {newest}")
        span = newest - prev
        print(f"  Roughly {new_in_last/span:.0f} new buildings per year")

    print(f"\n  Your LiDAR window is Nov 2019 - Nov 2023 (4 years).")
    print(f"  This dataset stops at {newest}, so it covers at best the first")
    print(f"  year of that window and says nothing about {newest+1}-2023.")

    rate = new_in_last / max(newest - (prev or newest - 4), 1)
    projected = rate * 4
    print(f"\n  If Hamilton kept building at that rate, a 4-year window would")
    print(f"  contain roughly {projected:.0f} new buildings.")

    if rate >= 400:
        print("\n  ENCOURAGING - Hamilton was building steadily. Worth proceeding")
        print("  to a pilot, then confirming against the actual LiDAR.")
    elif rate >= 150:
        print("\n  MARGINAL - some construction, but not a lot. A pilot will tell")
        print("  you quickly whether it is enough.")
    else:
        print("\n  WEAK - consider Tauranga, Queenstown-Lakes or Selwyn instead.")

    # Where is the growth? Useful for placing the pilot square.
    sub = defaultdict(int)
    for p in rows:
        bid = p.get("building_id")
        raw = p.get("capture_source_from")
        if not bid or not raw:
            continue
        try:
            if int(str(raw)[:4]) == newest and first_seen.get(bid) == newest:
                sub[p.get("suburb_locality") or "(unknown)"] += 1
        except ValueError:
            pass

    if sub:
        print(f"\n  Where the {newest} new buildings are:")
        for name, n in sorted(sub.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {name:<32} {n:>6}")
        print("\n  Put your pilot square in the top suburb.")


if __name__ == "__main__":
    main()
