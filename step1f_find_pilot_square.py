"""
STEP 1F - Let the data choose your pilot square
===============================================

Step 1e showed Hamilton's growth is INFILL in established suburbs, not
greenfield expansion. So the pilot square must sit where new buildings actually
cluster - not where I guessed from memory (Peacocke, which turned out to be
farmland).

This fetches the geometry of genuinely-new buildings, grids the AOI into 1 km
cells, and ranks them. The top cell is your pilot square.

Needs the cache written by step1e. Run that first.

RUN
---
    cd /d D:\\hamilton
    python step1f_find_pilot_square.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import requests

HERE = Path(__file__).parent
CACHE_IDS = HERE / "buildings_cache.json"
CACHE_GEOM = HERE / "new_buildings_geom.json"

LAYER = 101292
AOI = (1795069, 5809132, 1803069, 5823132)
PAGE = 10000
CELL = 1000        # metres - matches the pilot square size


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

def build_first_seen():
    """Earliest capture year per building_id, from the step1e cache."""
    if not CACHE_IDS.exists():
        sys.exit(f"{CACHE_IDS.name} missing. Run step1e_dedupe_buildings.py first.")

    first = {}
    for p in json.loads(CACHE_IDS.read_text()):
        bid, raw = p.get("building_id"), p.get("capture_source_from")
        if not bid or not raw:
            continue
        try:
            y = int(str(raw)[:4])
        except ValueError:
            continue
        if bid not in first or y < first[bid]:
            first[bid] = y
    return first


def fetch_geometry(key, newest_year):
    """
    Fetch geometry for rows from the newest capture campaign.

    A CQL filter keeps this to ~68k rows instead of all 200k. We still have to
    filter to genuinely-new buildings afterwards using the cache, because this
    campaign also re-captured every existing building.
    """
    if CACHE_GEOM.exists():
        print(f"Using cached geometry ({CACHE_GEOM.name}). Delete to re-fetch.")
        return json.loads(CACHE_GEOM.read_text())

    x0, y0, x1, y1 = AOI
    url = f"https://data.linz.govt.nz/services;key={key}/wfs"
    out, start = [], 0

    while True:
        r = requests.get(url, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": f"layer-{LAYER}", "outputFormat": "json",
            "bbox": f"{y0},{x0},{y1},{x1},urn:ogc:def:crs:EPSG::2193",
            "cql_filter": f"capture_source_from >= '{newest_year}-01-01'",
            "count": PAGE, "startIndex": start,
        }, timeout=300)

        if r.status_code != 200:
            sys.exit(f"HTTP {r.status_code}\n{r.text[:300]}")

        got = r.json().get("features", [])
        for f in got:
            g = f.get("geometry") or {}
            c = centroid(g)
            if c:
                out.append({"id": f.get("properties", {}).get("building_id"),
                            "suburb": f.get("properties", {}).get("suburb_locality"),
                            "e": c[0], "n": c[1]})
        print(f"  fetched {len(out)} with geometry...", end="\r")

        if len(got) < PAGE:
            break
        start += PAGE
        if start > 200000:
            break

    print(f"  fetched {len(out)} with geometry        ")
    CACHE_GEOM.write_text(json.dumps(out))
    return out


def centroid(geom):
    """Rough centroid of a polygon or multipolygon - good enough for gridding."""
    t = geom.get("type")
    coords = geom.get("coordinates")
    if not coords:
        return None
    if t == "Polygon":
        ring = coords[0]
    elif t == "MultiPolygon":
        ring = coords[0][0]
    else:
        return None
    if not ring:
        return None
    return (sum(p[0] for p in ring) / len(ring),
            sum(p[1] for p in ring) / len(ring))


def main():
    key = get_api_key()
    first_seen = build_first_seen()
    newest = max(first_seen.values())
    print(f"Newest capture campaign: {newest}")

    truly_new = {b for b, y in first_seen.items() if y == newest}
    print(f"Genuinely new buildings in that campaign: {len(truly_new)}\n")

    rows = fetch_geometry(key, newest)
    pts = [r for r in rows if r["id"] in truly_new]
    print(f"\n  of {len(rows)} rows, {len(pts)} are genuinely new buildings")

    if not pts:
        sys.exit("No new buildings with geometry - cannot place a pilot square.")

    # Grid the AOI, offset by half a cell as well so a cluster straddling a
    # boundary is not split in half and missed
    x0, y0, x1, y1 = AOI
    best = []
    for ox in (0, CELL // 2):
        for oy in (0, CELL // 2):
            grid = defaultdict(list)
            for p in pts:
                c = int((p["e"] - x0 - ox) // CELL)
                r = int((p["n"] - y0 - oy) // CELL)
                grid[(r, c)].append(p)
            for (r, c), lst in grid.items():
                e = x0 + ox + c * CELL
                n = y0 + oy + r * CELL
                if e < x0 or n < y0 or e + CELL > x1 or n + CELL > y1:
                    continue
                subs = defaultdict(int)
                for p in lst:
                    subs[p["suburb"] or "?"] += 1
                top = max(subs.items(), key=lambda kv: kv[1])[0]
                best.append((len(lst), e, n, top))

    # Deduplicate overlapping offsets - keep the strongest, then anything
    # 1 km clear of it
    best.sort(reverse=True)
    picked = []
    for n_b, e, n, sub in best:
        if all(abs(e - pe) >= CELL or abs(n - pn) >= CELL for _, pe, pn, _ in picked):
            picked.append((n_b, e, n, sub))
        if len(picked) == 6:
            break

    print("\nBEST 1 km PILOT SQUARES (ranked by new buildings)")
    print("-" * 72)
    print(f"{'rank':>4} {'new':>6}  {'suburb':<18} bbox NZTM")
    for i, (n_b, e, n, sub) in enumerate(picked, 1):
        print(f"{i:>4} {n_b:>6}  {sub:<18} ({e:.0f}, {n:.0f}, {e+CELL:.0f}, {n+CELL:.0f})")

    n_b, e, n, sub = picked[0]
    print("\n" + "=" * 72)
    print("RECOMMENDED PILOT SQUARE")
    print("=" * 72)
    print(f"  Suburb        : {sub}")
    print(f"  New buildings : {n_b} in 1 km2")
    print(f"  bbox          : ({e:.0f}, {n:.0f}, {e+CELL:.0f}, {n+CELL:.0f})")
    print(f"\n  To view it in the imagery comparison, patch step1b with:")
    print(f"    AREA_NZTM = ({e:.0f}, {n:.0f}, {e+CELL:.0f}, {n+CELL:.0f})")
    print(f"    ZOOM = 18")
    print("\n  Zoom 18 (~0.5 m/px) - infill buildings are small, so you need the")
    print("  detail. Zoom 15 was why the earlier texture test saw nothing.")


if __name__ == "__main__":
    main()
