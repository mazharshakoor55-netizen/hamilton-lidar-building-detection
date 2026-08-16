"""
STEP 4C - Proper cartography
============================

step4b produced technically-correct but badly-designed maps. Fixed here:

  NORTH ARROW        was missing entirely
  TYPE SCALE         furniture was sized for a 500 px image and rendered at
                     2100 px, so everything looked tiny. All text now scales
                     with output width.
  LEGEND COLLISION   legend labels overlapped the attribution line
  INVISIBLE FINDINGS the real failure. At 3.78 m/px an average 188 m2 building
                     is ~13 px - smaller than a full stop. Outlining it changes
                     nothing. At overview scale buildings are now drawn as
                     SYMBOLS with a minimum size and a dark halo, so they read
                     against the imagery. Cartographic convention: when the
                     feature is smaller than the minimum legible mark, switch
                     from true shape to symbol.
  NO HIERARCHY       added a headline-figure callout so the key number is the
                     first thing seen, not something to be inferred.
  FLAT BASEMAP       replaced blanket darkening with desaturation plus a milder
                     darken, which keeps ground detail readable while letting
                     saturated symbols dominate.

RUN
---
    cd /d D:\\hamilton
    python step4c_render_map_v2.py
"""

import io
import json
import math
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pyproj import Transformer

HERE = Path(__file__).parent
OUT = HERE / "outputs"

AERIAL_LAYER = 114136
TILE_URL = ("https://tiles-a.data-cdn.linz.govt.nz/services;key={key}"
            "/tiles/v4/layer={layer}/EPSG:3857/{z}/{x}/{y}.png")

OVERVIEW_ZOOM = 15
DETAIL_ZOOM = 18
DETAIL_SPAN_M = 700

RAMP = [(0.0, (255, 247, 188)), (3.0, (254, 196, 79)),
        (4.5, (254, 120, 46)), (6.0, (217, 39, 44)), (9.0, (128, 0, 38))]

_to_wgs = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)


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

def font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    names = (["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"] if bold
             else ["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    f = None
    for n in names:
        try:
            f = ImageFont.truetype(n, size)
            break
        except OSError:
            continue
    if f is None:
        f = ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def colour_for(g):
    if g <= RAMP[0][0]:
        return RAMP[0][1]
    for (v0, c0), (v1, c1) in zip(RAMP, RAMP[1:]):
        if g <= v1:
            t = (g - v0) / (v1 - v0) if v1 > v0 else 0
            return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return RAMP[-1][1]


def lonlat_to_px(lon, lat, z):
    n = 256 * 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_basemap(session, key, bbox, z):
    lon0, lat0 = _to_wgs.transform(bbox[0], bbox[1])
    lon1, lat1 = _to_wgs.transform(bbox[2], bbox[3])
    px0, py0 = lonlat_to_px(lon0, lat1, z)
    px1, py1 = lonlat_to_px(lon1, lat0, z)
    tx0, ty0 = int(px0 // 256), int(py0 // 256)
    tx1, ty1 = int(px1 // 256), int(py1 // 256)
    ncols, nrows = tx1 - tx0 + 1, ty1 - ty0 + 1
    total = ncols * nrows
    print(f"    {ncols} x {nrows} = {total} tiles at z{z}")

    canvas = Image.new("RGB", (ncols * 256, nrows * 256), (30, 30, 30))
    ok = 0
    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            url = TILE_URL.format(key=key, layer=AERIAL_LAYER, z=z, x=tx, y=ty)
            delay = 0.3
            for _ in range(5):
                try:
                    r = session.get(url, timeout=30)
                    if r.status_code == 200 and len(r.content) > 100:
                        canvas.paste(Image.open(io.BytesIO(r.content)).convert("RGB"),
                                     (i * 256, j * 256))
                        ok += 1
                        break
                    if r.status_code == 404:
                        break
                except Exception:
                    pass
                time.sleep(delay)
                delay *= 2
            print(f"    {ok}/{total}", end="\r")
            time.sleep(0.12)
    print(f"    {ok}/{total} tiles                ")
    return canvas, (tx0 * 256, ty0 * 256)


def north_arrow(d, x, y, size, scale):
    """Conventional north arrow: filled half, outlined half, N above."""
    h = size
    w = size * 0.42
    tipy = y - h / 2
    basey = y + h / 2
    d.polygon([(x, tipy), (x + w, basey), (x, basey - h * 0.22)],
              fill=(255, 255, 255), outline=(20, 20, 20))
    d.polygon([(x, tipy), (x - w, basey), (x, basey - h * 0.22)],
              fill=(30, 30, 30), outline=(20, 20, 20))
    f = font(int(20 * scale), bold=True)
    tw = d.textlength("N", font=f)
    d.text((x - tw / 2, tipy - 26 * scale), "N", fill=(255, 255, 255), font=f,
           stroke_width=max(2, int(3 * scale)), stroke_fill=(20, 20, 20))


def draw_map(feats, bbox, z, key, session, title, subtitle, stats, outfile,
             mode="outline"):
    base, (ox, oy) = fetch_basemap(session, key, bbox, z)

    # Desaturate and mildly darken - keeps ground detail legible while letting
    # saturated symbols dominate. Blanket darkening killed both.
    grey = base.convert("L").convert("RGB")
    base = Image.blend(base, grey, 0.45)
    base = Image.blend(base, Image.new("RGB", base.size, (0, 0, 0)), 0.22)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    halo = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    dh = ImageDraw.Draw(halo)

    m_per_px = 156543.03 * math.cos(math.radians(-37.79)) / (2 ** z)

    for f in feats:
        p = f["properties"]
        col = colour_for(p.get("height_gain", 0) or 0)
        ring = f["geometry"]["coordinates"][0]
        pts = [tuple(x - o for x, o in zip(lonlat_to_px(lon, lat, z), (ox, oy)))
               for lon, lat in ring]
        if len(pts) < 3:
            continue

        if mode == "symbol":
            # Feature smaller than the minimum legible mark -> draw a symbol
            cx = sum(x for x, _ in pts) / len(pts)
            cy = sum(y for _, y in pts) / len(pts)
            area = p.get("area_m2", 100)
            r = max(4.5, min(11.0, math.sqrt(area) / m_per_px * 1.5))
            dh.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3],
                       fill=(0, 0, 0, 170))
            d.ellipse([cx - r, cy - r, cx + r, cy + r],
                      fill=col + (255,), outline=(255, 255, 255, 210), width=1)
        else:
            dh.line(pts + [pts[0]], fill=(0, 0, 0, 190), width=7)
            d.polygon(pts, fill=col + (110,))
            d.line(pts + [pts[0]], fill=col + (255,), width=3)

    halo = halo.filter(ImageFilter.GaussianBlur(2))
    img = Image.alpha_composite(base.convert("RGBA"), halo)
    img = Image.alpha_composite(img, overlay).convert("RGB")

    lon0, lat0 = _to_wgs.transform(bbox[0], bbox[1])
    lon1, lat1 = _to_wgs.transform(bbox[2], bbox[3])
    x0, y0 = lonlat_to_px(lon0, lat1, z)
    x1, y1 = lonlat_to_px(lon1, lat0, z)
    img = img.crop((int(x0 - ox), int(y0 - oy), int(x1 - ox), int(y1 - oy)))

    W, H = img.size
    s = max(1.0, W / 1000.0)          # everything scales with output width

    head = int(112 * s)
    foot = int(126 * s)
    canvas = Image.new("RGB", (W, H + head + foot), (252, 252, 250))
    canvas.paste(img, (0, head))
    dd = ImageDraw.Draw(canvas)

    # ---- header ----
    # Shrink type until it fits - a title that runs off the edge is worse
    # than one a few points smaller.
    avail = W - int(52 * s)
    ts = int(34 * s)
    while ts > 12 and dd.textlength(title, font=font(ts, bold=True)) > avail:
        ts -= 1
    dd.text((int(26 * s), int(22 * s)), title,
            fill=(15, 15, 20), font=font(ts, bold=True))
    ss = int(19 * s)
    while ss > 9 and dd.textlength(subtitle, font=font(ss)) > avail:
        ss -= 1
    dd.text((int(26 * s), int(68 * s)), subtitle,
            fill=(105, 105, 115), font=font(ss))
    dd.line([(0, head - 2), (W, head - 2)], fill=(30, 30, 35), width=int(3 * s))

    # ---- north arrow, on the map, top right ----
    ax = W - int(62 * s)
    ay = head + int(72 * s)
    pad = int(46 * s)
    box = Image.new("RGBA", (pad * 2, pad * 2 + int(20 * s)), (255, 255, 255, 45))
    canvas.paste(Image.alpha_composite(
        canvas.crop((ax - pad, ay - pad - int(20 * s),
                     ax + pad, ay + pad)).convert("RGBA"), box).convert("RGB"),
        (ax - pad, ay - pad - int(20 * s)))
    north_arrow(ImageDraw.Draw(canvas), ax, ay, int(52 * s), s)

    # ---- headline callout, on the map, top left ----
    if stats:
        bw, bh = min(int(310 * s), W - int(150 * s)), int(84 * s)
        bx, by = int(24 * s), head + int(24 * s)
        panel = Image.new("RGBA", (bw, bh), (18, 18, 24, 205))
        region = canvas.crop((bx, by, bx + bw, by + bh)).convert("RGBA")
        canvas.paste(Image.alpha_composite(region, panel).convert("RGB"), (bx, by))
        dd.rectangle([bx, by, bx + bw, by + bh], outline=(255, 255, 255), width=int(2 * s))
        dd.text((bx + int(18 * s), by + int(10 * s)), stats[0],
                fill=(255, 255, 255), font=font(int(40 * s), bold=True))
        dd.text((bx + int(18 * s), by + int(56 * s)), stats[1],
                fill=(215, 215, 225), font=font(int(16 * s)))

    # ---- footer: legend, scale, credit ----
    fy = head + H + int(20 * s)
    dd.text((int(26 * s), fy), "Height gain above ground",
            fill=(30, 30, 35), font=font(int(17 * s), bold=True))

    bx = int(26 * s)
    by = fy + int(28 * s)
    bw, bh = int(330 * s), int(20 * s)
    for i in range(bw):
        dd.line([(bx + i, by), (bx + i, by + bh)], fill=colour_for(9.0 * i / bw))
    dd.rectangle([bx, by, bx + bw, by + bh], outline=(40, 40, 45), width=max(1, int(1.5 * s)))
    for v in (0, 3, 6, 9):
        px = bx + int(bw * v / 9.0)
        dd.line([(px, by + bh), (px, by + bh + int(6 * s))], fill=(40, 40, 45), width=max(1, int(1.5 * s)))
        lab = f"{v} m"
        fnt = font(int(15 * s))
        dd.text((px - dd.textlength(lab, font=fnt) / 2, by + bh + int(9 * s)),
                lab, fill=(55, 55, 60), font=fnt)

    # scale bar
    span_m = bbox[2] - bbox[0]
    mpp = span_m / W
    for target in (2000, 1000, 500, 250, 100, 50):
        bar = target / mpp
        if bar < W * 0.26:
            break
    sx = W - int(bar) - int(30 * s)
    sy = by + int(4 * s)
    if sx < bx + bw + int(40 * s):          # would overlap the legend ramp
        sx = W - int(bar) - int(30 * s)
        sy = by + bh + int(34 * s)
    dd.rectangle([sx, sy, sx + int(bar) / 2, sy + int(11 * s)],
                 fill=(30, 30, 35), outline=(30, 30, 35))
    dd.rectangle([sx + int(bar) / 2, sy, sx + int(bar), sy + int(11 * s)],
                 fill=(255, 255, 255), outline=(30, 30, 35))
    lab = f"{target} m" if target < 1000 else f"{target//1000} km"
    fnt = font(int(15 * s), bold=True)
    dd.text((sx + int(bar) / 2 - dd.textlength(lab, font=fnt) / 2,
             sy + int(16 * s)), lab, fill=(40, 40, 45), font=fnt)
    dd.text((sx, sy - int(17 * s)), "0", fill=(90, 90, 95), font=font(int(12 * s)))

    credit_full = ("Data: Toitu Te Whenua LINZ (CC BY 4.0)   |   "
                   "NZGD2000 / New Zealand Transverse Mercator 2000   |   "
                   "Method: nDSM change 2019-2023, validated with 2025 survey")
    credit_short = "Data: Toitu Te Whenua LINZ (CC BY 4.0)  |  NZTM2000"
    cf = font(int(13 * s))
    credit = credit_full if dd.textlength(credit_full, font=cf) < W - int(52 * s) \
        else credit_short
    dd.text((int(26 * s), head + H + foot - int(26 * s)), credit,
            fill=(140, 140, 148), font=cf)

    dd.rectangle([0, 0, W - 1, head + H + foot - 1], outline=(30, 30, 35), width=int(2 * s))
    canvas.save(outfile)
    print(f"    wrote {outfile.name}  ({canvas.width} x {canvas.height})")


def main():
    gj = OUT / "buildings_new.geojson"
    if not gj.exists():
        sys.exit("buildings_new.geojson missing - run step4a first.")
    feats = json.loads(gj.read_text())["features"]
    print(f"Loaded {len(feats)} new buildings")

    es = [f["properties"]["easting"] for f in feats]
    ns = [f["properties"]["northing"] for f in feats]
    pad = 250
    bbox = (min(es) - pad, min(ns) - pad, max(es) + pad, max(ns) + pad)

    key = get_key()
    session = requests.Session()
    session.headers.update({"User-Agent": "hamilton-lidar/1.0"})

    print("\nOverview...")
    draw_map(feats, bbox, OVERVIEW_ZOOM, key, session,
             "New buildings detected from LiDAR",
             "Hamilton, New Zealand  |  November 2019 to November 2023  |  34.6 km2",
             (f"{len(feats):,} buildings", "detected from elevation change alone"),
             OUT / "map_overview_v2.png", mode="symbol")

    print("\nFinding densest cluster...")
    best, bc = -1, None
    for cx in range(int(bbox[0]), int(bbox[2]), 250):
        for cy in range(int(bbox[1]), int(bbox[3]), 250):
            h = DETAIL_SPAN_M / 2
            n = sum(1 for e, nn in zip(es, ns)
                    if abs(e - cx) <= h and abs(nn - cy) <= h)
            if n > best:
                best, bc = n, (cx, cy)
    print(f"  {best} buildings at {bc}")

    h = DETAIL_SPAN_M / 2
    dbox = (bc[0] - h, bc[1] - h, bc[0] + h, bc[1] + h)
    sub = [f for f in feats if abs(f["properties"]["easting"] - bc[0]) <= h
           and abs(f["properties"]["northing"] - bc[1]) <= h]

    print("\nDetail...")
    draw_map(sub, dbox, DETAIL_ZOOM, key, session,
             "Detected footprints on 2023 aerial imagery",
             "Chartwell, Hamilton  |  detection outlines shown over the imagery "
             "they were never trained on",
             (f"{len(sub)} buildings", f"in a single {DETAIL_SPAN_M} m window"),
             OUT / "map_detail_v2.png", mode="outline")

    print("\n" + "=" * 58)
    print("  map_overview_v2.png  |  map_detail_v2.png")
    print("=" * 58)


if __name__ == "__main__":
    main()
