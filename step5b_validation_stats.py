"""
STEP 5B - Turn the review into a defensible number
==================================================

Reads validation/review.csv once you have filled in the verdict column and
produces the precision figure your report needs, with a Wilson confidence
interval and a breakdown by size and height.

Why Wilson rather than the textbook interval
--------------------------------------------
The usual p +/- 1.96*sqrt(p(1-p)/n) breaks down at small n and near p = 1. With
n = 100 and precision around 0.9 it can produce an upper bound above 1.0, which
is nonsense. Wilson stays inside [0,1] and has far better coverage. If you
report one interval in this project, make it this one.

RUN
---
    cd /d D:\\hamilton
    python step5b_validation_stats.py
"""

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
VAL = HERE / "validation"

VALID = ("correct", "wrong", "existing", "partial", "unclear")

# 'correct' and 'partial' both mean a real new building was found. 'partial'
# only says the footprint geometry is poor, which is a separate quality issue
# and is reported separately.
TRUE_POSITIVE = ("correct", "partial")


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bar(frac, width=40):
    return "#" * int(round(width * frac))


def main():
    path = VAL / "review.csv"
    if not path.exists():
        sys.exit(f"{path} not found - run step5a_build_review_pack.py first.")

    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    scored = [r for r in rows if (r.get("verdict") or "").strip().lower() in VALID]

    print(f"{len(rows)} rows, {len(scored)} scored")
    if not scored:
        sys.exit("\nNothing scored yet. Open the sheets and fill in the "
                 "verdict column, then run this again.")

    if len(scored) < len(rows):
        bad = {(r.get('verdict') or '').strip() for r in rows} - set(VALID) - {""}
        print(f"  {len(rows)-len(scored)} unscored or invalid")
        if bad:
            print(f"  unrecognised verdicts: {sorted(bad)}")
        print(f"  valid options: {', '.join(VALID)}")

    for r in scored:
        r["v"] = r["verdict"].strip().lower()
        r["area"] = float(r["area_m2"])
        r["gain"] = float(r["height_gain"])

    counts = Counter(r["v"] for r in scored)
    n = len(scored)

    print(f"\n{'='*62}")
    print("VERDICTS")
    print(f"{'='*62}")
    for k in VALID:
        c = counts.get(k, 0)
        print(f"  {k:<10}{c:>5}  {100*c/n:5.1f}%  {bar(c/n)}")

    # Precision excludes 'unclear' - counting them either way would be a guess
    decided = [r for r in scored if r["v"] != "unclear"]
    tp = sum(1 for r in decided if r["v"] in TRUE_POSITIVE)
    p, lo, hi = wilson(tp, len(decided))

    print(f"\n{'='*62}")
    print("PRECISION")
    print(f"{'='*62}")
    print(f"  decided (excluding 'unclear') : {len(decided)}")
    print(f"  true positives                : {tp}")
    print(f"  PRECISION                     : {100*p:.1f}%")
    print(f"  95% Wilson interval           : {100*lo:.1f}% - {100*hi:.1f}%")

    strict_tp = sum(1 for r in decided if r["v"] == "correct")
    sp, slo, shi = wilson(strict_tp, len(decided))
    print(f"\n  strict (good footprint only)  : {100*sp:.1f}% "
          f"({100*slo:.1f}-{100*shi:.1f}%)")

    n_part = counts.get("partial", 0)
    if n_part:
        print(f"  {n_part} real buildings with poor footprint geometry "
              f"({100*n_part/n:.0f}%)")

    print(f"\n{'='*62}")
    print("PRECISION BY FOOTPRINT SIZE")
    print(f"{'='*62}")
    bands = [(40, 80), (80, 120), (120, 180), (180, 300), (300, 1e9)]
    for lo_a, hi_a in bands:
        sub = [r for r in decided if lo_a <= r["area"] < hi_a]
        if not sub:
            continue
        k = sum(1 for r in sub if r["v"] in TRUE_POSITIVE)
        pp, l, h = wilson(k, len(sub))
        hi_lab = "+" if hi_a > 1e8 else f"-{int(hi_a)}"
        print(f"  {int(lo_a):>4}{hi_lab:<6} m2  n={len(sub):>3}  "
              f"{100*pp:>5.1f}%  ({100*l:.0f}-{100*h:.0f}%)  {bar(pp, 24)}")

    print(f"\n{'='*62}")
    print("PRECISION BY HEIGHT GAIN")
    print(f"{'='*62}")
    for lo_g, hi_g in [(0, 3), (3, 4), (4, 5), (5, 7), (7, 100)]:
        sub = [r for r in decided if lo_g <= r["gain"] < hi_g]
        if not sub:
            continue
        k = sum(1 for r in sub if r["v"] in TRUE_POSITIVE)
        pp, l, h = wilson(k, len(sub))
        print(f"  {lo_g:>2}-{hi_g:<4} m   n={len(sub):>3}  "
              f"{100*pp:>5.1f}%  ({100*l:.0f}-{100*h:.0f}%)  {bar(pp, 24)}")

    # Error taxonomy - what is actually going wrong
    errs = [r for r in decided if r["v"] not in TRUE_POSITIVE]
    if errs:
        print(f"\n{'='*62}")
        print("ERROR BREAKDOWN")
        print(f"{'='*62}")
        for k, c in Counter(r["v"] for r in errs).most_common():
            med_a = sorted(r["area"] for r in errs if r["v"] == k)[len(
                [r for r in errs if r["v"] == k]) // 2]
            print(f"  {k:<10}{c:>4}  median footprint {med_a:.0f} m2")
        notes = [r.get("note", "").strip() for r in errs if r.get("note", "").strip()]
        if notes:
            print("\n  reviewer notes on errors:")
            for t in notes[:12]:
                print(f"    - {t}")

    print(f"\n{'='*62}")
    print("FOR YOUR REPORT")
    print(f"{'='*62}")
    total = 1016
    est_lo, est_hi = int(total * lo), int(total * hi)
    print(f"  Manual validation of {len(decided)} randomly sampled detections,")
    print(f"  stratified by footprint size, gives a precision of {100*p:.0f}%")
    print(f"  (95% CI {100*lo:.0f}-{100*hi:.0f}%). Applied to the full set of")
    print(f"  {total:,} detections, this implies roughly {est_lo:,}-{est_hi:,}")
    print(f"  genuinely new buildings across the 34.6 km2 tile.")
    print(f"\n  Note this is PRECISION, not recall. Segmentation recall was")
    print(f"  separately estimated at ~67% against LINZ building outlines, so")
    print(f"  the true count of new buildings is higher than the detected count.")


if __name__ == "__main__":
    main()
