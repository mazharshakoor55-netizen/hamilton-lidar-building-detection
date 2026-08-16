# Detecting new buildings from LiDAR elevation change
### Hamilton, New Zealand — November 2019 to November 2023

Two methods for finding buildings constructed between two national LiDAR surveys,
using only elevation data: a geometric rule-based pipeline, and a U-Net trained on
LiDAR-derived surfaces. Both are validated against independent aerial imagery.

**No imagery is used for detection.** The methods see only height above ground,
slope and surface roughness. Imagery is used solely to check the answers.

---

## Result

| | Rule-based | U-Net |
|---|---|---|
| Building segmentation recall | 67% | **91%** |
| Building segmentation precision | — | 84% |
| Pixel IoU (held-out spatial blocks) | — | 0.78 |
| New buildings detected | 1,016 | 1,162 |
| Total new roof area | 18.3 ha | 22.8 ha |
| Median footprint | 168 m² | 151 m² |
| Detection precision (reviewed sample) | **54%** (n=99) | ~40% (n=20, partial) |

The CNN closed a 24-point segmentation recall gap. Its per-detection precision was
not fully measured — see *Honest limitations*.

**Study area:** 34.6 km² of Hamilton (LINZ tile BD33_10000_0305)
**Data:** three LiDAR surveys (2019, 2023, 2025), all open, all CC BY 4.0

---

## Why this problem

New Zealand's authoritative building layer is traced from aerial imagery, and the
most recent capture for Hamilton is **2020**. Any building constructed between 2021
and 2023 exists in the LiDAR but not in the national dataset.

That gap is what this project fills. Councils need current building inventories for
rating valuation, consent compliance, hazard exposure and emergency response.
Reflying imagery is expensive; the LiDAR is already captured and already open.

---

## Method

### 1. Feasibility before processing

Three checks were run before any elevation data was downloaded:

- **Coverage** — both 2019 and 2023 surveys cover the study area (112 km², 100%,
  378/378 tiles pairable by name)
- **Change exists** — LINZ building outlines, deduplicated by `building_id`,
  showed ~2,064 new buildings per year across the area
- **Change is visible** — before/after aerial imagery over a candidate square

The third check failed twice on areas chosen by intuition. It passed only once the
site was selected from the data — see *What went wrong* below.

### 2. Elevation processing

```
nDSM = DSM − DEM          height above ground
```

Derived per epoch: nDSM, slope (Horn's method), local height standard deviation.

**Data quality note.** The 2019 and 2023 surveys were flown by different vendors
four years apart. Their bare-earth models agree to a **median 2 cm, scatter 6 cm**
across 34.6 km². Any height change detected is therefore real signal, not survey
disagreement.

Both surveys were also flown in early November — 3–5 Nov 2019 and 10 Nov 2023 —
so vegetation phenology is matched. Seasonal difference, normally a major confound
in multi-temporal LiDAR, largely cancels.

### 3. Rule-based detection

Segment buildings from the 2023 surface, then classify each by its height history:

| Class | 2019 | 2023 | 2025 |
|---|---|---|---|
| Existing | 3.5 m | 3.5 m | 3.5 m |
| **New** | **0.0 m** | 3.4 m | 3.4 m |
| Vegetation | 3.4 m | 4.7 m | 5.9 m |

**The three-epoch trajectory test is the most novel part of this project.** A
building's height jumps once and holds. Vegetation keeps growing. Over 2019 → 2023
→ 2025 those trajectories are unmistakable — and the test requires **no reference
layer at all**, so it works in the post-2020 window where no building data exists.

### 4. U-Net segmentation

Error analysis on the rule-based output showed the bottleneck was segmentation
recall (67%), not change logic. So the task was reframed:

- **Train** a U-Net to segment buildings from a single epoch's elevation
- **Labels** 62,532 LINZ building outlines — human-digitised, authoritative
- **Apply** to 2019 and 2023 separately, then difference the masks

Architecture: U-Net, ImageNet-pretrained ResNet-34 encoder adapted to 3 physical
input bands, 24.4M parameters. Loss: Dice + boundary-weighted BCE.

Two design choices worth stating:

**Loss.** At 13.4% positive pixels, "predict all background" costs only 0.93 on
BCE — nearly free. On Dice it costs 0.998. The Dice term is what prevents collapse
to background. Boundary weighting at 2.5 lifts roof edges from 4% to 10% of total
loss weight, which is where object-level accuracy comes from.

**Augmentation is restricted to flips and 90° rotations.** No brightness, blur or
scale jitter — these bands are measurements in metres and degrees. A 6 m wall is
not a 3 m wall seen differently.

**Splits are spatially blocked** (~2 km blocks), not random. Adjacent patches share
suburb, roof material and flightline; a random split would test memorisation.

### 5. Validation

100 detections were sampled, stratified by footprint size, and reviewed against
four panels each: nDSM 2019, nDSM 2023, aerial 2020–21, aerial 2022–23. The last
two are independent of the detection.

**Rule-based: 54% precision (95% Wilson CI 44–63%, n=99 decided).**

Precision varies sharply with size:

| Footprint | Precision |
|---|---|
| 40–80 m² | 21% |
| 80–120 m² | 50% |
| 120 m²+ | 65% |

Raising the minimum footprint to 120 m² would lift precision to ~65% at the cost
of roughly a third of detections.

---

## Error taxonomy

From the reviewed sample:

| Error | n | Cause |
|---|---|---|
| Existing building | 27 | 2019 height sampled beside the old roof, not on it |
| Vegetation | 13 | trees and scrub passing the stability test |
| Demolition | 3 | strong negative 2025 growth, not caught |
| Spoil heap | 1 | compact stable earth mound — geometrically a building |
| **Specular return** | 1 | a water reservoir invisible to the 2019 LiDAR |
| Shade structure | 1 | fabric canopy over a pool |

The reservoir case is the most interesting. Its smooth surface reflected the 2019
laser away from the sensor, so the DSM has no return there. Every filter behaved
exactly as designed and produced a confident wrong answer. That is a sensor
limitation, not a tuning problem.

---

## What went wrong, and what it cost

Three bugs were found only by looking at images. None was visible in summary
statistics.

**1. Morphological closing merged an entire suburb.** Two iterations of a 3×3
closing kernel bridges gaps of ~4 m — exactly suburban house spacing. Measured on
synthetic data: threshold-only recovered 91% of buildings as separate objects,
closing ×1 gave 87%, **closing ×2 gave 1%**. Segmentation recall collapsed from
91% to near zero while every summary metric still looked plausible.

**2. Difference-then-segment produced fragments.** Labelling connected components
of a change raster makes patch shape depend on where the difference crossed a
threshold — roofs fragmented, neighbours merged. Reversing the order to
segment-then-classify raised detections from 278 to 1,016 on the same data.

**3. A missing height-gain floor inflated CNN detections by 60%.** Testing "clear
in 2019" and "tall in 2023" separately let flat model disagreements pass both.
Adding `h23 − h19 ≥ 2.0` removed 707 artifacts with near-zero height change.

**Two study areas were chosen by intuition and both were wrong.** The first landed
on farmland, the second missed the growth front by 200 m. The third was selected by
gridding actual new-building density — and independently rediscovered the same
location when the finished pipeline was asked where change concentrated.

---

## Honest limitations

- **CNN detection precision is not fully measured.** Only 20 of 100 samples were
  reviewed after the final filter, giving ~40% on a sample too small to report
  with confidence. The rule-based 54% figure is the only validated precision here.
- **Median roof height reads 3.4 m** where 4–6 m is expected, suggesting the DSM
  smooths roof peaks.
- **Segmentation recall is 67%** for the rule-based method, so true new-building
  counts are higher than detected counts.
- **The "clear in 2019" test excludes genuine rebuilds** — 247 candidates fell in
  this ambiguous class.
- **Test blocks had 5.8% building fraction versus 18.6% in training**, so test IoU
  is not directly comparable to validation IoU.
- **Validation was AI-assisted.** Verdicts were assigned by an AI assistant from
  the evidence panels, with reasoning recorded per row. This is not equivalent to
  independent human review and is reported as such.
- **One tile of 23.** 34.6 km² of a validated 112 km² study area.

---

## Reproduce

```bash
pip install requests pillow pyproj rasterio scipy torch segmentation-models-pytorch
```

A free LINZ API key is needed for imagery and building outlines:
https://data.linz.govt.nz/my/api/

```
step1b   confirm change is visible in imagery
step1e   count new buildings from LINZ outlines (deduplicated)
step1f   locate the densest change cluster
step2c   download DEM/DSM for 2019, 2023, 2025
step3e   segment buildings, classify by height history
step4a   export polygons
step4c   render maps
step5a   build validation review pack
step5b   compute precision with Wilson intervals
step6a   build CNN training data from LINZ outlines
step6b   train U-Net
step6c   apply to both epochs and difference
```

Elevation rasters come from the public `nz-elevation` S3 bucket — no credentials
required.

---

## Data

All open, all CC BY 4.0, sourced from Toitū Te Whenua Land Information New Zealand:

- Hamilton LiDAR 1 m DEM/DSM, 2019 / 2023 / 2025
- NZ Building Outlines (All Sources), layer 101292
- Hamilton 0.05 m Urban Aerial Photos, 2020–21 and 2022–23

Coordinate system: NZGD2000 / New Zealand Transverse Mercator 2000 (EPSG:2193).

---

## Author

**Mazhar Shakoor** — GIS Analyst & Remote Sensing Specialist
[LinkedIn](https://linkedin.com/in/mazhar-shakoor-55009524b) ·
[GitHub](https://github.com/mazharshakoor55-netizen)
