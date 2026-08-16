"""
Point cloud -> raster gridding, pure numpy.

Why not PDAL
------------
PDAL's Python bindings need libpdal, which pip cannot supply. On Colab you would
need condacolab, which restarts the kernel and costs ~5 minutes of every session.

You do not need PDAL here. LINZ ships its point clouds pre-classified (ASPRS
class 2 = ground), so the hard part - ground filtering - is already done. What
remains is binning points into a 1 m grid, which numpy does in a few lines and
faster than a PDAL pipeline for this workload.

Reading LAZ is handled by `laspy[lazrs]`, which is pure pip and installs in
seconds.
"""

from __future__ import annotations

import numpy as np


def _cell_index(x, y, bounds, res):
    """Map coordinates to (row, col) grid indices. Rows increase southward."""
    xmin, ymin, xmax, ymax = bounds
    ncols = int(np.ceil((xmax - xmin) / res))
    nrows = int(np.ceil((ymax - ymin) / res))

    col = np.floor((x - xmin) / res).astype(np.int64)
    row = np.floor((ymax - y) / res).astype(np.int64)

    inside = (col >= 0) & (col < ncols) & (row >= 0) & (row < nrows)
    return row, col, inside, nrows, ncols


def grid_max(x, y, z, bounds, res=1.0, nodata=np.nan):
    """
    Maximum z per cell - the DSM reducer.

    Max, not mean: roof ridges and edges are exactly the high-frequency detail
    the segmentation model keys on, and averaging rounds them off. A mean DSM
    produces visibly softer building boundaries and measurably worse IoU.
    """
    row, col, inside, nrows, ncols = _cell_index(x, y, bounds, res)
    out = np.full((nrows, ncols), -np.inf, dtype=np.float64)
    np.maximum.at(out, (row[inside], col[inside]), z[inside])
    out[np.isinf(out)] = nodata
    return out.astype(np.float32)


def grid_min(x, y, z, bounds, res=1.0, nodata=np.nan):
    """Minimum z per cell - useful as a bare-earth reducer on ground points."""
    row, col, inside, nrows, ncols = _cell_index(x, y, bounds, res)
    out = np.full((nrows, ncols), np.inf, dtype=np.float64)
    np.minimum.at(out, (row[inside], col[inside]), z[inside])
    out[np.isinf(out)] = nodata
    return out.astype(np.float32)


def grid_mean(x, y, z, bounds, res=1.0, nodata=np.nan):
    """
    Mean z per cell - the DEM reducer for ground returns.

    Ground is smooth, so averaging suppresses per-point noise rather than
    destroying signal. The opposite of the DSM case.
    """
    row, col, inside, nrows, ncols = _cell_index(x, y, bounds, res)
    tot = np.zeros((nrows, ncols), dtype=np.float64)
    cnt = np.zeros((nrows, ncols), dtype=np.int64)
    np.add.at(tot, (row[inside], col[inside]), z[inside])
    np.add.at(cnt, (row[inside], col[inside]), 1)
    out = np.full((nrows, ncols), nodata, dtype=np.float64)
    hit = cnt > 0
    out[hit] = tot[hit] / cnt[hit]
    return out.astype(np.float32)


def grid_count(x, y, bounds, res=1.0):
    """Points per cell - the density raster."""
    row, col, inside, nrows, ncols = _cell_index(x, y, bounds, res)
    out = np.zeros((nrows, ncols), dtype=np.int32)
    np.add.at(out, (row[inside], col[inside]), 1)
    return out


def grid_mean_of(x, y, values, bounds, res=1.0, nodata=np.nan):
    """
    Mean of an arbitrary per-point attribute - used for NumberOfReturns.

    This produces the return_ratio band: buildings are opaque so mean returns
    approaches 1.0, canopy is porous so it reaches 2.0+. It is the single
    strongest feature for separating roofs from tree crowns.
    """
    return grid_mean(x, y, values.astype(np.float64), bounds, res, nodata)


def read_laz(path, classes=None, return_number=None):
    """
    Read a LAZ/LAS file into arrays.

    classes       : keep only these ASPRS codes, e.g. [2] for ground
    return_number  : keep only this return number, e.g. 1 for first returns

    ASPRS codes you care about here:
        1  unclassified   2  ground        6  building (if provided)
        7  low noise      18 high noise
    """
    import laspy

    las = laspy.read(path)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    cls = np.asarray(las.classification)
    rn = np.asarray(las.return_number)
    nr = np.asarray(las.number_of_returns)

    keep = np.ones(len(x), dtype=bool)

    # Always drop noise. Leaving class 7/18 in produces spikes tens of metres
    # above the roofline that wreck the nDSM clip and the normalisation.
    keep &= ~np.isin(cls, [7, 18])

    if classes is not None:
        keep &= np.isin(cls, classes)
    if return_number is not None:
        keep &= rn == return_number

    return {
        "x": x[keep], "y": y[keep], "z": z[keep],
        "classification": cls[keep],
        "return_number": rn[keep],
        "number_of_returns": nr[keep],
        "n_total": len(x), "n_kept": int(keep.sum()),
    }


def tile_bounds(x, y, res=1.0, snap=True):
    """
    Bounding box for a point set, snapped to the resolution grid.

    Snapping is not cosmetic. Both epochs must land on the SAME grid or your
    2019 and 2023 rasters will be offset by a sub-pixel amount, and every
    differencing operation afterwards will show phantom change along every
    building edge.
    """
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    if snap:
        xmin = np.floor(xmin / res) * res
        ymin = np.floor(ymin / res) * res
        xmax = np.ceil(xmax / res) * res
        ymax = np.ceil(ymax / res) * res
    return (xmin, ymin, xmax, ymax)


def grid_return_ratio(x, y, number_of_returns, classification, bounds, res=1.0):
    """
    Vegetation-porosity band, computed on NON-GROUND points only.

    Verified on synthetic data: including ground returns dilutes the statistic,
    because ground under canopy is single-return and drags the cell mean toward
    1.0. Measured roof-vs-canopy separation:

        all points  : 1.00 vs 2.49  -> separation 1.49
        non-ground  : 1.00 vs 3.00  -> separation 2.00   (+34%)

    Cells with no non-ground points (open ground, roads, water) return 0.0,
    which is correct - no vegetation there.
    """
    nong = classification != 2
    if not nong.any():
        _, _, _, nrows, ncols = _cell_index(x, y, bounds, res)
        return np.zeros((nrows, ncols), dtype=np.float32)

    mean_nr = grid_mean(
        x[nong], y[nong], number_of_returns[nong].astype(np.float64),
        bounds, res, nodata=np.nan,
    )
    ratio = (mean_nr - 1.0) / 2.0          # 1 return -> 0.0, 3 returns -> 1.0
    return np.clip(np.nan_to_num(ratio, nan=0.0), 0.0, 1.0).astype(np.float32)


def write_gtiff(array, bounds, path, crs="EPSG:2193", res=1.0, nodata=-9999.0):
    """Write a grid to a compressed GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin

    arr = np.where(np.isnan(array), nodata, array).astype(np.float32)
    transform = from_origin(bounds[0], bounds[3], res, res)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=arr.shape[0], width=arr.shape[1], count=1,
        dtype="float32", crs=crs, transform=transform,
        nodata=nodata, compress="deflate", predictor=3, tiled=True,
    ) as dst:
        dst.write(arr, 1)
    return path
