#!/usr/bin/env python3
"""Bake CDIP MOP gridded nowcast+forecast into compact payloads for the browser.

CDIP's THREDDS server sends no Access-Control-Allow-Origin header, so the page
cannot read OPeNDAP directly. This script runs server-side (locally, or in a
scheduled GitHub Action) and commits a quantized binary the page can fetch.

Source: https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_grids/
  CA_0.01_{nowcast,forecast}.nc   statewide, 1000x800 @ 0.01 deg (~1 km)
  D_0.001_{nowcast,forecast}.nc   San Diego, 900x500 @ 0.001 deg (~100 m)

Each carries waveHs, waveTp, waveDp on a lat/lon grid. Nowcast is the trailing
~6 hours at hourly steps; the forecast steps every 6 hours to roughly four
days ahead. The merged axis is therefore non-uniform, and is shipped that way
-- the page interpolates its presentation to hourly rather than baking ~5x
more frames into every payload.

Output per domain, in ../data/:
  <id>.json      grid geometry, time axis, quantization ranges
  <id>.bin.gz    mask + land + depth + quantized frames (layout below)

Binary layout (after gunzip), all little-endian:
  uint8  mask[ny*nx]        1 = wet model cell, 0 = not
  uint8  shore[nwet]        1 = mask boundary in shallow water, i.e. coastline
  uint8  edge[nwet]         0..255 distance ramp from the model-domain edge
  uint8  depth[nwet]        quantized log10 depth, see meta.vars.depth
  then for each time t, in order:
    uint8 hs[nwet], uint8 tp[nwet], uint8 dp[nwet]

Wet cells are stored in row-major order of the mask; the page rebuilds the
scatter index once at load.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import random
import sys
import time

import netCDF4 as nc
import numpy as np

THREDDS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_grids"

# Quantization ranges. Hs and Tp are clipped, so these need headroom over any
# plausible California sea state -- 8 m Hs is well above the largest MOP
# nearshore hindcast values, and 25.5 s covers the longest forerunners.
HS_MAX = 8.0     # m,  -> 3.1 cm per level
TP_MAX = 25.5    # s,  -> 0.1 s per level
DP_MAX = 360.0   # deg, -> 1.41 deg per level
DEPTH_LOG_MIN, DEPTH_LOG_MAX = 0.0, 3.7  # log10 m, 1 m to ~5000 m

# Seconds to wait after CDIP's abuse filter denies a request, one entry per
# retry. Deliberately much longer than the plain-error retry -- the blocks seen
# so far clear on a scale of minutes to hours, not seconds -- and deliberately
# short: waiting out a block that lasts hours is not this job's business, and
# the four-minute budget here has to be paid per domain.
DENIAL_BACKOFF = (60.0, 180.0)

# Consecutive denied domains before the run stops asking. A block is a property
# of the source address, not of the dataset, so once two domains in a row are
# refused the remaining fourteen will be too; trying them anyway costs the job
# an hour it does not have and adds fourteen domains' worth of requests to
# whatever made CDIP object in the first place.
DENIAL_GIVE_UP_AFTER = 2

# The statewide grid plus every county grid CDIP publishes. Together the county
# grids tile the coast continuously from the Mexican border to Oregon, so the
# page can zoom to 100-200 m anywhere rather than only off San Diego. They are
# fetched one at a time, on zoom, so the total on disk is not what any visitor
# downloads.
#
# The 0.002-degree northern grids are ~200 m and the 0.001-degree southern ones
# ~100 m; decim is 1 throughout because these are already the fine products.
DOMAINS = {
    "ca": dict(stem="CA_0.01", label="California", decim=2),
    "dn": dict(stem="DN_0.002", label="Del Norte", decim=1),
    "hu": dict(stem="HU_0.002", label="Humboldt", decim=1),
    "mn": dict(stem="M_0.002", label="Mendocino", decim=1),
    "sn": dict(stem="SN_0.002", label="Sonoma", decim=1),
    "ma": dict(stem="MA_0.002", label="Marin", decim=1),
    "sf": dict(stem="SF_0.002", label="San Francisco", decim=1),
    "sm": dict(stem="SM_0.002", label="San Mateo", decim=1),
    "sc": dict(stem="SC_0.002", label="Santa Cruz", decim=1),
    "mo": dict(stem="MO_0.002", label="Monterey", decim=1),
    "sl": dict(stem="SL_0.002", label="San Luis Obispo", decim=1),
    "sb": dict(stem="B_0.001", label="Santa Barbara", decim=1),
    "ve": dict(stem="VE_0.001", label="Ventura", decim=1),
    "la": dict(stem="L_0.001", label="Los Angeles", decim=1),
    "oc": dict(stem="OC_0.001", label="Orange", decim=1),
    "sd": dict(stem="D_0.001", label="San Diego", decim=1),
}


def log(msg: str) -> None:
    print(f"[build] {msg}", file=sys.stderr, flush=True)


def is_denial(exc: BaseException) -> bool:
    """True if THREDDS refused us rather than failed.

    CDIP fronts THREDDS with an abuse filter that answers a blocked source with
    a JSON body ({"error": "Access Denied", ...}) where the DAP client expects a
    DDS. netCDF surfaces that as an authorization failure, errno -78, which is
    worth separating from a genuine server error: a 500 clears in seconds, a
    block does not, and retrying hard against one only deepens it.
    """
    return "Authorization failure" in str(exc) or "[Errno -78]" in str(exc)


def open_with_retry(url: str, tries: int = 4, delay: float = 5.0) -> nc.Dataset:
    """THREDDS intermittently 500s under load; a couple of retries is enough.

    A denial gets a different schedule. Five seconds is the right wait for a
    server hiccup and the wrong one for an abuse filter, so back off to minutes
    and give up sooner: the caller keeps the previous payload for this domain
    rather than the run failing outright.
    """
    denials = 0
    for attempt in range(1, tries + 1):
        try:
            return nc.Dataset(url)
        except OSError as exc:
            if is_denial(exc):
                denials += 1
                if denials > len(DENIAL_BACKOFF):
                    raise
                wait = DENIAL_BACKOFF[denials - 1]
                wait *= 1.0 + 0.25 * random.random()   # jitter, against a synchronized retry storm
                log(f"  THREDDS denied the request, retry {denials}/{len(DENIAL_BACKOFF)} in {wait:.0f}s")
            else:
                if attempt == tries:
                    raise
                wait = delay
                log(f"  open failed ({exc.__class__.__name__}), retry {attempt}/{tries - 1} in {wait:.0f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def fetch(stem: str, decim: int):
    """Merge nowcast + forecast onto one strictly increasing time axis."""
    lat = lon = depth = None
    frames: dict[int, dict[str, np.ndarray]] = {}

    for kind in ("nowcast", "forecast"):
        url = f"{THREDDS}/{stem}_{kind}.nc"
        log(f"  reading {stem}_{kind}.nc")
        d = open_with_retry(url)
        try:
            if lat is None:
                lat = np.asarray(d["metaLatitude"][:])[::decim]
                lon = np.asarray(d["metaLongitude"][:])[::decim]
                depth = d["metaWaterDepth"][::decim, ::decim]

            times = np.asarray(d["waveTime"][:]).astype(np.int64)
            hs = d["waveHs"][:, ::decim, ::decim]
            tp = d["waveTp"][:, ::decim, ::decim]
            dp = d["waveDp"][:, ::decim, ::decim]

            for k, t in enumerate(times):
                # Nowcast wins on overlap: it is the analysis, not a projection.
                if int(t) in frames and kind == "forecast":
                    continue
                frames[int(t)] = {"hs": hs[k], "tp": tp[k], "dp": dp[k]}
        finally:
            d.close()

    order = sorted(frames)
    log(f"  {len(order)} unique frames")
    return lat, lon, depth, order, frames


SHORE_DEPTH = 20.0   # m; boundary shallower than this is coastline
EDGE_FADE_CELLS = 18  # ramp width for the domain-edge alpha fade


def classify_boundary(wet: np.ndarray, depth: np.ndarray):
    """Separate the coastline from the model-domain edge, by depth.

    A flood fill cannot do this: the MOP domains are staircases of nested
    boxes, and at every step corner the offshore padding abuts the continent
    directly, so a fill seeded from either edge leaks into the other region.

    Depth separates them cleanly and physically. The mask boundary is the
    coast wherever the water is shallow, and the edge of the model domain
    wherever it is deep -- MOP is a nearshore model, so it simply stops at a
    few hundred meters. That gives the page a real coastline to draw, and a
    distance ramp to dissolve the fur into the background at the domain edge
    instead of cutting it off along a visible staircase.

    Returns (shore, edge_dist) over wet cells only.
    """
    dry = ~wet
    nbr_dry = np.zeros_like(dry)
    nbr_dry[:-1, :] |= dry[1:, :]
    nbr_dry[1:, :] |= dry[:-1, :]
    nbr_dry[:, :-1] |= dry[:, 1:]
    nbr_dry[:, 1:] |= dry[:, :-1]
    # Grid border counts as a boundary too, or the fur runs off the edge hard.
    nbr_dry[0, :] = nbr_dry[-1, :] = True
    nbr_dry[:, 0] = nbr_dry[:, -1] = True

    boundary = wet & nbr_dry
    shore = boundary & (depth < SHORE_DEPTH)
    edge = boundary & ~shore

    # Euclidean distance from the domain edge, in cells, capped at the ramp
    # width. Cheap and visually indistinguishable from a geodesic distance at
    # this scale.
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(~edge)
    dist = np.clip(dist, 0, EDGE_FADE_CELLS) / EDGE_FADE_CELLS

    return shore, dist


def quantize(a: np.ndarray, vmax: float, vmin: float = 0.0) -> np.ndarray:
    x = (np.asarray(a, dtype=np.float64) - vmin) / (vmax - vmin)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def build(domain_id: str, out_dir: pathlib.Path) -> dict:
    cfg = DOMAINS[domain_id]
    log(f"{domain_id}: {cfg['stem']} (decim {cfg['decim']})")
    lat, lon, depth, order, frames = fetch(cfg["stem"], cfg["decim"])

    ny, nx = len(lat), len(lon)
    # A cell is usable only where every frame has data; the model footprint is
    # static in practice, but an all-frames AND makes that assumption explicit
    # rather than trusting frame 0.
    wet = np.ones((ny, nx), dtype=bool)
    for t in order:
        wet &= ~np.ma.getmaskarray(frames[t]["hs"])
        wet &= ~np.ma.getmaskarray(frames[t]["dp"])
    nwet = int(wet.sum())
    if nwet == 0:
        raise RuntimeError(f"{domain_id}: no wet cells survived the all-frames mask")

    dep = np.clip(np.ma.filled(depth, 1e9), 1.0, None)
    shore, edge_dist = classify_boundary(wet, dep)
    log(f"  grid {ny}x{nx}, wet {nwet} ({100 * nwet / wet.size:.1f}%), "
        f"shore cells {int(shore.sum())}")

    chunks: list[bytes] = [wet.astype(np.uint8).tobytes()]
    chunks.append(shore[wet].astype(np.uint8).tobytes())
    chunks.append(np.clip(np.round(edge_dist[wet] * 255), 0, 255).astype(np.uint8).tobytes())
    chunks.append(quantize(np.log10(np.clip(dep, 1.0, 1e4))[wet],
                           DEPTH_LOG_MAX, DEPTH_LOG_MIN).tobytes())

    for t in order:
        f = frames[t]
        chunks.append(quantize(np.ma.filled(f["hs"], 0.0)[wet], HS_MAX).tobytes())
        chunks.append(quantize(np.ma.filled(f["tp"], 0.0)[wet], TP_MAX).tobytes())
        chunks.append(quantize(np.ma.filled(f["dp"], 0.0)[wet] % 360.0, DP_MAX).tobytes())

    raw = b"".join(chunks)
    packed = gzip.compress(raw, compresslevel=9)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{domain_id}.bin.gz").write_bytes(packed)

    # Percentiles over the whole window give the page a sensible default color
    # range without it having to scan the payload -- and without a fixed range
    # that washes out a small summer swell or clips a winter one.
    all_hs = np.concatenate([np.ma.filled(frames[t]["hs"], np.nan)[wet] for t in order])
    hs_lo, hs_hi = np.nanpercentile(all_hs, [2, 98])

    meta = {
        "id": domain_id,
        "label": cfg["label"],
        "source": f"{cfg['stem']} nowcast+forecast",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid": {
            "nx": nx, "ny": ny,
            # Cell centers; lat/lon are ascending in the CDIP files.
            "lon0": float(lon[0]), "dlon": float(lon[1] - lon[0]),
            "lat0": float(lat[0]), "dlat": float(lat[1] - lat[0]),
        },
        "nwet": nwet,
        "times": [int(t) for t in order],
        "vars": {
            "hs": {"max": HS_MAX, "unit": "m", "p2": float(hs_lo), "p98": float(hs_hi)},
            "tp": {"max": TP_MAX, "unit": "s"},
            "dp": {"max": DP_MAX, "unit": "deg"},
            "depth": {"logMin": DEPTH_LOG_MIN, "logMax": DEPTH_LOG_MAX, "unit": "m"},
        },
        "shoreDepth": SHORE_DEPTH,
        "edgeFadeCells": EDGE_FADE_CELLS,
        "bytes": {"raw": len(raw), "gz": len(packed)},
    }
    (out_dir / f"{domain_id}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"  wrote {len(packed) / 1e6:.2f} MB gz (from {len(raw) / 1e6:.2f} MB raw)")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=list(DOMAINS))
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--strict", action="store_true",
                    help="fail the run if any domain cannot be rebuilt, instead "
                         "of keeping that domain's previous payload")
    args = ap.parse_args()

    # A domain that cannot be rebuilt keeps whatever payload is already on
    # disk, and the run carries on. The alternative -- aborting -- takes the
    # whole site down over one upstream refusal, and there is nothing dishonest
    # about serving the older file: the page dates every domain from its own
    # time axis, so a stale payload shows up as "stale - N h" on its own.
    built, stale = [], []
    denied_in_a_row = 0
    for d in args.domains:
        exc = None
        if denied_in_a_row >= DENIAL_GIVE_UP_AFTER:
            log(f"{d}: skipped, CDIP is refusing this runner")
        else:
            try:
                built.append(build(d, args.out))
                denied_in_a_row = 0
                continue
            except Exception as err:
                exc = err
                denied_in_a_row = denied_in_a_row + 1 if is_denial(err) else 0

        prev = args.out / f"{d}.json"
        if args.strict or not prev.exists():
            if exc is not None:
                raise exc
            raise RuntimeError(f"{d}: no previous payload to fall back on")
        if exc is not None:
            log(f"{d}: rebuild failed ({exc.__class__.__name__}: {exc})")
        log(f"{d}: keeping the payload already on disk")
        built.append(json.loads(prev.read_text()))
        stale.append(d)
        if exc is not None and denied_in_a_row == DENIAL_GIVE_UP_AFTER:
            log(f"{DENIAL_GIVE_UP_AFTER} domains refused in a row; "
                f"keeping the rest at their previous build without asking again")

    if stale and len(stale) == len(args.domains):
        log(f"every domain failed to rebuild: {' '.join(stale)}")
        return 1

    def summary(m):
        g = m["grid"]
        # The page needs each domain's footprint and resolution to decide what
        # to load *before* loading it, so carry them in the index.
        return {
            "id": m["id"], "label": m["label"], "source": m["source"],
            "bytes": m["bytes"], "nt": len(m["times"]),
            "dlon": g["dlon"],
            "bbox": {
                "lonMin": g["lon0"], "lonMax": g["lon0"] + g["nx"] * g["dlon"],
                "latMin": g["lat0"], "latMax": g["lat0"] + g["ny"] * g["dlat"],
            },
        }

    index = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "domains": [summary(m) for m in built],
    }
    if stale:
        # Name them in the index so the staleness is legible from the payload
        # itself, not only from a workflow log that expires.
        index["stale"] = stale
    (args.out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    if stale:
        log(f"done, with {len(stale)}/{len(args.domains)} domains kept from the "
            f"previous build: {' '.join(stale)}")
    else:
        log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
