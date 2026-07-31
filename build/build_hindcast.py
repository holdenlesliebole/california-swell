#!/usr/bin/env python3
"""Bake the CDIP MOP alongshore hindcast into payloads for the coastal ribbon.

Two tiers, both from the same 11,594 alongshore MOP sites covering all 15
California coastal counties:

  overview   daily samples across the whole 2000 -> present record, via strided
             OPeNDAP. One file per year, loaded on demand.
  event      hourly, denser alongshore sampling, over a ~7-day window around a
             named swell. Events are *derived from the overview*, not asserted
             from memory -- see `events` subcommand.

Direction comes from the band-split wave energy flux, not from Dm. Following
estimateEnergyFlux() in read_MOPline2.m -- the author's MATLAB reader for
these files, not included here -- the flux is

    Fx = sum_band rho g Cg a1 E df        Fy = sum_band rho g Cg b1 E df

evaluated separately over that routine's canonical swell and sea bands. A single
mean direction is smooth but wrong for most of the record: measured at D0586,
sea and swell each carry 30-70% of the flux for 66% of hours while their flux
directions differ by 22 degrees on average. Reproducing the *total*-band flux
direction matches the file's own waveDm to a 1.80 deg mean offset with 1.38 deg
residual, which is the check that the quantity is right; the small remaining
difference is the Cg weighting, which waveDm does not carry.

Time routing reproduces read_MOPline2.m: the hindcast runs to 2025-03-31 23:00
and the alongshore *nowcast* picks up at 2025-04-01 and runs to the present, so
the two together are continuous and gapless. Note this is unlike the gridded
nowcast, which is only a 6-hour buffer.

Usage:
    python build_hindcast.py sites
    python build_hindcast.py overview --years 2000-2026 --stride 8
    python build_hindcast.py events --top 6
    python build_hindcast.py event --slug 2023-01-05 --stride 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import math
import pathlib
import re
import sys
import threading
import time
import urllib.request

import netCDF4 as nc
import numpy as np

DODS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model"
CAT = "https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model"

G = 9.81
RHO = 1028.0

# Canonical bands from read_MOPline2.m. The 0.0813-0.0900 Hz gap between them is
# deliberate in that routine and is preserved here so this stays consistent with
# the analysis it is meant to accompany.
SWELL_BAND = (0.02, 0.0813)
SEA_BAND = (0.0900, 0.400)

# read_MOPline2.m's tSwitch: last hindcast step is 2025-03-31 23:00 UTC.
TSWITCH = dt.datetime(2025, 4, 1, tzinfo=dt.timezone.utc)

# Fallback Hs scale only, for reading payloads written before the scale was
# derived. pack() now measures each payload's own maximum -- see the note there
# on why two hand-picked ceilings were both wrong.
HS_MAX = 12.0
# The work is entirely THREDDS round-trips -- a one-year build ran at 14% CPU --
# so concurrency is set well above the core count on purpose.
WORKERS = 32

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[hindcast] {msg}", file=sys.stderr, flush=True)


def here() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ sites


def fetch_site_table() -> dict:
    """Site name + position for all 11,594 alongshore MOP sites, in one request.

    The R_CA_coefficients catalog encodes each site's coordinates in its
    filename (B0001_34.36880-119.47874_ref.nc), and the site set is identical to
    the alongshore stations. Parsing the catalog therefore replaces 11,594
    metadata reads with a single HTTP GET; spot-checked against
    metaLatitude/metaLongitude, the coordinates agree to 0.1 m.
    """
    log("fetching site catalog")
    x = urllib.request.urlopen(f"{CAT}/R_CA_coefficients/catalog.xml", timeout=180).read().decode()
    rows = re.findall(
        r'urlPath="cdip/model/R_CA_coefficients/([A-Z]+)([0-9]+)_(-?[0-9.]+)(-[0-9.]+)_ref\.nc"', x)
    if not rows:
        raise RuntimeError("could not parse the R coefficient catalog")

    region = np.array([r[0] for r in rows])
    number = np.array([int(r[1]) for r in rows])
    lat = np.array([float(r[2]) for r in rows])
    lon = np.array([float(r[3]) for r in rows])
    log(f"  {len(rows)} sites, {len(set(region))} regions")

    order = order_alongshore(region, number, lat)
    return {
        # Site IDs are a fixed five characters, so the zero-padding depends on
        # the length of the region prefix: single-letter regions take four
        # digits (D0586) and two-letter regions three (OC001). The R catalog
        # writes them all four-wide, so reformatting is required here -- getting
        # this wrong silently loses every two-letter region, which is 6,358 of
        # the 11,594 sites.
        "name": [f"{region[i]}{number[i]:0{5 - len(region[i])}d}" for i in order],
        "lat": lat[order],
        "lon": lon[order],
        "region": region[order],
    }


def order_alongshore(region: np.ndarray, number: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Order sites into one continuous north-to-south chain.

    The ribbon is drawn as a connected coastline, so the sites have to be
    sequenced along it. Regions are ordered by latitude; within a region, MOP
    numbering runs along the coast but the sign of that relationship is a
    per-region convention, so it is measured rather than assumed.
    """
    out = []
    regions = sorted(set(region), key=lambda r: -lat[region == r].mean())
    for r in regions:
        idx = np.flatnonzero(region == r)
        idx = idx[np.argsort(number[idx])]
        # Positive correlation between site number and latitude means numbering
        # runs south-to-north, so reverse it to keep the chain heading south.
        if len(idx) > 2 and np.corrcoef(number[idx], lat[idx])[0, 1] > 0:
            idx = idx[::-1]
        out.append(idx)
    return np.concatenate(out)


def write_sites(sites: dict, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    lat = sites["lat"].astype(np.float32)
    lon = sites["lon"].astype(np.float32)
    (out / "sites.bin.gz").write_bytes(
        gzip.compress(lat.tobytes() + lon.tobytes(), 9))

    step = np.hypot(np.diff(sites["lat"]) * 111.0,
                    np.diff(sites["lon"]) * 111.0 * np.cos(np.radians(sites["lat"][:-1])))
    (out / "sites.json").write_text(json.dumps({
        "n": len(lat),
        "regions": sorted(set(sites["region"].tolist())),
        "bbox": {"latMin": float(lat.min()), "latMax": float(lat.max()),
                 "lonMin": float(lon.min()), "lonMax": float(lon.max())},
        "medianSpacingKm": float(np.median(step)),
        "names": sites["name"],
    }, indent=1) + "\n")
    log(f"  wrote sites.bin.gz, median alongshore spacing {np.median(step) * 1000:.0f} m")


def load_sites(out: pathlib.Path) -> dict:
    meta = json.loads((out / "sites.json").read_text())
    raw = gzip.decompress((out / "sites.bin.gz").read_bytes())
    n = meta["n"]
    lat = np.frombuffer(raw, np.float32, n, 0)
    lon = np.frombuffer(raw, np.float32, n, 4 * n)
    return {"meta": meta, "lat": lat, "lon": lon, "name": meta["names"]}


# ------------------------------------------------------------------ physics


def wavenumber(f: np.ndarray, h: float) -> np.ndarray:
    """Guo (2002) explicit fit to the linear dispersion relation, ~0.75% error."""
    om = 2 * np.pi * f
    x = om * np.sqrt(h / G)
    beta = 2.4908
    kh = x ** 2 * (1 - np.exp(-(x ** beta))) ** (-1 / beta)
    return kh / h


def group_speed(f: np.ndarray, h: float) -> np.ndarray:
    k = wavenumber(f, h)
    om = 2 * np.pi * f
    kh = np.clip(k * h, 1e-6, 20.0)
    n = 0.5 * (1 + 2 * kh / np.sinh(2 * kh))
    return om / k * n


def band_flux(E, a1, b1, f, df, cg, band):
    """Hs and energy-flux direction over one frequency band.

    Returns (hs, dir_deg). Direction is the compass bearing the energy is
    coming FROM, matching the convention of waveDp and waveDm.
    """
    i = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    w = (cg[i] * df[i])[None, :]
    fx = np.sum(w * a1[:, i] * E[:, i], axis=1)
    fy = np.sum(w * b1[:, i] * E[:, i], axis=1)
    m0 = np.sum(E[:, i] * df[i][None, :], axis=1)
    hs = 4.0 * np.sqrt(np.clip(m0, 0, None))
    ang = np.degrees(np.arctan2(fy, fx)) % 360.0
    return hs, ang


# ------------------------------------------------------------------ fetch


def read_site(name: str, t0: dt.datetime, t1: dt.datetime, stride: int):
    """Read one site over [t0, t1], spanning the hindcast/nowcast handoff.

    Mirrors the routing in read_MOPline2.m rather than reinventing it: the
    hindcast covers everything before 2025-04-01 and the alongshore nowcast
    everything after, with no gap between them.
    """
    pieces = []
    plan = []
    if t0 < TSWITCH:
        plan.append(("hindcast", t0, min(t1, TSWITCH)))
    if t1 > TSWITCH:
        plan.append(("nowcast", max(t0, TSWITCH), t1))

    depth = None
    freq = df = None
    for kind, a, b in plan:
        d = nc.Dataset(f"{DODS}/MOP_alongshore/{name}_{kind}.nc")
        try:
            tv = np.asarray(d["waveTime"][:]).astype(np.int64)
            sel = np.flatnonzero((tv >= a.timestamp()) & (tv <= b.timestamp()))
            if not len(sel):
                continue
            lo, hi = int(sel[0]), int(sel[-1]) + 1
            sl = slice(lo, hi, stride)
            if freq is None:
                freq = np.asarray(d["waveFrequency"][:], dtype=float)
                fb = np.asarray(d["waveFrequencyBounds"][:], dtype=float)
                # Bounds are (nfreq, 2) here; bandwidths are not constant.
                df = np.abs(fb[:, 1] - fb[:, 0])
                depth = float(np.ravel(d["metaWaterDepth"][:])[0])
            pieces.append((
                tv[sl],
                np.ma.filled(d["waveEnergyDensity"][sl, :], 0.0).astype(np.float32),
                np.ma.filled(d["waveA1Value"][sl, :], 0.0).astype(np.float32),
                np.ma.filled(d["waveB1Value"][sl, :], 0.0).astype(np.float32),
            ))
        finally:
            d.close()

    if not pieces:
        return None
    t = np.concatenate([p[0] for p in pieces])
    E = np.concatenate([p[1] for p in pieces])
    a1 = np.concatenate([p[2] for p in pieces])
    b1 = np.concatenate([p[3] for p in pieces])
    return t, E, a1, b1, freq, df, depth


def site_bands(name, t0, t1, stride):
    """Per-site band-split Hs and flux direction, or None if the site is empty."""
    got = read_site(name, t0, t1, stride)
    if got is None:
        return None
    t, E, a1, b1, f, df, depth = got
    cg = group_speed(f, depth)
    hs_sw, dir_sw = band_flux(E, a1, b1, f, df, cg, SWELL_BAND)
    hs_se, dir_se = band_flux(E, a1, b1, f, df, cg, SEA_BAND)
    return t, hs_sw, dir_sw, hs_se, dir_se


def _worker(job):
    """Module-level so it can be pickled to a worker process (see gather)."""
    i, name, t0, t1, stride = job
    for attempt in range(3):
        try:
            return i, site_bands(name, t0, t1, stride)
        except Exception as exc:
            if attempt == 2:
                log(f"  {name}: giving up ({exc.__class__.__name__}: {exc})")
                return i, None
            time.sleep(2 + 3 * attempt)


def gather(names, t0, t1, stride, label):
    """Fetch every site concurrently and align them onto a common time axis.

    Processes, not threads. netCDF4's OPeNDAP backend is not thread-safe --
    concurrent Dataset opens trip over libcurl's global state and fail with
    "a libcurl function was given a bad argument". Separate processes each get
    their own libcurl, and the work is I/O-bound enough that the extra overhead
    does not matter.
    """
    out = [None] * len(names)
    done = [0]
    t_start = time.time()
    jobs = [(i, names[i], t0, t1, stride) for i in range(len(names))]

    with cf.ProcessPoolExecutor(WORKERS) as ex:
        for i, res in ex.map(_worker, jobs, chunksize=4):
            out[i] = res
            done[0] += 1
            if done[0] % 200 == 0:
                el = time.time() - t_start
                log(f"  {label}: {done[0]}/{len(names)} sites, {el:.0f}s elapsed, "
                    f"~{el / done[0] * (len(names) - done[0]):.0f}s left")

    ok = [o for o in out if o is not None]
    if not ok:
        raise RuntimeError(f"{label}: every site failed")
    # Fail loudly on wholesale loss. A silent partial result looks like a valid
    # build and renders as a coastline with holes in it, which is much harder to
    # notice later than an exception here.
    lost = 1 - len(ok) / len(names)
    if lost > 0.05:
        raise RuntimeError(
            f"{label}: {lost:.0%} of sites returned nothing ({len(ok)}/{len(names)}). "
            f"Check site-ID padding and THREDDS availability before trusting this.")
    # Sites occasionally differ by a step at the record ends; intersect so every
    # column of the payload refers to the same instant at every site.
    common = ok[0][0]
    for o in ok[1:]:
        common = np.intersect1d(common, o[0], assume_unique=False)
    log(f"  {label}: {len(ok)}/{len(names)} sites, {len(common)} common steps")
    return out, common


def pack(out, common, names):
    """Quantize to uint8 and interleave: hs_sw, dir_sw, hs_se, dir_se per step.

    The Hs scale is measured from the payload, not fixed. Two hand-picked
    ceilings were wrong in a row -- 8 m clipped the January 2023 event, and 12 m
    still clipped 2015-12-11, where Humboldt sites reach 13.99 m -- because a
    few sampled sites say nothing about the extremes across 11,594 of them over
    26 years. Deriving the scale removes the failure mode rather than moving the
    constant, and every payload carries its own scale as meta.hsMax so the page
    never assumes one.
    """
    nt, ns = len(common), len(names)
    arrs = {k: np.zeros((nt, ns), np.uint8) for k in ("hsw", "dsw", "hse", "dse")}
    valid = np.zeros(ns, np.uint8)

    peak = 0.0
    for o in out:
        if o is None:
            continue
        for a in (o[1], o[3]):          # hs_swell, hs_sea
            finite = a[np.isfinite(a)]
            if finite.size:
                peak = max(peak, float(finite.max()))
    # Headroom so the largest sample does not land exactly on 255, and a floor
    # so a calm window does not get an absurdly tight scale.
    hs_max = max(4.0, math.ceil(peak * 1.02 * 2) / 2)
    log(f"  Hs scale: peak {peak:.2f} m -> hsMax {hs_max:.1f} m")

    for j, o in enumerate(out):
        if o is None:
            continue
        t, hsw, dsw, hse, dse = o
        idx = np.searchsorted(t, common)
        idx = np.clip(idx, 0, len(t) - 1)
        if not np.all(t[idx] == common):
            continue          # misaligned axis: drop rather than mis-timestamp
        q = lambda a, m: np.clip(np.round(a / m * 255), 0, 255).astype(np.uint8)
        arrs["hsw"][:, j] = q(np.nan_to_num(hsw[idx]), hs_max)
        arrs["dsw"][:, j] = q(np.nan_to_num(dsw[idx]) % 360, 360.0)
        arrs["hse"][:, j] = q(np.nan_to_num(hse[idx]), hs_max)
        arrs["dse"][:, j] = q(np.nan_to_num(dse[idx]) % 360, 360.0)
        valid[j] = 1

    # Clipping is invisible once quantized -- a saturated Hs still renders as a
    # perfectly plausible strand -- so check for it here rather than discovering
    # it later in a screenshot.
    for key in ("hsw", "hse"):
        col = arrs[key][:, valid.astype(bool)]
        sat = float((col == 255).mean()) if col.size else 0.0
        if sat > 0.001:
            raise RuntimeError(
                f"{key}: {sat:.3%} of samples hit the {hs_max} m quantization "
                f"ceiling, which the derived scale should make impossible. "
                f"Treat this as a bug in the scale derivation.")
        if sat > 0:
            log(f"  note: {key} touches the ceiling in {sat:.4%} of samples")

    chunks = [valid.tobytes()]
    for k in range(nt):
        for a in ("hsw", "dsw", "hse", "dse"):
            chunks.append(arrs[a][k].tobytes())
    return b"".join(chunks), int(valid.sum()), hs_max


def emit(path_stem: pathlib.Path, meta: dict, raw: bytes) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    gz = gzip.compress(raw, 9)
    path_stem.with_suffix(".bin.gz").write_bytes(gz)
    meta["bytes"] = {"raw": len(raw), "gz": len(gz)}
    path_stem.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    log(f"  wrote {path_stem.name}: {len(gz) / 1e6:.2f} MB gz "
        f"(from {len(raw) / 1e6:.2f} MB)")


# ------------------------------------------------------------------ products


def build_overview(years, stride, out):
    sites = load_sites(out)
    names = sites["name"][::stride]
    log(f"overview {years[0]}-{years[-1]}: {len(names)} sites (every {stride})")

    index = []
    for y in years:
        t0 = dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc)
        t1 = dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc) - dt.timedelta(hours=1)
        if t0 > dt.datetime.now(dt.timezone.utc):
            break
        # A full 2000-present build is a couple of hours of THREDDS round-trips,
        # so completed years are skipped and the run can simply be restarted.
        done_path = out / "history" / "overview" / f"{y}.json"
        if done_path.exists():
            prev = json.loads(done_path.read_text())
            if prev.get("siteStride") == stride:
                log(f"{y}: already built, skipping")
                index.append({"year": y, "nt": len(prev["times"])})
                continue
        # stride 24 on the hourly axis: one sample per day.
        out_rows, common = gather(names, t0, t1, 24, f"{y}")
        raw, nvalid, hs_max = pack(out_rows, common, names)
        emit(out / "history" / "overview" / str(y), {
            "kind": "overview", "year": y, "siteStride": stride,
            "nsites": len(names), "nvalid": nvalid, "hsMax": hs_max,
            "times": [int(t) for t in common],
        }, raw)
        index.append({"year": y, "nt": len(common)})

    (out / "history" / "overview" / "index.json").write_text(
        json.dumps({"siteStride": stride, "years": index}, indent=1) + "\n")


def pick_events(out, top, min_gap_days=20):
    """Choose events from the overview rather than from memory.

    Ranks days by the statewide 90th-percentile swell-band Hs, so an event has
    to be big along a broad stretch of coast rather than at one exposed site,
    then enforces a minimum separation so one storm does not fill the list.
    """
    root = out / "history" / "overview"
    idx = json.loads((root / "index.json").read_text())
    rows = []
    for y in idx["years"]:
        meta = json.loads((root / f"{y['year']}.json").read_text())
        raw = gzip.decompress((root / f"{y['year']}.bin.gz").read_bytes())
        ns, nt = meta["nsites"], len(meta["times"])
        valid = np.frombuffer(raw, np.uint8, ns, 0).astype(bool)
        body = np.frombuffer(raw, np.uint8, nt * 4 * ns, ns).reshape(nt, 4, ns)
        hsw = body[:, 0, :][:, valid] * meta.get('hsMax', HS_MAX) / 255
        rows += list(zip(meta["times"], np.percentile(hsw, 90, axis=1)))

    rows.sort(key=lambda r: -r[1])
    chosen = []
    for t, v in rows:
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
        if all(abs((d - c[0]).days) > min_gap_days for c in chosen):
            chosen.append((d, v))
        if len(chosen) >= top:
            break

    events = [{"slug": d.strftime("%Y-%m-%d"),
               "peak": d.strftime("%Y-%m-%dT%H:%MZ"),
               "statewideHs90": round(float(v), 2)} for d, v in chosen]
    (out / "history" / "events.json").write_text(json.dumps(events, indent=1) + "\n")
    for e in events:
        log(f"  event {e['slug']}  statewide p90 swell Hs = {e['statewideHs90']} m")
    return events


def build_event(slug, stride, out, days=7):
    sites = load_sites(out)
    names = sites["name"][::stride]
    peak = dt.datetime.strptime(slug, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    t0 = peak - dt.timedelta(days=days // 2)
    t1 = peak + dt.timedelta(days=days - days // 2)
    log(f"event {slug}: {len(names)} sites (every {stride}), {t0.date()} -> {t1.date()}")

    out_rows, common = gather(names, t0, t1, 1, slug)
    raw, nvalid, hs_max = pack(out_rows, common, names)
    emit(out / "history" / "events" / slug, {
        "kind": "event", "slug": slug, "siteStride": stride,
        "nsites": len(names), "nvalid": nvalid, "hsMax": hs_max,
        "times": [int(t) for t in common],
    }, raw)


# ------------------------------------------------------------------ cli


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sites", "overview", "events", "event"])
    ap.add_argument("--years", default="2000-2026")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--slug")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out", type=pathlib.Path, default=here() / "data")
    a = ap.parse_args()

    if a.cmd == "sites":
        write_sites(fetch_site_table(), a.out)
    elif a.cmd == "overview":
        lo, hi = (int(x) for x in a.years.split("-"))
        build_overview(list(range(lo, hi + 1)), a.stride, a.out)
    elif a.cmd == "events":
        pick_events(a.out, a.top)
    elif a.cmd == "event":
        if not a.slug:
            ap.error("--slug is required")
        build_event(a.slug, a.stride, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
