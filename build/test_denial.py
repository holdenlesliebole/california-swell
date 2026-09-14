#!/usr/bin/env python3
"""Regression test for how build_data.py behaves when CDIP refuses a request.

On 2026-09-14 two scheduled refreshes failed outright: CDIP's abuse filter
answered the first OPeNDAP open with its "Access Denied" page, netCDF raised
errno -78, and the exception took the whole deploy down. The site then served
whatever Pages held, with no rebuild attempted for six hours. Nothing was wrong
with the other fifteen domains, and nothing was wrong with the payloads already
on disk.

These tests pin the behavior that replaced it: a refusal costs that domain its
refresh, not the run. They stub netCDF entirely, so they need no network and
run in well under a second.

    python build/test_denial.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import build_data as B  # noqa: E402

DENIAL = OSError("[Errno -78] NetCDF: Authorization failure: "
                 "'https://thredds.cdip.ucsd.edu/thredds/dodsC/x.nc'")
PLAIN = OSError("[Errno -68] NetCDF: I/O failure")

FAKE_META = {
    "id": None, "label": "stand-in", "source": "previous build",
    "grid": {"nx": 10, "ny": 10, "dlon": 0.01, "dlat": 0.01,
             "lon0": -120.0, "lat0": 33.0},
    "bytes": {"raw": 1, "gz": 1},
    "times": [1_757_000_000, 1_757_003_600],
}


def seed(out: pathlib.Path, domains) -> None:
    """Write a previous build's payload for each domain, as a real run leaves."""
    for d in domains:
        meta = dict(FAKE_META, id=d)
        (out / f"{d}.json").write_text(json.dumps(meta))
        (out / f"{d}.bin.gz").write_bytes(b"previous")


def run(out: pathlib.Path, domains, *extra) -> int:
    sys.argv = ["build_data.py", "--domains", *domains, "--out", str(out), *extra]
    return B.main()


def test_denial_backoff_is_long_and_bounded(_out):
    """A denial waits minutes, not seconds, and stops after DENIAL_BACKOFF."""
    opens, slept = [], []
    B.time.sleep = slept.append

    class Denied:
        def __init__(self, url):
            opens.append(url)
            raise DENIAL

    B.nc.Dataset = Denied
    try:
        B.open_with_retry("http://example/x.nc")
    except OSError:
        pass
    else:
        raise AssertionError("open_with_retry should have re-raised the denial")

    assert len(opens) == 1 + len(B.DENIAL_BACKOFF), f"{len(opens)} attempts"
    assert len(slept) == len(B.DENIAL_BACKOFF), f"slept {slept}"
    for waited, base in zip(slept, B.DENIAL_BACKOFF):
        # Jittered upward by up to 25%, never below the nominal wait.
        assert base <= waited <= base * 1.25, f"{waited:.0f}s off schedule for {base:.0f}s"
    return f"{len(opens)} attempts, waits {[round(x) for x in slept]} s"


def test_plain_error_keeps_the_fast_retry(_out):
    """A server hiccup is not a block, and must not inherit the long backoff."""
    opens, slept = [], []
    B.time.sleep = slept.append

    class Broken:
        def __init__(self, url):
            opens.append(url)
            raise PLAIN

    B.nc.Dataset = Broken
    try:
        B.open_with_retry("http://example/x.nc")
    except OSError:
        pass
    else:
        raise AssertionError("open_with_retry should have re-raised")

    assert len(opens) == 4, f"{len(opens)} attempts"
    assert slept == [5.0, 5.0, 5.0], f"slept {slept}"
    return f"{len(opens)} attempts, waits {slept} s"


def test_one_refusal_costs_one_domain(out):
    """The other domains rebuild, the refused one keeps its previous payload."""
    doms = ["sd", "oc", "la"]
    seed(out, doms)
    B.time.sleep = lambda _s: None

    def fetch(stem, decim):
        if stem == "OC_0.001":
            raise DENIAL
        lat = B.np.linspace(33.0, 33.09, 10)
        lon = B.np.linspace(-120.0, -119.91, 10)
        depth = B.np.ma.masked_array(B.np.full((10, 10), 50.0))
        frame = {k: B.np.ma.masked_array(B.np.ones((10, 10))) for k in ("hs", "tp", "dp")}
        return lat, lon, depth, [1_757_010_000], {1_757_010_000: frame}

    B.fetch = fetch
    rc = run(out, doms)

    assert rc == 0, f"a partial refusal must not fail the run (rc={rc})"
    assert (out / "oc.bin.gz").read_bytes() == b"previous", "oc payload was overwritten"
    assert (out / "sd.bin.gz").read_bytes() != b"previous", "sd was not rebuilt"

    idx = json.loads((out / "index.json").read_text())
    assert idx["stale"] == ["oc"], f"stale list wrong: {idx.get('stale')}"
    assert [d["id"] for d in idx["domains"]] == doms, "a refused domain must stay in the index"
    return f"rc={rc}, stale={idx['stale']}, index intact"


def test_a_blocked_runner_stops_asking(out):
    """Once the source address is blocked, every domain is; do not ask sixteen times."""
    doms = ["sd", "oc", "la", "ve"]
    seed(out, doms)
    B.time.sleep = lambda _s: None
    tried = []

    def fetch(stem, decim):
        tried.append(stem)
        raise DENIAL

    B.fetch = fetch
    rc = run(out, doms)

    assert rc == 1, f"a total refusal must exit non-zero (rc={rc})"
    assert len(tried) == B.DENIAL_GIVE_UP_AFTER, f"asked {len(tried)} times: {tried}"
    # Nothing was rebuilt, so the index keeps its old "generated" stamp rather
    # than claiming a build happened at this hour.
    assert not (out / "index.json").exists(), "index.json must not be rewritten"
    return f"rc={rc}, asked {len(tried)} of {len(doms)} domains, index untouched"


def test_strict_still_fails_hard(out):
    """--strict is the local escape hatch: no silent fallback when debugging."""
    seed(out, ["sd"])
    B.time.sleep = lambda _s: None

    def fetch(stem, decim):
        raise DENIAL

    B.fetch = fetch
    try:
        run(out, ["sd"], "--strict")
    except OSError:
        return "denial propagated"
    raise AssertionError("--strict should have propagated the denial")


def test_no_previous_payload_fails_hard(out):
    """Falling back needs something to fall back to; an empty dir must not pass."""
    B.time.sleep = lambda _s: None

    def fetch(stem, decim):
        raise DENIAL

    B.fetch = fetch
    try:
        run(out, ["sd"])
    except OSError:
        return "denial propagated"
    raise AssertionError("a refusal with no previous payload should have raised")


TESTS = [
    test_denial_backoff_is_long_and_bounded,
    test_plain_error_keeps_the_fast_retry,
    test_one_refusal_costs_one_domain,
    test_a_blocked_runner_stops_asking,
    test_strict_still_fails_hard,
    test_no_previous_payload_fails_hard,
]


def main() -> int:
    real = (B.nc.Dataset, B.fetch, B.time.sleep)
    failed = 0
    for t in TESTS:
        B.nc.Dataset, B.fetch, B.time.sleep = real
        with tempfile.TemporaryDirectory() as tmp:
            try:
                note = t(pathlib.Path(tmp))
            except (Exception, SystemExit) as exc:
                # Not just AssertionError: against the pre-fix builder these
                # tests crash, or argparse exits over a flag that does not
                # exist yet. Either way the behavior is absent, so it is a
                # failure and the remaining tests should still run.
                print(f"FAIL  {t.__name__}: {exc.__class__.__name__}: {exc}")
                failed += 1
            else:
                print(f"ok    {t.__name__}  ({note})")
    B.nc.Dataset, B.fetch, B.time.sleep = real
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
