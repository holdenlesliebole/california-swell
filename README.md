# The California swell field

Two animated visualizations of CDIP's MOP nearshore wave model, built from the
Scripps THREDDS server. No dependencies, no build step for the front end — plain
ES2020 and canvas.

| page | what it is |
|---|---|
| `index.html` | **Live 2-D field.** Gridded nowcast + forecast over all sixteen grids — statewide at ~2 km, zooming to any county at ~100–200 m. Optional coastal sea/swell fringe. |
| `history.html` | **The 2000–present hindcast** as a two-layer coastal fringe: 11,594 alongshore MOP sites, swell and sea drawn separately. 27 years daily plus 20 events hourly. |

Data is CDIP's, which is public; the code and the derived payloads are what this
repo adds. Everything here is reproducible from `build/` against THREDDS.

## What it shows

Strands are traced through the model's peak-direction (Dp) field. In the
geometric-optics limit that makes them **wave rays**, so strand convergence is
refractive focusing rather than a rendering artifact. Strand *speed* is the wave
group velocity, solved per cell from the linear dispersion relation, so strands
slow and crowd as they shoal.

Sixteen grids, fetched one at a time on zoom — 11.9 MB in total, but a visitor
only ever downloads the statewide grid plus whichever county they look at:

| tier | grids | resolution | payload each |
|---|---|---|---:|
| statewide | `CA_0.01`, decimated 2× | ~2 km | 0.51 MB |
| northern counties | `DN` `HU` `M` `SN` `MA` `SF` `SM` `SC` `MO` `SL` | ~200 m | 0.18–0.72 MB |
| southern counties | `B` `VE` `L` `OC` `D` | ~100 m | 0.92–2.54 MB |

Together the county grids tile the coast continuously from the Mexican border to
Oregon.

**A band split is not possible from the grids.** They publish bulk
`Hs/Tp/Dp/Ta` only: the `waveFrequency` dimension is present but no data
variable references it, so there is no per-frequency energy and no
`a1`/`b1` to partition — at any resolution. Publishing spectra on the grid would
cost ~54 MB per time step for San Diego alone and ~4.9 GB per run statewide,
which is presumably why it is not done. The alongshore stations *do* carry
spectra, so the live page's sea/swell fringe is computed there
(`build_hindcast.py live`) and drawn as discrete coastal vectors over the field
rather than blended into it. It updates every three hours, the alongshore
forecast's own cadence.

## Why there is a build step

CDIP's THREDDS server sends no `Access-Control-Allow-Origin` header, so the page
cannot read OPeNDAP directly from the browser. `build/build_data.py` runs
server-side, merges nowcast with forecast onto one time axis, and quantizes
Hs/Tp/Dp to a byte each over a wet-cell mask. The merged axis is not uniform:
the nowcast is hourly, the forecast 6-hourly, which is all the gridded products
publish. The page presents it hourly anyway — clock, readout, and Hs trace
interpolate linearly between the 6-hourly forecast frames, and the footer says
so. Baking interpolated hourly frames into the payloads instead would multiply
every download by roughly five for data the model never produced.

```bash
python build/build_data.py                 # both domains
python build/build_data.py --domains sd    # just San Diego
```

Requires `netCDF4`, `numpy`, `scipy`.

`.github/workflows/refresh-swell.yml` reruns this every six hours and commits
the result. Without it the page quietly becomes a snapshot while still claiming
to be live.

## Notes on the data

**The gridded products are nowcast + forecast only — there is no gridded
hindcast.** Confirmed against the catalog: `MOP_grids` contains zero files
matching `hindcast`, only `nowcast`, `forecast`, `seaswellnc` and `seaswellfc`.
The window is therefore rolling — about six hours back to four days ahead —
and cannot be extended backwards.

The historical record exists only as the **1-D alongshore MOP stations**:
**11,594 sites covering all 15 California coastal counties** at ~115 m spacing.

**"Nowcast" means two different things on this server, and conflating them will
bite.** For the *grids* it is a 6-hour buffer. For the *alongshore* stations it
is a rolling multi-year archive that begins exactly where the hindcast stops:

| file | steps | span |
|---|---:|---|
| `_hindcast` | 221,328 | 2000-01-01 → 2025-03-31 23:00 |
| `_nowcast` | 11,663 | 2025-04-01 00:00 → present |
| `_forecast` | 80 | present − 3 d → present + 6 d |

So hindcast + nowcast is a **continuous hourly record from 2000 to now, ~26.6
years, with no gap** — the handoff is seamless at 2025-04-01, which is the
switchover hardcoded in `read_MOPline2.m` (a private MATLAB reader for these files). Use that routine's routing
logic as the reference implementation; a build script should reproduce it rather
than reinvent it.

### Hindcast product — `build/build_hindcast.py`, `history.html`

```bash
python build/build_hindcast.py sites                          # one HTTP request
python build/build_hindcast.py overview --years 2000-2026 --stride 8
python build/build_hindcast.py events --top 20                # derived, not remembered
python build/build_hindcast.py event --slug 2023-01-06 --stride 4
python build/build_hindcast.py live --stride 4                # coastal sea/swell split
```

Overview years are skipped if already built, so a long run can just be
restarted. Two traps worth knowing about, both of which produced plausible-
looking wrong output before being caught:

- **Site IDs are five characters, so padding depends on the prefix length** —
  `D0586` but `OC001`. The R catalog writes them all four-wide. Getting this
  wrong silently drops every two-letter region, which is 6,358 of 11,594 sites,
  and the result still renders as a perfectly convincing coastline. `gather()`
  now raises if more than 5% of sites return nothing.
- **No fixed Hs ceiling survived the record, so the scale is derived.** An 8 m
  ceiling clipped the January 2023 event; 12 m still clipped 2015-12-11, where
  Humboldt sites reach 13.99 m hourly. Both were set from a handful of sampled
  sites, which says nothing about the extremes across 11,594 of them over 26
  years — and a clipped value still renders as a believable strand. `pack()`
  now measures each payload's own peak and sets `hsMax = ceil(peak × 1.02)`,
  shipped in the metadata the page reads. Across the built set that lands
  between 6.5 m and 15.5 m, with zero saturated samples. The assert remains as
  a backstop against a bug in the derivation itself.

- **The overview is a snapshot, not a daily summary.** Years sample every 24th
  hourly step, so each frame is 00:00 UTC — not a mean, not a maximum. The
  December 2015 event reads 13.56 m in the yearly view against 14.77 m in the
  hourly event view. This also means event *ranking* runs on attenuated peaks,
  so the shortlist is defensible but not a definitive ordering.

- **`gather()` takes a strict time intersection, so one bad site drops a step
  for everyone.** 2017 and 2018 each come out at 364 daily steps instead of 365
  (missing 2017-09-01 and 2018-01-21) because one or two sites of 1,450 lack
  that hour. Timestamps are read per frame from `meta.times`, so labels stay
  correct and the only effect is a one-day skip in 365. Left as is; relaxing the
  intersection would mean per-site time indices for a 0.3% gain.

- **Sea-band coverage is not uniform across the record.** MOP's band estimates
  depend on which offshore buoys were feeding the model (`waveModelInputSource`),
  and that changed: Del Norte has no sea-band estimate until 2005-10, Humboldt
  carries one in ~84% of months, Santa Barbara 100%. Gaps arrive as exactly zero
  energy, indistinguishable from flat calm once drawn, so the page computes and
  reports per-frame coverage and leaves missing sites blank. Worth knowing for
  any analysis using early-record northern sea bands.

`netCDF4`'s OPeNDAP backend is not thread-safe; concurrent opens fail with
"a libcurl function was given a bad argument". The fetch uses processes.

### Design notes

Two tiers, both driven by the alongshore stations:

- **Events** — hourly, all 11,594 sites, full spectra, ~7-day windows around
  named swells. Measured: 0.74 s per site, so ~9 min at 16× concurrency.
- **Overview** — daily samples across the whole 2000→now record via strided
  OPeNDAP (`[::24]`), bulk variables only. Measured: ~1 min for 1,449 sites
  (every 8th, ~1 km spacing). Ship one year per file, lazily loaded.

**Direction should come from the band-split energy flux, not from `Dm`.**
Following `estimateEnergyFlux` in `read_MOPline2.m` — flux is
`ρg·Cg·a1·E` and `ρg·Cg·b1·E` summed over a band — with that file's canonical
bands (swell 0.02–0.0813 Hz, sea 0.09–0.400 Hz). Reproducing the total-band
flux direction matches the file's own `waveDm` to a 1.80° mean offset with 1.38°
residual, which confirms the quantity is right.

Measured over 1,200 hours at D0586: `Dp` jumps >30° in 0.83% of hours, while
`Dm` and both flux directions never do — so `Dm` alone would fix the
discontinuity. But sea and swell each carry 30–70% of the flux for **796 of
those 1,200 hours (66%)**, and their flux directions differ by **22° on average,
up to 39°**. For two-thirds of the record a single mean direction is therefore
smooth but wrong: it points where neither system is going. The split is not a
refinement, it is the signal — and it comes almost free, since the two band sums
share one fetch. Flux magnitude per band gives each layer a natural weight.

Crucially the alongshore files carry far more than the grids do: `waveDm`
(energy-weighted mean direction), `waveEnergyDensity` and `waveMeanDirection`
over 20 frequency bands, the `waveA1/B1/A2/B2` directional moments, and
`waveSxx`/`waveSxy`. The grids carry only bulk `Hs/Tp/Ta/Dp`. **A hindcast
product is therefore better conditioned than the live one**: mean direction is
continuous, so the bimodal-front problem that forces the coherence fade here
disappears by construction, and the per-frequency spectra allow sea and swell to
be split into two independent layers of fur.

**Coastline vs. model-domain edge is separated by depth, not by a flood fill.**
The MOP domains are staircases of nested boxes, and at every step corner the
offshore padding abuts the continent directly, so a fill seeded from either edge
leaks into the other region — an early version marked 83% of the statewide grid
as "land". A mask boundary in shallow water is coast; in deep water it is the
edge of the model. That also gives the distance ramp used to dissolve the fur at
the domain edge instead of cutting it off along a visible staircase.

There are 15 further county-level grids on THREDDS (`B`, `L`, `VE`, `OC`, `SL`,
`SF`, `MA`, `MO`, `SC`, `HU`, `DN`, …), so the zoom-to-detail architecture
extends to the whole coast without changing the renderer.

`R_CA_coefficients` holds the MOP transfer operator itself — one file per site
(the same 11,594), giving `swl_et/ec/es/ec2/es2` over 10 swell bands × 72
directions plus 17 sea bands, along with each site's `shoreNormal`, `depth` and
a `shoreFlag` marking which sites sit on planar natural coastline. That is the
linear map from an offshore directional spectrum to a local one, so a hindcast
can in principle be *recomputed* for any hour with an offshore spectrum rather
than only read back. It does not, however, yield a gridded field: the
coefficients exist at those same coastal sites, not on a 2-D mesh.

## What it is not

Strands are rays, not water-particle trajectories and not orbital motion —
nothing here advects a physical parcel. It traces an already-computed Dp field
rather than integrating the eikonal equation, so caustics are implied, not
resolved. Strand pacing is scaled for legibility, not real time. Color ranges
are percentile-clipped per build, so a calm summer day and a winter event are
not on a common scale.

## Implementation

Plain ES2020, no dependencies, no build step for the front end. Four stacked
canvases (bathymetry wash, animated fur, coastline, pointer UI). Strand count is
steered by an adaptive controller targeting ~50 fps, since per-strand cost
varies by an order of magnitude across machines. Segments are batched into 32
color buckets by a counting sort into preallocated buffers, so a frame costs 32
canvas strokes regardless of strand count.
