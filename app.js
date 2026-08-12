/* CDIP MOP swell field — animated wave-ray visualization.
 *
 * Strands are traced through the peak-direction (Dp) field. In the
 * geometric-optics limit that makes them wave rays, so where strands converge
 * you are looking at refractive focusing, not a rendering artifact.
 *
 * Strand speed is the wave group velocity c_g, solved per cell from the linear
 * dispersion relation using the local peak period and water depth. Strands
 * therefore slow and bunch as they climb the shelf, which is the same shoaling
 * that concentrates energy at the coast.
 *
 * Plain ES2020, no dependencies, no build step.
 */
'use strict';

const G = 9.81;
const DEG = Math.PI / 180;

// Reference group speed (m/s) that maps to unit strand pace on screen —
// roughly a 15 s swell in shelf depths off California.
const CG_REF = 9.0;
// On-screen strand pace, in pixels per second, at unit relative group speed.
const PX_PER_SEC = 105;

// Playback rate in simulated seconds per real second: six hours of sea state
// per second, so the ~5-day gridded window loops in about twenty seconds. Held
// constant across the axis so the hourly and 6-hourly sections of the window
// advance at the same physical rate (see _loop).
const SIM_SEC_PER_SEC = 6 * 3600;

// Directional-coherence window for fading out bimodal fronts (see coherence()).
// A smoothly refracting field scores ~0.999 — neighbors differ by a degree or
// two — so the cut sits high and still leaves genuine refraction fully drawn.
const COH_LO = 0.94, COH_HI = 0.995;
const COH_INV = 1 / (COH_HI - COH_LO);

// ---------------------------------------------------------------- palettes

// Perceptually-ordered ramps sampled to 32 buckets. Strands are batched by
// bucket so a frame costs ~32 canvas strokes instead of ~20k.
const NBUCKET = 32;

function ramp(stops) {
  const out = [];
  for (let i = 0; i < NBUCKET; i++) {
    const t = i / (NBUCKET - 1) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    const f = t - k;
    const a = stops[k], b = stops[k + 1];
    out.push([
      Math.round(a[0] + (b[0] - a[0]) * f),
      Math.round(a[1] + (b[1] - a[1]) * f),
      Math.round(a[2] + (b[2] - a[2]) * f),
    ]);
  }
  return out;
}

const PALETTES = {
  // Cool-to-hot, dark-background safe. Reads as "small swell" to "big swell".
  hs: ramp([[26, 51, 92], [32, 110, 168], [56, 178, 186], [140, 214, 142],
            [242, 214, 96], [240, 148, 60], [222, 74, 62]]),
  // Period: short wind sea (gray-green) through long-period groundswell (violet).
  tp: ramp([[70, 88, 92], [78, 140, 130], [110, 190, 150], [186, 214, 130],
            [232, 178, 128], [214, 124, 168], [156, 96, 200]]),
  // Deliberately NOT circular. Direction is a circular quantity, but what is
  // displayed is a clamped sub-range (see _colorRange) covering the swell
  // window that actually reaches California — so a ramp that returned to its
  // starting color would give two different bearings the same hue for no
  // benefit. Distinct endpoints keep the mapping one-to-one over the range
  // shown, which is what makes the shoreward rotation of refraction readable.
  dp: ramp([[232, 122, 64], [222, 196, 84], [120, 204, 120], [64, 178, 214],
            [110, 132, 226], [166, 110, 214], [212, 96, 168]]),
};

const SCALARS = {
  hs: { label: 'Significant wave height', unit: 'm', short: 'Hs' },
  tp: { label: 'Peak period', unit: 's', short: 'Tp' },
  dp: { label: 'Peak direction', unit: '°', short: 'Dp' },
};

// ---------------------------------------------------------------- geometry

// Web Mercator in y so a 10-degree-tall California is not visibly stretched.
const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + lat * DEG / 2));
const invMercY = (y) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / DEG;

// ---------------------------------------------------------------- payload

/** Fetch a .bin.gz, transparently handling servers that pre-decompress it. */
async function fetchGz(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const h = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (h[0] === 0x1f && h[1] === 0x8b) {
    if (typeof DecompressionStream !== 'function') {
      throw new Error('This browser cannot decompress the data.');
    }
    buf = await new Response(
      new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'))
    ).arrayBuffer();
  }
  return buf;
}

/** Decode one domain's .bin.gz into typed arrays plus a scatter index. */
async function loadDomain(meta, onProgress) {
  const res = await fetch(`data/${meta.id}.bin.gz`);
  if (!res.ok) throw new Error(`fetch ${meta.id}: HTTP ${res.status}`);

  let buf = await res.arrayBuffer();
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  // Some hosts set Content-Encoding: gzip and the browser has already
  // decompressed for us; others hand back the raw file. Sniff the magic
  // number rather than trusting either behavior.
  if (head[0] === 0x1f && head[1] === 0x8b) {
    if (typeof DecompressionStream !== 'function') {
      throw new Error('This browser cannot decompress the data (needs DecompressionStream).');
    }
    const ds = new DecompressionStream('gzip');
    buf = await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
  }
  if (onProgress) onProgress();

  const { nx, ny } = meta.grid;
  const ncell = nx * ny;
  const nwet = meta.nwet;
  const nt = meta.times.length;

  let o = 0;
  const mask = new Uint8Array(buf, o, ncell); o += ncell;
  const shore = new Uint8Array(buf, o, nwet); o += nwet;
  const edge = new Uint8Array(buf, o, nwet); o += nwet;
  const depthQ = new Uint8Array(buf, o, nwet); o += nwet;

  const expected = o + nt * 3 * nwet;
  if (expected !== buf.byteLength) {
    throw new Error(`${meta.id}: payload is ${buf.byteLength} B, expected ${expected} B`);
  }

  const frames = [];
  for (let t = 0; t < nt; t++) {
    frames.push({
      hs: new Uint8Array(buf, o, nwet),
      tp: new Uint8Array(buf, o + nwet, nwet),
      dp: new Uint8Array(buf, o + 2 * nwet, nwet),
    });
    o += 3 * nwet;
  }

  // cellToWet[c] = index into the packed wet arrays, or -1 on land.
  const cellToWet = new Int32Array(ncell).fill(-1);
  const wetToCell = new Int32Array(nwet);
  for (let c = 0, w = 0; c < ncell; c++) {
    if (mask[c]) { cellToWet[c] = w; wetToCell[w] = c; w++; }
  }

  // Depth in meters, per wet cell — needed for the dispersion solve.
  const { logMin, logMax } = meta.vars.depth;
  const depth = new Float32Array(nwet);
  for (let w = 0; w < nwet; w++) {
    depth[w] = Math.pow(10, logMin + (depthQ[w] / 255) * (logMax - logMin));
  }

  return {
    meta, mask, shore, edge, depth, frames, cellToWet, wetToCell,
    nx, ny, ncell, nwet, nt,
    bbox: {
      lonMin: meta.grid.lon0,
      lonMax: meta.grid.lon0 + nx * meta.grid.dlon,
      latMin: meta.grid.lat0,
      latMax: meta.grid.lat0 + ny * meta.grid.dlat,
    },
    cache: new Map(),   // frame index -> expanded field
  };
}

// ---------------------------------------------------- dispersion & fields

/** Wavenumber from period and depth: Guo (2002) explicit fit, ~0.75% error. */
function waveNumber(omega, h) {
  const x = omega * Math.sqrt(h / G);
  const beta = 2.4908;
  const kh = x * x * Math.pow(1 - Math.exp(-Math.pow(x, beta)), -1 / beta);
  return kh / h;
}

/** Group velocity from the linear dispersion relation. */
function groupSpeed(period, h) {
  if (!(period > 0) || !(h > 0)) return 0;
  const omega = 2 * Math.PI / period;
  const k = waveNumber(omega, h);
  if (!(k > 0)) return 0;
  const kh = k * h;
  // 2kh/sinh(2kh) -> 1 in shallow water, 0 in deep; guard the overflow.
  const n = kh > 20 ? 0.5 : 0.5 * (1 + (2 * kh) / Math.sinh(2 * kh));
  return (omega / k) * n;
}

// Scratch for the coherence pass. Module-scope and reused: these are only
// needed while building a frame, and allocating two ncell-sized arrays per
// frame would churn ~4 MB a time on the San Diego grid.
let _ex = null, _ey = null;

function scratch(ncell) {
  if (!_ex || _ex.length < ncell) {
    _ex = new Float32Array(ncell);
    _ey = new Float32Array(ncell);
  }
  return [_ex, _ey];
}

/**
 * Directional coherence: how well the peak direction agrees with its neighbors.
 *
 * Dp is a *peak* statistic, so in a bimodal sea it is discontinuous. Where a
 * decaying swell and a rising wind sea carry comparable energy, the spectral
 * peak flips between the two systems from one cell to the next, and Dp jumps
 * tens of degrees across a sharp front. Measured on a real frame, cells either
 * side of such a front differ in Tp by ~6 s against ~0.05 s elsewhere, which is
 * the giveaway that this is two wave systems trading places rather than noise.
 *
 * Traced faithfully, those fronts throw adjacent strands off at wildly
 * different angles and read as glitchy straight streaks. Smoothing Dp would be
 * worse than the disease: the average of two distinct wave systems is a
 * direction in which no wave is traveling. Instead we measure the local
 * agreement and let the renderer fade strands out where the peak is ambiguous
 * — the field goes quiet exactly where a single direction stops being a
 * meaningful description of the sea state.
 *
 * Returns the resultant length of the neighborhood's unit direction vectors:
 * 1 where the field is locally parallel, falling toward 0 as it disagrees.
 * Smooth refraction gradients score near 1, so real physics is untouched.
 */
function coherence(dom, EX, EY, VALID) {
  const { nx, ncell } = dom;
  const COH = new Float32Array(ncell);
  for (let w = 0; w < dom.nwet; w++) {
    const c = dom.wetToCell[w];
    let sx = EX[c], sy = EY[c], n = 1;
    if (c >= nx && VALID[c - nx]) { sx += EX[c - nx]; sy += EY[c - nx]; n++; }
    if (c + nx < ncell && VALID[c + nx]) { sx += EX[c + nx]; sy += EY[c + nx]; n++; }
    if (VALID[c - 1]) { sx += EX[c - 1]; sy += EY[c - 1]; n++; }
    if (VALID[c + 1]) { sx += EX[c + 1]; sy += EY[c + 1]; n++; }
    COH[c] = Math.sqrt(sx * sx + sy * sy) / n;
  }
  return COH;
}

/**
 * Expand one time frame into full-grid float arrays.
 *
 * U,V are the propagation direction scaled by group speed, in grid columns and
 * rows per second — the column step carries the 1/cos(lat) metric correction
 * so strands run true on a Mercator canvas. VALID marks usable cells.
 */
function expandFrame(dom, t) {
  const hit = dom.cache.get(t);
  if (hit) return hit;

  const { nx, ny, ncell } = dom;
  const f = dom.frames[t];
  const U = new Float32Array(ncell);
  const V = new Float32Array(ncell);
  const C = new Float32Array(ncell);
  const VALID = new Uint8Array(ncell);

  const hsMax = dom.meta.vars.hs.max;
  const tpMax = dom.meta.vars.tp.max;
  const dlat = dom.meta.grid.dlat;
  const lat0 = dom.meta.grid.lat0;
  const [EX, EY] = scratch(ncell);

  for (let w = 0; w < dom.nwet; w++) {
    const c = dom.wetToCell[w];
    const row = (c / nx) | 0;
    const lat = lat0 + row * dlat;

    const tp = (f.tp[w] / 255) * tpMax;
    const cg = groupSpeed(tp, dom.depth[w]);

    // Dp is the compass direction waves come FROM; propagation is the reverse.
    const th = (f.dp[w] / 255) * 360 * DEG;
    const east = -Math.sin(th), north = -Math.cos(th);

    // Magnitude is group speed *relative* to a typical California value. Real
    // c_g spans only ~4-12 m/s here, so pacing the animation in absolute m/s
    // would be either invisible or a blur depending on zoom; carrying the
    // ratio instead keeps the physics that matters -- strands visibly slowing
    // and crowding as they shoal -- and lets the view set the overall tempo.
    const rel = cg / CG_REF;
    // The 1/cos(lat) here is the Mercator metric: with it, a column step and a
    // row step cover the same number of screen pixels, so strands run true.
    U[c] = (east * rel) / Math.cos(lat * DEG);
    V[c] = north * rel;
    C[c] = (f.hs[w] / 255) * hsMax;
    VALID[c] = 1;
    EX[c] = east; EY[c] = north;
  }

  const out = { U, V, C, VALID, COH: coherence(dom, EX, EY, VALID) };
  dom.cache.set(t, out);
  // Playback only ever needs the two bracketing frames; keep a little slack.
  if (dom.cache.size > 4) dom.cache.delete(dom.cache.keys().next().value);
  return out;
}

/** Bilinear sample that renormalizes over whichever corners are in water. */
function sampleBilinear(A, VALID, nx, ny, col, row, out) {
  const j0 = Math.floor(col), i0 = Math.floor(row);
  if (j0 < 0 || i0 < 0 || j0 >= nx - 1 || i0 >= ny - 1) return false;
  const fx = col - j0, fy = row - i0;
  let acc = 0, wsum = 0;
  for (let di = 0; di < 2; di++) {
    for (let dj = 0; dj < 2; dj++) {
      const c = (i0 + di) * nx + (j0 + dj);
      if (!VALID[c]) continue;
      const w = (dj ? fx : 1 - fx) * (di ? fy : 1 - fy);
      acc += A[c] * w; wsum += w;
    }
  }
  if (wsum < 0.35) return false;   // mostly land: treat as out of bounds
  out.v = acc / wsum;
  return true;
}

// ---------------------------------------------------------------- the app

class SwellView {
  constructor(root) {
    this.root = root;
    this.layers = {
      bathy: root.querySelector('#l-bathy'),
      fur: root.querySelector('#l-fur'),
      coast: root.querySelector('#l-coast'),
      bands: root.querySelector('#l-bands'),
      ui: root.querySelector('#l-ui'),
    };
    this.ctx = {};
    for (const k of Object.keys(this.layers)) {
      this.ctx[k] = this.layers[k].getContext('2d');
    }

    this.domains = {};        // id -> decoded payload
    this.metas = {};
    this.active = null;
    this.scalar = 'hs';
    this.playing = true;
    this.tf = 0;              // fractional frame index
    this.density = 1.0;
    this.particles = null;
    this.pinned = null;       // clicked cell for the time series
    this.hover = null;
    this.loadingDetail = false;

    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this._bindEvents();
  }

  // --- projection -------------------------------------------------------

  /** Fit a bbox into the canvas, preserving Mercator aspect. */
  setView(bbox, padFrac = 0.02) {
    const w = this.cssW, h = this.cssH;
    const y0 = mercY(bbox.latMin), y1 = mercY(bbox.latMax);
    const lonSpan = bbox.lonMax - bbox.lonMin;
    const ySpan = y1 - y0;
    // Mercator x is in radians of longitude to match y's units.
    const xSpan = lonSpan * DEG;
    const s = Math.min(w / xSpan, h / ySpan) * (1 - padFrac);
    this.view = {
      cx: (bbox.lonMin + bbox.lonMax) / 2 * DEG,
      cy: (y0 + y1) / 2,
      s,
    };
    this._afterViewChange();
  }

  toScreen(lon, lat) {
    const { cx, cy, s } = this.view;
    return [
      this.cssW / 2 + (lon * DEG - cx) * s,
      this.cssH / 2 - (mercY(lat) - cy) * s,
    ];
  }

  toLonLat(px, py) {
    const { cx, cy, s } = this.view;
    return [
      (cx + (px - this.cssW / 2) / s) / DEG,
      invMercY(cy + (this.cssH / 2 - py) / s),
    ];
  }

  /** Screen pixels per grid cell, used to pick the right domain and density. */
  pxPerCell(dom) {
    const d = dom || this.active;
    if (!d) return 1;
    return d.meta.grid.dlon * DEG * this.view.s;
  }

  /**
   * Per-row / per-column screen coordinates for the active grid.
   *
   * Screen x is exactly linear in column, so it needs two numbers. Screen y is
   * not linear in row — that is the whole point of Mercator — but it is smooth,
   * so a per-row table with linear interpolation between entries is exact to
   * well under a pixel. This turns the inverse projection from a transcendental
   * per particle per frame into a table lookup, which is what lets the strand
   * count and the visible-cell scan stay cheap while panning.
   */
  _buildLUT() {
    const d = this.active;
    const { lat0, dlat, lon0, dlon } = d.meta.grid;
    const syRow = new Float32Array(d.ny);
    for (let r = 0; r < d.ny; r++) {
      syRow[r] = this.cssH / 2 - (mercY(lat0 + r * dlat) - this.view.cy) * this.view.s;
    }
    this.lut = {
      syRow,
      x0: this.cssW / 2 + (lon0 * DEG - this.view.cx) * this.view.s,
      dx: dlon * DEG * this.view.s,
    };
  }

  _projX(col) { return this.lut.x0 + col * this.lut.dx; }

  _projY(row) {
    const t = this.lut.syRow;
    if (row <= 0) return t[0] + row * (t[1] - t[0]);
    const n = t.length;
    if (row >= n - 1) return t[n - 1] + (row - n + 1) * (t[n - 1] - t[n - 2]);
    const i = row | 0;
    return t[i] + (row - i) * (t[i + 1] - t[i]);
  }

  /**
   * Wet cells currently on screen, for seeding strands.
   *
   * Seeding from the whole domain and rejecting off-screen candidates looks
   * fine statewide and collapses when zoomed in: almost every draw misses the
   * viewport, particles get parked, and the field goes sparse exactly where the
   * detail matters. Sampling from a precomputed visible set is uniform at every
   * zoom and costs one pass per view change.
   */
  _rebuildVisible() {
    const d = this.active;
    const m = 24;
    const vis = [];
    for (let w = 0; w < d.nwet; w++) {
      const c = d.wetToCell[w];
      const row = (c / d.nx) | 0, col = c - row * d.nx;
      const y = this._projY(row);
      if (y < -m || y > this.cssH + m) continue;
      const x = this._projX(col);
      if (x < -m || x > this.cssW + m) continue;
      vis.push(w);
    }
    this.visible = vis;
  }

  // --- lifecycle --------------------------------------------------------

  async init() {
    const idx = await (await fetch('data/index.json')).json();
    this.index = idx;
    for (const d of idx.domains) this.metas[d.id] = d;

    this._status(`loading statewide field — ${(idx.domains[0].bytes.gz / 1e6).toFixed(2)} MB`);
    const meta = await (await fetch('data/ca.json')).json();
    this.domains.ca = await loadDomain(meta);
    this.active = this.domains.ca;

    this._buildCountyPicker();
    this.resize();
    this.setView(this.active.bbox);
    this._buildParticles();
    this._status(null);
    this._syncChrome();
    this._loop();
  }

  resize() {
    const box = this.root.querySelector('.stage').getBoundingClientRect();
    this.cssW = Math.max(320, Math.floor(box.width));
    this.cssH = Math.max(320, Math.floor(box.height));
    for (const k of Object.keys(this.layers)) {
      const cv = this.layers[k];
      // The bathymetry wash is written with putImageData, which bypasses the
      // canvas transform — so it lives at CSS resolution and is stretched by
      // the browser. It is a soft background; 1x is indistinguishable.
      const scale = k === 'bathy' ? 1 : this.dpr;
      cv.width = Math.floor(this.cssW * scale);
      cv.height = Math.floor(this.cssH * scale);
      cv.style.width = this.cssW + 'px';
      cv.style.height = this.cssH + 'px';
      this.ctx[k].setTransform(scale, 0, 0, scale, 0, 0);
    }
  }

  _status(msg) {
    const el = this.root.querySelector('#status');
    el.textContent = msg || '';
    el.style.display = msg ? 'block' : 'none';
  }

  // --- particles --------------------------------------------------------

  _targetCount() {
    const area = this.cssW * this.cssH;
    return Math.round(Math.min(34000, area / 32) * this.density);
  }

  _buildParticles() {
    const n = this._targetCount();
    this.particles = {
      col: new Float32Array(n),
      row: new Float32Array(n),
      age: new Float32Array(n),
      life: new Float32Array(n),
      n,
      // Strands actually stepped this frame. The adaptive controller moves
      // this, not the allocation, so quality changes cost no garbage.
      active: n,
    };
    // Scratch for the per-frame counting sort of segments into color buckets.
    // Preallocated because this is the hot path: growing 32 plain arrays by
    // ~100k pushes every frame was the single largest cost in the loop.
    this.seg = {
      xy: new Float32Array(n * 4),
      bucket: new Uint8Array(n),
      sorted: new Float32Array(n * 4),
      count: new Int32Array(NBUCKET),
      offset: new Int32Array(NBUCKET + 1),
    };
    for (let i = 0; i < n; i++) this._respawn(i);
  }

  /** Drop a particle on a random wet cell that is currently on screen. */
  _respawn(i) {
    const d = this.active, p = this.particles;
    const vis = this.visible;
    if (!vis || !vis.length) {
      // Nothing of the model is in view; park it and retry next frame.
      p.col[i] = -1; p.row[i] = -1; p.age[i] = 0; p.life[i] = 1;
      return;
    }
    const w = vis[(Math.random() * vis.length) | 0];
    const c = d.wetToCell[w];
    const row = (c / d.nx) | 0, col = c - row * d.nx;
    p.col[i] = col + Math.random() - 0.5;
    p.row[i] = row + Math.random() - 0.5;
    p.age[i] = 0;
    // Lifetime must outlast the trail, or strands are cut off mid-hair.
    p.life[i] = 65 + Math.random() * 95;
  }

  // --- domain switching -------------------------------------------------

  /** The county grid whose footprint contains a point, if any. */
  _countyAt(lon, lat) {
    let best = null, bestArea = Infinity;
    for (const m of this.index.domains) {
      if (m.id === 'ca') continue;
      const b = m.bbox;
      if (lon <= b.lonMin || lon >= b.lonMax || lat <= b.latMin || lat >= b.latMax) continue;
      // County footprints overlap slightly at their shared edges; prefer the
      // smaller one so a point near a boundary lands in the tighter grid.
      const area = (b.lonMax - b.lonMin) * (b.latMax - b.latMin);
      if (area < bestArea) { bestArea = area; best = m; }
    }
    return best;
  }

  async _maybeSwitchDomain() {
    if (this.loadingDetail) return;

    const [lonC, latC] = this.toLonLat(this.cssW / 2, this.cssH / 2);
    const county = this._countyAt(lonC, latC);
    // Switch on the coarse grid's own resolution, not on the view's width in
    // degrees: a wide, short viewport can be very zoomed in and still span a
    // lot of longitude. Once a statewide cell covers more than a couple of
    // screen pixels that grid is visibly blocking, which is exactly where the
    // county grid starts to pay for itself. The two thresholds are hysteresis,
    // so a view sitting near the boundary does not flap between grids.
    const coarsePx = this.pxPerCell(this.domains.ca);
    const onCounty = this.active.meta.id !== 'ca';

    if (county && coarsePx > 2.6 && this.active.meta.id !== county.id) {
      if (!this.domains[county.id]) {
        this.loadingDetail = true;
        this._status(`loading ${county.label} — ${(county.bytes.gz / 1e6).toFixed(2)} MB`);
        try {
          const meta = await (await fetch(`data/${county.id}.json`)).json();
          this.domains[county.id] = await loadDomain(meta);
        } catch (err) {
          this._status(`${county.label} grid failed: ${err.message}`);
          setTimeout(() => this._status(null), 4000);
          this.loadingDetail = false;
          return;
        }
        this._status(null);
        this.loadingDetail = false;
      }
      this._setActive(this.domains[county.id]);
    } else if (onCounty && (!county || coarsePx < 1.7)) {
      this._setActive(this.domains.ca);
    }
  }

  _setActive(dom) {
    if (this.active === dom) return;
    // Keep the same wall-clock instant across the switch; the two grids cover
    // the same window but not necessarily from the same start time.
    const nowT = this._timeAt(this.tf);
    this.active = dom;
    this.tf = this._tfAtTime(nowT);
    // Order matters: the visible-cell set is rebuilt for the new grid here,
    // and seeding strands depends on it.
    this._afterViewChange();
    this._buildParticles();
    this._syncChrome();
  }

  // --- static layers ----------------------------------------------------

  _afterViewChange() {
    if (!this.active) return;
    this._buildLUT();
    this._rebuildVisible();
    this._drawBathy();
    this._drawCoast();
    this._drawBands();
    this._clearFur();
  }

  _clearFur() {
    const c = this.ctx.fur;
    c.clearRect(0, 0, this.cssW, this.cssH);
  }

  /** Subtle depth shading so the shelf and the canyons read under the fur. */
  _drawBathy() {
    const d = this.active, ctx = this.ctx.bathy;
    ctx.clearRect(0, 0, this.cssW, this.cssH);

    // Render at CSS resolution, not device resolution: this is a soft
    // background wash, and at 2x it costs 4x for no visible gain.
    const W = this.cssW, H = this.cssH;
    const img = ctx.createImageData(W, H);
    const px = img.data;
    const { lon0, lat0, dlon, dlat } = d.meta.grid;

    // The inverse projection is separable — x maps to longitude linearly, and
    // only the y axis needs the inverse Mercator. Hoisting it out of the inner
    // loop turns W*H transcendentals into H of them, which is what makes this
    // cheap enough to redraw while panning.
    const cols = new Int32Array(W);
    for (let x = 0; x < W; x++) {
      const lon = (this.view.cx + (x - W / 2) / this.view.s) / DEG;
      cols[x] = Math.round((lon - lon0) / dlon);
    }
    const rows = new Int32Array(H);
    for (let y = 0; y < H; y++) {
      const lat = invMercY(this.view.cy + (H / 2 - y) / this.view.s);
      rows[y] = Math.round((lat - lat0) / dlat);
    }

    for (let y = 0; y < H; y++) {
      const row = rows[y];
      if (row < 0 || row >= d.ny) continue;
      const base = row * d.nx;
      for (let x = 0; x < W; x++) {
        const col = cols[x];
        if (col < 0 || col >= d.nx) continue;
        const w = d.cellToWet[base + col];
        if (w < 0) continue;
        const o = (y * W + x) * 4;
        // log depth 1 m -> 1000 m mapped to a narrow dark blue band
        const t = Math.max(0, Math.min(1, (Math.log10(d.depth[w]) - 0.3) / 2.7));
        const shade = 1 - t;                    // shallow = slightly brighter
        px[o] = 18 + 22 * shade;
        px[o + 1] = 31 + 38 * shade;
        px[o + 2] = 50 + 52 * shade;
        // Alpha must reach zero at the model-domain edge, not merely dim: any
        // floor here redraws the staircase of the domain boundary as a hard
        // line, which reads as geography and is not.
        const e = d.edge[w] / 255;
        px[o + 3] = 172 * e * e;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  _drawCoast() {
    const d = this.active, ctx = this.ctx.coast;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const r = Math.max(0.8, Math.min(2.2, this.pxPerCell() * 0.7));

    ctx.fillStyle = 'rgba(214, 228, 240, 0.9)';
    for (let w = 0; w < d.nwet; w++) {
      if (!d.shore[w]) continue;
      const c = d.wetToCell[w];
      const row = (c / d.nx) | 0, col = c - row * d.nx;
      const sy = this._projY(row);
      if (sy < -4 || sy > this.cssH + 4) continue;
      const sx = this._projX(col);
      if (sx < -4 || sx > this.cssW + 4) continue;
      ctx.fillRect(sx - r / 2, sy - r / 2, r, r);
    }
  }

  // --- sea / swell at the coast ----------------------------------------

  /**
   * Load the coastal band split.
   *
   * This cannot come from the grid. The gridded MOP products publish bulk
   * Hs/Tp/Dp/Ta only -- their waveFrequency dimension carries no data variable,
   * so there is no per-frequency energy and no a1/b1 to partition. The
   * alongshore stations do carry spectra, so the split is computed there and
   * drawn as a fringe over the field. The two sources are different products,
   * which is exactly why the fringe is drawn as discrete vectors rather than
   * blended into the fur.
   */
  async _loadBands() {
    if (this.bands) return true;
    this._status('loading coastal sea/swell split…');
    try {
      const meta = await (await fetch('data/live_bands.json')).json();
      const sm = await (await fetch('data/sites.json')).json();
      const sbuf = await fetchGz('data/sites.bin.gz');
      const buf = await fetchGz('data/live_bands.bin.gz');
      const ns = meta.nsites, nt = meta.times.length;
      const need = ns + nt * 4 * ns;
      if (buf.byteLength !== need) {
        throw new Error(`payload ${buf.byteLength} B, expected ${need} B`);
      }
      const n = sm.n;
      const frames = [];
      for (let k = 0; k < nt; k++) {
        const o = ns + k * 4 * ns;
        frames.push({
          hsw: new Uint8Array(buf, o, ns), dsw: new Uint8Array(buf, o + ns, ns),
          hse: new Uint8Array(buf, o + 2 * ns, ns), dse: new Uint8Array(buf, o + 3 * ns, ns),
        });
      }
      this.bands = {
        meta, frames,
        valid: new Uint8Array(buf, 0, ns),
        lat: new Float32Array(sbuf, 0, n),
        lon: new Float32Array(sbuf, 4 * n, n),
        stride: meta.siteStride,
        ns,
      };
      this._status(null);
      return true;
    } catch (err) {
      this._status(`sea/swell split failed: ${err.message}`);
      setTimeout(() => this._status(null), 4000);
      return false;
    }
  }

  /** Nearest band frame to the grid's current instant. */
  _bandFrame() {
    const want = this._timeAt(this.tf);
    const t = this.bands.meta.times;
    let best = 0, bd = Infinity;
    for (let i = 0; i < t.length; i++) {
      const d = Math.abs(t[i] - want);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  _drawBands() {
    const ctx = this.ctx.bands;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    if (!this.showBands || !this.bands) return;

    const B = this.bands;
    const F = B.frames[this._bandFrame()];
    const hm = B.meta.hsMax;
    // Long enough to read against the fur, short enough not to smother it.
    const LEN = 26;

    // Thin by on-screen spacing. The sites are ~430 m apart, so zoomed into one
    // county they would otherwise stack into a solid band and bury the field
    // this overlay is meant to annotate. Target roughly one vector per 11 px.
    const spacingPx = 430 * this.view.s * DEG / 111320;
    const step = Math.max(1, Math.round(11 / Math.max(spacingPx, 0.01)));

    for (const [key, dkey, col] of [['hsw', 'dsw', 'rgba(96,178,232,0.95)'],
                                    ['hse', 'dse', 'rgba(236,168,92,0.95)']]) {
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      for (let c = 0; c < B.ns; c += step) {
        if (!B.valid[c]) continue;
        const site = c * B.stride;
        const [x, y] = this.toScreen(B.lon[site], B.lat[site]);
        if (x < -30 || y < -30 || x > this.cssW + 30 || y > this.cssH + 30) continue;
        const hs = F[key][c] / 255 * hm;
        if (hs < 0.05) continue;
        const th = F[dkey][c] / 255 * 360 * DEG;
        // Compass bearing the energy comes FROM; screen y runs south.
        const L = LEN * Math.sqrt(hs);
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.sin(th) * L, y - Math.cos(th) * L);
      }
      ctx.stroke();
    }
  }

  // --- animation --------------------------------------------------------

  /**
   * Hold the frame rate by trading strand count for it.
   *
   * The cost per strand varies by an order of magnitude across machines, and
   * the right count is not knowable at build time. Measuring the achieved
   * frame time and steering to it is both more robust than a tuned constant
   * and self-correcting when the user zooms into the denser grid.
   */
  _autoQuality(dt) {
    const p = this.particles;
    this._ema = this._ema ? this._ema * 0.9 + dt * 0.1 : dt;
    if ((this._tick = (this._tick || 0) + 1) < 20) return;
    this._tick = 0;

    const floor = Math.round(p.n * 0.2);
    if (this._ema > 1 / 42 && p.active > floor) {
      p.active = Math.max(floor, Math.round(p.active * 0.9));
    } else if (this._ema < 1 / 57 && p.active < p.n) {
      p.active = Math.min(p.n, Math.round(p.active * 1.06) + 32);
    }
  }

  /** Epoch seconds at a fractional frame index. The axis is non-uniform, so
   *  this is where "frame 7.5" becomes an actual instant. */
  _timeAt(tf) {
    const times = this.active.meta.times;
    const i = Math.max(0, Math.min(times.length - 2, Math.floor(tf)));
    return times[i] + (times[i + 1] - times[i]) * (tf - i);
  }

  /** Fractional frame index at an epoch time, clamped to the window. */
  _tfAtTime(t) {
    const times = this.active.meta.times;
    const n = times.length;
    if (t <= times[0]) return 0;
    if (t >= times[n - 1]) return n - 1;
    let i = 0;
    while (times[i + 1] < t) i++;
    return i + (t - times[i]) / (times[i + 1] - times[i]);
  }

  _loop() {
    const step = (now) => {
      const dt = Math.min(0.05, (now - (this._last || now)) / 1000);
      this._last = now;
      this._autoQuality(dt);

      if (this.playing) {
        // Advance at a constant rate in *simulated* time, not in frames. The
        // gridded axis is not uniform: the nowcast is hourly, then the forecast
        // drops to 6-hourly and runs about four days out. Stepping by frame
        // index would race through the forecast six times faster than the
        // nowcast while the strands claim to move at the group velocity.
        const times = this.active.meta.times;
        const i = Math.max(0, Math.min(times.length - 2, Math.floor(this.tf)));
        const gapSec = Math.max(1, times[i + 1] - times[i]);
        this.tf += dt * SIM_SEC_PER_SEC / gapSec;
        if (this.tf >= this.active.nt - 1) this.tf = 0;
        this._syncTime();
      }
      this._advect(dt);
      if (this.showBands) {
        const bf = this._bandFrame();
        if (bf !== this._lastBandFrame) { this._lastBandFrame = bf; this._drawBands(); }
      }
      this._drawReadout();
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  _advect(dt) {
    const d = this.active, p = this.particles, ctx = this.ctx.fur;

    // Fade the previous strands rather than clearing, which is what turns
    // moving points into fur.
    ctx.globalCompositeOperation = 'destination-out';
    // Strand length is pace x persistence: at ~105 px/s this leaves a tail of
    // roughly 70 px, which is what reads as fur rather than as moving dots.
    ctx.fillStyle = `rgba(0,0,0,${1 - Math.pow(0.974, dt * 60)})`;
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    ctx.globalCompositeOperation = 'source-over';

    const i0 = Math.min(d.nt - 1, Math.floor(this.tf));
    const i1 = Math.min(d.nt - 1, i0 + 1);
    const fT = this.tf - i0;
    const A = expandFrame(d, i0), B = expandFrame(d, i1);

    const pal = PALETTES[this.scalar];
    const seg = this.seg;
    seg.count.fill(0);
    let nseg = 0;
    const wantHs = this.scalar === 'hs';

    // Convert the on-screen pace into grid cells, so strands move at the same
    // apparent speed whether the view is the whole state or one beach.
    const stepCells = (dt * PX_PER_SEC) / Math.max(0.02, this.pxPerCell());
    const o = { v: 0 };

    const cRange = this._colorRange();
    const cLo = cRange[0], cSpan = 1 / (cRange[1] - cRange[0]);

    // Hoisted out of the loop: these are property chains on a hot path that
    // runs ~25k times a frame, and leaving them inline costs several ms.
    const nx = d.nx, ny = d.ny, ncell = d.ncell;
    const cellToWet = d.cellToWet, edgeArr = d.edge;
    const hs0 = d.frames[i0].hs, hs1 = d.frames[i1].hs;
    const hsScale = d.meta.vars.hs.max / 255;
    const AU = A.U, AV = A.V, BU = B.U, BV = B.V, AVAL = A.VALID, BVAL = B.VALID;
    const COH = A.COH;
    const segXY = seg.xy, segB = seg.bucket, segC = seg.count;

    for (let i = 0; i < p.active; i++) {
      const col = p.col[i], row = p.row[i];
      if (col < 0) { this._respawn(i); continue; }

      const okU = sampleBilinear(AU, AVAL, nx, ny, col, row, o);
      const uA = o.v;
      if (!okU) { this._respawn(i); continue; }
      sampleBilinear(AV, AVAL, nx, ny, col, row, o); const vA = o.v;
      sampleBilinear(BU, BVAL, nx, ny, col, row, o); const uB = o.v;
      sampleBilinear(BV, BVAL, nx, ny, col, row, o); const vB = o.v;

      // Lerp the vector components, not the angle: no branch cut at 0/360.
      const u = uA + (uB - uA) * fT;
      const v = vA + (vB - vA) * fT;

      const nc = col + u * stepCells;
      const nr = row + v * stepCells;

      const x0 = this._projX(col), y0 = this._projY(row);
      const x1 = this._projX(nc), y1 = this._projY(nr);

      p.col[i] = nc; p.row[i] = nr;
      p.age[i] += dt * 60;
      if (p.age[i] > p.life[i]) { this._respawn(i); continue; }

      if (x1 < -50 || y1 < -50 || x1 > this.cssW + 50 || y1 > this.cssH + 50) continue;

      // Stochastic transparency: dissolve strands at the model-domain edge and
      // at the ends of their life without needing a per-strand alpha, which
      // would break the color batching.
      const cell = ((nr + 0.5) | 0) * nx + ((nc + 0.5) | 0);
      const wIdx = (cell >= 0 && cell < ncell) ? cellToWet[cell] : -1;
      if (wIdx < 0) continue;
      const edgeA = edgeArr[wIdx] / 255;
      const lifeA = Math.min(1, Math.min(p.age[i], p.life[i] - p.age[i]) / 12);
      // Fade out across bimodal fronts, where the spectral peak is flipping
      // between two wave systems and Dp is not a meaningful single direction.
      let cohA = (COH[cell] - COH_LO) * COH_INV;
      cohA = cohA < 0 ? 0 : cohA > 1 ? 1 : cohA;
      if (Math.random() > edgeA * lifeA * cohA) continue;

      // Color is read at the cell, not bilinearly: it only picks one of 32
      // buckets, so interpolating it would cost two more samples per particle
      // to change nothing visible.
      const scalarVal = wantHs
        ? (hs0[wIdx] + (hs1[wIdx] - hs0[wIdx]) * fT) * hsScale
        : this._scalarAt(d, i0, i1, fT, wIdx);
      if (scalarVal === null) continue;

      let t = (scalarVal - cLo) * cSpan;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const b = t === 1 ? NBUCKET - 1 : (t * NBUCKET) | 0;

      const k = nseg * 4;
      segXY[k] = x0; segXY[k + 1] = y0; segXY[k + 2] = x1; segXY[k + 3] = y1;
      segB[nseg] = b;
      segC[b]++;
      nseg++;
    }

    // Counting sort so each color's segments end up contiguous, and the whole
    // frame costs 32 strokes instead of one per strand.
    let acc = 0;
    for (let b = 0; b < NBUCKET; b++) { seg.offset[b] = acc; acc += seg.count[b]; }
    seg.offset[NBUCKET] = acc;
    const cursor = seg.count;   // reused as a write cursor
    for (let b = 0; b < NBUCKET; b++) cursor[b] = seg.offset[b];
    for (let i = 0; i < nseg; i++) {
      const b = seg.bucket[i];
      const dst = cursor[b]++ * 4, src = i * 4;
      seg.sorted[dst] = seg.xy[src];
      seg.sorted[dst + 1] = seg.xy[src + 1];
      seg.sorted[dst + 2] = seg.xy[src + 2];
      seg.sorted[dst + 3] = seg.xy[src + 3];
    }

    ctx.lineWidth = Math.max(0.6, Math.min(1.5, this.pxPerCell() * 0.5));
    ctx.lineCap = 'round';
    const S = seg.sorted;
    for (let b = 0; b < NBUCKET; b++) {
      const lo = seg.offset[b], hi = seg.offset[b + 1];
      if (hi === lo) continue;
      const [r, g, bl] = pal[b];
      ctx.strokeStyle = `rgba(${r},${g},${bl},0.78)`;
      ctx.beginPath();
      for (let k = lo * 4; k < hi * 4; k += 4) {
        ctx.moveTo(S[k], S[k + 1]);
        ctx.lineTo(S[k + 2], S[k + 3]);
      }
      ctx.stroke();
    }
  }

  /** Tp / Dp read straight from the packed frames (Hs is already expanded). */
  _scalarAt(d, i0, i1, fT, wIdx) {
    if (wIdx < 0) return null;
    const key = this.scalar;
    const max = d.meta.vars[key].max;
    const a = (d.frames[i0][key][wIdx] / 255) * max;
    const b = (d.frames[i1][key][wIdx] / 255) * max;
    if (key === 'dp') {
      // Shortest-arc interpolation across the 0/360 seam.
      let diff = ((b - a + 540) % 360) - 180;
      return (a + diff * fT + 360) % 360;
    }
    return a + (b - a) * fT;
  }

  _colorRange() {
    const d = this.active;
    if (this.scalar === 'hs') {
      const v = d.meta.vars.hs;
      return [Math.max(0, v.p2 ?? 0), Math.max(0.4, v.p98 ?? v.max)];
    }
    if (this.scalar === 'tp') return [5, 22];
    return [180, 330];
  }

  // --- readout ----------------------------------------------------------

  _cellAt(px, py) {
    const d = this.active;
    const [lon, lat] = this.toLonLat(px, py);
    const { lon0, lat0, dlon, dlat } = d.meta.grid;
    const col = Math.round((lon - lon0) / dlon);
    const row = Math.round((lat - lat0) / dlat);
    if (col < 0 || row < 0 || col >= d.nx || row >= d.ny) return null;
    const w = d.cellToWet[row * d.nx + col];
    if (w < 0) return null;
    return { w, col, row, lon, lat };
  }

  /** Readout values at the current instant, interpolated between the two
   *  bracketing frames — the panel ticks hourly through a forecast that is
   *  only stored 6-hourly. Dp takes the shortest arc across the 0/360 seam. */
  _valuesAt(w) {
    const d = this.active;
    const i0 = Math.max(0, Math.min(d.nt - 1, Math.floor(this.tf)));
    const i1 = Math.min(d.nt - 1, i0 + 1);
    const fT = Math.min(1, Math.max(0, this.tf - i0));
    const f0 = d.frames[i0], f1 = d.frames[i1];
    const a = (f0.dp[w] / 255) * 360, b = (f1.dp[w] / 255) * 360;
    const arc = ((b - a + 540) % 360) - 180;
    return {
      hs: ((f0.hs[w] + (f1.hs[w] - f0.hs[w]) * fT) / 255) * d.meta.vars.hs.max,
      tp: ((f0.tp[w] + (f1.tp[w] - f0.tp[w]) * fT) / 255) * d.meta.vars.tp.max,
      dp: (a + arc * fT + 360) % 360,
      depth: d.depth[w],
    };
  }

  _drawReadout() {
    const ctx = this.ctx.ui;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const target = this.pinned || this.hover;
    if (!target) { this._panel(null); return; }

    const d = this.active;
    const { lon0, lat0, dlon, dlat } = d.meta.grid;
    const [sx, sy] = this.toScreen(lon0 + target.col * dlon, lat0 + target.row * dlat);

    ctx.strokeStyle = this.pinned ? 'rgba(88,166,255,0.95)' : 'rgba(230,237,243,0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, this.pinned ? 6 : 4, 0, Math.PI * 2);
    ctx.stroke();

    if (this.pinned) {
      ctx.beginPath();
      ctx.moveTo(sx - 11, sy); ctx.lineTo(sx - 7, sy);
      ctx.moveTo(sx + 7, sy); ctx.lineTo(sx + 11, sy);
      ctx.moveTo(sx, sy - 11); ctx.lineTo(sx, sy - 7);
      ctx.moveTo(sx, sy + 7); ctx.lineTo(sx, sy + 11);
      ctx.stroke();
    }
    this._panel(target);
  }

  _panel(target) {
    const el = this.root.querySelector('#readout');
    if (!target) { el.classList.remove('on'); return; }
    el.classList.add('on');

    const d = this.active;
    const v = this._valuesAt(target.w);
    const ns = target.lat >= 0 ? 'N' : 'S';
    const ew = target.lon >= 0 ? 'E' : 'W';

    el.querySelector('.rd-pos').textContent =
      `${Math.abs(target.lat).toFixed(3)}°${ns}  ${Math.abs(target.lon).toFixed(3)}°${ew}`;
    el.querySelector('.rd-depth').textContent = `${v.depth.toFixed(0)} m depth`;
    el.querySelector('.rd-hs').textContent = `${v.hs.toFixed(2)} m`;
    el.querySelector('.rd-tp').textContent = `${v.tp.toFixed(1)} s`;
    el.querySelector('.rd-dp').textContent = `${v.dp.toFixed(0)}°`;
    el.querySelector('.rd-hint').textContent = this.pinned
      ? 'pinned — click again to release'
      : 'click to pin a time series';

    this._sparkline(target.w);
  }

  /** Hs trace across the window at the pinned cell, drawn against true time.
   *  Index spacing would stretch the hourly nowcast to look as long as the
   *  6-hourly forecast; here an hour is an hour anywhere on the axis. */
  _sparkline(w) {
    const cv = this.root.querySelector('#spark');
    const ctx = cv.getContext('2d');
    const d = this.active;
    const W = cv.width = 232 * this.dpr, H = cv.height = 46 * this.dpr;
    cv.style.width = '232px'; cv.style.height = '46px';
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const w0 = 232, h0 = 46;
    ctx.clearRect(0, 0, w0, h0);

    const vals = [];
    for (let t = 0; t < d.nt; t++) vals.push((d.frames[t].hs[w] / 255) * d.meta.vars.hs.max);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const span = Math.max(0.12, hi - lo);
    const times = d.meta.times;
    const t0 = times[0], tSpan = Math.max(1, times[d.nt - 1] - t0);
    const X = (t) => 3 + ((t - t0) / tSpan) * (w0 - 6);
    const Y = (v) => h0 - 10 - ((v - lo) / span) * (h0 - 20);

    // Divider at the present, when the window straddles it.
    const nowS = Date.now() / 1000;
    if (nowS > t0 && nowS < times[d.nt - 1]) {
      ctx.strokeStyle = 'rgba(139,148,158,0.45)';
      ctx.setLineDash([2, 3]);
      ctx.beginPath(); ctx.moveTo(X(nowS), 4); ctx.lineTo(X(nowS), h0 - 6); ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    vals.forEach((v, i) => (i ? ctx.lineTo(X(times[i]), Y(v)) : ctx.moveTo(X(times[i]), Y(v))));
    ctx.stroke();

    const cx = X(this._timeAt(this.tf)), cy = Y(this._valuesAt(w).hs);
    ctx.fillStyle = '#e6edf3';
    ctx.beginPath(); ctx.arc(cx, cy, 2.6, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = '#8b949e';
    ctx.font = '9px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.fillText(`${hi.toFixed(2)}`, 3, 9);
    ctx.fillText(`${lo.toFixed(2)}`, 3, h0 - 1);
  }

  // --- chrome -----------------------------------------------------------

  _syncTime() {
    const d = this.active;
    // The clock reads the interpolated instant, snapped to the hour. The
    // forecast frames sit six hours apart, but the playback clock should not:
    // the field between them is already being drawn by interpolation, so the
    // clock ticks through every hour it crosses rather than jumping six.
    const tSec = this._timeAt(this.tf);
    const dt = new Date(Math.round(tSec / 3600) * 3.6e6);
    const fmt = new Intl.DateTimeFormat([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
      timeZone: 'America/Los_Angeles',
    });
    this.root.querySelector('#clock').textContent = fmt.format(dt) + ' PT';

    // Say whether the data is actually current rather than asserting it. The
    // payload is rebuilt on a schedule, and a page that keeps printing
    // "nowcast" after that schedule has failed is worse than one that admits
    // it does not know: the window is supposed to straddle the present, so if
    // the whole series now lies in the past the build has stopped running.
    const last = d.meta.times[d.nt - 1] * 1000;
    const staleH = (Date.now() - last) / 3.6e6;
    const badge = this.root.querySelector('#kind');
    if (staleH > 0) {
      const age = staleH < 48 ? `${Math.round(staleH)} h` : `${Math.round(staleH / 24)} d`;
      badge.textContent = `stale · ${age}`;
      badge.className = 'badge st';
      badge.title = `The forecast window ends ${age} ago, so the scheduled rebuild `
                  + `has not run. Values shown are a snapshot, not current conditions.`;
    } else {
      const ahead = tSec * 1000 > Date.now();
      badge.textContent = ahead ? 'forecast' : 'nowcast';
      badge.className = 'badge ' + (ahead ? 'fc' : 'nc');
      badge.title = '';
    }

    const sc = this.root.querySelector('#scrub');
    if (document.activeElement !== sc) {
      sc.value = String((tSec - d.meta.times[0]) / 3600);
    }
  }

  _syncChrome() {
    const d = this.active;
    // The scrub runs in hours, not frame indices: on a non-uniform axis equal
    // slider travel should mean equal time, and each step is one clock hour.
    const times = d.meta.times;
    this.root.querySelector('#scrub').max = String((times[d.nt - 1] - times[0]) / 3600);
    // Resolution is read from the grid spacing itself — which already reflects
    // any decimation applied at build time — so adding a county needs no
    // matching change here.
    // Rounded to a readable step rather than the literal spacing: 0.002 deg is
    // 223 m of latitude, but the product is universally called 200 m.
    const raw = d.meta.grid.dlon * 111320;
    const m = raw >= 1000 ? Math.round(raw / 500) * 500 : Math.round(raw / 50) * 50;
    const res = m >= 1000 ? `~${(m / 1000).toFixed(m % 1000 ? 1 : 0)} km` : `~${m} m`;
    this.root.querySelector('#domain').textContent = `${d.meta.label} · ${res}`;
    this._syncTime();
    this._legend();
  }

  /** Counties, ordered north to south, from the index alone. */
  _buildCountyPicker() {
    const sel = this.root.querySelector('#county');
    const counties = this.index.domains
      .filter((d) => d.id !== 'ca')
      .sort((a, b) => b.bbox.latMax - a.bbox.latMax);
    sel.innerHTML = '<option value="">Zoom to county…</option>';
    for (const c of counties) {
      const o = document.createElement('option');
      o.value = c.id;
      o.textContent = `${c.label} · ${(c.bytes.gz / 1e6).toFixed(1)} MB`;
      sel.appendChild(o);
    }
  }

  _legend() {
    const cv = this.root.querySelector('#legend');
    const ctx = cv.getContext('2d');
    const W = cv.width = 168 * this.dpr, H = cv.height = 10 * this.dpr;
    cv.style.width = '168px'; cv.style.height = '10px';
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const pal = PALETTES[this.scalar];
    for (let i = 0; i < NBUCKET; i++) {
      const [r, g, b] = pal[i];
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect((i / NBUCKET) * 168, 0, 168 / NBUCKET + 1, 10);
    }
    const [lo, hi] = this._colorRange();
    const u = SCALARS[this.scalar].unit;
    this.root.querySelector('#leg-lo').textContent = `${lo.toFixed(lo < 10 ? 1 : 0)}${u}`;
    this.root.querySelector('#leg-hi').textContent = `${hi.toFixed(hi < 10 ? 1 : 0)}${u}`;
    this.root.querySelector('#leg-name').textContent = SCALARS[this.scalar].label;
  }

  // --- input ------------------------------------------------------------

  _bindEvents() {
    const stage = this.root.querySelector('.stage');

    let dragging = false, lastX = 0, lastY = 0, moved = 0;

    stage.addEventListener('pointerdown', (e) => {
      dragging = true; moved = 0;
      lastX = e.clientX; lastY = e.clientY;
      stage.setPointerCapture(e.pointerId);
    });

    stage.addEventListener('pointermove', (e) => {
      const r = stage.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;

      if (dragging) {
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);
        this.view.cx -= dx / this.view.s;
        this.view.cy += dy / this.view.s;
        lastX = e.clientX; lastY = e.clientY;
        this._afterViewChange();
        this._maybeSwitchDomain();
      } else {
        this.hover = this._cellAt(px, py);
      }
    });

    stage.addEventListener('pointerup', (e) => {
      dragging = false;
      stage.releasePointerCapture(e.pointerId);
      if (moved < 5) {
        const r = stage.getBoundingClientRect();
        const hit = this._cellAt(e.clientX - r.left, e.clientY - r.top);
        this.pinned = (this.pinned && hit && this.pinned.w === hit.w) ? null : hit;
      }
    });

    stage.addEventListener('pointerleave', () => { this.hover = null; });

    stage.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = stage.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const [lon, lat] = this.toLonLat(px, py);
      const k = Math.exp(-e.deltaY * 0.0016);
      this.view.s = Math.max(20, Math.min(4.2e6, this.view.s * k));
      // Keep the point under the cursor fixed while zooming.
      this.view.cx = lon * DEG - (px - this.cssW / 2) / this.view.s;
      this.view.cy = mercY(lat) - (this.cssH / 2 - py) / this.view.s;
      this._afterViewChange();
      this._maybeSwitchDomain();
    }, { passive: false });

    this.root.querySelector('#play').addEventListener('click', (e) => {
      this.playing = !this.playing;
      e.currentTarget.textContent = this.playing ? 'Pause' : 'Play';
    });

    this.root.querySelector('#scrub').addEventListener('input', (e) => {
      const t0 = this.active.meta.times[0];
      this.tf = this._tfAtTime(t0 + parseFloat(e.target.value) * 3600);
      this._syncTime();
    });

    this.root.querySelectorAll('[data-scalar]').forEach((b) => {
      b.addEventListener('click', () => {
        this.scalar = b.dataset.scalar;
        this.root.querySelectorAll('[data-scalar]').forEach(
          (o) => o.classList.toggle('on', o === b));
        this._legend();
        this._clearFur();
      });
    });

    this.root.querySelector('#bands').addEventListener('click', async (e) => {
      const b = e.currentTarget;
      if (!this.showBands) {
        b.textContent = 'loading…';
        if (!(await this._loadBands())) { b.textContent = 'Off'; return; }
      }
      this.showBands = !this.showBands;
      b.textContent = this.showBands ? 'On' : 'Off';
      b.classList.toggle('on', this.showBands);
      this._lastBandFrame = -1;
      this._drawBands();
    });

    this.root.querySelector('#density').addEventListener('input', (e) => {
      this.density = parseFloat(e.target.value);
      this._buildParticles();
    });

    this.root.querySelector('#reset').addEventListener('click', () => {
      this._setActive(this.domains.ca);
      this.setView(this.domains.ca.bbox);
      this.pinned = null;
    });

    this.root.querySelector('#county').addEventListener('change', async (e) => {
      const m = this.index.domains.find((d) => d.id === e.target.value);
      if (!m) return;
      const b = m.bbox;
      // Pad the county box so the zoom lands inside the switch threshold and
      // the fur is not pressed against the frame.
      const padLon = (b.lonMax - b.lonMin) * 0.04, padLat = (b.latMax - b.latMin) * 0.04;
      this.setView({
        lonMin: b.lonMin - padLon, lonMax: b.lonMax + padLon,
        latMin: b.latMin - padLat, latMax: b.latMax + padLat,
      });
      await this._maybeSwitchDomain();
    });


    let rt;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => {
        this.resize();
        this._buildParticles();
        this._afterViewChange();
      }, 160);
    });
  }
}

// ---------------------------------------------------------------- bootstrap

window.addEventListener('DOMContentLoaded', () => {
  const app = new SwellView(document.querySelector('#app'));
  window.swell = app;   // handy for poking at the field from the console
  app.init().catch((err) => {
    console.error(err);
    const el = document.querySelector('#status');
    el.style.display = 'block';
    el.textContent = `Could not start: ${err.message}`;
  });
});
