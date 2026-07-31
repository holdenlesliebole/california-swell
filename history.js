/* CDIP MOP alongshore hindcast — the California coast as a two-layer fringe.
 *
 * Unlike the live page, this is not a 2-D field: the hindcast exists only at
 * the 11,594 alongshore MOP sites, a single chain of points along the coast.
 * So each site emits its own strand, streaming inshore along the direction the
 * energy is actually arriving from.
 *
 * Two layers, drawn independently: swell (0.02-0.0813 Hz) and sea
 * (0.09-0.400 Hz). Direction for each is the band-split energy flux, not the
 * mean direction -- across the record the two systems each carry 30-70% of the
 * flux about two thirds of the time while pointing ~22 degrees apart, so a
 * single mean direction is smooth but describes neither.
 *
 * Plain ES2020, no dependencies, no build step.
 */
'use strict';

const DEG = Math.PI / 180;
// Fallback only. The real scale is carried per payload as meta.hsMax, so that
// raising the build's quantization ceiling cannot silently rescale this page.
const HS_MAX_DEFAULT = 12.0;

const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + lat * DEG / 2));
const invMercY = (y) => (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / DEG;

// Swell reads cool and deep, sea warm and bright: the two layers have to stay
// separable at a glance even where they overlap.
const SWELL_COL = [96, 178, 232];
const SEA_COL = [236, 168, 92];

async function fetchBin(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  let buf = await res.arrayBuffer();
  const h = new Uint8Array(buf, 0, 2);
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

/** One payload (a year of daily samples, or an event of hourly ones). */
function decodeFrames(meta, buf) {
  const ns = meta.nsites, nt = meta.times.length;
  const valid = new Uint8Array(buf, 0, ns);
  const need = ns + nt * 4 * ns;
  if (buf.byteLength !== need) {
    throw new Error(`${meta.slug || meta.year}: payload ${buf.byteLength} B, expected ${need} B`);
  }
  const frames = [];
  for (let k = 0; k < nt; k++) {
    const o = ns + k * 4 * ns;
    frames.push({
      hsw: new Uint8Array(buf, o, ns),
      dsw: new Uint8Array(buf, o + ns, ns),
      hse: new Uint8Array(buf, o + 2 * ns, ns),
      dse: new Uint8Array(buf, o + 3 * ns, ns),
    });
  }
  // Per-frame coverage, computed here rather than shipped, so it stays correct
  // for payloads built before this existed.
  //
  // MOP's band estimates depend on which offshore buoys were feeding the model
  // at the time (see waveModelInputSource), and that changed over the record:
  // Del Norte has no sea-band estimate at all until 2005-10, Humboldt covers
  // ~84% of months, Santa Barbara 100%. Those gaps arrive as exactly zero
  // energy, which is indistinguishable from flat calm once drawn -- so the page
  // has to say which it is rather than let the absence speak.
  const nValid = Array.prototype.reduce.call(valid, (a, b) => a + b, 0) || 1;
  const cov = { swell: new Float32Array(nt), sea: new Float32Array(nt) };
  for (let k = 0; k < nt; k++) {
    let sw = 0, se = 0;
    for (let c = 0; c < ns; c++) {
      if (!valid[c]) continue;
      if (frames[k].hsw[c] > 0) sw++;
      if (frames[k].hse[c] > 0) se++;
    }
    cov.swell[k] = sw / nValid;
    cov.sea[k] = se / nValid;
  }

  return { meta, valid, frames, ns, nt, cov, nValid };
}

class Ribbon {
  constructor(root) {
    this.root = root;
    this.ctx = {};
    for (const k of ['coast', 'fur', 'ui']) {
      this.ctx[k] = root.querySelector('#h-' + k).getContext('2d');
    }
    this.layers = { swell: true, sea: true };
    this.playing = true;
    this.tf = 0;
    this.speed = 6;          // samples per second
    this.set = null;         // active decoded payload
    this.dpr = Math.min(2, window.devicePixelRatio || 1);
    this.hover = null;
    this._bind();
  }

  async init() {
    this.sitesMeta = await (await fetch('data/sites.json')).json();
    const sb = await fetchBin('data/sites.bin.gz');
    const n = this.sitesMeta.n;
    this.lat = new Float32Array(sb, 0, n);
    this.lon = new Float32Array(sb, 4 * n, n);

    this.events = await (await fetch('data/history/events.json')).json().catch(() => []);
    this.overview = await (await fetch('data/history/overview/index.json')).json();

    this.resize();
    this.fitView();
    this._buildPicker();

    const first = this.overview.years[this.overview.years.length - 1];
    await this.load({ kind: 'overview', year: first.year });
    this._loop();
  }

  // --- projection -------------------------------------------------------

  fitView() {
    const b = this.sitesMeta.bbox;
    const y0 = mercY(b.latMin), y1 = mercY(b.latMax);
    const xs = (b.lonMax - b.lonMin) * DEG, ys = y1 - y0;
    // Leave room to the west: strands stream in from offshore, so the fringe
    // needs clear space on the seaward side or it renders against the bezel.
    const s = Math.min(this.cssW / (xs * 1.9), this.cssH / (ys * 1.06));
    // Park the coastline right of center by 18% of the width, so the fringe
    // has somewhere to go.
    this.view = {
      cx: (b.lonMin + b.lonMax) / 2 * DEG - 0.18 * this.cssW / s,
      cy: (y0 + y1) / 2,
      s,
    };
    this._project();
  }

  _project() {
    const n = this.sitesMeta.n;
    if (!this.sx) { this.sx = new Float32Array(n); this.sy = new Float32Array(n); }
    const { cx, cy, s } = this.view;
    for (let i = 0; i < n; i++) {
      this.sx[i] = this.cssW / 2 + (this.lon[i] * DEG - cx) * s;
      this.sy[i] = this.cssH / 2 - (mercY(this.lat[i]) - cy) * s;
    }
    this._drawCoast();
  }

  resize() {
    const box = this.root.querySelector('.stage').getBoundingClientRect();
    this.cssW = Math.max(320, Math.floor(box.width));
    this.cssH = Math.max(320, Math.floor(box.height));
    for (const k of ['coast', 'fur', 'ui']) {
      const cv = this.root.querySelector('#h-' + k);
      cv.width = Math.floor(this.cssW * this.dpr);
      cv.height = Math.floor(this.cssH * this.dpr);
      cv.style.width = this.cssW + 'px';
      cv.style.height = this.cssH + 'px';
      this.ctx[k].setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
  }

  // --- data -------------------------------------------------------------

  async load(sel) {
    const path = sel.kind === 'overview'
      ? `data/history/overview/${sel.year}`
      : `data/history/events/${sel.slug}`;
    this._status(`loading ${sel.kind === 'overview' ? sel.year : sel.slug}…`);
    try {
      const meta = await (await fetch(path + '.json')).json();
      const buf = await fetchBin(path + '.bin.gz');
      this.set = decodeFrames(meta, buf);
      this.sel = sel;
      this.tf = 0;
      // Payloads are strided over the full site list; map payload column -> site.
      this.colToSite = new Int32Array(this.set.ns);
      for (let c = 0; c < this.set.ns; c++) this.colToSite[c] = c * meta.siteStride;
      this._buildParticles();
      this._status(null);
      this._syncChrome();
    } catch (err) {
      this._status(`could not load: ${err.message}`);
    }
  }

  _status(m) {
    const el = this.root.querySelector('#h-status');
    el.textContent = m || '';
    el.style.display = m ? 'block' : 'none';
  }

  // --- strands ----------------------------------------------------------

  /**
   * One strand per site per layer. A strand is a point traveling inshore
   * along the arrival direction at its own site; there is no field to
   * integrate through here, so its path is the straight local ray.
   */
  _buildParticles() {
    const ns = this.set.ns;
    this.p = {
      t: new Float32Array(ns * 2),      // 0..1 along the ray, per layer
      jit: new Float32Array(ns * 2),
    };
    for (let i = 0; i < ns * 2; i++) {
      this.p.t[i] = Math.random();
      this.p.jit[i] = 0.6 + Math.random() * 0.8;
    }
  }

  _drawCoast() {
    const c = this.ctx.coast;
    c.clearRect(0, 0, this.cssW, this.cssH);
    c.strokeStyle = 'rgba(200,216,230,0.5)';
    c.lineWidth = 1;
    c.beginPath();
    const n = this.sitesMeta.n;
    let pen = false;
    for (let i = 0; i < n; i++) {
      const x = this.sx[i], y = this.sy[i];
      if (i > 0) {
        // Break the stroke at region joins so the chain does not draw a chord
        // across a bay it never surveyed.
        const d = Math.hypot(x - this.sx[i - 1], y - this.sy[i - 1]);
        if (d > 24) pen = false;
      }
      if (!pen) { c.moveTo(x, y); pen = true; } else c.lineTo(x, y);
    }
    c.stroke();
  }

  // --- animation --------------------------------------------------------

  _loop() {
    const step = (now) => {
      const dt = Math.min(0.05, (now - (this._last || now)) / 1000);
      this._last = now;
      if (this.playing && this.set) {
        this.tf += dt * this.speed;
        if (this.tf >= this.set.nt - 1) this.tf = 0;
        this._syncTime();
      }
      if (this.set) this._draw(dt);
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  _draw(dt) {
    const ctx = this.ctx.fur;
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = `rgba(0,0,0,${1 - Math.pow(0.90, dt * 60)})`;
    ctx.fillRect(0, 0, this.cssW, this.cssH);
    ctx.globalCompositeOperation = 'source-over';

    const S = this.set;
    const HS_MAX = S.meta.hsMax || HS_MAX_DEFAULT;
    const i0 = Math.min(S.nt - 1, Math.floor(this.tf));
    const i1 = Math.min(S.nt - 1, i0 + 1);
    const f = this.tf - i0;
    const A = S.frames[i0], B = S.frames[i1];

    // Strand length in pixels at Hs = 1 m. Rays are drawn seaward of the site,
    // so this also sets how far the fringe stands off the coast.
    const LEN = 42 * (this.view.s / this._baseScale());

    for (const layer of ['swell', 'sea']) {
      if (!this.layers[layer]) continue;
      const sw = layer === 'swell';
      const hA = sw ? A.hsw : A.hse, hB = sw ? B.hsw : B.hse;
      const dA = sw ? A.dsw : A.dse, dB = sw ? B.dsw : B.dse;
      const col = sw ? SWELL_COL : SEA_COL;
      const base = sw ? 0 : S.ns;
      // Swell runs at roughly twice the group speed of local sea, so let the
      // layers advance at different rates -- the contrast is the point.
      const rate = sw ? 0.85 : 0.45;

      ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},0.62)`;
      ctx.lineWidth = 1.1;
      ctx.beginPath();

      for (let c = 0; c < S.ns; c++) {
        if (!S.valid[c]) continue;
        const site = this.colToSite[c];
        const x0 = this.sx[site], y0 = this.sy[site];
        if (x0 < -60 || x0 > this.cssW + 60 || y0 < -60 || y0 > this.cssH + 60) continue;

        const hs = (hA[c] + (hB[c] - hA[c]) * f) / 255 * HS_MAX;
        if (hs < 0.05) continue;

        // Interpolate direction on the unit vector, not the angle.
        const aA = dA[c] / 255 * 360 * DEG, aB = dB[c] / 255 * 360 * DEG;
        const ex = Math.sin(aA) + (Math.sin(aB) - Math.sin(aA)) * f;
        const ey = Math.cos(aA) + (Math.cos(aB) - Math.cos(aA)) * f;
        const m = Math.hypot(ex, ey) || 1;

        // Direction is where the energy comes FROM, so the ray extends seaward
        // from the site along +e and the strand travels back down it.
        const L = LEN * Math.sqrt(hs) * this.p.jit[base + c];
        const k = base + c;
        this.p.t[k] += dt * rate;
        if (this.p.t[k] > 1) this.p.t[k] -= 1;

        const t = this.p.t[k];
        const seg = 0.30;
        const t1 = t, t2 = Math.max(0, t - seg);
        // sin/cos of a compass bearing give (east, north); screen y runs south,
        // so the north component is negated. Without this the northern half of
        // the coast grows its fringe inland.
        const ux = (ex / m) * L, uy = -(ey / m) * L;
        ctx.moveTo(x0 + ux * t1, y0 + uy * t1);
        ctx.lineTo(x0 + ux * t2, y0 + uy * t2);
      }
      ctx.stroke();
    }
    this._drawHover();
  }

  _baseScale() {
    const b = this.sitesMeta.bbox;
    return Math.min(this.cssW / ((b.lonMax - b.lonMin) * DEG * 1.9),
                    this.cssH / ((mercY(b.latMax) - mercY(b.latMin)) * 1.06));
  }

  // --- readout ----------------------------------------------------------

  _nearestSite(px, py) {
    let best = -1, bd = 1e9;
    for (let c = 0; c < this.set.ns; c++) {
      if (!this.set.valid[c]) continue;
      const i = this.colToSite[c];
      const d = (this.sx[i] - px) ** 2 + (this.sy[i] - py) ** 2;
      if (d < bd) { bd = d; best = c; }
    }
    return bd < 900 ? best : -1;
  }

  _drawHover() {
    const ctx = this.ctx.ui;
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const el = this.root.querySelector('#h-readout');
    if (this.hover == null || this.hover < 0) { el.classList.remove('on'); return; }
    el.classList.add('on');

    const c = this.hover, site = this.colToSite[c];
    ctx.strokeStyle = 'rgba(88,166,255,0.95)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(this.sx[site], this.sy[site], 5, 0, Math.PI * 2);
    ctx.stroke();

    const i = Math.round(this.tf);
    const F = this.set.frames[i];
    const HS_MAX = this.set.meta.hsMax || HS_MAX_DEFAULT;
    const q = (a) => a[c] / 255;
    el.querySelector('.rd-site').textContent = this.sitesMeta.names[site];
    el.querySelector('.rd-pos').textContent =
      `${this.lat[site].toFixed(3)}°N ${Math.abs(this.lon[site]).toFixed(3)}°W`;
    el.querySelector('.rd-hsw').textContent = (q(F.hsw) * HS_MAX).toFixed(2) + ' m';
    el.querySelector('.rd-dsw').textContent = Math.round(q(F.dsw) * 360) + '°';
    el.querySelector('.rd-hse').textContent = (q(F.hse) * HS_MAX).toFixed(2) + ' m';
    el.querySelector('.rd-dse').textContent = Math.round(q(F.dse) * 360) + '°';
  }

  // --- chrome -----------------------------------------------------------

  _buildPicker() {
    const sel = this.root.querySelector('#h-source');
    sel.innerHTML = '';
    const og1 = document.createElement('optgroup');
    og1.label = 'Events (hourly)';
    for (const e of this.events) {
      const o = document.createElement('option');
      o.value = `event:${e.slug}`;
      o.textContent = `${e.slug} — statewide p90 ${e.statewideHs90} m`;
      og1.appendChild(o);
    }
    if (this.events.length) sel.appendChild(og1);

    const og2 = document.createElement('optgroup');
    og2.label = 'Years (daily)';
    for (const y of [...this.overview.years].reverse()) {
      const o = document.createElement('option');
      o.value = `overview:${y.year}`;
      o.textContent = `${y.year}`;
      og2.appendChild(o);
    }
    sel.appendChild(og2);

    sel.addEventListener('change', () => {
      const [kind, v] = sel.value.split(':');
      this.load(kind === 'event' ? { kind, slug: v } : { kind, year: +v });
    });
  }

  _syncTime() {
    const S = this.set;
    const i = Math.round(this.tf);
    const d = new Date(S.meta.times[i] * 1000);
    const hourly = S.meta.kind === 'event';
    this.root.querySelector('#h-clock').textContent = new Intl.DateTimeFormat([], {
      year: 'numeric', month: 'short', day: 'numeric',
      ...(hourly ? { hour: '2-digit', minute: '2-digit' } : {}),
      timeZone: 'UTC',
    }).format(d) + (hourly ? ' UTC' : '');
    const sc = this.root.querySelector('#h-scrub');
    if (document.activeElement !== sc) sc.value = String(this.tf);

    // Say when a layer is absent rather than merely quiet.
    const pct = (x) => Math.round(x * 100);
    const cs = pct(S.cov.swell[i]), ce = pct(S.cov.sea[i]);
    const el = this.root.querySelector('#h-cov');
    el.textContent = (cs === 100 && ce === 100)
      ? 'full coverage'
      : `estimated at ${cs}% of sites (swell), ${ce}% (sea)`;
    el.classList.toggle('warn', cs < 95 || ce < 95);
  }

  _syncChrome() {
    const S = this.set;
    this.root.querySelector('#h-scrub').max = String(S.nt - 1);
    this.root.querySelector('#h-kind').textContent =
      S.meta.kind === 'event' ? 'hourly' : 'daily';
    this.root.querySelector('#h-sites').textContent =
      `${S.meta.nvalid.toLocaleString()} sites · every ${S.meta.siteStride}`;
    this.speed = S.meta.kind === 'event' ? 8 : 14;
    this._syncTime();
  }

  _bind() {
    const stage = this.root.querySelector('.stage');
    stage.addEventListener('pointermove', (e) => {
      if (!this.set) return;
      const r = stage.getBoundingClientRect();
      this.hover = this._nearestSite(e.clientX - r.left, e.clientY - r.top);
    });
    stage.addEventListener('pointerleave', () => { this.hover = null; });

    this.root.querySelector('#h-play').addEventListener('click', (e) => {
      this.playing = !this.playing;
      e.currentTarget.textContent = this.playing ? 'Pause' : 'Play';
    });
    this.root.querySelector('#h-scrub').addEventListener('input', (e) => {
      this.tf = parseFloat(e.target.value);
      this._syncTime();
    });
    this.root.querySelectorAll('[data-layer]').forEach((b) => {
      b.addEventListener('click', () => {
        this.layers[b.dataset.layer] = !this.layers[b.dataset.layer];
        b.classList.toggle('on', this.layers[b.dataset.layer]);
      });
    });

    let rt;
    window.addEventListener('resize', () => {
      clearTimeout(rt);
      rt = setTimeout(() => { this.resize(); this.fitView(); }, 160);
    });
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const app = new Ribbon(document.querySelector('#hist'));
  window.ribbon = app;
  app.init().catch((err) => {
    console.error(err);
    const el = document.querySelector('#h-status');
    el.style.display = 'block';
    el.textContent = `Could not start: ${err.message}`;
  });
});
