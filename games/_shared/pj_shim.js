// PlayJev shim. Injected before any game script (Playwright add_init_script).
// Gives every game: a virtual clock (the game only advances when pj.tick runs), a seeded
// Math.random, synthetic key events, muted audio, no CSS animations, and the pj.* API that the
// Python driver and the demo page call. Per-game hooks call pj.register({...}).
(function () {
  if (window.pj) return;
  const real = {
    setTimeout: window.setTimeout.bind(window), clearTimeout: window.clearTimeout.bind(window),
    setInterval: window.setInterval.bind(window), clearInterval: window.clearInterval.bind(window),
    requestAnimationFrame: window.requestAnimationFrame.bind(window),
    Date: window.Date, performanceNow: performance.now.bind(performance), random: Math.random,
  };
  const FRAME_MS = 1000 / 60;
  const EPS = 1e-6;                                  // see tick()
  const q = (x) => Math.round(x * 1e6) / 1e6;        // 1 microsecond lattice for all virtual times
  const EPOCH0 = 1700000000000;
  let now = 0, seq = 0, nextFrameAt = FRAME_MS;
  const timers = new Map(); // id -> {cb, args, at, interval}
  let rafs = new Map();     // id -> cb

  // ---- virtual timers ----
  window.setTimeout = function (cb, delay, ...args) {
    const id = ++seq; const d = Math.max(0, Number(delay) || 0);
    timers.set(id, { cb, args, at: q(now + d), interval: null }); return id;
  };
  window.setInterval = function (cb, delay, ...args) {
    const id = ++seq; const d = Math.max(1, Number(delay) || 1);
    timers.set(id, { cb, args, at: q(now + d), interval: d }); return id;
  };
  window.clearTimeout = window.clearInterval = function (id) { timers.delete(id); };
  window.requestAnimationFrame = function (cb) { const id = ++seq; rafs.set(id, cb); return id; };
  window.cancelAnimationFrame = function (id) { rafs.delete(id); };
  window.webkitRequestAnimationFrame = window.mozRequestAnimationFrame = window.requestAnimationFrame;
  window.webkitCancelAnimationFrame = window.mozCancelAnimationFrame = window.cancelAnimationFrame;

  // ---- virtual Date / performance ----
  const RealDate = real.Date;
  function FakeDate(...a) {
    if (!(this instanceof FakeDate)) return new RealDate(EPOCH0 + now).toString();
    return a.length === 0 ? new RealDate(EPOCH0 + now) : new RealDate(...a);
  }
  FakeDate.prototype = RealDate.prototype;
  FakeDate.now = () => EPOCH0 + now;
  FakeDate.parse = RealDate.parse; FakeDate.UTC = RealDate.UTC;
  window.Date = FakeDate;
  performance.now = () => now;

  function runTimer(t) {
    try { typeof t.cb === 'function' ? t.cb(...t.args) : (0, eval)(String(t.cb)); }
    catch (e) { pj.errors.push(String(e && e.stack || e)); }
  }
  // Times are kept on a 1 microsecond lattice so interval accumulation never lands a hair past a frame boundary
  // (a 33.33 ms setInterval under frames(6) used to fire 2 or 4 times instead of 3, about 0.4 percent of steps).
  function tick(ms) {
    const target = q(now + Math.max(0, ms)); let guard = 0;
    for (;;) {
      let nextTimerAt = Infinity;
      for (const t of timers.values()) if (t.at < nextTimerAt) nextTimerAt = t.at;
      const nextAt = Math.min(nextTimerAt, rafs.size ? nextFrameAt : Infinity);
      if (nextAt > target + EPS) break;
      now = nextAt;
      if (rafs.size && Math.abs(nextAt - nextFrameAt) <= EPS) {
        const batch = rafs; rafs = new Map();
        for (const cb of batch.values()) { try { cb(now); } catch (e) { pj.errors.push(String(e && e.stack || e)); } }
      }
      while (nextFrameAt <= now + EPS) nextFrameAt = q(nextFrameAt + FRAME_MS);
      for (const [id, t] of Array.from(timers.entries())) {
        if (t.at <= now + EPS) {
          if (t.interval) t.at = q(t.at + t.interval); else timers.delete(id);
          runTimer(t);
        }
      }
      if (++guard > 20000) { pj.errors.push('tick guard tripped'); break; }
    }
    now = target;
    while (nextFrameAt <= now + EPS) nextFrameAt = q(nextFrameAt + FRAME_MS);
    pj.t = now;
  }

  // ---- storage shadow ----
  // Pages of one browser context share localStorage and it survives the reload that starts an episode, so a game
  // that saves its level, settings or high score leaks state across pages and episodes. Each page load gets a
  // private in-memory stand-in instead (Proxy so both getItem/setItem and property-style access work).
  function memStorage() {
    const m = new Map();
    const api = {
      getItem: (k) => (m.has(String(k)) ? m.get(String(k)) : null),
      setItem: (k, v) => { m.set(String(k), String(v)); },
      removeItem: (k) => { m.delete(String(k)); },
      clear: () => m.clear(),
      key: (i) => Array.from(m.keys())[i] ?? null,
      get length() { return m.size; },
    };
    return new Proxy(api, {
      get(t, k) { if (k in t) return t[k]; return m.has(String(k)) ? m.get(String(k)) : undefined; },
      set(t, k, v) { if (k in t) return false; m.set(String(k), String(v)); return true; },
      deleteProperty(t, k) { m.delete(String(k)); return true; },
      has(t, k) { return k in t || m.has(String(k)); },
      ownKeys() { return Array.from(m.keys()); },
      getOwnPropertyDescriptor(t, k) { return m.has(String(k)) ? { value: m.get(String(k)), enumerable: true, configurable: true, writable: true } : undefined; },
    });
  }
  for (const name of ['localStorage', 'sessionStorage']) {
    try { Object.defineProperty(window, name, { value: memStorage(), configurable: true, enumerable: true, writable: false }); } catch (e) {}
  }

  // ---- seeded random (mulberry32, seed scrambled splitmix-style so neighbouring seeds diverge at once) ----
  let rs = 0x9e3779b9;
  function rnd() { rs |= 0; rs = rs + 0x6D2B79F5 | 0; let t = Math.imul(rs ^ rs >>> 15, 1 | rs);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
  Math.random = rnd;

  // ---- keys ----
  const KEYS = {
    ArrowLeft: [37, 'ArrowLeft'], ArrowUp: [38, 'ArrowUp'], ArrowRight: [39, 'ArrowRight'], ArrowDown: [40, 'ArrowDown'],
    Space: [32, 'Space', ' '], Enter: [13, 'Enter'], Escape: [27, 'Escape'], Shift: [16, 'ShiftLeft'],
    Control: [17, 'ControlLeft'], Alt: [18, 'AltLeft'], Tab: [9, 'Tab'], Backspace: [8, 'Backspace'],
  };
  for (let i = 0; i < 26; i++) { const c = String.fromCharCode(97 + i); KEYS[c] = [65 + i, 'Key' + c.toUpperCase(), c]; }
  for (let i = 0; i < 10; i++) KEYS[String(i)] = [48 + i, 'Digit' + i, String(i)];
  const held = new Set();
  function keyEvent(type, name, target) {
    const spec = KEYS[name]; if (!spec) throw new Error('unknown key ' + name);
    const [keyCode, code, key = name] = spec;
    const ev = new KeyboardEvent(type, { key, code, keyCode, which: keyCode, bubbles: true, cancelable: true });
    for (const p of ['keyCode', 'which', 'charCode']) Object.defineProperty(ev, p, { get: () => keyCode });
    (target || pj.keyTarget || document.activeElement || document.body || document).dispatchEvent(ev);
  }

  // ---- audio + css ----
  try { HTMLMediaElement.prototype.play = function () { return Promise.resolve(); }; } catch (e) {}
  try { Object.defineProperty(HTMLMediaElement.prototype, 'muted', { get: () => true, set() {} }); } catch (e) {}
  document.addEventListener('DOMContentLoaded', () => {
    const s = document.createElement('style');
    s.textContent = '*,*::before,*::after{transition:none!important;animation:none!important;caret-color:transparent!important}';
    document.head.appendChild(s);
  });

  // ---- pj API ----
  let hook = null, resolveReady;
  const pj = window.pj = {
    version: 1, t: 0, errors: [], keyTarget: null, hook: null,
    real, FRAME_MS, now: () => now, tick, frames: (k) => tick(k * FRAME_MS),
    seed(s) {
      let z = ((Number(s) >>> 0) + 0x9e3779b9) | 0;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b); z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); z ^= z >>> 16;
      rs = z || 1; for (let i = 0; i < 4; i++) rnd();
    },
    keyDown(name, target) { held.add(name); keyEvent('keydown', name, target); },
    keyUp(name, target) { held.delete(name); keyEvent('keyup', name, target); },
    press(name, target) { keyEvent('keydown', name, target); keyEvent('keypress', name, target); keyEvent('keyup', name, target); },
    releaseAll() { for (const k of Array.from(held)) pj.keyUp(k); },
    ready: null,
    register(h) {
      hook = pj.hook = h;
      if (!h.actions || !h.actions.length) throw new Error('hook needs actions');
      pj.actions = h.actions.map((a, i) => ({ index: i, name: a.name, description: a.description || a.name, keys: a.keys || [] }));
      resolveReady && resolveReady(true);
    },
    // Start a fresh episode on an already loaded page. The driver reloads the page before calling this.
    async start(seed) {
      await pj.ready; pj.seed(seed == null ? 1 : seed);
      if (hook.start) await hook.start(seed);
      pj.episodeSteps = 0; pj.startScore = pj.score();
      return pj.obs();
    },
    score() { try { return Number(hook.score()) || 0; } catch (e) { pj.errors.push(String(e)); return 0; } },
    done() { try { return !!hook.done(); } catch (e) { pj.errors.push(String(e)); return true; } },
    // Default action semantics: hold the action's keys for k frames, then release.
    step(i, k) {
      const a = pj.actions[i]; if (!a) throw new Error('bad action ' + i);
      k = k == null ? (hook.stepFrames || 4) : k;
      const before = pj.score();
      if (hook.applyAction) hook.applyAction(i, k);
      else { for (const key of a.keys) pj.keyDown(key); pj.frames(k); pj.releaseAll(); }
      if (hook.afterStep) hook.afterStep();
      pj.episodeSteps++;
      const o = pj.obs(); o.reward = o.score - before; return o;
    },
    obs() {
      const o = { t: now, steps: pj.episodeSteps || 0, score: pj.score(), done: pj.done(), errors: pj.errors.splice(0) };
      if (hook.info) { try { o.info = hook.info(); } catch (e) {} }
      if (hook.canvas) { o.frame = pj.frame(); } else { o.bbox = pj.bbox(); }
      return o;
    },
    frameSize: 448, frameQuality: 0.85,
    // JPEG data URL of the game canvas scaled so the long side is frameSize (aspect kept).
    frame() {
      const src = hook.canvas(); if (!src) return null;
      const sw = src.width, sh = src.height; const s = pj.frameSize / Math.max(sw, sh);
      const w = Math.max(1, Math.round(sw * s)), h = Math.max(1, Math.round(sh * s));
      if (!pj._off || pj._off.width !== w || pj._off.height !== h) { pj._off = document.createElement('canvas'); pj._off.width = w; pj._off.height = h; }
      const ctx = pj._off.getContext('2d'); ctx.imageSmoothingEnabled = true; ctx.drawImage(src, 0, 0, w, h);
      return pj._off.toDataURL('image/jpeg', pj.frameQuality);
    },
    bbox() {
      const el = hook.element ? hook.element() : document.body; const r = el.getBoundingClientRect();
      return { x: Math.max(0, r.left), y: Math.max(0, r.top), width: Math.ceil(r.width), height: Math.ceil(r.height) };
    },
  };
  pj.ready = new Promise((res) => { resolveReady = res; });
})();
