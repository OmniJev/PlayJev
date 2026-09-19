// PlayJev demo page (docs/DEMO.md). Data comes from data.js (window.PJ_DEMO, written by scripts/build_demo.py);
// each game runs in its own same-origin iframe with the training shim and hook, driven over postMessage by the
// bridge (pj_bridge.js), so the picture is the real game replaying the recorded action sequence.
(function () {
  'use strict';
  const D = window.PJ_DEMO;
  if (!D) { document.body.insertAdjacentHTML('afterbegin', '<p>data.js is missing: run scripts/build_demo.py</p>'); return; }
  const qs = new URLSearchParams(location.search);
  const SERVER = normaliseServer(qs.get('server'));
  const WANT = wantedGame(qs.get('game'));   // ?game=snake: the per-game link, opens the featured board on that game
  const INSTRUCTIONS = 'Which move should the player make next?';
  const BOOT_TIMEOUT_MS = 60000, LIVE_TIMEOUT_MS = 30000, END_PAUSE_MS = 1800;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fmtScore = (x) => (x == null ? '' : Number.isInteger(x) ? String(x) : (Math.round(x * 10) / 10).toString());
  const confidence = (p) => { const K = p.length; if (K < 2) return 0; return (Math.max(...p) - 1 / K) / (1 - 1 / K); };
  const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
  const shortPolicy = (p) => String(p).replace(/^playjev-[\d.]+b-/i, '');
  const policyRank = (n) => { const s = String(n).toLowerCase(); return s === 'live' ? -1 : s.startsWith('playjev') ? 0 : s === 'teacher' ? 1 : s === 'random' ? 2 : 3; };
  const remember = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } };
  const remembered = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };
  const NS = 'http://www.w3.org/2000/svg';
  const svgEl = (parent, tag, attrs, cls) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (cls) e.setAttribute('class', cls); parent.appendChild(e); return e; };
  const svgText = (parent, x, y, s, anchor) => { const t = svgEl(parent, 'text', { x, y }, 'txt'); if (anchor) t.setAttribute('text-anchor', anchor); t.textContent = s; return t; };
  const tip = (node, s) => { const t = document.createElementNS(NS, 'title'); t.textContent = s; node.appendChild(t); return node; };

  // ---------------------------------------------------------------- language
  // The page is written in English (index.html) and i18n.js carries the Chinese version. Switching reloads, which
  // keeps one code path for everything drawn once, the SVG axes included. Move names and the prompt are never
  // translated: they are the model's own input.
  const I18N = window.PJ_I18N || { en: {}, zh: {}, games: {} };
  const LANG = (function pickLang() {
    const q = (qs.get('lang') || '').toLowerCase();
    if (q.indexOf('zh') === 0) return 'zh';
    if (q.indexOf('en') === 0) return 'en';
    const saved = remembered('pj-lang');
    if (saved === 'zh' || saved === 'en') return saved;
    return /^zh/i.test(navigator.language || '') ? 'zh' : 'en';
  })();
  const T = (k, fb) => (I18N[LANG] && I18N[LANG][k]) || (I18N.en && I18N.en[k]) || fb || k;
  const zhTitle = (id, fb) => (LANG === 'zh' && I18N.games && I18N.games[id]) || fb;
  const gameTitle = (g) => zhTitle(g.id, g.title);
  (function applyLanguage() {
    document.documentElement.lang = LANG === 'zh' ? 'zh-CN' : 'en';
    if (LANG !== 'en') {
      for (const node of document.querySelectorAll('[data-i18n]')) {
        const v = I18N[LANG][node.dataset.i18n]; if (v) node.textContent = v;
      }
      for (const node of document.querySelectorAll('[data-i18n-html]')) {
        const v = I18N[LANG][node.dataset.i18nHtml]; if (v) node.innerHTML = v;
      }
    }
    const btn = document.getElementById('lang'); if (!btn) return;
    btn.textContent = T('ui.lang'); btn.title = T('ui.langTitle');
    btn.addEventListener('click', () => { remember('pj-lang', LANG === 'zh' ? 'en' : 'zh'); location.reload(); });
  })();

  // A game page that puts focus on itself scrolls this page down, which slides the top of the board under the
  // sticky bar. Hold the page at the top until the reader moves it themselves.
  (function keepTop() {
    if (location.hash) return;
    let free = false;
    const release = () => { free = true; };
    for (const ev of ['wheel', 'touchstart', 'keydown', 'mousedown', 'pointerdown']) {
      window.addEventListener(ev, release, { passive: true, once: true });
    }
    const t0 = Date.now();
    const iv = setInterval(() => {
      if (free || Date.now() - t0 > 15000) { clearInterval(iv); return; }
      if (window.scrollY > 0) window.scrollTo({ top: 0, behavior: 'auto' });
    }, 100);
  })();

  function normaliseServer(s) {
    if (!s) return null;
    try {
      const u = new URL(s, location.href);
      if (u.pathname === '/' || u.pathname === '') u.pathname = '/v1/systemone';
      return u.toString();
    } catch (e) { console.warn('bad ?server= url', s); return null; }
  }

  // ?game=<id> names one game for the featured board. The id is what the README links, and the title is accepted
  // too, so ?game=floppy-bird and ?game=Floppy%20Bird land on the same board.
  function wantedGame(s) {
    if (!s) return null;
    const key = (x) => String(x).toLowerCase().replace(/[^a-z0-9]/g, '');
    const k = key(s), list = D.games || [];
    const g = list.find((x) => key(x.id) === k) || list.find((x) => key(x.title) === k) ||
              (k.length >= 4 ? list.find((x) => key(x.title).includes(k)) : null);
    if (!g) console.warn('unknown ?game=', s);
    return g ? g.id : null;
  }

  // ---------------------------------------------------------------- chrome
  (function theme() {
    const root = document.documentElement, btn = document.getElementById('theme');
    let stored = null; try { stored = localStorage.getItem('pj-theme'); } catch (e) {}
    if (stored === 'dark' || stored === 'light') root.dataset.theme = stored;
    const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    const label = () => { btn.textContent = isDark() ? T('ui.light') : T('ui.dark'); btn.title = isDark() ? T('ui.lightTitle') : T('ui.darkTitle'); };
    btn.addEventListener('click', () => { root.dataset.theme = isDark() ? 'light' : 'dark'; remember('pj-theme', root.dataset.theme); label(); });
    label();
  })();
  (function stickyLine() {
    const top = document.querySelector('.top'); if (!top) return;
    const onScroll = () => top.classList.toggle('stuck', window.scrollY > 4);
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  })();

  // ---------------------------------------------------------------- replay loading (fetch, or a script tag on file://)
  const replayCache = new Map(), replayWaiters = new Map();
  window.PJ_DEMO_REPLAY = (rec) => {
    const key = `${rec.game}/${rec.policy}_${rec.seed}`;
    replayCache.set(key, rec);
    const w = replayWaiters.get(key); if (w) { replayWaiters.delete(key); w.forEach((f) => f.resolve(rec)); }
  };
  async function loadReplay(entry, gameId) {
    const key = `${gameId}/${entry.policy}_${entry.seed}`;
    if (replayCache.has(key)) return replayCache.get(key);
    if (location.protocol !== 'file:') {
      try {
        const r = await fetch('replays/' + entry.file, { cache: 'no-cache' });
        if (r.ok) { const rec = await r.json(); replayCache.set(key, rec); return rec; }
      } catch (e) { /* fall through to the script tag */ }
    }
    return new Promise((resolve, reject) => {
      const list = replayWaiters.get(key) || []; list.push({ resolve, reject }); replayWaiters.set(key, list);
      if (list.length > 1) return;
      const s = document.createElement('script');
      s.src = 'replays/' + entry.file.replace(/\.json$/, '.js');
      s.onerror = () => { replayWaiters.delete(key); list.forEach((f) => f.reject(new Error('could not load ' + s.src))); };
      document.head.appendChild(s);
    });
  }

  // ---------------------------------------------------------------- tiles
  const tiles = [];
  let applyMode = null;  // set by buildGrid: switches the grid between all-at-once and one-at-a-time
  let lazyOff = false;   // pjDemo.startAll() turns the lazy sections on for good, for screenshots and verification

  // Run these tiles only while `host` is near the viewport, so a section far down the page costs nothing until read.
  // `want` decides which of them start when the section comes into view (in one-at-a-time mode, only the active one).
  function lazyWhileVisible(ts, host, want) {
    if (typeof IntersectionObserver !== 'function') return;
    for (const t of ts) t.suspended = true;
    let on = false;
    const io = new IntersectionObserver((entries) => {
      if (lazyOff) return;
      const vis = entries.some((e) => e.isIntersecting);
      if (vis === on) return;
      on = vis;
      for (const t of ts) { if (!vis) t.suspend(); else if (!want || want(t)) t.start(); }
    }, { rootMargin: '300px' });
    io.observe(host);
  }
  window.addEventListener('message', (ev) => {
    const m = ev.data; if (!m || m.pjDemo !== true) return;
    for (const t of tiles) if (t.iframe && ev.source === t.iframe.contentWindow) { t.onMessage(m); return; }
  });

  class Tile {
    constructor(game, entries, opts) {
      opts = opts || {};
      this.game = game; this.entries = entries || []; this.K = game.actions.length;
      this.speed = 1; this.playing = true; this.gen = 0; this.pending = new Map(); this.seq = 0;
      this.iframe = null; this.rect = null; this.errors = []; this.result = null; this.autoAdvance = true; this.suspended = false;
      this.hero = !!opts.hero; this.fixed = !!opts.fixed; this.tag = opts.tag || null;
      // a free board is sized to the game's own shape instead of the grid's square cell
      this.free = !!(opts.hero || opts.free); this.onFit = opts.onFit || null;
      // the table's trained-model policy for this game plays by default; other recordings stay selectable
      const row = ((D.results && D.results.rows) || []).find((r) => r.game === this.game.id);
      const preferred = row && row.model ? row.model.policy : null;
      const rank = (n) => (n === preferred ? -0.5 : policyRank(n));
      const pols = [...new Set(this.entries.map((e) => e.policy))].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
      if (SERVER) pols.unshift('live');
      this.policies = pols; this.policy = pols[0] || null; this.epi = 0;
      this.build();
    }

    build() {
      const g = this.game;
      const root = this.root = el('section', 'tile'); 
      // Only the grid tiles carry data-game, so `.tile[data-game=x]` names exactly one board on the page.
      if (this.tag) root.dataset[this.tag] = g.id; else root.dataset.game = g.id;
      if (this.hero) root.classList.add('hero');

      const head = el('div', 'head');
      head.appendChild(el('span', 'title', gameTitle(g)));
      const mark = el('span', 'mark'); mark.title = T('ui.mismatch'); head.appendChild(mark);
      this.scoreEl = el('span', 'score', T('ui.score') + ' 0'); head.appendChild(this.scoreEl);
      root.appendChild(head);

      // Play and speed sit in the readout, under the name and the score, on the boards that have a readout column.
      // The grid's small tiles keep the same buttons on the board itself, out of the way until the pointer is there.
      this.ppBtns = [];
      const deck = this.deck = el('div', 'deck');
      const deckPP = el('button', 'pp', T('ui.pause'));
      deckPP.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
      this.ppBtns.push(deckPP);
      const deckSeg = el('span', 'seg speeds'); this.speedBtns = {};
      for (const sp of [0.5, 1, 2, 4, 8]) {
        const b = el('button', sp === 1 ? 'on' : '', sp + 'x');
        b.addEventListener('click', () => this.setSpeed(sp));
        deckSeg.appendChild(b); this.speedBtns[sp] = b;
      }
      deck.append(deckPP, el('span', 'decklabel', T('ui.speed')), deckSeg);
      root.appendChild(deck);

      const board = this.board = el('div', 'board');
      this.view = el('div', 'view'); this.overlay = el('div', 'overlay', T('ui.loading'));
      this.view.appendChild(this.overlay); board.appendChild(this.view);
      const c = this.ctrl = el('div', 'controls');
      this.ppBtn = el('button', 'pp', T('ui.pause')); this.ppBtn.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
      this.ppBtns.push(this.ppBtn);
      const restart = el('button', '', T('ui.restart')); restart.addEventListener('click', () => this.restart());
      const next = el('button', '', T('ui.next')); next.title = T('ui.nextTitle'); next.addEventListener('click', () => this.next());
      c.append(this.ppBtn, restart, next);
      if (this.policies.length > 1) {
        const sel = el('select'); sel.title = T('ui.whichrec');
        for (const p of this.policies) { const o = el('option', '', p === 'live' ? T('ui.live') : shortPolicy(p)); o.value = p; o.title = p; sel.appendChild(o); }
        sel.value = this.policy; sel.addEventListener('change', () => this.selectPolicy(sel.value)); c.appendChild(sel); this.sel = sel;
      }
      board.appendChild(c); root.appendChild(board);

      this.bars = el('div', 'bars'); this.rows = [];
      for (const a of g.actions) {
        const row = el('div', 'row'); row.title = a.description;
        const lbl = el('span', 'lbl', a.name), bar = el('span', 'bar'), fill = el('i'), val = el('span', 'val', '');
        bar.appendChild(fill); row.append(lbl, bar, val); this.bars.appendChild(row);
        this.rows.push({ row, fill, val });
      }
      root.appendChild(this.bars);
      this.conf = el('div', 'conf'); root.appendChild(this.conf);

      // The featured board shows the whole episode's confidence under the bars, with the current step marked.
      // Every tile carries one, since any of the ten becomes the featured board in one-at-a-time mode.
      {
        const tr = this.trace = el('div', 'trace');
        tr.appendChild(el('span', 'tlabel', T('ui.trace')));
        const svg = this.traceSvg = document.createElementNS(NS, 'svg');
        svg.setAttribute('aria-hidden', 'true');
        tr.appendChild(svg); root.appendChild(tr);
        this.traceInit(null);
      }

      const foot = el('div', 'foot');
      this.stepEl = el('span', 'step', T('ui.step') + ' 0'); this.recEl = el('span', 'rec', ''); this.warnEl = el('span', 'warn', '');
      foot.append(this.stepEl, this.recEl, this.warnEl); root.appendChild(foot);

      if (!this.entries.length && !SERVER) { restart.disabled = true; next.disabled = true; this.ppBtn.disabled = true; }
      this.renderBars(null, -1); this.renderRec();
    }

    // ---- iframe plumbing
    call(op, args) {
      return new Promise((resolve, reject) => {
        if (!this.iframe || !this.iframe.contentWindow) return reject(new Error('no frame'));
        const id = ++this.seq; this.pending.set(id, { resolve, reject });
        this.iframe.contentWindow.postMessage(Object.assign({ pjDemoCall: true, id, op }, args || {}), '*');
      });
    }
    onMessage(m) {
      if (m.id != null) { const p = this.pending.get(m.id); if (!p) return; this.pending.delete(m.id); m.error ? p.reject(new Error(m.error)) : p.resolve(m.result); return; }
      if (m.ready) { this.actionsSeen = m.actions; if (this._ready) this._ready(); return; }
      if (m.pageerror) { this.noteError(m.pageerror); }
    }
    noteError(msg) {
      this.errors.push(msg); console.warn(`[${this.game.id}] page error: ${msg}`);
      if (this.errors.length === 1) this.warnEl.textContent = T('ui.pageerror');
    }
    async boot() {
      // A fresh frame per episode, as the driver reloads the page before every pj.start.
      for (const p of this.pending.values()) p.reject(new Error('frame replaced')); this.pending.clear();
      if (this.iframe) this.iframe.remove();
      const f = this.iframe = document.createElement('iframe');
      f.width = this.game.viewport.width; f.height = this.game.viewport.height;
      f.style.width = this.game.viewport.width + 'px'; f.style.height = this.game.viewport.height + 'px';
      f.setAttribute('scrolling', 'no'); f.setAttribute('tabindex', '-1'); f.title = gameTitle(this.game) + ' (' + T('ui.gameframe') + ')';
      const ready = new Promise((res) => { this._ready = res; });
      f.src = this.game.entry;
      this.view.insertBefore(f, this.overlay);
      const t = await Promise.race([ready.then(() => 'ok'), sleep(BOOT_TIMEOUT_MS).then(() => 'timeout')]);
      if (t !== 'ok') throw new Error('game did not load (hook never registered)');
      if (this.game.hide) await this.call('style', { css: this.game.hide });
    }
    async measure() {
      const r = await this.call('rect', { selector: this.game.view });
      if (r && r.w > 0 && r.h > 0) this.rect = r;
      this.fit();
    }
    fit() {
      if (!this.iframe) return;
      const r = this.rect || { x: 0, y: 0, w: this.game.viewport.width, h: this.game.viewport.height };
      if (this.free) this.root.style.setProperty('--ar', (r.w / r.h).toFixed(4));
      const BW = this.view.clientWidth || 200, BH = this.view.clientHeight || BW;
      const s = Math.min(BW / r.w, BH / r.h);
      const tx = (BW - r.w * s) / 2 - r.x * s, ty = (BH - r.h * s) / 2 - r.y * s;
      const V = this.game.viewport;
      // Only the game area shows; the rest of the game page (menus, footers, score panels) is clipped away.
      this.iframe.style.clipPath = `inset(${r.y.toFixed(2)}px ${(V.width - r.x - r.w).toFixed(2)}px ${(V.height - r.y - r.h).toFixed(2)}px ${r.x.toFixed(2)}px)`;
      this.iframe.style.transform = `translate(${tx.toFixed(2)}px, ${ty.toFixed(2)}px) scale(${s.toFixed(5)})`;
      // Where the game area actually landed against where it should sit. The frame is laid out at the board's
      // top-left, but a browser can place it a pixel or ten off (a snake board came out half a cell high), and that
      // shows as a cut-off row at the top and a black strip at the bottom. Measure the residual and take it out.
      {
        const vb = this.view.getBoundingClientRect(), ib = this.iframe.getBoundingClientRect();
        const dx = (vb.left + (BW - r.w * s) / 2) - (ib.left + r.x * s);
        const dy = (vb.top + (BH - r.h * s) / 2) - (ib.top + r.y * s);
        if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
          this.iframe.style.transform = `translate(${(tx + dx).toFixed(2)}px, ${(ty + dy).toFixed(2)}px) scale(${s.toFixed(5)})`;
        }
      }
      this.iframe.classList.add('shown');
      // The trace is sized by the room left in the readout, so a resize redraws it at the new size.
      if (this.traceSteps) {
        const b = this.traceSvg.getBoundingClientRect();
        if (Math.abs(b.width - this.traceW) > 1 || Math.abs(b.height - this.traceH) > 1) {
          const last = this.traceLast, steps = this.traceSteps;
          this.traceInit(steps);
          if (last) this.traceAt(last[0], last[1], last[2]);
        }
      }
      if (this.onFit) this.onFit(r);
    }
    showOverlay(text, err) { if (text == null) { this.overlay.hidden = true; return; } this.overlay.hidden = false; this.overlay.textContent = text; this.overlay.classList.toggle('err', !!err); }

    // ---- rendering
    renderBars(p, taken, s2) {
      for (let i = 0; i < this.K; i++) {
        const v = p ? p[i] : 0; const r = this.rows[i];
        r.fill.style.width = (Math.max(0, Math.min(1, v)) * 100).toFixed(1) + '%';
        r.val.textContent = p ? v.toFixed(2) : '';
        r.row.classList.toggle('taken', i === taken);
      }
      this.bars.classList.toggle('s2', !!s2);
      this.conf.innerHTML = p ? `confidence <b>${confidence(p).toFixed(2)}</b>` + (s2 ? ' <span class="s2tag">the teacher decided this one</span>' : '')
                              + (this.latency != null ? ` <span>latency ${Math.round(this.latency)} ms</span>` : '') : ' ';
    }
    // Draw one episode's confidence: the whole line faint, the part already played in blue, coral ticks on the
    // steps the teacher decided. traceAt() then only moves the clip and the dot, once per step.
    traceInit(steps) {
      if (!this.trace) return;
      const svg = this.traceSvg; while (svg.firstChild) svg.removeChild(svg.firstChild);
      this.traceRect = null; this.traceLast = null; this.traceSteps = steps || null;
      // The box is whatever the readout column has left over, and the line is drawn in its own pixels.
      const box = svg.getBoundingClientRect();
      const W = this.traceW = Math.max(120, Math.round(box.width) || 240);
      const H = this.traceH = Math.max(64, Math.round(box.height) || 96);
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      const TOP = 5, BASE = H - 16;
      svgEl(svg, 'line', { x1: 0, y1: BASE, x2: W, y2: BASE }, 'tbase');
      if (!steps || steps.length < 2) return;
      const N = steps.length;
      const X = (i) => (i / (N - 1)) * W, Y = (c) => TOP + (1 - Math.max(0, Math.min(1, c))) * (BASE - TOP);
      // A long episode has more steps than the box has pixels, so each point is the mean of the steps behind it:
      // taking every nth step instead would turn a thousand-step game into noise.
      const M = Math.max(2, Math.min(N, Math.round(W / 2)));
      let d = '', x0 = 0, x1 = 0;
      for (let b = 0; b < M; b++) {
        const s0 = Math.floor((b * N) / M), s1 = Math.max(s0 + 1, Math.floor(((b + 1) * N) / M));
        let sum = 0;
        for (let i = s0; i < s1; i++) sum += confidence(steps[i].p);
        const x = X((s0 + s1 - 1) / 2);
        if (!d) x0 = x;
        x1 = x;
        d += (d ? 'L' : 'M') + x.toFixed(1) + ' ' + Y(sum / (s1 - s0)).toFixed(1);
      }
      // The same shape closed down to the baseline: the fill is what carries the line across a tall box.
      const da = d + 'L' + x1.toFixed(1) + ' ' + BASE + 'L' + x0.toFixed(1) + ' ' + BASE + 'Z';
      svgEl(svg, 'path', { d: da }, 'tafull');
      svgEl(svg, 'path', { d }, 'tfull');
      this.traceUid = (this.traceUid || 0) + 1;
      const cid = 'tc-' + this.game.id + '-' + this.traceUid;
      const clip = document.createElementNS(NS, 'clipPath'); clip.setAttribute('id', cid); svg.appendChild(clip);
      this.traceRect = svgEl(clip, 'rect', { x: -4, y: 0, width: 0, height: H });
      svgEl(svg, 'path', { d: da, 'clip-path': 'url(#' + cid + ')' }, 'taplayed');
      svgEl(svg, 'path', { d, 'clip-path': 'url(#' + cid + ')' }, 'tplayed');
      let ticks = '';
      for (let i = 0; i < N; i++) if (steps[i].h) ticks += 'M' + X(i).toFixed(1) + ' ' + (BASE + 5) + 'v7';
      if (ticks) svgEl(svg, 'path', { d: ticks }, 'ttick');
      this.traceNow = svgEl(svg, 'line', { x1: -9, y1: TOP, x2: -9, y2: BASE }, 'tnow');
      this.traceDot = svgEl(svg, 'circle', { cx: -9, cy: -9, r: 3.2 }, 'tdot');
      this.traceX = X; this.traceY = Y;
    }
    traceAt(i, c, h) {
      if (!this.traceRect) return;
      this.traceLast = [i, c, h];
      const x = this.traceX(i), y = this.traceY(c);
      this.traceRect.setAttribute('width', (x + 4).toFixed(1));
      this.traceNow.setAttribute('x1', x.toFixed(1)); this.traceNow.setAttribute('x2', x.toFixed(1));
      this.traceDot.setAttribute('cx', x.toFixed(1)); this.traceDot.setAttribute('cy', y.toFixed(1));
      this.traceDot.classList.toggle('h', !!h);
    }
    renderRec() {
      const pol = this.policy;
      if (!pol) { this.recEl.textContent = T('ui.norec'); this.recEl.classList.remove('random'); return; }
      this.recEl.textContent = pol === 'live' ? 'live: ' + new URL(SERVER).host : shortPolicy(pol);
      this.recEl.title = pol === 'live' ? SERVER : T('ui.recording') + ': ' + pol;
      this.recEl.classList.toggle('random', pol === 'random');
    }
    renderStatus(i, total, score, tail) {
      this.stepEl.textContent = T('ui.step') + ' ' + i + (total != null ? ' / ' + total : '') + (tail ? ' ' + tail : '');
      this.scoreEl.textContent = T('ui.score') + ' ' + fmtScore(score);
    }
    // One-at-a-time mode promotes the running tile to the featured board: its own shape, the readout beside it.
    setFeatured(on) {
      on = !!on; if (this.featured === on) return;
      this.featured = on; this.free = on || this.hero;
      this.root.classList.toggle('hero', this.free);
      if (!this.free) this.root.style.removeProperty('--ar');
      this.fit();
    }
    setMismatch(text) { this.root.classList.toggle('mismatch', !!text); this.warnEl.textContent = text || ''; }

    // ---- controls
    setSpeed(s) { this.speed = s; for (const k in this.speedBtns) this.speedBtns[k].classList.toggle('on', Number(k) === s); }
    pause() { this.playing = false; for (const b of this.ppBtns) b.textContent = T('ui.play'); }
    play() { if (this.suspended) return; this.playing = true; for (const b of this.ppBtns) b.textContent = T('ui.pause'); if (this._resume) { const r = this._resume; this._resume = null; r(); } else if (!this.running) this.start(); }
    // Stop this tile and throw its game frame away. Ten live games is more than most machines can paint at once, so
    // the page can keep one running and hold the rest here; start() brings a tile back.
    suspend() {
      this.suspended = true; this.gen++;
      for (const p of this.pending.values()) p.reject(new Error('tile suspended')); this.pending.clear();
      if (this.iframe) { this.iframe.remove(); this.iframe = null; }
      this.running = false; this.rect = null; this._resume = null;
      this.renderBars(null, -1); this.traceInit(null); this.setMismatch(''); this.showOverlay(T('ui.paused'));
    }
    waitResume() { return new Promise((res) => { this._resume = res; }); }
    currentEntries() { return this.entries.filter((e) => e.policy === this.policy); }
    selectPolicy(p) { this.policy = p; this.epi = 0; this.renderRec(); if (this.sel) this.sel.value = p; if (!this.suspended) this.start(); }
    restart() { this.start(); }
    next() { const n = this.currentEntries().length; if (n) this.epi = (this.epi + 1) % n; this.start(); }

    // Start (or restart) the current episode; keeps the page cycling through the recordings while playing.
    async start() {
      const gen = ++this.gen; this.running = true; this._resume = null; this.suspended = false;
      if (!this.playing) this.play();
      if (location.protocol === 'file:' && this.game.modules) {
        this.showOverlay('this game is written as ES modules, which browsers refuse to load from file://; open the page over http (a local server or GitHub Pages)', true);
        this.running = false; return;
      }
      try {
        while (gen === this.gen) {
          let res;
          if (this.policy === 'live') res = await this.runLive(gen);
          else if (this.currentEntries().length) res = await this.runEpisode(this.currentEntries()[this.epi % this.currentEntries().length], gen);
          else { await this.idle(gen); return; }
          if (gen !== this.gen) return;
          if (!this.autoAdvance) return;
          await sleep(END_PAUSE_MS); if (gen !== this.gen) return;
          if (!this.playing) await this.waitResume(); if (gen !== this.gen) return;
          if (this.policy !== 'live') { const n = this.currentEntries().length; this.epi = (this.epi + 1) % Math.max(1, n); }
        }
      } catch (e) {
        if (gen !== this.gen) return;
        console.error(`[${this.game.id}]`, e); this.showOverlay(String(e.message || e), true);
      } finally { if (gen === this.gen) this.running = false; }
    }

    // Game loaded and started with seed 1 but nothing to replay yet.
    async idle(gen) {
      this.showOverlay(T('ui.loading')); await this.boot(); if (gen !== this.gen) return;
      const obs = await this.call('start', { seed: 1 }); await this.measure(); this.showOverlay(null);
      this.renderStatus(0, null, obs.score); this.renderBars(null, -1); this.traceInit(null); this.renderRec();
    }

    // Replay one recording; resolves with the verification record when the episode is over.
    async runEpisode(entry, gen, opts) {
      opts = opts || {};
      const g = this.game; const t0 = performance.now();
      this.latency = null; this.setMismatch(''); this.renderRec(); this.errors = [];
      this.showOverlay(T('ui.loading'));
      const rec = await loadReplay(entry, g.id); if (gen !== this.gen) return null;
      await this.boot(); if (gen !== this.gen) return null;
      let obs = await this.call('start', { seed: rec.seed }); if (gen !== this.gen) return null;
      await this.measure(); this.showOverlay(null);
      const steps = rec.steps, N = steps.length; let i = 0, divergedAt = null, handed = 0;
      this.traceInit(steps);
      this.renderStatus(0, N, obs.score); this.renderBars(null, -1);
      const stepMs = g.step_ms || 150;
      while (i < N) {
        if (gen !== this.gen) return null;
        if (!this.playing) { await this.waitResume(); if (gen !== this.gen) return null; }
        const tick = performance.now(); const st = steps[i];
        // The bars belong to the picture on screen: decision i was made from the frame before action i, so show
        // it first, hold for the step's duration, then apply the move (otherwise the model looks one beat late).
        if (st.h) handed++;
        this.renderBars(st.p, st.a, !!st.h); this.renderStatus(i, N, obs.score, handed ? `(teacher took ${handed} of ${i + 1})` : '');
        this.traceAt(i, confidence(st.p), !!st.h);
        const speed0 = opts.speed || this.speed;
        const hold = stepMs / speed0 - (performance.now() - tick); if (hold > 0) await sleep(hold);
        if (gen !== this.gen) return null;
        if (!this.playing) { await this.waitResume(); if (gen !== this.gen) return null; }
        obs = await this.call('step', { a: st.a }); if (gen !== this.gen) return null;
        i++;
        this.renderStatus(i, N, obs.score, handed ? `(teacher took ${handed} of ${i})` : '');
        if (obs.errors && obs.errors.length) for (const e of obs.errors) this.noteError(e);
        if (divergedAt == null && st.score != null && obs.score !== st.score) {
          divergedAt = i; console.warn(`[${g.id}] ${entry.policy}_${entry.seed}: score ${obs.score} at step ${i}, recording says ${st.score}`);
        }
        if (obs.done) break;
      }
      const match = obs.score === rec.final_score && (i === N);
      const result = { game: g.id, policy: rec.policy, seed: rec.seed, steps_played: i, steps_recorded: N, final: obs.score,
                       expected: rec.final_score, done: !!obs.done, truncated: !!rec.truncated, match, diverged_at: divergedAt,
                       errors: this.errors.slice(), seconds: Math.round((performance.now() - t0) / 100) / 10 };
      if (!match) {
        console.warn(`[${g.id}] replay mismatch:`, result);
        this.setMismatch(i < N ? `ended at step ${i} of ${N}, score ${fmtScore(obs.score)}, recording ${fmtScore(rec.final_score)}` : `score ${fmtScore(obs.score)}, recording says ${fmtScore(rec.final_score)}`);
      } else if (divergedAt != null) { this.setMismatch(`score path differed at step ${divergedAt}, same final score`); }
      this.renderStatus(i, N, obs.score, (obs.done ? '(over)' : i === N ? '(end of recording)' : '') + (handed ? ` the teacher took ${handed} of ${i}` : ''));
      this.result = result; return result;
    }

    // Live mode: the game frame goes to a PlayJev/OpenJev server, its probabilities pick the move.
    async runLive(gen) {
      const g = this.game; this.setMismatch(''); this.renderRec(); this.errors = []; this.showOverlay('loading');
      await this.boot(); if (gen !== this.gen) return null;
      await this.call('frames', { on: true });
      const seed = 1 + Math.floor(Math.random() * 1e6);
      let obs = await this.call('start', { seed }); if (gen !== this.gen) return null;
      await this.measure(); this.showOverlay(null);
      const criteria = {}; for (const a of g.actions) criteria[a.name] = a.description;
      let i = 0; const stepMs = g.step_ms || 150, cap = g.max_steps || 3000; let prevFrame = null;
      this.renderStatus(0, null, obs.score, 'live'); this.renderBars(null, -1);
      while (!obs.done && i < cap) {
        if (gen !== this.gen) return null;
        if (!this.playing) { await this.waitResume(); if (gen !== this.gen) return null; }
        const tick = performance.now();
        const frame = obs.frame || (await this.call('frame'));
        // (previous, current) once there is a previous frame: a single-frame server takes the last one, a two-frame server uses both
        const body = { model: 'playjev-latest', state: { frames: prevFrame ? [prevFrame, frame] : [frame] }, questions: { q: { type: 'choice', instructions: INSTRUCTIONS, criteria } } };
        prevFrame = frame;
        const ctl = new AbortController(); const timer = setTimeout(() => ctl.abort(), LIVE_TIMEOUT_MS);
        let p;
        try {
          const r = await fetch(SERVER, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: ctl.signal });
          if (!r.ok) throw new Error('server answered ' + r.status);
          const j = await r.json(); const probs = j.answers.q.probabilities;
          p = g.actions.map((a) => Number(probs[a.name]) || 0);
        } catch (e) { clearTimeout(timer); throw new Error('live server: ' + (e.message || e)); }
        clearTimeout(timer);
        this.latency = performance.now() - tick;
        if (gen !== this.gen) return null;
        const a = argmax(p);
        this.renderBars(p, a);  // the answer to the picture on screen, shown before the move is applied
        const hold = stepMs / this.speed - (performance.now() - tick); if (hold > 0) await sleep(hold);
        if (gen !== this.gen) return null;
        obs = await this.call('step', { a }); if (gen !== this.gen) return null;
        i++; this.renderStatus(i, null, obs.score, 'live');
      }
      this.renderStatus(i, null, obs.score, obs.done ? '(over)' : '(step cap)');
      return { game: g.id, policy: 'live', seed, steps_played: i, final: obs.score, done: !!obs.done };
    }

    // Verification: every recording of this tile at a given speed, sequentially; returns the records.
    async verify(speed, policies) {
      // The featured board is a second copy of a game that the grid already verifies; it steps aside instead.
      if (this.hero) { this.gen++; this.pause(); return []; }
      const out = []; const gen = ++this.gen; this.running = true; this.autoAdvance = false; this.playing = true; for (const b of this.ppBtns) b.textContent = T('ui.pause');
      try {
        for (let k = 0; k < this.entries.length; k++) {
          const e = this.entries[k]; if (policies && !policies.includes(e.policy)) continue;
          this.policy = e.policy; if (this.sel) this.sel.value = e.policy; this.epi = this.currentEntries().indexOf(e);
          let r;
          try { r = await this.runEpisode(e, gen, { speed }); }
          catch (err) { r = { game: this.game.id, policy: e.policy, seed: e.seed, match: false, error: String(err.message || err), errors: this.errors.slice() }; this.showOverlay(String(err.message || err), true); }
          if (gen !== this.gen) return out;
          if (r) out.push(r);
        }
      } finally { if (gen === this.gen) this.running = false; this.autoAdvance = true; }
      return out;
    }
  }

  // ---------------------------------------------------------------- the featured board
  // One large board at the top with a name for every game next to it. Switching throws the old frame away and
  // builds a new tile, so only one featured game is ever running.
  function buildStage() {
    const stage = document.getElementById('stage'), picker = document.getElementById('picker');
    if (!stage || !picker) return;
    const games = (D.games || []).slice();
    if (!games.length) return;
    const nRec = (g) => ((D.replays && D.replays[g.id]) || []).length;
    let cur = null;
    const show = (g, start) => {
      if (cur) { cur.suspend(); cur.root.remove(); const i = tiles.indexOf(cur); if (i >= 0) tiles.splice(i, 1); }
      const t = new Tile(g, (D.replays && D.replays[g.id]) || [], { hero: true, tag: 'heroGame' });
      tiles.push(t); stage.appendChild(t.root); cur = t;
      for (const b of picker.children) b.classList.toggle('on', b.dataset.game === g.id);
      remember('pj-hero', g.id);
      if (start) t.start();
    };
    for (const g of games) {
      const b = el('button', '', gameTitle(g)); b.type = 'button'; b.dataset.game = g.id;
      b.addEventListener('click', () => show(g, true)); picker.appendChild(b);
    }
    const want = WANT || remembered('pj-hero');
    show(games.find((g) => g.id === want && nRec(g)) || games.find((g) => g.id === 'snake' && nRec(g)) || games.find(nRec) || games[0], false);
  }

  // Two boards of the same game from the same seed, side by side: the model on its own and the model with the
  // teacher behind it. Everything else is identical, so the only difference on screen is who took the hard steps.
  function buildDuel() {
    const host = document.getElementById('duel'); if (!host) return null;
    const block = host.closest('.duel-block') || host;
    const rows = (D.results && D.results.rows) || [];
    const table = [...new Set(rows.filter((r) => r.model).map((r) => r.model.policy))];
    let cand = [];
    for (const g of D.games || []) {
      const eps = (D.replays && D.replays[g.id]) || [];
      for (const s2 of [...new Set(eps.map((e) => e.policy).filter((p) => /-s2$/.test(p)))]) {
        const base = s2.replace(/-s2$/, '');
        for (const a of eps.filter((e) => e.policy === base)) {
          const b = eps.find((e) => e.policy === s2 && e.seed === a.seed);
          if (b && a.final_score != null && b.final_score > a.final_score) cand.push({ g, a, b, base, gap: b.final_score / Math.max(1, a.final_score) });
        }
      }
    }
    if (!cand.length) { block.remove(); return null; }
    // one checkpoint for the pair, the table's if it was recorded with a teacher behind it
    const bases = [...new Set(cand.map((c) => c.base))];
    const useBase = bases.find((b) => table.includes(b)) || bases.sort((x, y) => policyRank(x) - policyRank(y) || y.localeCompare(x))[0];
    cand = cand.filter((c) => c.base === useBase);
    // Tetris first, because a board filling up reads at a glance; then the seed whose gap is the median of that
    // game's seeds, so the pair on screen is a typical one rather than the best one.
    const byGame = {}; for (const c of cand) (byGame[c.g.id] = byGame[c.g.id] || []).push(c);
    const gid = byGame.tetris ? 'tetris' : Object.keys(byGame).sort((x, y) => byGame[y].length - byGame[x].length)[0];
    const seeds = byGame[gid].sort((x, y) => x.gap - y.gap);
    const { g, a, b } = seeds[Math.floor(seeds.length / 2)];
    const nSeeds = seeds.length;
    const sameAsTable = table.includes(useBase);
    const capH = () => Math.min(360, 0.42 * window.innerHeight);
    const sizeHost = (r) => {
      const panel = Math.max(272, Math.min(520, Math.round(capH() * (r.w / r.h)) + 30));
      host.style.maxWidth = (panel * 2 + 18) + 'px';
    };
    const mk = (entry, title, lead) => {
      const t = new Tile(g, [entry], { fixed: true, free: true, onFit: sizeHost, tag: 'duelGame' }); t.autoAdvance = true;
      t.root.querySelector('.head .title').textContent = title;
      if (lead) t.root.classList.add('lead-board');
      tiles.push(t); host.appendChild(t.root); return t;
    };
    const ta = mk(a, T('ui.duelAlone'), false);
    const tb = mk(b, T('ui.duelS2'), true);
    lazyWhileVisible([ta, tb], host);
    const note = document.getElementById('duel-note');
    if (note) note.textContent = `Both boards are ${g.title} from seed ${a.seed}, the same game with the same `
      + `pieces in the same order. On the left the model plays every step and scores ${fmtScore(a.final_score)}. On `
      + `the right it keeps the steps it is confident about and the teacher takes the rest, and the same seed reaches `
      + `${fmtScore(b.final_score)}${b.truncated ? ' by the time the recording hits the step cap' : ''}. This seed is `
      + `the median of the ${nSeeds} recorded here, not the widest gap.`
      + (sameAsTable ? '' : ` Both recordings are ${useBase}, the checkpoint the handover sweep below ran on, `
        + `rather than the newer one in the table above.`);
    return { g, a, b };
  }

  // ---------------------------------------------------------------- page sections
  function buildGrid() {
    const grid = document.getElementById('grid');
    const section = document.getElementById('gallery');
    for (const g of D.games) { const t = new Tile(g, (D.replays && D.replays[g.id]) || []); tiles.push(t); grid.appendChild(t.root); }
    const all = document.getElementById('all');
    document.getElementById('all-pp').addEventListener('click', (ev) => {
      const gt = tiles.filter((t) => t.root.parentElement === grid);
      const anyPlaying = gt.some((t) => t.playing);
      for (const t of gt) (anyPlaying ? t.pause() : t.play());
      ev.target.textContent = anyPlaying ? 'play all' : 'pause all';
    });
    for (const b of all.querySelectorAll('[data-speed]')) b.addEventListener('click', () => {
      for (const t of tiles) t.setSpeed(Number(b.dataset.speed));
      for (const x of all.querySelectorAll('[data-speed]')) x.classList.toggle('on', x === b);
    });
    // who plays on every tile: the model alone, the model with the teacher behind it (the teacher takes the
    // low-confidence steps), or random; a tile without such a recording keeps what it has
    const pick = (t, which) => {
      const row = ((D.results && D.results.rows) || []).find((r) => r.game === t.game.id);
      const base = row && row.model ? row.model.policy : (t.policies.find((p) => String(p).startsWith('playjev')) || null);
      if (which === 'random') return t.policies.includes('random') ? 'random' : null;
      if (which === 's2') return t.policies.find((p) => p === base + '-s2') || t.policies.find((p) => /-s2$/.test(p)) || null;
      return base;
    };
    const hasS2 = tiles.some((t) => pick(t, 's2'));
    for (const b of all.querySelectorAll('[data-who]')) {
      if (b.dataset.who === 's2' && !hasS2) { b.remove(); continue; }
      b.addEventListener('click', () => {
        for (const t of tiles) { if (t.fixed) continue; const p = pick(t, b.dataset.who); if (p && p !== t.policy) t.selectPolicy(p); }
        for (const x of all.querySelectorAll('[data-who]')) x.classList.toggle('on', x === b);
      });
    }
    // How many games run at once. Ten live iframes is more than a laptop can paint, so the page offers to keep one
    // game running and collapse the other nine to names; the choice is remembered per browser.
    const gridTiles = () => tiles.filter((t) => t.root.parentElement === grid);
    const modeBtns = [...all.querySelectorAll('[data-mode]')];
    let solo = false, activeTile = null;
    const mark = (t, on) => { t.root.classList.toggle('active', on); t.setFeatured(on); };
    function activate(t) {
      if (!solo || !t || t === activeTile) return;
      for (const o of gridTiles()) if (o !== t) { mark(o, false); if (!o.suspended) o.suspend(); }
      activeTile = t; mark(t, true); remember('pj-tile-game', t.game.id); t.start();
    }
    applyMode = (next, initial) => {
      solo = !!next; grid.classList.toggle('solo', solo); remember('pj-tile-mode', solo ? 'solo' : 'all');
      for (const b of modeBtns) b.classList.toggle('on', (b.dataset.mode === 'solo') === solo);
      const gt = gridTiles();
      if (solo) {
        const want = WANT || remembered('pj-tile-game');
        const keep = (activeTile && gt.includes(activeTile) ? activeTile : null) || gt.find((o) => o.game.id === want) || gt[0];
        for (const o of gt) if (o !== keep) { mark(o, false); if (!o.suspended) o.suspend(); else o.suspended = true; }
        activeTile = null; if (keep) { if (initial) { mark(keep, true); activeTile = keep; remember('pj-tile-game', keep.game.id); } else activate(keep); }
      } else {
        activeTile = null;
        for (const o of gt) { mark(o, false); if (o.suspended && !initial) o.start(); }
      }
    };
    for (const b of modeBtns) b.addEventListener('click', () => applyMode(b.dataset.mode === 'solo', false));
    for (const t of gridTiles()) t.root.addEventListener('click', (ev) => {
      if (!solo || ev.target.closest('.controls')) return;   // the tile's own buttons keep working
      activate(t);
    });
    applyMode(remembered('pj-tile-mode') === 'solo', true);
    // The ten only run while the section is on screen: the featured board above it is what loads first.
    lazyWhileVisible(gridTiles(), section, (t) => !solo || t.root.classList.contains('active'));

    let resizeTimer = null;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => tiles.forEach((t) => t.fit()), 60); });
    const notice = document.getElementById('notice');
    const nRec = Object.values(D.replays || {}).reduce((s, l) => s + l.length, 0);
    if (SERVER) { notice.textContent = T('ui.liveMode').replace('{url}', SERVER); }
    else if (!nRec) notice.textContent = T('ui.noRecs');
    // Everything else the reader needs about the recordings is in the "How it works" section, so the lead stays short.
  }

  function buildResults() {
    const R = D.results; if (!R || !R.rows) return;
    const tb = document.getElementById('results-body');
    const cell = (v, cls) => { const td = el('td', cls); if (v == null) { td.textContent = T('ui.pending'); td.classList.add('pending'); } else td.textContent = v; return td; };
    for (const r of R.rows) {
      const tr = el('tr');
      const name = el('td'); const a = el('a', '', zhTitle(r.game, r.title)); a.href = 'https://github.com/' + (r.upstream || ''); a.target = '_blank'; a.rel = 'noopener'; name.appendChild(a);
      name.appendChild(el('span', 'sub', ` ${r.k} moves`)); tr.appendChild(name);
      tr.appendChild(cell(r.random ? fmtScore(r.random.score) : null, 'num'));
      tr.appendChild(cell(r.model ? fmtScore(r.model.score) : null, 'num'));
      tr.appendChild(cell(r.teacher && r.teacher.score != null ? fmtScore(r.teacher.score) : null, 'num'));
      tr.appendChild(cell(r.vs_teacher != null ? (Math.round(r.vs_teacher * 100) / 100).toFixed(2) : null, 'num'));
      tb.appendChild(tr);
    }
    // The same column as a chart: 0 is the random player's score, 1 is the teacher's, the bar is where the model lands.
    const host = document.getElementById('score-chart');
    if (host) {
      const done = R.rows.filter((r) => r.vs_teacher != null).sort((a, b) => b.vs_teacher - a.vs_teacher);
      const rest = R.rows.filter((r) => r.vs_teacher == null);
      const top = Math.max(1, ...done.map((r) => r.vs_teacher));
      const at = (v) => (Math.max(0, Math.min(v, top)) / top * 100).toFixed(2) + '%';
      const scale = el('div', 'scale');
      const ticks = el('div', 'ticks');
      const tick = (v, s) => { const x = el('span', '', s); x.style.left = at(v); ticks.appendChild(x); };
      tick(0, 'random play'); if (top > 1.02) tick(top, fmtScore(Math.round(top * 100) / 100)); tick(1, 'the teacher');
      scale.append(el('span'), ticks, el('span'));
      const rows = el('div', 'rows');
      for (const r of done.concat(rest)) {
        rows.appendChild(el('div', 'name', zhTitle(r.game, r.title)));
        const lane = el('div', 'lane');
        for (const v of [0, 1]) { const gl = el('i', 'grid-line'); gl.style.left = at(v); lane.appendChild(gl); }
        if (r.vs_teacher != null) {
          const fill = el('b', 'fill' + (r.vs_teacher <= 0 ? ' zero' : '')); fill.style.width = at(r.vs_teacher);
          lane.title = `${r.title}: random ${r.random ? fmtScore(r.random.score) : '?'}, the model `
            + `${r.model ? fmtScore(r.model.score) : '?'}, the teacher ${r.teacher && r.teacher.score != null ? fmtScore(r.teacher.score) : '?'}`;
          lane.appendChild(fill);
        }
        rows.appendChild(lane);
        rows.appendChild(el('div', 'value' + (r.vs_teacher == null ? ' pending' : ''), r.vs_teacher == null ? 'pending' : (Math.round(r.vs_teacher * 100) / 100).toFixed(2)));
      }
      host.append(scale, rows);
    }
    const hasModel = R.rows.some((r) => r.model);
    const foot = document.getElementById('results-note');
    if (foot) foot.textContent = (R.note || 'Mean score per policy through the same harness, episodes capped at 1500 steps.') + ' '
      + (hasModel ? `The model here is ${[...new Set(R.rows.filter((r) => r.model).map((r) => r.model.policy))].join(', ')}.` : 'The model columns fill in when its results land.');
  }

  function buildCalibration() {
    const rows = ((D.results && D.results.rows) || []).filter((r) => r.calibration && r.calibration.bins && r.calibration.bins.length);
    const sec = document.getElementById('calibration'); if (!sec) return;
    if (!rows.length) { sec.remove(); return; }
    const box = sec.querySelector('.calib');
    const W = 184, P = 24, S = W - 2 * P;
    for (const r of rows) {
      const fig = el('figure');
      const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${W}`);
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${r.title}: stated probability against how often the move was the teacher's`);
      svgEl(svg, 'line', { x1: P, y1: W - P, x2: W - P, y2: W - P }, 'axis');
      svgEl(svg, 'line', { x1: P, y1: P, x2: P, y2: W - P }, 'axis');
      svgEl(svg, 'line', { x1: P, y1: W - P, x2: W - P, y2: P }, 'diag');
      svgText(svg, P, W - 8, '0', 'middle'); svgText(svg, W - P, W - 8, '1', 'middle');
      svgText(svg, P - 5, W - P + 3, '0', 'end'); svgText(svg, P - 5, P + 3, '1', 'end');
      svgText(svg, P + S / 2, W - 1, T('ui.axP'), 'middle');
      const maxN = Math.max(1, ...r.calibration.bins.map((b) => b.n || 0));
      for (const b of r.calibration.bins) {
        const c = svgEl(svg, 'circle', { cx: (P + b.p * S).toFixed(1), cy: (W - P - b.acc * S).toFixed(1), r: (3 + 4.5 * Math.sqrt((b.n || 0) / maxN)).toFixed(1) }, 'pt');
        tip(c, `said ${b.p.toFixed(2)}, agreed with the teacher ${b.acc.toFixed(2)} of the time, ${b.n} decisions`);
      }
      fig.appendChild(svg); fig.appendChild(el('figcaption', '', zhTitle(r.game, r.title))); box.appendChild(fig);
    }
    const pols = [...new Set(rows.map((r) => r.calibration.policy).filter(Boolean))];
    const note = document.getElementById('calib-note');
    if (note) note.textContent = 'Dots on the diagonal mean the stated probability is the rate it achieves, above it means it is underselling itself.'
      + (pols.length ? ` These bins come from ${pols.join(', ')}.` : '');
  }

  function buildHandover() {
    const H = D.results && D.results.handover;
    const sec = document.getElementById('help');
    const box = sec && sec.querySelector('.curves');
    if (!box) return;
    const block = box.closest('.curves-block') || box;
    if (!H || !H.models || !H.models[H.model]) { block.remove(); return; }
    const refs = {};
    for (const r of ((D.results && D.results.rows) || [])) {
      if (r.random && r.teacher && r.teacher.score != null && r.teacher.score !== r.random.score) {
        refs[r.game] = { lo: r.random.score, hi: r.teacher.score, title: zhTitle(r.game, r.title) };
      }
    }
    const per = H.models[H.model];
    // most headroom first: the games the model already plays as well as the teacher have a flat curve and say nothing
    const solo = (g) => { const c = per[g].confidence[0]; return c ? (c.score - refs[g].lo) / (refs[g].hi - refs[g].lo) : 1; };
    const ids = (D.games || []).map((g) => g.id).filter((g) => per[g] && per[g].confidence.length && refs[g])
      .sort((a, b) => (per[b].random.length > 0) - (per[a].random.length > 0) || solo(a) - solo(b));
    if (!ids.length) { block.remove(); return; }
    const W = 200, P = 34, S = W - 2 * P;
    for (const g of ids) {
      const ref = refs[g]; const norm = (v) => (v - ref.lo) / (ref.hi - ref.lo);
      const pts = per[g].confidence.map((d) => ({ x: d.rate, y: norm(d.score), tau: d.tau, score: d.score }));
      const ctl = per[g].random.map((d) => ({ x: d.rate, y: norm(d.score), score: d.score }));
      const ys = pts.concat(ctl).map((d) => d.y);
      const top = Math.max(1, ...ys), bot = Math.min(0, ...ys);
      const px = (x) => P + x * S, py = (y) => W - P - ((y - bot) / (top - bot)) * S;
      const fig = el('figure');
      const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${W}`);
      svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${ref.title}: score against the share of steps the teacher took`);
      svgEl(svg, 'line', { x1: P, y1: W - P, x2: W - P, y2: W - P }, 'axis');
      svgEl(svg, 'line', { x1: P, y1: P, x2: P, y2: W - P }, 'axis');
      svgEl(svg, 'line', { x1: P, y1: py(1), x2: W - P, y2: py(1) }, 'diag');      // the teacher's own score
      const path = (list, cls) => { if (list.length > 1) svgEl(svg, 'path', { d: list.map((d, i) => `${i ? 'L' : 'M'}${px(d.x).toFixed(1)} ${py(d.y).toFixed(1)}`).join(' ') }, cls); };
      path(ctl, 'ctlline'); path(pts, 'line');
      for (const d of ctl) tip(svgEl(svg, 'circle', { cx: px(d.x).toFixed(1), cy: py(d.y).toFixed(1), r: 4 }, 'ctl'), `the same share handed over at random: ${(d.x * 100).toFixed(0)}% of steps, score ${fmtScore(d.score)}`);
      for (const d of pts) tip(svgEl(svg, 'circle', { cx: px(d.x).toFixed(1), cy: py(d.y).toFixed(1), r: 4 }, 'pt'), `below confidence ${d.tau}: the teacher takes ${(d.x * 100).toFixed(0)}% of steps, score ${fmtScore(d.score)}`);
      svgText(svg, P, W - P + 12, T('ui.axModel')); svgText(svg, W - P, W - P + 12, T('ui.axTeacher'), 'end');
      svgText(svg, P - 4, py(1) + 3, T('ui.axTeacher'), 'end'); svgText(svg, P - 4, py(0) + 3, T('ui.axAlone'), 'end');
      fig.appendChild(svg); fig.appendChild(el('figcaption', '', ref.title)); box.appendChild(fig);
    }
    const legend = el('div', 'legend');
    const key = (cls, text) => { const s = el('span'); const i = el('i', cls); s.append(i, document.createTextNode(text)); legend.appendChild(s); };
    key('', T('ui.byConf'));
    key('c', T('ui.atRandom'));
    box.parentNode.insertBefore(legend, box.nextSibling);
    const note = document.getElementById('handover-note');
    const withCtl = ids.filter((g) => per[g].random.length);
    if (note) note.textContent = (H.note || '') + ` Checkpoint: ${H.model}.`
      + (withCtl.length ? ` The control ran for ${withCtl.length} of ${ids.length} games.` : '');
  }

  function buildTransfer() {
    const TR = D.results && D.results.transfer;
    const sec = document.getElementById('transfer');
    if (!sec) return;
    if (!TR || !TR.games || !TR.games.length) { sec.remove(); return; }
    const tb = document.getElementById('transfer-body');
    const INIT = { base: 'the base model', hold8: 'eight other games', shuf: 'eight other games, labels shuffled' };
    const titles = {}; for (const r of ((D.results && D.results.rows) || [])) titles[r.game] = zhTitle(r.game, r.title);

    // One panel per held-out game: how close each starting point gets to the teacher's moves as it sees more frames.
    // Only the default learning rate is drawn; every run, including the other learning rates, stays in the table.
    const box = sec.querySelector('.panels.transfer');
    const KEY = [{ init: 'hold8', line: 'line', dot: 'pt' }, { init: 'base', line: 'ctlline', dot: 'ctl' }];
    const kFrames = (n) => (n >= 1000 ? n / 1000 + 'k' : String(n));
    let drew = 0;
    if (box) {
      const W = 236, H = 198, PL = 36, PR = 12, PT = 12, PB = 34;
      for (const g of TR.games) {
        const series = KEY.map((k) => Object.assign({}, k, {
          pts: g.runs.filter((r) => r.init === k.init && !r.lr && r.agreement != null && r.frames > 0)
            .map((r) => ({ x: r.frames, y: r.agreement, score: r.score })).sort((a, b) => a.x - b.x),
        })).filter((k) => k.pts.length);
        const xs = [...new Set(series.flatMap((k) => k.pts.map((d) => d.x)))].sort((a, b) => a - b);
        if (series.length < 2 || xs.length < 2) continue;
        const px = (x) => PL + (xs.indexOf(x) / (xs.length - 1)) * (W - PL - PR);
        const py = (y) => H - PB - y * (H - PT - PB);
        const fig = el('figure');
        const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        svg.setAttribute('role', 'img');
        svg.setAttribute('aria-label', `${titles[g.game] || g.game}: how often the fine-tuned model picks the teacher's move, against the number of frames it trained on`);
        svgEl(svg, 'line', { x1: PL, y1: H - PB, x2: W - PR, y2: H - PB }, 'axis');
        svgEl(svg, 'line', { x1: PL, y1: PT, x2: PL, y2: H - PB }, 'axis');
        svgText(svg, PL - 5, H - PB + 3, '0', 'end'); svgText(svg, PL - 5, PT + 3, '1', 'end');
        for (const x of xs) svgText(svg, px(x), H - PB + 14, kFrames(x), 'middle');
        svgText(svg, PL + (W - PL - PR) / 2, H - 3, T('ui.axFrames'), 'middle');
        for (const k of series) {
          if (k.pts.length > 1) svgEl(svg, 'path', { d: k.pts.map((d, i) => `${i ? 'L' : 'M'}${px(d.x).toFixed(1)} ${py(d.y).toFixed(1)}`).join(' ') }, k.line);
          for (const d of k.pts) {
            tip(svgEl(svg, 'circle', { cx: px(d.x).toFixed(1), cy: py(d.y).toFixed(1), r: 4 }, k.dot),
              `${INIT[k.init] || k.init}, ${d.x.toLocaleString()} frames: picks the teacher's move ${(d.y * 100).toFixed(0)}% of the time`
              + (d.score != null ? `, score ${fmtScore(d.score)}` : ''));
          }
        }
        fig.appendChild(svg); fig.appendChild(el('figcaption', '', titles[g.game] || g.game)); box.appendChild(fig);
        drew++;
      }
      if (!drew) box.remove();
      else {
        const legend = el('div', 'legend');
        const key = (cls, text) => { const sp = el('span'); sp.append(el('i', cls), document.createTextNode(text)); legend.appendChild(sp); };
        key('', T('ui.fromEight'));
        key('c', T('ui.fromBase'));
        box.parentNode.insertBefore(legend, box.nextSibling);
      }
    }
    for (const g of TR.games) {
      let first = true;
      for (const r of g.runs) {
        const tr = el('tr'); if (first) tr.classList.add('sep');
        tr.appendChild(el('td', '', first ? (titles[g.game] || g.game) : ''));
        first = false;
        tr.appendChild(el('td', '', (INIT[r.init] || r.init) + (r.lr ? ` (lr ${r.lr})` : '')));
        tr.appendChild(el('td', 'num', r.frames.toLocaleString()));
        const cell = (v, cls) => { const td = el('td', 'num ' + (cls || '')); if (v == null) { td.textContent = T('ui.pending'); td.classList.add('pending'); } else td.textContent = v; return td; };
        tr.appendChild(cell(r.agreement != null ? r.agreement.toFixed(3) : null));
        tr.appendChild(cell(r.score != null ? fmtScore(r.score) : null));
        tr.appendChild(cell(r.vs_teacher != null ? r.vs_teacher.toFixed(2) : null, r.init === 'hold8' ? 'hi' : ''));
        tb.appendChild(tr);
      }
    }
    const hasLr = TR.games.some((g) => g.runs.some((r) => r.lr));
    const tnote = document.getElementById('transfer-note');
    if (tnote) tnote.textContent = (TR.note || '') + ' '
      + TR.games.map((g) => `${titles[g.game] || g.game}: random ${fmtScore(g.random)}, teacher ${fmtScore(g.teacher)}.`).join(' ')
      + (drew && hasLr ? ' The panels show the default learning rate; the table has every run, including the other learning rates tried at 10,000 frames.' : '');
  }

  function buildPrompt() {
    const pre = document.getElementById('prompt'), sel = document.getElementById('prompt-game');
    for (const g of D.games) { const o = el('option', '', gameTitle(g)); o.value = g.id; sel.appendChild(o); }
    const render = () => {
      const g = D.games.find((x) => x.id === sel.value) || D.games[0];
      pre.textContent = ''; const parts = g.prompt.split('<|vision_start|><|image_pad|><|vision_end|>');
      pre.appendChild(document.createTextNode(parts[0]));
      const ph = el('span', 'ph', '<|vision_start|><|image_pad|><|vision_end|>'); ph.title = 'the game frame: one token per merged 32x32 pixel block';
      pre.appendChild(ph); pre.appendChild(document.createTextNode(parts.slice(1).join('')));
    };
    sel.value = D.games.some((g) => g.id === 'snake') ? 'snake' : D.games[0].id; sel.addEventListener('change', render); render();
    for (const a of document.querySelectorAll('a[data-link]')) if (D.links && D.links[a.dataset.link]) a.href = D.links[a.dataset.link];
    // Weights and paper are announced before they are published: build_demo.py's LINKS carries an empty string until
    // then, and the label turns into a real link the moment a URL lands there.
    for (const node of document.querySelectorAll('[data-soon]')) {
      const url = D.links && D.links[node.dataset.soon]; if (!url) continue;
      const a = el('a'); a.href = url; a.innerHTML = node.innerHTML;
      const tag = a.querySelector('em'); if (tag && node.dataset.soonLabel) tag.textContent = node.dataset.soonLabel;
      node.replaceWith(a);
    }
  }

  // ---------------------------------------------------------------- go
  buildGrid(); buildStage(); buildResults(); buildCalibration(); buildDuel(); buildHandover(); buildTransfer(); buildPrompt();
  // the handover section is two independent blocks; if the data for neither is here, the heading goes too
  const help = document.getElementById('help');
  if (help && !help.querySelector('.duel-block, .curves-block')) help.remove();
  document.getElementById('generated').textContent = D.generated ? T('ui.built') + ' ' + D.generated.replace('T', ' ') : '';
  for (const t of tiles) if (!t.suspended && !t.running) t.start();

  window.pjDemo = {
    tiles, data: D,
    // Replays every recording of every game (in parallel across games) and returns one record per recording.
    async verifyAll(opts) {
      opts = opts || {}; const speed = opts.speed || 4;
      const res = await Promise.all(tiles.map((t) => t.verify(speed, opts.policies || null)));
      return res.flat();
    },
    pauseAll() { tiles.forEach((t) => t.pause()); }, playAll() { tiles.forEach((t) => t.play()); },
    // Wake every parked tile and keep it awake: what the screenshot and verification runs want.
    startAll() { lazyOff = true; if (applyMode) applyMode(false, false); for (const t of tiles) if (t.suspended) t.start(); },
  };
})();
