// PlayJev demo page (docs/DEMO.md). Data comes from data.js (window.PJ_DEMO, written by scripts/build_demo.py);
// each game runs in its own same-origin iframe with the training shim and hook, driven over postMessage by the
// bridge (pj_bridge.js), so the picture is the real game replaying the recorded action sequence.
(function () {
  'use strict';
  const D = window.PJ_DEMO;
  if (!D) { document.body.insertAdjacentHTML('afterbegin', '<p>data.js is missing: run scripts/build_demo.py</p>'); return; }
  const qs = new URLSearchParams(location.search);
  const SERVER = normaliseServer(qs.get('server'));
  const INSTRUCTIONS = 'Which move should the player make next?';
  const BOOT_TIMEOUT_MS = 60000, LIVE_TIMEOUT_MS = 30000, END_PAUSE_MS = 1800;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const fmtScore = (x) => (x == null ? '' : Number.isInteger(x) ? String(x) : (Math.round(x * 10) / 10).toString());
  const confidence = (p) => { const K = p.length; if (K < 2) return 0; return (Math.max(...p) - 1 / K) / (1 - 1 / K); };
  const argmax = (p) => p.reduce((b, v, i) => (v > p[b] ? i : b), 0);
  const policyRank = (n) => { const s = String(n).toLowerCase(); return s === 'live' ? -1 : s.startsWith('playjev') ? 0 : s === 'teacher' ? 1 : s === 'random' ? 2 : 3; };

  function normaliseServer(s) {
    if (!s) return null;
    try {
      const u = new URL(s, location.href);
      if (u.pathname === '/' || u.pathname === '') u.pathname = '/v1/systemone';
      return u.toString();
    } catch (e) { console.warn('bad ?server= url', s); return null; }
  }

  // ---------------------------------------------------------------- theme
  (function theme() {
    const root = document.documentElement, btn = document.getElementById('theme');
    let stored = null; try { stored = localStorage.getItem('pj-theme'); } catch (e) {}
    if (stored === 'dark' || stored === 'light') root.dataset.theme = stored;
    const isDark = () => root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    const label = () => { btn.textContent = isDark() ? 'light mode' : 'dark mode'; };
    btn.addEventListener('click', () => { root.dataset.theme = isDark() ? 'light' : 'dark'; try { localStorage.setItem('pj-theme', root.dataset.theme); } catch (e) {} label(); });
    label();
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
  window.addEventListener('message', (ev) => {
    const m = ev.data; if (!m || m.pjDemo !== true) return;
    for (const t of tiles) if (t.iframe && ev.source === t.iframe.contentWindow) { t.onMessage(m); return; }
  });

  class Tile {
    constructor(game, entries) {
      this.game = game; this.entries = entries || []; this.K = game.actions.length;
      this.speed = 1; this.playing = true; this.gen = 0; this.pending = new Map(); this.seq = 0;
      this.iframe = null; this.rect = null; this.errors = []; this.result = null; this.autoAdvance = true;
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
      const root = this.root = el('section', 'tile'); root.dataset.game = g.id;
      const head = el('div', 'head'); head.appendChild(el('span', 'title', g.title));
      const mark = el('span', 'mark'); mark.title = 'replayed score differs from the recording'; head.appendChild(mark);
      root.appendChild(head);
      this.view = el('div', 'view'); this.overlay = el('div', 'overlay', 'loading'); this.view.appendChild(this.overlay); root.appendChild(this.view);
      this.bars = el('div', 'bars'); this.rows = [];
      for (const a of g.actions) {
        const row = el('div', 'row'); row.title = a.description;
        const lbl = el('span', 'lbl', a.name), bar = el('span', 'bar'), fill = el('i'), val = el('span', 'val', '');
        bar.appendChild(fill); row.append(lbl, bar, val); this.bars.appendChild(row);
        this.rows.push({ row, fill, val });
      }
      root.appendChild(this.bars);
      this.conf = el('div', 'conf'); root.appendChild(this.conf);
      this.bars = root.querySelector('.bars');
      this.status = el('div', 'status');
      this.stepEl = el('span', 'step', 'step 0'); this.scoreEl = el('span', 'score', 'score 0'); this.recEl = el('span', 'rec', ''); this.warnEl = el('span', 'warn', '');
      this.status.append(this.stepEl, this.scoreEl, this.recEl, this.warnEl); root.appendChild(this.status);
      const c = el('div', 'controls');
      this.ppBtn = el('button', 'pp', 'pause'); this.ppBtn.addEventListener('click', () => (this.playing ? this.pause() : this.play()));
      const seg = el('span', 'seg'); this.speedBtns = {};
      for (const s of [1, 2, 4]) { const b = el('button', s === 1 ? 'on' : '', s + 'x'); b.addEventListener('click', () => this.setSpeed(s)); seg.appendChild(b); this.speedBtns[s] = b; }
      const restart = el('button', '', 'restart'); restart.addEventListener('click', () => this.restart());
      const next = el('button', '', 'next'); next.title = 'next recorded episode'; next.addEventListener('click', () => this.next());
      c.append(this.ppBtn, seg, restart, next);
      if (this.policies.length > 1) {
        const sel = el('select'); sel.title = 'which recording to replay';
        for (const p of this.policies) { const o = el('option', '', p === 'live' ? 'live server' : p); o.value = p; sel.appendChild(o); }
        sel.value = this.policy; sel.addEventListener('change', () => this.selectPolicy(sel.value)); c.appendChild(sel); this.sel = sel;
      }
      root.appendChild(c);
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
      if (this.errors.length === 1) this.warnEl.textContent = 'page error, see console';
    }
    async boot() {
      // A fresh frame per episode, as the driver reloads the page before every pj.start.
      for (const p of this.pending.values()) p.reject(new Error('frame replaced')); this.pending.clear();
      if (this.iframe) this.iframe.remove();
      const f = this.iframe = document.createElement('iframe');
      f.width = this.game.viewport.width; f.height = this.game.viewport.height;
      f.style.width = this.game.viewport.width + 'px'; f.style.height = this.game.viewport.height + 'px';
      f.setAttribute('scrolling', 'no'); f.setAttribute('tabindex', '-1'); f.title = this.game.title + ' (game frame)';
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
      const B = this.view.clientWidth || 200, r = this.rect || { x: 0, y: 0, w: this.game.viewport.width, h: this.game.viewport.height };
      const s = Math.min(B / r.w, B / r.h);
      const tx = (B - r.w * s) / 2 - r.x * s, ty = (B - r.h * s) / 2 - r.y * s;
      const V = this.game.viewport;
      // Only the game area shows; the rest of the game page (menus, footers, score panels) is clipped away.
      this.iframe.style.clipPath = `inset(${r.y.toFixed(2)}px ${(V.width - r.x - r.w).toFixed(2)}px ${(V.height - r.y - r.h).toFixed(2)}px ${r.x.toFixed(2)}px)`;
      this.iframe.style.transform = `translate(${tx.toFixed(2)}px, ${ty.toFixed(2)}px) scale(${s.toFixed(5)})`;
      this.iframe.classList.add('shown');
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
      this.conf.innerHTML = p ? `confidence <b>${confidence(p).toFixed(2)}</b>` + (s2 ? ' <span class="s2tag">System Two decided</span>' : '')
                              + (this.latency != null ? ` <span>latency ${Math.round(this.latency)} ms</span>` : '') : '\u00a0';
    }
    renderRec() {
      const pol = this.policy;
      if (!pol) { this.recEl.textContent = 'no recording yet'; this.recEl.classList.remove('random'); return; }
      this.recEl.textContent = pol === 'live' ? 'live: ' + new URL(SERVER).host : 'recording: ' + pol;
      this.recEl.classList.toggle('random', pol === 'random');
    }
    renderStatus(i, total, score, tail) {
      this.stepEl.textContent = 'step ' + i + (total != null ? ' / ' + total : '') + (tail ? ' ' + tail : '');
      this.scoreEl.textContent = 'score ' + fmtScore(score);
    }
    setMismatch(text) { this.root.classList.toggle('mismatch', !!text); this.warnEl.textContent = text || ''; }

    // ---- controls
    setSpeed(s) { this.speed = s; for (const k in this.speedBtns) this.speedBtns[k].classList.toggle('on', Number(k) === s); }
    pause() { this.playing = false; this.ppBtn.textContent = 'play'; }
    play() { this.playing = true; this.ppBtn.textContent = 'pause'; if (this._resume) { const r = this._resume; this._resume = null; r(); } else if (!this.running) this.start(); }
    waitResume() { return new Promise((res) => { this._resume = res; }); }
    currentEntries() { return this.entries.filter((e) => e.policy === this.policy); }
    selectPolicy(p) { this.policy = p; this.epi = 0; this.renderRec(); if (this.sel) this.sel.value = p; this.start(); }
    restart() { this.start(); }
    next() { const n = this.currentEntries().length; if (n) this.epi = (this.epi + 1) % n; this.start(); }

    // Start (or restart) the current episode; keeps the page cycling through the recordings while playing.
    async start() {
      const gen = ++this.gen; this.running = true; this._resume = null;
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
      this.showOverlay('loading'); await this.boot(); if (gen !== this.gen) return;
      const obs = await this.call('start', { seed: 1 }); await this.measure(); this.showOverlay(null);
      this.renderStatus(0, null, obs.score); this.renderBars(null, -1); this.renderRec();
    }

    // Replay one recording; resolves with the verification record when the episode is over.
    async runEpisode(entry, gen, opts) {
      opts = opts || {};
      const g = this.game; const t0 = performance.now();
      this.latency = null; this.setMismatch(''); this.renderRec(); this.errors = [];
      this.showOverlay('loading');
      const rec = await loadReplay(entry, g.id); if (gen !== this.gen) return null;
      await this.boot(); if (gen !== this.gen) return null;
      let obs = await this.call('start', { seed: rec.seed }); if (gen !== this.gen) return null;
      await this.measure(); this.showOverlay(null);
      const steps = rec.steps, N = steps.length; let i = 0, divergedAt = null, handed = 0;
      this.renderStatus(0, N, obs.score); this.renderBars(null, -1);
      const stepMs = g.step_ms || 150;
      while (i < N) {
        if (gen !== this.gen) return null;
        if (!this.playing) { await this.waitResume(); if (gen !== this.gen) return null; }
        const tick = performance.now(); const st = steps[i];
        // The bars belong to the picture on screen: decision i was made from the frame before action i, so show
        // it first, hold for the step's duration, then apply the move (otherwise the model looks one beat late).
        if (st.h) handed++;
        this.renderBars(st.p, st.a, !!st.h); this.renderStatus(i, N, obs.score, handed ? `(System Two ${handed} of ${i + 1})` : '');
        const speed0 = opts.speed || this.speed;
        const hold = stepMs / speed0 - (performance.now() - tick); if (hold > 0) await sleep(hold);
        if (gen !== this.gen) return null;
        if (!this.playing) { await this.waitResume(); if (gen !== this.gen) return null; }
        obs = await this.call('step', { a: st.a }); if (gen !== this.gen) return null;
        i++;
        this.renderStatus(i, N, obs.score, handed ? `(System Two ${handed} of ${i})` : '');
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
      this.renderStatus(i, N, obs.score, (obs.done ? '(over)' : i === N ? '(end of recording)' : '') + (handed ? ` System Two decided ${handed} of ${i} steps` : ''));
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
      const out = []; const gen = ++this.gen; this.running = true; this.autoAdvance = false; this.playing = true; this.ppBtn.textContent = 'pause';
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

  // ---------------------------------------------------------------- page sections
  function buildGrid() {
    const grid = document.getElementById('grid');
    for (const g of D.games) { const t = new Tile(g, (D.replays && D.replays[g.id]) || []); tiles.push(t); grid.appendChild(t.root); }
    const all = document.getElementById('all');
    document.getElementById('all-pp').addEventListener('click', (ev) => {
      const anyPlaying = tiles.some((t) => t.playing);
      for (const t of tiles) (anyPlaying ? t.pause() : t.play());
      ev.target.textContent = anyPlaying ? 'play all' : 'pause all';
    });
    for (const b of all.querySelectorAll('[data-speed]')) b.addEventListener('click', () => {
      for (const t of tiles) t.setSpeed(Number(b.dataset.speed));
      for (const x of all.querySelectorAll('[data-speed]')) x.classList.toggle('on', x === b);
    });
    // who plays on every tile: the model alone, the model with System Two (the teacher takes the low-confidence
    // steps), or random; a tile without such a recording keeps what it has
    const pick = (t, which) => {
      const row = ((D.results && D.results.rows) || []).find((r) => r.game === t.game.id);
      const base = row && row.model ? row.model.policy : (t.policies.find((p) => String(p).startsWith('playjev')) || null);
      if (which === 'random') return t.policies.includes('random') ? 'random' : null;
      if (which === 's2') return t.policies.find((p) => p === base + '-s2') || null;
      return base;
    };
    const hasS2 = tiles.some((t) => pick(t, 's2'));
    for (const b of all.querySelectorAll('[data-who]')) {
      if (b.dataset.who === 's2' && !hasS2) { b.remove(); continue; }
      b.addEventListener('click', () => {
        for (const t of tiles) { const p = pick(t, b.dataset.who); if (p && p !== t.policy) t.selectPolicy(p); }
        for (const x of all.querySelectorAll('[data-who]')) x.classList.toggle('on', x === b);
      });
    }
    let resizeTimer = null;
    window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => tiles.forEach((t) => t.fit()), 60); });
    const notice = document.getElementById('notice');
    const pols = (D.policies || []).filter((p) => p !== 'live');
    const nRec = Object.values(D.replays || {}).reduce((s, l) => s + l.length, 0);
    if (SERVER) { notice.textContent = `Live mode: frames go to ${SERVER} and its probabilities pick every move.`; notice.classList.add('live'); }
    else if (!nRec) notice.textContent = 'No recordings have been built into this page yet; the games load and wait.';
    else notice.textContent = `${nRec} recorded episode${nRec === 1 ? '' : 's'} on this page, from: ${pols.join(', ')}. ` +
      (pols.some((p) => p.startsWith('playjev')) ? '' : 'The trained model has no recordings here yet; what plays is the recorded policy named on each tile.');
  }

  function buildResults() {
    const R = D.results; const tb = document.getElementById('results-body'); if (!R || !R.rows) return;
    const cell = (v, cls) => { const td = el('td', cls); if (v == null) { td.textContent = 'pending'; td.classList.add('pending'); } else td.textContent = v; return td; };
    for (const r of R.rows) {
      const tr = el('tr');
      const name = el('td'); const a = el('a', '', r.title); a.href = 'https://github.com/' + (r.upstream || ''); a.target = '_blank'; a.rel = 'noopener'; name.appendChild(a);
      name.appendChild(el('span', 'sub', ` ${r.k} moves`)); tr.appendChild(name);
      tr.appendChild(cell(r.random ? fmtScore(r.random.score) : null, 'num'));
      tr.appendChild(cell(r.model ? fmtScore(r.model.score) : null, 'num'));
      tr.appendChild(cell(r.teacher && r.teacher.score != null ? fmtScore(r.teacher.score) : null, 'num'));
      tr.appendChild(cell(r.vs_teacher != null ? (Math.round(r.vs_teacher * 100) / 100).toFixed(2) : null, 'num'));
      tb.appendChild(tr);
    }
    const hasModel = R.rows.some((r) => r.model);
    const foot = document.getElementById('results-note');
    foot.textContent = (R.note || 'Mean score per policy through the same harness, episodes capped at 1500 steps.') + ' '
      + (hasModel ? `Trained model: ${[...new Set(R.rows.filter((r) => r.model).map((r) => r.model.policy))].join(', ')}.` : 'The trained model columns fill in when its results land.');
  }

  function buildCalibration() {
    const rows = ((D.results && D.results.rows) || []).filter((r) => r.calibration && r.calibration.bins && r.calibration.bins.length);
    const sec = document.getElementById('calibration'); if (!rows.length) { sec.remove(); return; }
    const box = sec.querySelector('.calib');
    for (const r of rows) {
      const fig = el('figure'); const W = 150, P = 18, S = W - 2 * P; const ns = 'http://www.w3.org/2000/svg';
      const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${W}`);
      const line = (x1, y1, x2, y2, cls) => { const l = document.createElementNS(ns, 'line'); l.setAttribute('x1', x1); l.setAttribute('y1', y1); l.setAttribute('x2', x2); l.setAttribute('y2', y2); l.setAttribute('class', cls); svg.appendChild(l); };
      const text = (x, y, s, anchor) => { const t = document.createElementNS(ns, 'text'); t.setAttribute('x', x); t.setAttribute('y', y); t.setAttribute('class', 'txt'); if (anchor) t.setAttribute('text-anchor', anchor); t.textContent = s; svg.appendChild(t); };
      line(P, W - P, W - P, W - P, 'axis'); line(P, P, P, W - P, 'axis'); line(P, W - P, W - P, P, 'diag');
      text(P, W - 4, '0', 'middle'); text(W - P, W - 4, '1', 'middle'); text(P - 4, W - P + 3, '0', 'end'); text(P - 4, P + 3, '1', 'end');
      text(W / 2, W - 4, 'p(chosen move)', 'middle');
      const maxN = Math.max(1, ...r.calibration.bins.map((b) => b.n || 0));
      for (const b of r.calibration.bins) {
        const c = document.createElementNS(ns, 'circle'); c.setAttribute('cx', P + b.p * S); c.setAttribute('cy', W - P - b.acc * S);
        c.setAttribute('r', (2 + 4 * Math.sqrt((b.n || 0) / maxN)).toFixed(1)); c.setAttribute('class', 'pt');
        const t = document.createElementNS(ns, 'title'); t.textContent = `p ${b.p.toFixed(2)}, agreed ${b.acc.toFixed(2)}, n ${b.n}`; c.appendChild(t); svg.appendChild(c);
      }
      fig.appendChild(svg); fig.appendChild(el('figcaption', '', r.title + (r.calibration.policy ? ` (${r.calibration.policy})` : ''))); box.appendChild(fig);
    }
  }

  function buildPrompt() {
    const pre = document.getElementById('prompt'), sel = document.getElementById('prompt-game');
    for (const g of D.games) { const o = el('option', '', g.title); o.value = g.id; sel.appendChild(o); }
    const render = () => {
      const g = D.games.find((x) => x.id === sel.value) || D.games[0];
      pre.textContent = ''; const parts = g.prompt.split('<|vision_start|><|image_pad|><|vision_end|>');
      pre.appendChild(document.createTextNode(parts[0]));
      const ph = el('span', 'ph', '<|vision_start|><|image_pad|><|vision_end|>'); ph.title = 'the game frame: one token per merged 32x32 pixel block';
      pre.appendChild(ph); pre.appendChild(document.createTextNode(parts.slice(1).join('')));
    };
    sel.value = D.games.some((g) => g.id === 'snake') ? 'snake' : D.games[0].id; sel.addEventListener('change', render); render();
    for (const a of document.querySelectorAll('a[data-link]')) if (D.links && D.links[a.dataset.link]) a.href = D.links[a.dataset.link];
  }

  // ---------------------------------------------------------------- go
  buildGrid(); buildResults(); buildCalibration(); buildPrompt();
  document.getElementById('generated').textContent = D.generated ? 'built ' + D.generated.replace('T', ' ') : '';
  for (const t of tiles) t.start();

  window.pjDemo = {
    tiles, data: D,
    // Replays every recording of every game (in parallel across games) and returns one record per recording.
    async verifyAll(opts) {
      opts = opts || {}; const speed = opts.speed || 4;
      const res = await Promise.all(tiles.map((t) => t.verify(speed, opts.policies || null)));
      return res.flat();
    },
    pauseAll() { tiles.forEach((t) => t.pause()); }, playAll() { tiles.forEach((t) => t.play()); },
  };
})();
