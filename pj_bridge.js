// PlayJev demo bridge. Loaded inside each game iframe right after pj_shim.js and pj_hook.js (the build script
// puts the three tags at the start of the entry page's <head>). It relays the parent page's calls to the pj API
// over postMessage, so the demo works from file:// as well as from GitHub Pages (a file:// parent cannot touch a
// frame's window directly). Nothing here changes how the game runs: the shim and the hook are the training ones.
(function () {
  if (!window.pj) { console.error('pj_bridge: pj_shim.js must load first'); return; }
  const pj = window.pj;
  const post = (m) => { try { window.parent.postMessage(Object.assign({ pjDemo: true }, m), '*'); } catch (e) {} };

  // pj.obs() attaches a JPEG of the canvas to every step (the training driver needs it). Replay does not, and on
  // file:// toDataURL throws on a canvas that has drawn a file:// image, so frames are only produced on request.
  const realFrame = pj.frame.bind(pj);
  let wantFrames = false;
  pj.frame = function () { return wantFrames ? realFrame() : null; };

  function strip(o) {
    if (!o || typeof o !== 'object') return o;
    const out = { t: o.t, steps: o.steps, score: o.score, done: o.done, reward: o.reward, errors: o.errors || [] };
    if (wantFrames && o.frame) out.frame = o.frame;
    return out;
  }

  function rect(selector) {
    let el = null;
    if (selector) el = document.querySelector(selector);
    if (!el) { try { const c = pj.hook && pj.hook.canvas && pj.hook.canvas(); if (c && document.contains(c)) el = c; } catch (e) {} }
    if (!el) el = document.body || document.documentElement;
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height, inner: [window.innerWidth, window.innerHeight] };
  }

  function style(css) {
    let s = document.getElementById('pj-demo-style');
    if (!s) { s = document.createElement('style'); s.id = 'pj-demo-style'; (document.head || document.documentElement).appendChild(s); }
    s.textContent = css || '';
    return true;
  }

  async function handle(m) {
    switch (m.op) {
      case 'meta':
        await pj.ready;
        return { actions: pj.actions.map((a) => ({ name: a.name, description: a.description })), stepFrames: (pj.hook && pj.hook.stepFrames) || null, title: document.title };
      case 'start': return strip(await pj.start(m.seed));
      case 'step': return strip(pj.step(m.a, m.k == null ? undefined : m.k));
      case 'frame': return realFrame();                       // one JPEG data URL of the game canvas (live mode)
      case 'frames': wantFrames = !!m.on; return wantFrames;  // attach a frame to every start/step result (live mode)
      case 'rect': return rect(m.selector);
      case 'style': return style(m.css);
      case 'score': return pj.score();
      case 'done': return pj.done();
      default: throw new Error('pj_bridge: unknown op ' + m.op);
    }
  }

  window.addEventListener('message', (ev) => {
    const m = ev.data;
    if (!m || m.pjDemoCall !== true || ev.source !== window.parent) return;
    Promise.resolve().then(() => handle(m)).then(
      (result) => post({ id: m.id, result }),
      (err) => post({ id: m.id, error: String((err && err.stack) || err) }));
  });

  window.addEventListener('error', (ev) => post({ pageerror: String((ev && ev.message) || ev) }));
  window.addEventListener('unhandledrejection', (ev) => post({ pageerror: 'unhandled rejection: ' + String(ev && ev.reason) }));

  // The driver's goto(load) then pj.start; announce the same moment: page loaded and the hook registered.
  window.addEventListener('load', () => {
    pj.ready.then(() => post({ ready: true, actions: pj.actions.map((a) => ({ name: a.name, description: a.description })) }));
    // Real clock (the shim's setTimeout is virtual): report a hook that never registers.
    pj.real.setTimeout(() => { if (!pj.hook) post({ pageerror: 'hook did not register within 15 s' }); }, 15000);
  });
})();
