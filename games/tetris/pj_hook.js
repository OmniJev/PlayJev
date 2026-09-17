// PlayJev hook for jakesgordon/javascript-tetris (canvas game, requestAnimationFrame loop whose
// dt comes from new Date). Everything lives in globals declared by the page's inline script:
//   playing  true while a game is running (lose() sets it false)
//   score    points (10 per landed piece, 100/200/400/800 per 1/2/3/4 lines), rows  lines cleared
//   step     seconds of game time before gravity pulls the piece down one row (0.6 -> 0.1)
//   current/next  piece objects {type, dir, x, y}; blocks[x][y]  locked cells; nx/ny  court size
// The game reads keys with a keydown listener on document and queues one action per frame.
// One env step = one row of descent, the way a player thinks about a move.
(function () {
  const LETTER = { cyan: 'I', blue: 'J', orange: 'L', yellow: 'O', green: 'S', purple: 'T', red: 'Z' };
  const S = { started: false, tex: undefined, pat: null };
  let cv = null;

  function court() { return document.getElementById('canvas'); }
  function tap(name) { pj.keyDown(name, document); pj.keyUp(name, document); }
  // frames to allow for one gravity drop, plus slack
  function dropFrames() { return Math.ceil((Number(window.step) || 0.6) * 60) + 12; }

  function loadTexture() {
    if (S.tex !== undefined) return Promise.resolve(S.tex);
    return new Promise((res) => {
      const img = new Image(); let done = false;
      const fin = (v) => { if (!done) { done = true; S.tex = v; res(v); } };
      img.onload = () => fin(img); img.onerror = () => fin(null);
      pj.real.setTimeout(() => fin(null), 3000);   // real clock: the virtual one never runs here
      img.src = 'texture.jpg';
    });
  }

  function letterOf(type) { return (type && LETTER[type.color]) || '?'; }

  function pieceCells(p) {
    const out = [];
    if (p && typeof window.eachblock === 'function') {
      window.eachblock(p.type, p.x, p.y, p.dir, function (x, y) { out.push([x, y]); });
    }
    return out;
  }

  window.addEventListener('DOMContentLoaded', () => {
    pj.register({
      id: 'tetris',

      actions: [
        { name: 'left', description: 'shift the falling piece one column to the left', keys: ['ArrowLeft'] },
        { name: 'right', description: 'shift the falling piece one column to the right', keys: ['ArrowRight'] },
        { name: 'rotate', description: 'turn the falling piece a quarter turn clockwise', keys: ['ArrowUp'] },
        { name: 'drop', description: 'send the falling piece straight down until it lands', keys: ['ArrowDown'] },
        { name: 'none', description: 'leave the piece alone and let it sink one row', keys: [] },
      ],

      async start(seed) {
        await loadTexture();
        // Refill the piece bag on the seeded RNG so the whole sequence, first piece included,
        // follows the seed (the two pieces drawn while the page loaded are pre-seed).
        if (Array.isArray(window.pieces) && typeof window.randomPiece === 'function') {
          window.pieces.length = 0;
          window.next = window.randomPiece();
        }
        tap('Space');            // start screen -> play(): hides the prompt and resets the court
        pj.frames(1);            // one frame so the court is drawn with the first piece
        S.started = true;
      },

      // Turn-based: act, then run the game until the piece has sunk one row (or locked).
      applyAction(i) {
        const a = this.actions[i];
        if (!window.playing) return;
        if (a.name === 'drop') {                       // soft-drop every frame until it lands
          const piece = window.current;
          for (let n = 0; n < window.ny + 6; n++) {
            tap('ArrowDown');
            pj.frames(1);
            if (!window.playing || window.current !== piece) break;
          }
          return;
        }
        for (const k of a.keys) tap(k);
        const piece = window.current, y0 = piece ? piece.y : 0, limit = dropFrames();
        for (let n = 0; n < limit; n++) {
          pj.frames(1);
          if (!window.playing) break;
          if (window.current !== piece || window.current.y > y0) break;
        }
      },

      score() { return Math.max(0, Number(window.score) || 0); },
      done() { return S.started === true && window.playing === false; },

      // The page paints the court on a transparent canvas over a CSS stone texture, so copy the
      // texture in first and the game canvas on top; otherwise the frame comes out on black.
      canvas() {
        const src = court(), w = src.width, h = src.height;
        if (!cv || cv.width !== w || cv.height !== h) { cv = document.createElement('canvas'); cv.width = w; cv.height = h; }
        const ctx = cv.getContext('2d');
        if (S.tex && !S.pat) S.pat = ctx.createPattern(S.tex, 'repeat');
        ctx.fillStyle = S.pat || '#b5ada1';
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(src, 0, 0);
        return cv;
      },

      info() {
        const nx = window.nx, ny = window.ny, b = window.blocks || [];
        const grid = [];
        for (let y = 0; y < ny; y++) {
          let row = '';
          for (let x = 0; x < nx; x++) { const v = b[x] && b[x][y]; row += v ? letterOf(v) : '.'; }
          grid.push(row);
        }
        const cur = window.current, nxt = window.next;
        return {
          cols: nx, rows_high: ny,
          score: Math.max(0, Number(window.score) || 0),
          lines: Number(window.rows) || 0,
          drop_interval: Number(window.step) || 0,
          piece: cur ? { kind: letterOf(cur.type), x: cur.x, y: cur.y, dir: cur.dir, cells: pieceCells(cur) } : null,
          next: nxt ? letterOf(nxt.type) : null,
          grid: grid.join('|'),
        };
      },
    });
  });
})();
