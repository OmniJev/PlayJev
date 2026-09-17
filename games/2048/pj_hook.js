// PlayJev hook for gabrielecirulli/2048 (DOM game, turn based, no game loop).
// The whole game state lives on one GameManager instance: grid.cells[x][y] holds Tile objects,
// .score, .over, .won. Moves happen synchronously inside the document keydown handler; only the
// DOM repaint is deferred to requestAnimationFrame (HTMLActuator.actuate).
//
// Reaching the instance without patching the vendored code: application.js does
// `new GameManager(...)` inside a rAF callback and keeps no reference. The shim's rAF is virtual,
// so at DOMContentLoaded that callback has not run yet and we can wrap
// GameManager.prototype.setup (called from the constructor) to capture `this` on the first frame.
//
// Persistence: LocalStorageManager would restore a half-played game after a reload, and pages in
// one Playwright context share localStorage. We force localStorageSupported() to false so the
// game uses the per-page in-memory window.fakeStorage; every page load therefore starts fresh.
(function () {
  const SIZE = 4;
  const BOARD = 500;   // .game-container width/height (style/main.css, desktop layout)
  const PAD = 15;      // .game-container padding
  const STEP = 121.25; // .grid-cell 106.25 + 15 margin
  const CELL = 107;    // .tile width/height

  let gm = null;           // the GameManager instance
  let cv = null;           // offscreen canvas for frame()
  const styleCache = new Map();

  // Read a tile's real colours/font out of the page CSS by measuring a throwaway element
  // that carries the same classes the actuator would give it.
  function tileStyle(value) {
    if (styleCache.has(value)) return styleCache.get(value);
    const host = document.querySelector('.tile-container');
    const wrap = document.createElement('div');
    wrap.setAttribute('class', 'tile tile-' + value + ' tile-position-1-1' + (value > 2048 ? ' tile-super' : ''));
    wrap.style.visibility = 'hidden';
    const inner = document.createElement('div');
    inner.className = 'tile-inner';
    inner.textContent = String(value);
    wrap.appendChild(inner);
    host.appendChild(wrap);
    const cs = getComputedStyle(inner);
    const s = {
      bg: cs.backgroundColor,
      fg: cs.color,
      font: cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily,
    };
    host.removeChild(wrap);
    styleCache.set(value, s);
    return s;
  }

  function boardStyle() {
    if (styleCache.has('_board')) return styleCache.get('_board');
    const s = {
      bg: getComputedStyle(document.querySelector('.game-container')).backgroundColor,
      cell: getComputedStyle(document.querySelector('.grid-cell')).backgroundColor,
    };
    styleCache.set('_board', s);
    return s;
  }

  // grid.cells is column major: cells[x][y]. Return row major numbers, 0 for empty.
  function gridRows() {
    const out = [];
    for (let y = 0; y < SIZE; y++) {
      const row = [];
      for (let x = 0; x < SIZE; x++) {
        const t = gm && gm.grid ? gm.grid.cells[x][y] : null;
        row.push(t ? t.value : 0);
      }
      out.push(row);
    }
    return out;
  }

  window.addEventListener('DOMContentLoaded', () => {
    // No stored game ever comes back: use the in-memory fakeStorage, which is per page load.
    LocalStorageManager.prototype.localStorageSupported = function () { return false; };
    // Capture the instance the moment it builds its first grid.
    const setup = GameManager.prototype.setup;
    GameManager.prototype.setup = function () { gm = this; return setup.apply(this, arguments); };

    pj.register({
      id: '2048',
      actions: [
        { name: 'up', description: 'slide every tile as far up as it can go and merge equal tiles that meet', keys: ['ArrowUp'] },
        { name: 'down', description: 'slide every tile as far down as it can go and merge equal tiles that meet', keys: ['ArrowDown'] },
        { name: 'left', description: 'slide every tile as far left as it can go and merge equal tiles that meet', keys: ['ArrowLeft'] },
        { name: 'right', description: 'slide every tile as far right as it can go and merge equal tiles that meet', keys: ['ArrowRight'] },
      ],

      async start() {
        pj.keyTarget = document.body;
        // The numbers are drawn with the page's own webfont; wait for it so every page paints
        // identical frames instead of racing the font load.
        try { await document.fonts.load('bold 55px "Clear Sans"'); await document.fonts.ready; } catch (e) {}
        // application.js builds the GameManager inside a rAF callback; one virtual frame runs it.
        for (let i = 0; i < 4 && !gm; i++) pj.frames(1);
        if (!gm) throw new Error('GameManager never constructed');
        gm.restart();     // fresh grid, cleared message, score 0
        pj.frames(2);     // let the actuator's rAF chain repaint the DOM
      },

      // One decision = one move. The grid updates synchronously inside the keydown handler;
      // two frames flush the actuator so the DOM matches the grid as well.
      applyAction(i) {
        pj.press(this.actions[i].keys[0], document.body);
        pj.frames(2);
      },

      score() { return gm ? gm.score : 0; },
      done() { return !!(gm && (gm.over || gm.won)); },

      canvas() {
        if (!cv) { cv = document.createElement('canvas'); cv.width = BOARD; cv.height = BOARD; }
        const ctx = cv.getContext('2d');
        const b = boardStyle();
        ctx.fillStyle = b.bg;
        ctx.fillRect(0, 0, BOARD, BOARD);
        const rows = gridRows();
        for (let y = 0; y < SIZE; y++) {
          for (let x = 0; x < SIZE; x++) {
            const px = PAD + x * STEP, py = PAD + y * STEP;
            const v = rows[y][x];
            if (!v) { ctx.fillStyle = b.cell; ctx.fillRect(px, py, CELL, CELL); continue; }
            const s = tileStyle(v);
            ctx.fillStyle = s.bg;
            ctx.fillRect(px, py, CELL, CELL);
            ctx.fillStyle = s.fg;
            ctx.font = s.font;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(v), px + CELL / 2, py + CELL / 2 + 2);
          }
        }
        return cv;
      },

      info() {
        return {
          grid: gridRows(),           // grid[row][col], 0 = empty
          score: gm ? gm.score : 0,
          over: !!(gm && gm.over),
          won: !!(gm && gm.won),
          empty: gm && gm.grid ? gm.grid.availableCells().length : 16,
        };
      },
    });
  });
})();
