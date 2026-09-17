// PlayJev hook for patorjk/JavaScript-Snake (DOM game, moves on a setTimeout chain).
// State comes from mySnakeBoard.grid: 0 empty, 1 snake or edge wall, -1 food.
// The board is DOM divs, so frame() paints the same blocks onto a canvas with the theme's
// computed colours (a page screenshot costs 30 ms, this costs about 2 ms).
(function () {
  const BLOCK = 20;
  function board() { return mySnakeBoard; }
  function headEl() { return document.getElementById('snake-snakehead-alive') || document.querySelector('.snake-snakebody-dead'); }
  function headCell() {
    const el = headEl(); if (!el) return null;
    return { row: Math.round(parseInt(el.style.top, 10) / BLOCK), col: Math.round(parseInt(el.style.left, 10) / BLOCK) };
  }
  let cv = null, colors = null;
  function themeColors(field) {
    if (colors) return colors;
    const cs = (el) => el && getComputedStyle(el);
    const body = document.querySelector('.snake-snakebody-block'); const food = document.querySelector('.snake-food-block');
    colors = {
      field: cs(field).backgroundColor, border: cs(field).borderColor,
      body: body ? cs(body).backgroundColor : '#ffff00', food: food ? cs(food).backgroundColor : '#ff0000',
    };
    return colors;
  }
  window.addEventListener('DOMContentLoaded', () => {
    pj.register({
      id: 'snake',
      actions: [
        { name: 'up', description: 'turn the snake to move up', keys: ['ArrowUp'] },
        { name: 'down', description: 'turn the snake to move down', keys: ['ArrowDown'] },
        { name: 'left', description: 'turn the snake to move left', keys: ['ArrowLeft'] },
        { name: 'right', description: 'turn the snake to move right', keys: ['ArrowRight'] },
      ],
      start() {
        pj.keyUp('Space', window);  // closes the welcome dialog, board state -> READY
        pj.tick(50);
        window.__pjStarted = true; window.__pjMax = 0;
      },
      // One decision = one snake move: press the arrow, then advance until the head moves.
      applyAction(i) {
        const key = this.actions[i].keys[0]; const tgt = board().getBoardContainer();
        const before = headCell();
        pj.keyDown(key, tgt); pj.keyUp(key, tgt);
        const speed = board().getSpeed ? board().getSpeed() : 80;
        for (let t = 0; t < speed * 3; t += 5) {
          pj.tick(5);
          const h = headCell();
          if (!h || !before || h.row !== before.row || h.col !== before.col || board().getBoardState() === 0) break;
        }
      },
      // grid marks the edge ring and the snake both as 1; length = ones minus the ring.
      // Score = longest length reached this episode (the grid drops the tail cell on the fatal move).
      score() { const g = board().grid; let n = 0; for (const r of g) for (const v of r) if (v === 1) n++;
        const len = n - (2 * g.length + 2 * g[0].length - 4); window.__pjMax = Math.max(window.__pjMax || 0, len); return window.__pjMax; },
      done() { return window.__pjStarted === true && board().getBoardState() === 0; },
      canvas() {
        const field = board().getPlayingFieldElement(); const c = themeColors(field); const g = board().grid;
        const w = field.offsetWidth, h = field.offsetHeight;
        if (!cv || cv.width !== w || cv.height !== h) { cv = document.createElement('canvas'); cv.width = w; cv.height = h; }
        const ctx = cv.getContext('2d');
        ctx.fillStyle = c.field; ctx.fillRect(0, 0, w, h);
        const head = headCell(); const dead = !!document.querySelector('.snake-snakebody-dead');
        for (let r = 1; r < g.length - 1; r++) for (let col = 1; col < g[r].length - 1; col++) {
          const v = g[r][col]; if (v === 0) continue;
          const x = (col - 1) * BLOCK, y = (r - 1) * BLOCK;   // playing field starts one block in
          if (v === -1) { ctx.fillStyle = c.food; ctx.fillRect(x, y, BLOCK, BLOCK); continue; }
          const isHead = head && head.row === r && head.col === col;
          ctx.fillStyle = isHead ? (dead ? '#888888' : '#ffffff') : c.body;
          ctx.fillRect(x + 1, y + 1, BLOCK - 2, BLOCK - 2);
        }
        return cv;
      },
      info() {
        const g = board().grid; let food = null;
        for (let r = 0; r < g.length; r++) for (let c = 0; c < g[r].length; c++) if (g[r][c] === -1) food = [r, c];
        return { rows: g.length, cols: g[0].length, head: headCell(), food, grid: g.map((r) => r.map((v) => (v === -1 ? 'F' : v === 1 ? '#' : '.')).join('')).join('|') };
      },
    });
  });
})();
