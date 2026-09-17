// PlayJev hook for jakesgordon/javascript-breakout (canvas game, setInterval loop at 60 fps whose
// dt comes from new Date). Game.start() returns the Breakout instance as Game.current:
//   Game.current.current      state machine: 'menu' before play and after the last life, 'game' while playing
//   .score.score / .lives     points (a brick is worth (25 - row) * 5) and lives left (3 to start, +1 per level)
//   .level                    index into Breakout.Levels (10 layouts, wraps to 0 after the last)
//   .ball {x,y,dx,dy,speed,moving}   .paddle {x,y,w,h}   .court {left,top,right,bottom,chunk,bricks[]}
// Keys: keydown/keyup on document, read through ev.keyCode. Left/Right start and stop the paddle,
// Space (on keyup) starts the game from the menu and launches a waiting ball during play.
(function () {
  const S = { started: false, lastLevel: 0, levelsDone: 0, tex: undefined, pat: null };
  const LEVEL_BONUS = 1000;
  let cv = null;

  function game() { return window.Game && Game.current; }
  function front() { return document.getElementById('canvas'); }
  function tapSpace() { pj.keyUp('Space', document); }   // the game binds Space to keyup

  // Walls plus court: the play area inside the 640x480 canvas (448x378 at this viewport).
  function playRect(g) {
    const c = g.court, s = c.wall.size;
    return { x: c.left - s, y: c.top - 2 * s, w: c.width + 2 * s, h: c.height + 2 * s };
  }

  // The canvas is 50% translucent grey over the page's stone texture; paint the texture under it so
  // the frame looks like the page instead of dark grey on JPEG black.
  function loadTexture() {
    if (S.tex !== undefined) return Promise.resolve(S.tex);
    return new Promise((res) => {
      const img = new Image(); let done = false;
      const fin = (v) => { if (!done) { done = true; S.tex = v; res(v); } };
      img.onload = () => fin(img); img.onerror = () => fin(null);
      pj.real.setTimeout(() => fin(null), 3000);   // real clock: the virtual one never runs here
      img.src = 'images/paving.jpg';
    });
  }

  function round(v, d) { const m = Math.pow(10, d == null ? 3 : d); return Math.round(Number(v) * m) / m; }

  // On document, not window: the game constructs itself from a DOMContentLoaded listener on
  // document, and this one is registered earlier (init script), so it runs first.
  document.addEventListener('DOMContentLoaded', () => {
    // The game keeps level and high score in runner.storage(), which is window.localStorage when
    // available: shared by every page of a browser context and kept across reloads, so one page's
    // episode would leak into another's (setLevel() re-reads it on lose, the HUD draws the high
    // score). Runner.storage() returns this property when it is already set; a fresh plain object
    // per page load keeps each page's storage its own. start() checks that it took.
    Game.Runner.localStorage = {};
    pj.register({
      id: 'breakout',
      stepFrames: 5,

      actions: [
        { name: 'left', description: 'move the paddle to the left', keys: ['ArrowLeft'] },
        { name: 'right', description: 'move the paddle to the right', keys: ['ArrowRight'] },
        { name: 'stay', description: 'keep the paddle where it is', keys: [] },
      ],

      async start(seed) {
        await loadTexture();
        const g = game();
        if (g.storage === window.localStorage) pj.errors.push('breakout: storage is window.localStorage, pages would share level and high score');
        // Level from the seed so episodes vary; paddle position from the seeded RNG (the game rolls
        // it once at page load, before the seed, and does not roll again on play).
        g.setLevel((Number(seed) >>> 0) % Breakout.Levels.length);
        g.paddle.reset();
        pj.keyTarget = document;
        tapSpace();            // menu -> game: score to 0, ball on the paddle, 2 s countdown starts
        tapSpace();            // launch now instead of waiting out the countdown
        pj.frames(1);          // one loop() so the canvas shows the game
        S.started = true; S.lastLevel = g.level; S.levelsDone = 0;
      },

      // Hold the action's keys for k frames. A ball parked on the paddle (after a lost life or a
      // cleared level) is launched at once, which is what a player does with Space anyway.
      applyAction(i, k) {
        const g = game(), a = this.actions[i];
        if (g.is('game') && !g.ball.moving) tapSpace();
        for (const key of a.keys) pj.keyDown(key);
        for (let n = 0; n < k; n++) pj.frames(1);   // one frame at a time keeps the 60 Hz interval aligned
        pj.releaseAll();
      },

      // A cleared level shows as a level change while still playing (winLevel -> nextLevel(true)).
      // lose() also calls setLevel(), so only count while the state is 'game'.
      afterStep() {
        const g = game();
        if (g.is('game') && g.level !== S.lastLevel) { S.levelsDone++; S.lastLevel = g.level; }
      },

      // Game score plus a bonus per cleared level. lose() keeps score.score, so the final step reads it.
      score() {
        const g = game(); if (!g) return 0;
        return Math.max(0, (Number(g.score.score) || 0) + LEVEL_BONUS * S.levelsDone);
      },
      done() { return S.started === true && game().is('menu'); },

      canvas() {
        const g = game(), src = front(), r = playRect(g);
        if (!cv || cv.width !== r.w || cv.height !== r.h) { cv = document.createElement('canvas'); cv.width = r.w; cv.height = r.h; }
        const ctx = cv.getContext('2d');
        if (S.tex && !S.pat) S.pat = ctx.createPattern(S.tex, 'repeat');
        ctx.fillStyle = S.pat || '#dcdcd4';
        ctx.fillRect(0, 0, r.w, r.h);
        ctx.drawImage(src, r.x, r.y, r.w, r.h, 0, 0, r.w, r.h);
        return cv;
      },

      info() {
        const g = game(), c = g.court, b = g.ball, p = g.paddle, done = this.done();
        const rows = [];
        for (let y = 0; y < c.cfg.ychunks; y++) rows.push(new Array(c.cfg.xchunks).fill('.'));
        for (const br of c.bricks) {
          if (br.hit) continue;
          for (let x = br.pos.x1; x <= br.pos.x2 && x < c.cfg.xchunks; x++) rows[br.pos.y][x] = br.c;
        }
        return {
          level: g.level, lives: done ? 0 : g.score.lives, score: g.score.score, levels_done: S.levelsDone,
          court: { left: c.left, top: c.top, right: c.right, bottom: c.bottom, chunk: c.chunk },
          ball: { x: round(b.x), y: round(b.y), dx: round(b.dx), dy: round(b.dy), speed: round(b.speed), moving: !!b.moving },
          paddle: { x: round(p.x), y: p.y, w: p.w },
          bricks_left: c.numbricks - c.numhits,
          bricks: rows.map((r) => r.join('')).join('|'),
        };
      },
    });
  });
})();
