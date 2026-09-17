// PlayJev hook for nebez/floppybird (jQuery DOM game: bird and pipes are absolutely positioned divs).
//
// The game's own physics runs in `gameloop` (setInterval at 1000/60 ms) and pipes spawn in
// `updatePipes` (setInterval 1400 ms); both are top-level function declarations, so they are
// writable properties of window and can be wrapped here without touching main.js.
//
// Pipe motion in the upstream game is a pure CSS animation (`animPipe`: left 900px -> -100px over
// 7500 ms) that the shim switches off and that would run on wall-clock time anyway. The wrapper
// below sets each pipe's `left` from the virtual clock with the same linear law before every
// physics tick, so collisions and scoring read exactly the positions the animation would give.
//
// Frames: the play area (sky + pipes + bird + land, 320 x 525) is painted onto an offscreen canvas
// from DOM state and the game's sprite sheets, the same way Snake paints its grid (about 2 ms).
(function () {
  const W = 320, SKY_H = 420, LAND_H = 105;          // #flyarea is 420 px; land is 20% of a 525 px viewport
  const PIPE_X0 = 900, PIPE_X1 = -100, PIPE_MS = 7500; // animPipe keyframes
  const PIPE_W = 52, CAP_H = 26, BIRD_W = 34, BIRD_H = 24, BIRD_LEFT = 60;
  const pxPerMs = (PIPE_X0 - PIPE_X1) / PIPE_MS;     // 0.1333 px/ms = 2.22 px per 60 Hz frame

  const img = {}; let imagesReady = null;
  function loadImages() {
    const names = ['sky', 'land', 'bird', 'pipe', 'pipe-up', 'pipe-down'];
    for (let d = 0; d < 10; d++) names.push('font_big_' + d);
    imagesReady = Promise.all(names.map((n) => new Promise((res) => {
      const im = new Image(); im.onload = res; im.onerror = res; im.src = 'assets/' + n + '.png'; img[n] = im;
    })));
  }

  // ---- pipe motion on the virtual clock (replaces the CSS animation) ----
  function placePipes() {
    const t = pj.now();
    const els = document.querySelectorAll('#flyarea .pipe');
    for (const el of els) {
      if (el._pjT0 == null) el._pjT0 = t;                       // creation time, stamped on first sight
      const left = Math.max(PIPE_X1, PIPE_X0 - (t - el._pjT0) * pxPerMs);
      el.style.left = left + 'px';
    }
  }
  function pipeGeom(el) {
    const up = el.firstElementChild;
    const gapTop = parseFloat(up.style.height);
    return { x: parseFloat(el.style.left), gapTop, gapBottom: gapTop + pipeheight };
  }

  // ---- painting ----
  let cv = null, deadAt = null;
  function drawRepeatX(ctx, im, offset, y, h) {
    if (!im || !im.naturalWidth) return;
    const w = im.naturalWidth; let x = ((offset % w) + w) % w - w;
    for (; x < W; x += w) ctx.drawImage(im, x, y, w, h == null ? im.naturalHeight : h);
  }
  function paint() {
    if (!cv) { cv = document.createElement('canvas'); cv.width = W; cv.height = SKY_H + LAND_H; }
    const ctx = cv.getContext('2d');
    const t = pj.now(); const dead = currentstate === states.ScoreScreen;
    if (dead && deadAt == null) deadAt = t;
    const ta = dead ? deadAt : t;                                 // decorations freeze on death, like the game
    // sky: flat colour with the sky strip aligned to the bottom of the sky, scrolling 275 px per 7 s
    ctx.fillStyle = '#4ec0ca'; ctx.fillRect(0, 0, W, SKY_H);
    if (img.sky.naturalHeight) drawRepeatX(ctx, img.sky, -(ta * 275 / 7000), SKY_H - img.sky.naturalHeight);
    // pipes: body is the 52 x 1 strip stretched, caps are 52 x 26
    for (const el of document.querySelectorAll('#flyarea .pipe')) {
      const g = pipeGeom(el); const x = Math.round(g.x);
      if (x > W || x < -PIPE_W) continue;
      const loTop = g.gapBottom;                                   // lower pipe starts where the gap ends
      ctx.drawImage(img.pipe, x, 0, PIPE_W, g.gapTop);
      ctx.drawImage(img['pipe-down'], x, g.gapTop - CAP_H);
      ctx.drawImage(img.pipe, x, loTop, PIPE_W, SKY_H - loTop);
      ctx.drawImage(img['pipe-up'], x, loTop);
    }
    // land: flat colour plus the land strip scrolling 335 px per 2516 ms (the pipe speed)
    ctx.fillStyle = '#ded895'; ctx.fillRect(0, SKY_H, W, LAND_H);
    drawRepeatX(ctx, img.land, -(ta * 335 / 2516), SKY_H);
    // bird: layout top from style.top, rotation and death drop from the computed transform matrix
    const el = document.getElementById('player');
    let angle = 0, tx = 0, ty = 0;
    const m = /matrix\(([^)]+)\)/.exec(getComputedStyle(el).transform);
    if (m) { const v = m[1].split(',').map(Number); angle = Math.atan2(v[1], v[0]); tx = v[4]; ty = v[5]; }
    const top = parseFloat(el.style.top) || 0;
    const flap = Math.floor(ta / 75) % 4;                         // animBird: 300 ms, steps(4)
    ctx.save(); ctx.translate(BIRD_LEFT + BIRD_W / 2 + tx, top + BIRD_H / 2 + ty); ctx.rotate(angle);
    ctx.drawImage(img.bird, 0, flap * BIRD_H, BIRD_W, BIRD_H, -BIRD_W / 2, -BIRD_H / 2, BIRD_W, BIRD_H);
    ctx.restore();
    // running score, top centre, in the game's big digit font
    const digits = String(score).split('');
    let tw = 0; for (const d of digits) tw += img['font_big_' + d].naturalWidth + 2;
    let dx = Math.round((W - tw) / 2);
    for (const d of digits) { const di = img['font_big_' + d]; ctx.drawImage(di, dx, 20); dx += di.naturalWidth + 2; }
    return cv;
  }

  window.addEventListener('DOMContentLoaded', () => {
    loadImages();
    const origLoop = window.gameloop, origPipes = window.updatePipes;
    window.gameloop = function () { placePipes(); return origLoop.apply(this, arguments); };
    window.updatePipes = function () { const r = origPipes.apply(this, arguments); placePipes(); return r; };

    pj.register({
      id: 'flappy',
      actions: [
        { name: 'flap', description: 'flap the wings once to rise', keys: ['Space'] },
        { name: 'wait', description: 'do nothing and let the bird fall', keys: [] },
      ],
      stepFrames: 5,   // 5 physics ticks = 83 ms of game time; a pipe moves 11 px per step
      async start() {
        if (currentstate !== states.SplashScreen) pj.tick(1);   // jQuery ready normally ran on DOMContentLoaded
        await imagesReady;
        // the painter assumes the 320 x 525 viewport from pj.json: sky (= #flyarea) 0..420, land 420..525
        const land = document.getElementById('land'), fly = document.getElementById('flyarea');
        if (land.offsetTop !== SKY_H || fly.offsetTop !== 0 || flyArea !== SKY_H)
          throw new Error('flappy layout mismatch: land top ' + land.offsetTop + ', flyarea top ' + fly.offsetTop);
        deadAt = null;
        pj.keyDown('Space'); pj.keyUp('Space');                    // $(document).keydown -> screenClick -> startGame
        if (currentstate !== states.GameScreen) throw new Error('flappy did not leave the splash screen');
      },
      score() { return score; },
      done() { return currentstate === states.ScoreScreen; },
      canvas() { return paint(); },
      info() {
        const all = Array.from(document.querySelectorAll('#flyarea .pipe')).map(pipeGeom);
        const next = pipes[0] ? pipeGeom(pipes[0][0]) : null;
        return { birdX: BIRD_LEFT, birdY: position, velocity, rotation, nextPipe: next, pipes: all,
                 flyArea, landTop: SKY_H, pipeWidth: PIPE_W, gap: pipeheight, t: pj.now() };
      },
    });
  });
})();
