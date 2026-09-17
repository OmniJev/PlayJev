// PlayJev hook for daleharvey/pacman (canvas game, one setInterval main loop at 30 Hz).
//
// All game state lives in closures (the PACMAN module, Pacman.User, Pacman.Ghost, Pacman.Map);
// pacman.js carries a small additive patch that exposes read-only getters for it (see NOTES.md).
// PACMAN.init runs from a setTimeout(0) in index.html, and its loop only starts once six audio
// files report canplaythrough; the hook replaces Pacman.Audio with a silent stub so that the start
// is instant and independent of media decoding (the shim mutes audio anyway).
(function () {
  const TICK_MS = 1000 / 30;                     // Pacman.FPS = 30
  const DIR = { 3: 'up', 1: 'down', 2: 'left', 11: 'right', 4: 'none' };
  const STATE = { 5: 'waiting', 6: 'pause', 7: 'playing', 8: 'countdown', 9: 'eaten_pause', 10: 'dying' };
  let startLives = 3;

  function user() { return PACMAN.getUser(); }
  function tickUntil(pred, maxTicks) {
    for (let n = 0; n < maxTicks && !pred(); n++) pj.tick(TICK_MS);
    return pred();
  }
  // Tick through the "Starting in: 4..1" countdown (120 ticks). Its last tick only redraws the maze; the
  // game paints the sprites on the next tick (which also moves them and eats the pellet next to the
  // spawn). Paint Pac-Man and the ghosts now with the game's own draw routines, on the game's canvas,
  // so the first observation is complete while the clock and the score stay at the start of play.
  function enterPlaying() {
    if (!tickUntil(() => PACMAN.getState() === PLAYING, 200)) return false;
    const ctx = document.querySelector('#pacman canvas').getContext('2d');
    for (const g of PACMAN.getGhosts()) g.draw(ctx);
    user().draw(ctx);
    return true;
  }

  window.addEventListener('DOMContentLoaded', () => {
    Pacman.Audio = function () {
      return { load(name, path, cb) { if (typeof cb === 'function') cb(); }, play() {}, pause() {}, resume() {}, disableSound() {} };
    };

    pj.register({
      id: 'pacman',
      actions: [
        { name: 'up', description: 'head up at the next opening', keys: ['ArrowUp'] },
        { name: 'down', description: 'head down at the next opening', keys: ['ArrowDown'] },
        { name: 'left', description: 'head left at the next opening', keys: ['ArrowLeft'] },
        { name: 'right', description: 'head right at the next opening', keys: ['ArrowRight'] },
      ],
      stepFrames: 6,   // 6 shim frames = 100 ms = 3 main-loop ticks; Pac-Man moves 0.6 of a block per step
      start() {
        pj.tick(1);                                  // fires index.html's setTimeout(PACMAN.init, 0): build game, start loop
        if (typeof PACMAN.getState !== 'function') throw new Error('pacman.js patch missing (no PACMAN.getState)');
        if (!PACMAN.getUser()) throw new Error('PACMAN.init did not run (Modernizr check failed or setTimeout not fired)');
        if (PACMAN.getState() !== WAITING) throw new Error('pacman not in WAITING state after init: ' + PACMAN.getState());
        pj.keyDown('n'); pj.keyUp('n');              // startNewGame -> COUNTDOWN ("Starting in: 4..1", 120 ticks)
        if (!enterPlaying()) throw new Error('pacman countdown did not finish');
        startLives = user().getLives();
      },
      // One step = k/2 main-loop ticks (3 by default), counted on the game's own tick counter so a
      // step is always whole ticks regardless of float drift between 16.67 ms frames and 33.33 ms ticks.
      applyAction(i, k) {
        const key = this.actions[i].keys[0];
        const n = Math.max(1, Math.round(k * pj.FRAME_MS / TICK_MS));
        pj.keyDown(key);                             // document keydown (capture) -> user.keyDown sets `due`
        const t0 = PACMAN.getTick();
        for (let g = 0; g < n * 8 && PACMAN.getTick() < t0 + n; g++) pj.tick(TICK_MS / 4);
        pj.releaseAll();
      },
      // A cleared level goes WAITING -> COUNTDOWN -> PLAYING; skip the 4 s countdown so steps keep meaning.
      afterStep() {
        if (PACMAN.getState() === COUNTDOWN && user().getLives() === startLives) enterPlaying();
      },
      score() { return user().theScore(); },
      // One life per episode: over as soon as the collision puts the game in DYING (same tick), or if lives dropped.
      done() { const s = PACMAN.getState(); return s === DYING || user().getLives() < startLives; },
      canvas() { return document.querySelector('#pacman canvas'); },
      info() {
        const u = user(), m = PACMAN.getMap(), up = u.getPosition();
        let pellets = 0, pills = 0; const rows = [];
        for (let y = 0; y < m.height; y++) {
          let r = '';
          for (let x = 0; x < m.width; x++) {
            const b = m.block({ y, x });
            if (b === Pacman.BISCUIT) { pellets++; r += '.'; } else if (b === Pacman.PILL) { pills++; r += 'o'; }
            else if (b === Pacman.WALL) r += '#'; else if (b === Pacman.BLOCK) r += '='; else r += ' ';
          }
          rows.push(r);
        }
        return {
          state: STATE[PACMAN.getState()] || PACMAN.getState(), lives: u.getLives(), level: PACMAN.getLevel(), tick: PACMAN.getTick(),
          pac: { x: up.x / 10, y: up.y / 10, dir: DIR[u.getDirection()] },
          ghosts: PACMAN.getGhosts().map((g) => { const p = g.getPosition();
            return { x: p.x / 10, y: p.y / 10, dir: DIR[g.getDirection()], vulnerable: g.isVunerable(), eaten: !g.isDangerous() }; }),
          pellets, pills, map: rows.join('|'),
        };
      },
    });
  });
})();
