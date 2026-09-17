// PlayJev hook for StrykerKKD/SpaceInvaders (Phaser 2.0.2 + RequireJS, canvas game, entry index.html).
//
// The game object is a local `var _game` inside main.js's require callback, but Phaser registers
// every instance in the global `Phaser.GAMES`, so no vendor patch is needed. Score, lives and
// health live in the HUD AMD module's closure and are only pushed out through
// HUD.updateScoreText / updateLivesText / updateHealthText, so start() wraps those methods on the
// module object (fetched with the synchronous RequireJS `require('module/HUD')`) and latches the
// values. Ship and alien positions are read from `game.world.children` by texture key.
//
// Loop: Phaser's RequestAnimationFrame runner -> game.update(Date.now()). Both the rAF and Date.now
// are the shim's, so the game is frozen between pj.step calls. RequireJS module fetches and the
// Phaser image loader run on real network time (they finish after window `load`), so start()
// interleaves real sleeps with single virtual frames, and never ticks while the loader is busy,
// which keeps the virtual clock at the same value in every episode.
//
// The ship fires by itself (Player.startShooting -> game.time.events.loop every 300 ms); the only
// keys the Play state reads are the left and right cursors. So there is no fire action: steering
// is the whole decision, and staying put is a real choice (stay lined up under a column, or sit
// between two incoming shots).
(function () {
  const ANCHOR_MS = 2000;      // virtual time at which every episode leaves the title screen
  const BOOT_TIMEOUT_MS = 30000;
  const ALIEN_ROW_PX = 50;     // Aliens.js: alien j sits at y = j * 50 inside the group

  function game() { return (window.Phaser && Phaser.GAMES && Phaser.GAMES[0]) || null; }
  function stateOf() {
    const g = game(); if (!g || !g.state) return null;
    const p = g.state._pendingState;
    return typeof p === 'string' ? p : g.state.current;
  }
  function world() { const g = game(); return g && g.world ? g.world.children : null; }
  function groupWithKey(key) {
    const ch = world(); if (!ch) return null;
    for (const c of ch) if (c.children && c.children.length && c.children[0].key === key) return c;
    return null;
  }
  function spriteWithKey(key) {
    const ch = world(); if (!ch) return null;
    for (const c of ch) if (c.key === key && !(c.children && c.children.length)) return c;   // sprites carry an empty children array
    return null;
  }
  function realSleep(ms) { return new Promise((r) => pj.real.setTimeout(r, ms)); }

  // Values latched from the HUD module calls (exact, and they survive the End state, which clears
  // the world). Reset by HUD.createStat, which Play.create calls once per game.
  const hud = { score: 0, lives: 3, health: 100 };
  function wrapHud() {
    let HUD = null;
    try { HUD = window.require('module/HUD'); } catch (e) { HUD = null; }
    if (!HUD || HUD.__pjWrapped) return !!HUD;
    const o = { createStat: HUD.createStat, updateScoreText: HUD.updateScoreText, updateLivesText: HUD.updateLivesText, updateHealthText: HUD.updateHealthText };
    HUD.createStat = function (score, health, lives) { hud.score = score; hud.health = health; hud.lives = lives; return o.createStat.apply(this, arguments); };
    HUD.updateScoreText = function (delta) { hud.score += Number(delta) || 0; return o.updateScoreText.apply(this, arguments); };
    HUD.updateLivesText = function (lives) { hud.lives = lives; return o.updateLivesText.apply(this, arguments); };
    HUD.updateHealthText = function (health) { hud.health = health; return o.updateHealthText.apply(this, arguments); };
    HUD.__pjWrapped = true;
    return true;
  }

  // Last snapshot of the Play world; the End state wipes the world, so info() keeps reporting the
  // positions the episode finished with (lives/health/score come from the HUD latch, so those are exact).
  let last = null;
  function snapshot() {
    if (stateOf() !== 'Play') return last;
    const ship = spriteWithKey('ship');
    const aliens = groupWithKey('invader');
    if (!ship || !aliens) return last;
    let alive = 0, lowestRow = -1;
    const pos = [], cols = new Array(10).fill(0);
    for (const a of aliens.children) {
      if (!a.alive) continue;
      alive++;
      const row = Math.round(a.y / ALIEN_ROW_PX);
      if (row > lowestRow) lowestRow = row;
      const col = Math.round(a.x / 48); if (col >= 0 && col < cols.length) cols[col]++;
      pos.push([Math.round(aliens.x + a.x), Math.round(aliens.y + a.y)]);
    }
    const shots = [];
    const eb = groupWithKey('enemyBullet');
    if (eb) for (const b of eb.children) if (b.alive) shots.push([Math.round(b.x), Math.round(b.y), Math.round(b.body.velocity.x), Math.round(b.body.velocity.y)]);
    const mine = [];
    const pb = groupWithKey('bullet');
    if (pb) for (const b of pb.children) if (b.alive) mine.push([Math.round(b.x), Math.round(b.y)]);
    last = {
      playerX: Math.round(ship.x), playerY: Math.round(ship.y),
      aliensAlive: alive, aliensTotal: aliens.children.length,
      lowestAlienRow: lowestRow,                       // 0 = top row, 3 = bottom row, -1 = none left
      lowestAlienY: lowestRow < 0 ? -1 : Math.round(aliens.y + lowestRow * ALIEN_ROW_PX),
      alienBlockX: Math.round(aliens.x),               // the block tweens between x = 100 and 200
      alienCols: cols,                                 // aliens alive per column, left to right
      aliens: pos,                                     // alive alien centres [x, y]
      enemyShots: shots,                               // [x, y, vx, vy]
      myShots: mine,
    };
    return last;
  }

  window.addEventListener('DOMContentLoaded', () => {
    pj.register({
      id: 'invaders',
      stepFrames: 5,     // 5 frames = 83 ms; the ship moves 200 px/s, so one step is about 17 px
      actions: [
        { name: 'left', description: 'move the ship left', keys: ['ArrowLeft'] },
        { name: 'right', description: 'move the ship right', keys: ['ArrowRight'] },
        { name: 'noop', description: 'keep the ship where it is', keys: [] },
      ],
      async start(seed) {
        last = null; hud.score = 0; hud.lives = 3; hud.health = 100;
        pj.keyTarget = window;   // Phaser binds keydown/keyup on window
        const t0 = pj.real.Date.now();
        // Scripts and images load on real time, but RequireJS defers module execution with
        // setTimeout(fn, 4) (virtual), and Phaser's state machine needs virtual frames. So: tiny
        // ticks while the modules load, no ticks while Phaser's image loader is busy, single frames
        // otherwise. The virtual time spent here is small and gets padded to ANCHOR_MS below.
        while (stateOf() !== 'Start') {
          if (pj.real.Date.now() - t0 > BOOT_TIMEOUT_MS) throw new Error('invaders: title screen never appeared (state ' + stateOf() + ')');
          const g = game();
          if (!g) { pj.tick(5); await realSleep(5); continue; }
          // main.js asks for Phaser.AUTO, which picks WebGL. In headless Chromium every WebGL page
          // funnels through the single GPU process (SwiftShader) and each frame grab is a GPU
          // readback: 29 env-steps/s on 8 pages. Phaser's own Canvas renderer draws the same
          // sprites in the page's process: 450 env-steps/s. Set before boot, where AUTO is resolved.
          if (!g.isBooted) g.renderType = Phaser.CANVAS;
          if (g.load && g.load.isLoading) { await realSleep(5); continue; }
          if (g.stage) g.stage.disableVisibilityChange = true;   // never pause on blur/hidden in headless
          pj.frames(1);
          await realSleep(1);
        }
        window.__pjBootVirtualMs = pj.now();
        const g = game();
        g.stage.disableVisibilityChange = true;
        if (g.paused) g.paused = false;
        if (!wrapHud()) throw new Error('invaders: HUD module not reachable through require');
        // Safety net: pad to a fixed virtual time so an unusual load path cannot shift the clock.
        pj.tick(ANCHOR_MS * Math.ceil((pj.now() + 1) / ANCHOR_MS) - pj.now());
        // Phaser seeds game.rnd from Date.now()*Math.random() at boot, i.e. identically for every
        // episode; re-sowing makes which alien shoots depend on the episode seed.
        g.rnd.sow([String(seed == null ? 1 : seed)]);
        pj.keyDown('Space'); pj.frames(2); pj.keyUp('Space');
        pj.frames(8);        // StateManager swaps in Play and builds the wave
        if (stateOf() !== 'Play') throw new Error('invaders: Play state did not start (state ' + stateOf() + ')');
        snapshot();
      },
      score() { return hud.score; },
      // Out of lives -> End state (Player.js). All aliens dead -> End too, but only when the alien
      // firing timer next runs (up to 200 ms later), so also flag the wave being cleared directly.
      done() {
        const st = stateOf();
        if (st === 'End') return true;
        const s = snapshot();
        return st === 'Play' && !!s && s.aliensAlive === 0;
      },
      canvas() { const g = game(); return g ? g.canvas : null; },
      info() {
        const s = snapshot();
        return Object.assign({ state: stateOf(), score: hud.score, lives: hud.lives, health: hud.health }, s || {});
      },
    });
  });
})();
