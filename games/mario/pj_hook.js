// PlayJev hook for robertkleffner/mariohtml5 (Infinite Mario in HTML5, ported from Notch's Java).
//
// Loop: Enjine.GameTimer runs setInterval(tick, 1000/30), so the game advances at 30 Hz and one
// game tick is two shim frames. State lives on the Application instance (captured below, because
// main.html keeps no reference to it) and on the globals Mario.MarioCharacter and Mario.GlobalMapState.
// States: LoadingState -> TitleState -> MapState -> LevelState (-> LoseState). start() walks that
// chain with real key presses: S on the title screen, arrows along the world map roads to the
// nearest playable level tile, S again to enter it.
(function () {
  'use strict';

  var JUMP = 's', RUN = 'a', LEFT = 'ArrowLeft', RIGHT = 'ArrowRight', UP = 'ArrowUp', DOWN = 'ArrowDown';
  var TICK_MS = 1000 / 30;            // Enjine.GameTimer.FramesPerSecond
  var INTRO_TICKS = 30;               // LevelState opens with a shrinking black circle, gone after 26 ticks
  var WIN_BONUS = 1000;               // added to the pixel progress when the level is finished
  var startX = 32, maxX = 32;         // Mario.Character.Initialize puts Mario at X = 32
  var tickCount = 0;                  // Application.Update calls, counted by the wrapper below
  var mapFallback = false;

  // main.html boots through jQuery from a CDN. Keep a stand-in so the page also works offline;
  // the real jQuery overwrites it when it loads, and start() boots the app itself if neither ran.
  if (!window.$) {
    window.$ = function () {
      return { ready: function (fn) {
        if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
        else fn();
      } };
    };
  }

  function app() { return window.__pjApp; }
  function state() { var a = app(); return a && a.stateContext ? a.stateContext.State : null; }
  function mario() { return window.Mario && Mario.MarioCharacter; }
  function inLevel() { return state() instanceof Mario.LevelState; }

  // Advance the virtual clock until exactly n game ticks have run. Half-interval increments so a
  // single increment never fires two ticks, whatever the floating point remainder of the timer is.
  function ticks(n) {
    var target = tickCount + n, guard = 0;
    while (tickCount < target && guard++ < 8 * n + 8) pj.tick(TICK_MS / 2);
    if (tickCount < target) pj.errors.push('mario: game timer did not advance (' + tickCount + '/' + target + ')');
  }

  function bootIfNeeded() {
    if (app()) return;
    // LoadingState.Enter registers the sprite sheets; if that already happened, an Application we
    // did not see is running. Booting another one would run two games on one canvas.
    if (Object.keys(Enjine.Resources.Images).length) { pj.errors.push('mario: page booted an Application the hook did not capture'); return; }
    var a = new Enjine.Application(); a.Initialize(new Mario.LoadingState(), 320, 240);
  }

  // LoadingState waits for the sprite sheets, which load over the real network, not the fake clock.
  function waitForImages() {
    return new Promise(function (resolve) {
      var tries = 0;
      (function poll() {
        var imgs = Enjine.Resources.Images, names = Object.keys(imgs), ok = names.length > 0;
        for (var i = 0; i < names.length; i++) if (!imgs[names[i]].complete) ok = false;
        if (ok || ++tries > 400) return resolve(ok);
        pj.real.setTimeout(poll, 15);
      })();
    });
  }

  // ---- world map navigation -------------------------------------------------
  function tileOf(ms) { return { x: (ms.XMario / 16) | 0, y: (ms.YMario / 16) | 0 }; }

  function walkable(ms, x, y) {
    if (!ms.Level[x] || ms.Level[x][y] === undefined) return false;
    var t = ms.Level[x][y];
    return t === Mario.MapTile.Road || t === Mario.MapTile.Level;
  }

  // MapState.Update enters a level tile when its Data is not -11 (Mario's start tile),
  // not 0 (already cleared) and greater than -10 (cleared fortress).
  function enterable(ms, x, y) {
    if (!ms.Level[x] || ms.Level[x][y] !== Mario.MapTile.Level) return false;
    var d = ms.Data[x][y];
    return d !== -11 && d !== 0 && d > -10;
  }

  // Breadth-first over road and level tiles; returns the arrow key for the first hop
  // towards the closest level Mario is allowed to enter.
  function stepTowardsLevel(ms) {
    var dirs = [[1, 0, RIGHT], [-1, 0, LEFT], [0, -1, UP], [0, 1, DOWN]];
    var t = tileOf(ms), seen = {}, q = [[t.x, t.y, null]], i = 0;
    seen[t.x + ',' + t.y] = 1;
    while (i < q.length) {
      var node = q[i++], x = node[0], y = node[1], first = node[2];
      if (first && enterable(ms, x, y)) return first;
      for (var d = 0; d < dirs.length; d++) {
        var nx = x + dirs[d][0], ny = y + dirs[d][1], k = nx + ',' + ny;
        if (seen[k] || !walkable(ms, nx, ny)) continue;
        seen[k] = 1;
        q.push([nx, ny, first || dirs[d][2]]);
      }
    }
    return null;
  }

  // One arrow press makes Mario walk a whole road segment (MapState.TryWalking), so hold the key
  // for a single tick and then wait out the walk before deciding again.
  function walkToLevel() {
    var ms = state();
    for (var hop = 0; hop < 40; hop++) {
      var t = tileOf(ms);
      if (enterable(ms, t.x, t.y)) return true;
      var key = stepTowardsLevel(ms);
      if (!key) return false;
      pj.keyDown(key); ticks(1); pj.keyUp(key);
      for (var i = 0; i < 300 && (ms.MoveTime > 0 || ms.XMarioA !== 0 || ms.YMarioA !== 0); i++) ticks(1);
      var after = tileOf(ms);
      if (after.x === t.x && after.y === t.y) return false;   // the hop did nothing, give up
    }
    return false;
  }

  // ---- tile window for info() ----------------------------------------------
  function tileChar(level, x, y) {
    if (x < 0 || y < 0 || x >= level.Width || y >= level.Height) return '#';
    var b = Mario.Tile.Behaviors[level.Map[x][y] & 0xff] | 0;
    if (b & Mario.Tile.PickUpable) return 'o';
    if (b & (Mario.Tile.Special | Mario.Tile.Bumpable)) return '?';
    if (b & Mario.Tile.BlockAll) return '#';
    if (b & Mario.Tile.BlockUpper) return '^';
    return '.';
  }

  var ENEMY_NAMES = ['redKoopa', 'greenKoopa', 'goomba', 'spiky', 'flower'];
  function spriteKind(s) {
    // FlowerEnemy extends Enemy, so it must be tested first or piranha plants report as their Type's name.
    if (s instanceof Mario.FlowerEnemy) return 'piranha';
    if (s instanceof Mario.Enemy) return ENEMY_NAMES[s.Type] || 'enemy';
    if (s instanceof Mario.Shell) return 'shell';
    if (s instanceof Mario.BulletBill) return 'bullet';
    if (s instanceof Mario.Mushroom) return 'mushroom';
    if (s instanceof Mario.FireFlower) return 'fireFlower';
    if (s instanceof Mario.Fireball) return 'fireball';
    return null;
  }

  // Listen on document, not window: jQuery's ready handler (which creates the Application) is a
  // document-level DOMContentLoaded listener, and this init script runs before jQuery loads, so
  // this listener is registered first and fires first. A window listener would fire after the
  // app already existed and the hook would boot a second game on the same canvas.
  document.addEventListener('DOMContentLoaded', function () {
    var origInit = Enjine.Application.prototype.Initialize;
    Enjine.Application.prototype.Initialize = function () {
      origInit.apply(this, arguments);
      window.__pjApp = this;
    };
    // One call = one game tick. Also keep the running maximum of Mario's X per tick, so progress
    // between two observations is not lost.
    var origUpdate = Enjine.Application.prototype.Update;
    Enjine.Application.prototype.Update = function () {
      tickCount++;
      var r = origUpdate.apply(this, arguments);
      var m = mario();
      if (m && this.stateContext && this.stateContext.State instanceof Mario.LevelState && m.DeathTime === 0 && m.X > maxX) maxX = m.X;
      return r;
    };

    pj.register({
      id: 'mario',
      stepFrames: 6,          // 6 shim frames = 100 ms = 3 game ticks
      actions: [
        { name: 'noop', description: 'do nothing for a moment', keys: [] },
        { name: 'left', description: 'walk to the left', keys: [LEFT] },
        { name: 'right', description: 'walk to the right', keys: [RIGHT] },
        { name: 'jump', description: 'jump up', keys: [JUMP] },
        { name: 'right jump', description: 'jump while moving to the right', keys: [RIGHT, JUMP] },
        { name: 'right run', description: 'run to the right at full speed', keys: [RIGHT, RUN] },
        { name: 'right run jump', description: 'run to the right at full speed and jump', keys: [RIGHT, RUN, JUMP] },
      ],

      async start() {
        pj.keyTarget = document.body;   // Enjine.KeyboardInput listens on document.onkeydown/onkeyup
        pj.releaseAll();
        bootIfNeeded();
        await waitForImages();
        mapFallback = false;

        for (var i = 0; i < 400 && !(state() instanceof Mario.TitleState); i++) ticks(1);

        // title -> world map
        pj.keyDown(JUMP);
        for (i = 0; i < 60 && !(state() instanceof Mario.MapState); i++) ticks(1);
        pj.keyUp(JUMP);
        ticks(2);              // MapState only arms S once it has seen the key released

        var reached = (state() instanceof Mario.MapState) && walkToLevel();
        if (reached) {
          pj.keyDown(JUMP);
          for (i = 0; i < 30 && !inLevel(); i++) ticks(1);
          pj.keyUp(JUMP);
        }
        if (!inLevel()) {
          // No reachable level on the generated map: drop into an overground level directly.
          mapFallback = true;
          app().stateContext.ChangeState(new Mario.LevelState(1, Mario.LevelType.Overground));
          ticks(1);
        }

        ticks(INTRO_TICKS);    // let the opening iris finish so the first frame shows the level
        startX = maxX = mario().X;
        window.__pjJumpHeld = false;
      },

      // Hold the action's keys for k shim frames (k/2 game ticks). Extra rule for the jump key:
      // Mario can only start a new jump on a tick where it was up (Character.Move sets
      // MayJump = (OnGround || Sliding) && !S), so when he is on the ground or sliding down a wall
      // with the key still held from the previous step, give the key one tick off first (that step
      // is then 4 ticks). While he is airborne the key stays down, which is what makes a jump go
      // higher the longer it is held.
      applyAction(i, k) {
        var keys = pj.actions[i].keys, wantJump = keys.indexOf(JUMP) >= 0, m = mario();
        var n = Math.max(1, Math.round(k / 2));
        if (wantJump && window.__pjJumpHeld && m && (m.OnGround || m.Sliding) && !m.MayJump) {
          pj.releaseAll();
          for (var j = 0; j < keys.length; j++) if (keys[j] !== JUMP) pj.keyDown(keys[j]);
          ticks(1);
        }
        pj.releaseAll();
        for (var q = 0; q < keys.length; q++) pj.keyDown(keys[q]);
        ticks(n);
        pj.releaseAll();
        window.__pjJumpHeld = wantJump;
      },

      // Progress along the level, in pixels, as a running maximum, plus a bonus for reaching
      // the exit. The game's own on-screen score is always 00000000, and coins are too sparse
      // to learn from, so distance is the signal.
      score() {
        var m = mario();
        var s = Math.max(0, Math.floor(maxX - startX));
        if (m && m.WinTime > 0) s += WIN_BONUS;
        return s;
      },

      done() {
        var m = mario();
        if (!m || !inLevel()) return true;
        return m.DeathTime > 0 || m.WinTime > 0;
      },

      canvas() { return document.getElementById('canvas'); },

      info() {
        var m = mario(), st = state();
        var out = {
          x: m ? Math.round(m.X * 100) / 100 : null,
          y: m ? Math.round(m.Y * 100) / 100 : null,
          xa: m ? Math.round(m.Xa * 100) / 100 : null,
          ya: m ? Math.round(m.Ya * 100) / 100 : null,
          maxX: Math.round(maxX),
          coins: m ? m.Coins : 0,
          lives: m ? m.Lives : 0,
          onGround: m ? !!m.OnGround : false,
          large: m ? !!m.Large : false,
          fire: m ? !!m.Fire : false,
          dead: m ? m.DeathTime > 0 : false,
          won: m ? m.WinTime > 0 : false,
          ticks: tickCount,
          mapFallback: mapFallback,
        };
        if (!(st instanceof Mario.LevelState) || !st.Level) return out;
        out.timeLeft = Math.round(st.TimeLeft);
        out.levelWidth = st.Level.Width * 16;
        out.exitX = st.Level.ExitX * 16;
        out.levelType = st.LevelType;
        out.levelString = m.LevelString;

        var mx = (m.X / 16) | 0, my = (m.Y / 16) | 0, rows = [], x, y;
        for (y = my - 7; y <= my + 5; y++) {
          var row = '';
          for (x = mx - 8; x <= mx + 12; x++) row += tileChar(st.Level, x, y);
          rows.push(row);
        }
        out.tiles = { originX: (mx - 8) * 16, originY: (my - 7) * 16, cell: 16, rows: rows };

        var near = [];
        for (var i = 0; i < st.Sprites.Objects.length && near.length < 12; i++) {
          var s = st.Sprites.Objects[i];
          if (s === m) continue;
          var kind = spriteKind(s);
          if (!kind) continue;
          near.push({ kind: kind, dx: Math.round(s.X - m.X) || 0, dy: Math.round(s.Y - m.Y) || 0 });   // || 0 turns -0 into 0
        }
        out.sprites = near;
        return out;
      },
    });
  });
})();
