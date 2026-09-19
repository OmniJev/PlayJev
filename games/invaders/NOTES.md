# invaders (StrykerKKD/SpaceInvaders, MIT)

Phaser 2.0.2 (bundled `phaser-no-physics.min.js`, Arcade physics only) plus RequireJS. Entry `index.html`
loads the readable AMD modules from `assets/javascript/`; `indexOpt.html` loads the r.js-compiled bundle
`assets/javascript/built/main-built.js`, the same code minified into one file. The plain entry is used
because the hook reaches the HUD module through `require('module/HUD')`, which is the same in both, but the
readable sources are what NOTES and any future debugging refer to. Vendored tree untouched (no patch).

## Loop and state
- Loop: Phaser's `RequestAnimationFrame` runner, `updateRAF -> game.update(Date.now())`. Both are the shim's,
  so nothing moves between steps. `Time.update` derives `physicsElapsed` from successive `Date.now()` values,
  which are exact frame multiples under the shim, so physics is repeatable.
- Timers: `game.time.events.loop` drives the ship's auto-fire (every 300 ms) and the aliens' fire (every 200 ms).
  The alien block is a Phaser tween (`x` 100 <-> 200, 2 s each way, yoyo). Aliens never descend; the only way
  an episode ends is the ship losing its 3 lives (100 health each, enemy bullet = 10 damage) or all 40 aliens
  dying (10 points each, so 400 is the maximum score). No new wave spawns: the game goes to its End state.
- The `Phaser.Game` instance is a local in main.js; the hook takes it from `Phaser.GAMES[0]`.
- Score/lives/health live in the HUD module closure and are only visible through
  `HUD.updateScoreText/updateLivesText/updateHealthText/createStat`. The hook wraps those four methods on the
  module object (fetched with the synchronous RequireJS `require('module/HUD')`) and latches the values, so
  the final score and `lives: 0` survive the End state, which clears the world. Positions (ship, aliens, bullets)
  are read from `game.world.children` by texture key (`ship`, `invader`, `bullet`, `enemyBullet`).
- Keys: `Phaser.Keyboard` listens on `window` and reads `keyCode`; `pj.keyTarget = window`.

## start(seed)
RequireJS defers module execution with `setTimeout(fn, 4)` and Phaser boots from a `setTimeout(0)`, both virtual;
Phaser's image loader runs on real network time and finishes after window `load`. So start() loops: 5 ms virtual
ticks while the modules load, no ticks while `game.load.isLoading`, single frames otherwise, until the Start
state exists (about 75 ms of virtual time). It then pads to a fixed anchor (2000 ms) so every episode has the same
clock, re-sows Phaser's RNG with the seed (`game.rnd.sow([seed])`; Phaser seeds it from `Date.now()*Math.random()`
at boot, i.e. identically for every page), presses Space, and gives the StateManager 10 frames to build the wave.
`game.stage.disableVisibilityChange = true` so a blur or hidden tab never pauses the game in headless runs.

Renderer: main.js asks for `Phaser.AUTO`, which resolves to WebGL. In headless Chromium all WebGL pages funnel
through the single GPU process (SwiftShader) and each frame grab is a GPU readback: 29 env-steps/s on 8 pages.
The hook sets `game.renderType = Phaser.CANVAS` before boot (Phaser's own fallback; same sprites, same output):
492 env-steps/s. Game logic is unaffected (identical score trajectories under both renderers).

## Step and actions
- One step = 5 frames = 83 ms. The ship moves at 200 px/s, so a step is about 17 px of travel; a bullet flies
  500 px/s and needs about 0.6 s to reach the alien block; enemy shots fly at 200 px/s.
- Actions: `left`, `right`, `noop` (hold position). The ship fires by itself (`Player.startShooting`), and the
  Play state only reads the left and right cursor keys, so the task sheet's `fire` / `left+fire` / `right+fire`
  would be dead options (the contract forbids those). Making fire key-driven would mean rewriting game logic,
  which the brief forbids. `noop` is a real choice: it keeps the ship under a column, or between two shots.
- score() = HUD score latch (10 per alien, monotone). done() = End state reached, or Play state with zero aliens
  alive (the game itself only switches to End when the alien fire timer next runs, up to 200 ms later).
- info(): `score, lives, health, playerX, playerY, aliensAlive, aliensTotal, lowestAlienRow (0..3), lowestAlienY,
  alienBlockX, alienCols (alive per column), aliens ([x,y] centres), enemyShots ([x,y,vx,vy]), myShots, state`.

## Numbers (local workstation, 8 pages)
- Bench: 449 to 492 env-steps/s across runs (the machine is shared), reset of 8 pages 0.3 to 1.2 s. Random policy:
  4 episodes in 200 steps, mean score 187.5, mean episode length 179 steps;
  in longer runs random play dies after 160 to 320 steps (13 to 27 s of game time) with 170 to 340 points.
- Greedy teacher (steer under the nearest lowest alien, sidestep incoming shots): clears the wave in about
  330 to 340 steps, score 400, 1 to 2 lives left.
- Determinism (`check_det.py`): same seed and action sequence gives identical score and info() on both pages at
  every step, through the end of the episode; a different seed gives a different trajectory.

## Quirks
- `Aliens._fireBullet` draws `integerInRange(0, n)` inclusive, so about 1 in n alien shots is skipped
  (undefined shooter). Harmless.
- The HUD (Score/Health/Lives text) is drawn inside the game canvas by the game itself, so it is part of the frame.
- Frames are 36 KB JPEGs because of the starfield noise (Mario's are 30 KB).
