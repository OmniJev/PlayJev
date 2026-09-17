# mario: hook notes

Upstream: robertkleffner/mariohtml5 (Infinite Mario in HTML5, a port of Notch's Java game), vendored at
commit 1dcf623 ("Remove MIDIJS since it had crypto miner"). Entry `main.html`, which loads the unminified
`Enjine/*.js` and `code/*.js` (the `*.min.js` bundles are not used). Vendored files are untouched;
`git status` inside `games/mario/` shows only the pj files as untracked.

## Loop and timing

- `Enjine.GameTimer.Start` runs `setInterval(tick, 1000/30)`; each tick computes `delta` from `new Date()`
  (faked by the shim, so delta is 33 or 34 ms, deterministic) and calls `Application.Update`, which runs
  `state.CheckForChange`, `state.Update(delta)`, then draws into a 320x240 back buffer and blits it onto
  the 640x480 `<canvas id="canvas">`. No requestAnimationFrame anywhere.
- One env step = exactly 3 game ticks = 100 ms of game time (`stepFrames: 6`; `applyAction` converts k
  frames to `round(k/2)` ticks). The hook counts `Application.Update` calls and advances the virtual
  clock in half-tick increments until the count is reached, instead of trusting `pj.frames(6)`:
  the 33.333 ms interval accumulates floating-point error in the shim's timer schedule, and a plain
  `pj.frames(6)` occasionally fired 2 or 4 ticks. Because of the half-tick stepping, `obs.t` advances by
  100 ms most steps and 83 or 117 ms once in a while; game time is the tick count (`info.ticks`).
- A jump step that follows a landing with the jump key still held costs 4 ticks (see Actions).
- Level timer: `TimeLeft` starts at 200 s and `Die()` fires at 0, so an episode is at most 6000 ticks
  = 2000 steps even if nothing else ends it.

## State

- `window.__pjApp`: the `Enjine.Application` (main.html never keeps a reference; the hook captures it by
  wrapping `Application.prototype.Initialize`). `__pjApp.stateContext.State` is the current state:
  `Mario.LoadingState` -> `Mario.TitleState` -> `Mario.MapState` -> `Mario.LevelState` (-> `LoseState`).
- `Mario.MarioCharacter`: the single player object (X, Y, Xa, Ya, OnGround, MayJump, Sliding, Coins, Lives,
  Large, Fire, DeathTime, WinTime, LevelString). `Mario.GlobalMapState`: the world map (tiles in
  `Level[x][y]`, per-tile `Data`, Mario's map position `XMario/YMario`).
- `LevelState.Level`: `Width` 320 tiles, `Height` 15, `Map[x][y]` tile bytes (behaviour bits via
  `Mario.Tile.Behaviors`), `ExitX` (win when `X > ExitX*16`, usually tile 260 to 280). `LevelState.Sprites.Objects`
  holds enemies, shells, items, fireballs and Mario.
- Keys: `Enjine.KeyboardInput` sets `document.onkeydown/onkeyup` and reads `event.keyCode`; the hook sets
  `pj.keyTarget = document.body` so the events bubble to document. Map: arrows 37..40, S (83) jump, A (65) run.

## start(seed)

The driver reloads the page, then `pj.start(seed)` seeds `Math.random` and calls the hook's start:

1. Wait (real time) until the 15 sprite sheets report `complete`; they are already loaded at `load`.
2. Tick until `TitleState` (LoadingState needs two ticks). `TitleState.Enter` regenerates
   `Mario.GlobalMapState` and creates `Mario.MarioCharacter`, both after the seed, so the world map and
   every level differ per seed.
3. Hold S until `MapState`, release, tick twice (MapState arms S only after seeing it released).
4. Walk the map: BFS over road and level tiles from Mario's tile to the nearest enterable level tile
   (`Data` not -11 start, not 0 cleared, > -10). One arrow press walks a whole road segment
   (`TryWalking` sets `MoveTime`), so press for one tick and wait until `MoveTime` and the velocity are 0.
   Then hold S until `LevelState`. Over 40 seeds the nearest level was `1-1` 37 times, a numbered level
   with a different number twice (one of them underground), and a castle (`1-#`) once. Zero fallbacks;
   if the walk ever fails, the hook drops straight into `new LevelState(1, Overground)` and sets
   `info.mapFallback = true`.
5. Run 30 ticks so the opening iris (26 ticks) is gone and Mario, who spawns at (32, 0) and falls, is
   standing on the ground. Reset the progress baseline. Whole start: about 45 to 50 ticks, 0.15 s wall.

## Actions (7, mirror the Jev Mario demo)

`noop` [], `left` [ArrowLeft], `right` [ArrowRight], `jump` [s], `right jump` [ArrowRight, s],
`right run` [ArrowRight, a], `right run jump` [ArrowRight, a, s]. Keys are held for the 3 ticks and
released; since no tick runs between the release and the next step's press, holding the same action
across steps is a continuous hold. Left only exists to back off from an enemy or line up a jump; the
level scrolls right and X is clamped at 0.

Jump detail: `Character.Move` sets `MayJump = (OnGround || Sliding) && !S`, so a new jump needs a tick
with S up while grounded. If the previous step held S and Mario is now on the ground (or wall sliding)
with `MayJump` false, `applyAction` runs one tick with S released (other keys of the action held), then
the normal 3 ticks. Holding S across consecutive steps keeps `JumpTime` counting down (7 ticks), which
is what makes the full-height jump; a single jump step is a short hop. Checked: `jump` repeated gives
3 to 7 liftoffs in 40 steps, and `right run jump` repeated reaches x 2000 to 3700 in 73 to 150 steps
and won the level on seed 3 (score 5282).

`noop` is a real choice (waiting lets an enemy walk past or into you: on seed 1 noop dies at step 17).

## Score and done

- `score()` = running maximum of Mario's X in the level minus the start X (32), in pixels, floor'ed,
  plus 1000 once `WinTime > 0`. The max is updated inside the `Update` wrapper every tick. The game's
  own score display is a constant `00000000`, and coins are too sparse to learn from; `info.coins`
  reports them anyway.
- `done()` = not in a `LevelState`, or `DeathTime > 0`, or `WinTime > 0`. `Die()` sets `DeathTime = 1`
  on the tick of the collision, fall, or timer expiry, so done flips within the step and the frame of
  that step shows the hit. The death and win irises (about 95 and 67 ticks later, then a state change
  back to the map) are never reached because the driver reloads.

## info()

x, y, xa, ya, maxX, coins, lives, onGround, large, fire, dead, won, ticks, mapFallback, timeLeft,
levelWidth, exitX, levelType (0 overground, 1 underground, 2 castle), levelString, a 21x13 tile window
around Mario (`.` empty, `#` solid, `^` platform top only, `?` bumpable or item block, `o` coin, with
`originX/originY/cell` in pixels) and up to 12 nearby sprites as `{kind, dx, dy}`. About 750 bytes.

## Quirks and bugs found

- Double application (bug in the previous partial hook, fixed): main.html boots from jQuery's
  `$(document).ready`, and jQuery 1.4.2 registers its DOMContentLoaded handler on `document`. The old
  hook wrapped `Application.prototype.Initialize` inside a `window` DOMContentLoaded listener, which fires
  after the document one, so the app was created unwrapped, `__pjApp` stayed null, and `bootIfNeeded()`
  started a second Application: two games on one canvas, two timers (6 Updates per step), a shared
  `Mario.MarioCharacter`, 133 env-steps/s. The hook now listens on `document` (the init script runs before
  jQuery loads, so it is registered first) and refuses to boot if sprite sheets were already requested by
  an app it did not see. Verified: one app, 3 ticks per step, and the offline path (CDN request aborted,
  jQuery absent, the hook's `$` stand-in boots the app) behaves identically.
- jQuery comes from `ajax.googleapis.com`. The page works without it (stand-in `$`), and a fast failure
  keeps `goto(load)` quick; a hanging connection would slow resets in a sandboxed network.
- `Enjine.Keys.Z` is 80 (same as P) and `LevelGenerator` writes `Odds[Mario.Odds.Cannon]` (undefined
  property, the constant is `Cannons`), so cannons never come from the odds table. Vendor quirks, harmless.
- The 640x480 canvas is a 2x blit of the 320x240 back buffer; `pj.frame()` scales it to 448x336. The
  in-game HUD (lives, coins, world, time) is part of the game's canvas and stays in the frame.

## Numbers (this machine, headless chromium, 8 pages)

- `python -m playjev.bench mario --pages 8 --steps 200`: 258 env-steps/s (221 to 258 across runs),
  reset of 8 pages 1.4 to 1.7 s, no error lines, 12 episodes ended, mean random score 450.
- Random policy over 8 pages x 500 steps: 38 episodes, all deaths, mean score 769 px (min 55, max 3522),
  mean length 89 steps (min 6, max 379), i.e. about 9 s of game time.
- `python games/mario/check_det.py` (seeds 1 2 3 11 42, 300 steps each, 18 episode boundaries): PASS,
  scores, `info()` and JPEG frames byte-identical on both pages at every step.

## Vendor patch

None. `pj.json`, `pj_hook.js`, `check_det.py`, `NOTES.md` are the only additions to `games/mario/`.

## Suggestions for the shim, driver and HARNESS.md

1. Shim `tick()`: compare with a tolerance (`if (nextAt > target + 1e-6) break;`) or round timer `at`
   values. A 33.333 ms `setInterval` driven by `pj.frames(6)` should give exactly 3 ticks per step; today
   floating-point accumulation in `t.at = now + t.interval` makes it 2 or 4 now and then (measured on a
   blank page with only the shim: 3000 steps gave 2989 x 3 ticks, 6 x 2, 5 x 4, the first odd one at step 1).
   Any setInterval game written against the shim without its own tick counter will see this jitter.
2. HARNESS.md: for games that boot from a jQuery ready handler (or any document-level DOMContentLoaded
   handler), a hook that must wrap prototypes before the game boots has to register on `document`, not
   `window`; the window listener fires after the game has already started.
3. Driver (optional): `GamePage.open` could `page.route` third-party hosts to abort, so a vendored page
   never depends on a CDN being reachable and resets stay fast on an air-gapped node.
