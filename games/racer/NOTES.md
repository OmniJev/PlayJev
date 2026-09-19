# racer (jakesgordon/javascript-racer, MIT)

Entry `v4.final.html` (the finished version: curves, hills, sprites, traffic) plus `common.js` (helpers, game
loop, renderer) and `stats.js` (FPS counter in the DOM). Vendored tree untouched (no patch). The page's settings
panel, links, HUD and FPS box are DOM outside the canvas; the frame is the `<canvas id="canvas">` (1024 x 768
logical, scaled to 448 x 336), so none of that is visible. The page auto-starts once its two images load; the
settings UI needs no interaction (defaults: high resolution, 3 lanes, road width 2000, draw distance 300).

## Loop and state
- Loop: `Game.run` (common.js) loads `background.png` and `sprites.png` (real `<img>` loads), calls `ready()`
  which builds the road, then runs `frame()` on `requestAnimationFrame`. `frame()` uses
  `Util.timestamp() = new Date().getTime()` and a fixed-step accumulator (`step = 1/60`), so under the shim each
  rAF is 16.67 ms and about one physics update (integer-millisecond Dates alternate 16 and 17 ms; the
  accumulator absorbs it, and the sequence is the same every run).
- State: globals of the inline script. `position` (camera z, wraps at `trackLength`), `speed` (0..`maxSpeed`
  = 12000 units/s), `playerX` (-1..1 is the road, clamped to -3..3), `segments` (6705 x 200 units, so
  `trackLength` = 1,341,000), `cars` (200), `currentLapTime`, `lastLapTime`, `keyLeft/keyRight/keyFaster/keySlower`.
  `findSegment`, `resetRoad`, `reset` are global functions.
- Keys: keydown/keyup listeners on `document`, matched by `keyCode` (arrows and WASD). `pj.keyTarget = document.body`.
- Randomness: `resetRoad()` places scenery and the 200 cars with `Math.random`, at image-load time (before
  `pj.seed`). start() calls the game's own `resetRoad()` again after seeding, so traffic and scenery differ per
  seed while the road layout (curves, hills, length) is the game's fixed recipe. `Render.player` also draws
  `Math.random` for the car bounce; that is render-only and its call count per step is fixed.

## Step, actions, score, done
- One step = 5 frames = 83 ms. At top speed the car covers 1000 units = 5 segments per step; steering shifts
  `playerX` by `dt * 2 * speed/maxSpeed` per frame, 0.17 per step at top speed. Acceleration is `maxSpeed/5`
  per second (0 to top speed in 5 s), braking `-maxSpeed`, coasting `-maxSpeed/5`, off-road `-maxSpeed/2` down
  to `maxSpeed/4`. Hitting a roadside sprite while off-road pins speed to `maxSpeed/5` and holds position at the
  segment start until you steer back onto the road; hitting a slower car drops you to its speed and behind it.
- Actions (the six from the task sheet): `left`, `right`, `faster`, `slower`, `left faster`, `right faster`.
  `left`/`right` alone coast (natural deceleration) while steering. No `noop`: plain coasting is a subset of
  what `left`/`right` do and the task sheet fixed the set at six; it could be added if a policy needs it.
- Score: the game has none, so score = distance travelled along the track in road segments (200 units), one
  decimal, accumulated across laps from `position` deltas (wrap-aware), kept as a running maximum so the push-back
  on a collision never lowers it. A lap is about 6713 (6705 segments plus the `playerZ` offset of the start line).
- done(): the game records a lap (`lastLapTime !== null`, set by `update()` when the camera passes the start
  line the second time), or `MAX_STEPS = 2400` steps (200 s of game time at 5 frames per step). A clean lap at
  top speed is 112 s; a simple teacher (accelerate, hug the inside of the coming curve, swerve around slower
  cars) laps in 124 to 130 s = 1500 to 1570 steps. Random play never laps, so its episodes are exactly 2400 steps.
- info(): `position, trackLength, segmentIndex, segmentsTotal, distance, speed, maxSpeed, playerX, offRoad,
  curve, curveAhead ([10, 30, 60] segments ahead), slope, lapTime, lastLapTime, carsAhead (up to 6 within 60
  segments: dz in units, x offset, speed), steps`.

## Numbers (local workstation, 8 pages)
- Bench: 305 to 362 env-steps/s across runs (the machine is shared), reset of 8 pages 0.35 s, no errors. In-page cost is about 1.3 ms per step (5 updates,
  5 renders, JPEG); the rest is the Playwright round trip. Rendering at the game's 480 x 360 "Low" setting
  saved nothing measurable, so the default 1024 x 768 is kept.
- Random policy: never laps, 2400-step episodes, mean score 380 segments (range 339 to 434 over 8 seeds), about
  5.7 percent of a lap; the car mostly sits near zero speed because `slower` brakes five times harder than
  `faster` accelerates.
- Teacher (see check_det.py `drive`, or the greedy policy used during development): full lap, score about 6713.
- Determinism (`check_det.py`, both `--policy random` and `--policy drive` through 2400 steps of traffic and
  collisions): score and info() identical on both pages at every step; a different seed gives different traffic.

## Quirks
- `Dom.storage` is localStorage: `fast_lap_time` persists across reloads within a browser context and only
  affects the DOM HUD (outside the frame), never game logic.
- The FPS box calls `setInterval(5000)` (virtual) and the music element is muted by the shim.
- The hills can crest above the horizon and clip billboards at the top of the frame; that is the game's renderer.
