# flappy (nebez/floppybird, Apache-2.0)

Entry `index.html`, code `js/main.js` (jQuery 1.10.2 + jquery.transit 0.9.9 + buzz). A DOM game: the
bird and the pipes are absolutely positioned divs inside `#flyarea` (420 px tall, anchored to the
bottom of `#sky`), the land is `#land` (20% of the viewport). With the 320 x 525 viewport from
`pj.json` the sky is exactly the fly area (0..420) and the land is 420..525; `start()` asserts this.

## Loop

- `startGame()` sets `loopGameloop = setInterval(gameloop, 1000/60)` (physics: gravity 0.25 px per
  tick squared, `velocity += 0.25; position += velocity`, flap sets `velocity = -4.6`) and
  `loopPipeloop = setInterval(updatePipes, 1400)` (one new pipe every 1400 ms, random gap top in
  `[80, 250)` via `Math.random`, so pipes are seeded).
- `playerDead()` clears both intervals; the death drop is a transit transition that the shim's
  `transition:none` makes instant. `showScore()` (scoreboard overlay) is chained behind the "ended"
  event of the hit sound, which never fires with muted audio, so no scoreboard is ever shown.
- Pipe motion upstream is a CSS animation only: `.pipe {animation: animPipe 7500ms linear}` with
  keyframes `left: 900px -> -100px`. The shim disables CSS animations (and they would run on
  wall-clock time regardless), so the hook wraps `window.gameloop` and `window.updatePipes` (both
  are top-level function declarations, hence writable window properties resolved at call time
  inside `startGame`) and sets every `.pipe`'s `left` from the virtual clock with the same linear
  law: `left = max(-100, 900 - 0.1333 px/ms * age)`. It runs before each physics tick and right
  after a pipe is created, so `gameloop`'s collision code (`nextpipeupper.offset().left`) reads what
  the animation would have produced. The removal filter in `updatePipes` (`position().left <= -100`)
  works unchanged.
- The bird's collision box is `#player.getBoundingClientRect()` (layout-dependent, includes the
  rotation transform); `$("#land").offset().top` is the ground. All layout, no wall-clock.

## State globals

`currentstate` (0 splash, 1 playing, 2 dead / score screen), `position` (bird top, px in the fly
area), `velocity`, `rotation`, `score`, `pipes` (jQuery objects of pipes not yet passed; `pipes[0]`
is the next one), `pipeheight` (gap, 90), `pipewidth` (52), `flyArea` (420). The bird's left is
fixed at 60 px; its sprite is 34 x 24.

## One step

`stepFrames = 5`: five physics ticks = 83 ms of game time; a pipe advances 11.1 px per step, the
bird falls 0.25 * 15 = 3.75 px more per step than the step before. Pipes spawn far off screen at
x = 900 (the upstream game is full-window), so the first pipe enters the 320 px frame at about
t = 5.75 s (step 69) and reaches the bird at about t = 7.4 s (step 90). Actions: `flap` (one Space
keydown on the document; the game listens on `$(document).keydown` for keyCode 32; there is no
auto-repeat, so holding is one flap) and `wait` (no key). Both are real choices in this game.

## Score / done / start

- `score()` = the game's `score` (pipes passed). Monotone, starts at 0.
- `done()` = `currentstate === 2`, set synchronously in `playerDead()` inside the physics tick that
  detects the hit, so it flips within the step and the score at that step is final.
- `start()`: `showSplash()` has already run on DOMContentLoaded (jQuery 1.x resolves ready
  callbacks synchronously), so a Space keydown/keyup goes `screenClick -> startGame`; the game
  starts with a jump (`velocity = -4.6`). Awaits the sprite images (loaded once per page load) and
  throws if the game is not in state 1 afterwards.

## Frame

Painted onto an offscreen 320 x 525 canvas from DOM state and `assets/*.png`: sky colour and the
sky strip (scrolling 275 px / 7 s like `animSky`), pipes (52 x 1 body strip stretched, 52 x 26 caps)
at their `style.left` with `style.height` of `.pipe_upper`, land colour plus the land strip
(335 px / 2516 ms like `animLand`), the bird from the 34 x 96 sprite sheet (wing frame
`floor(t/75) % 4` like `animBird`, rotation and the death drop read from the computed transform
matrix), and the running score in the game's big digit font at the top centre. Decorative scrolling
freezes at death, as the game pauses its animations. About 2 ms per frame. The ceiling strip is not
visible at this viewport height in the real game either. No splash, scoreboard or footer links.

## Vendor patch

`index.html`: removed the one line `<script defer src="https://yummy.nebez.dev/script.js" ...>`
(third-party analytics). It delayed the `load` event by about 1 s per page reload and would have
pinged an external server on every episode reset. The driver now also aborts every off-origin
request, so this is belt and braces; the page still should not carry a tracker. Nothing else changed
(`git diff` in games/flappy).

## Numbers (bench, 8 pages, 200 steps, local workstation)

About 630 to 800 env-steps/s over a 200-step bench (about 1050 in steady state; page reloads at
episode ends, 1.2 s for 8 pages, dominate the average because random episodes are short). Random
policy: mean score 0.00, every episode ends at step 90 or 91 (random flapping keeps the bird near
the ceiling, where the first pipe kills it; the gap is never that high). A one-line heuristic (flap when the bird is below the next gap centre) scores 2 to 3, so
the pipes are passable and the score moves. Determinism (`check_det.py`): PASS, 4 seeds x 2 pages
plus a same-page replay, identical score/done/info at every step; different seeds give different
gaps.

## Quirks

- With the shim's microsecond time lattice every step runs exactly 5 physics ticks (pipe x moves
  11.11 px per step, checked over 85 steps); before the lattice a tick could slip across a step
  boundary, so `info().nextPipe.x` was occasionally one tick (2.2 px) behind `t`.
- The bird is clamped to the top (`position = 0`) when its box touches the ceiling; it does not die
  there, only at pipes or the ground.
- The upstream game draws pipes (z-index 10) above the bird; the painter draws the bird last.
