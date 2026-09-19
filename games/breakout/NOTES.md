# breakout (jakesgordon/javascript-breakout, MIT)

Entry `index.html`; code in `game.js` (runner, state machine, maths), `breakout.js` (the game),
`levels.js` (10 layouts). Nothing in the vendored files was changed.

## Loop

`Game.start('canvas', Breakout)` runs from a `DOMContentLoaded` listener on `document` and
constructs `Game.Runner`, which constructs the `Breakout` instance and its state machine. The
state machine's initial transition fires `onstartup`, which calls `runner.start()`:
`setInterval(loop, 1000/60)`. `loop()` takes `dt = (Game.timestamp() - lastFrame) / 1000` with
`Game.timestamp() = new Date().getTime()` (integer ms, so dt alternates 16 and 17 ms), then
`update(dt)` and `draw()`. No requestAnimationFrame. Under the shim one `pj.frames(1)` fires the
interval exactly once; the hook ticks one frame at a time so a step of k frames is k updates.

Keys: `keydown` and `keyup` listeners on `document`, matched by `ev.keyCode` against
`Breakout.Defaults.keys`. Left/Right (or A/D) start the paddle on keydown and stop it on keyup.
Space/Return act on **keyup**: `play()` in the menu, `ball.launchNow()` while playing. Esc opens a
confirm dialog (abandon), Up/Down change the level in the menu; the hook uses none of those.

## State

Everything hangs off `Game.current` (the Breakout instance):

| field | meaning |
|---|---|
| `current` | state machine state: `'menu'` at load and after the last life, `'game'` while playing |
| `score.score`, `score.lives` | points and lives (3 to start, +1 per cleared level, max 5) |
| `level` | index into `Breakout.Levels` (0 to 9, wraps to 0 after 9) |
| `ball.x/y`, `ball.dx/dy` (unit direction), `ball.speed` (px/s), `ball.moving` | ball; `moving` false while parked on the paddle |
| `paddle.x/y/w/h`, `paddle.minX/maxX` | paddle, 84 x 14 px, 280 px/s |
| `court.left/top/right/bottom`, `court.chunk` | court 420 x 350 px at (110, 65), 14 px chunks; walls 14 px, top wall 28 px |
| `court.bricks[]` | `{pos:{x1,x2,y}, c, hit, score, color}`; `court.numbricks`, `court.numhits` |
| `storage` | `runner.storage()`, see Quirks |

A brick is worth `(25 - row) * 5` points (rows 0 to 24 from the top). Ball speed starts at 15
chunks/s (210 px/s) and grows with each brick hit towards 1.5x.

## What one step means

Real-time: `stepFrames` 5, so one step is 5 game updates = 83 ms of game time. The ball moves 17 px
per step at the start (26 px at top speed), the paddle 23 px; the court is 420 px wide, the paddle
84 px. `applyAction` holds the arrow for the 5 frames and releases it, the same as the shim
default, plus the relaunch described under Actions.

## Actions

`left`, `right`, `stay`. `stay` is a real choice (the paddle is where you want it).

There is no `launch` action. The game parks the ball on the paddle after `play()`, after a lost
life and after a cleared level, then launches it by itself two seconds later ('ready... set..
go!'); Space launches it at once. `start()` presses Space twice (play, then launch) so the ball is
in flight from the first step, and `applyAction` taps Space when it finds the ball parked, so a
lost life costs one step of dead time instead of 24. This is the game's own key binding, and it
is what a player does. The launch direction is always (1, -1) from the paddle centre, so the paddle
position at that moment is what decides the trajectory. The 'ready/set/go' labels never show.

## Episode start and end

`start(seed)`:

- `setLevel(seed % 10)`: the level comes from the seed so episodes vary (level 0 is the classic
  five rows, 30 bricks / 2550 points; levels 7 to 9 have 208 to 252 bricks and 16k to 19k points).
- `paddle.reset()`: the game rolls the paddle's start position with `Game.random` once, at page
  load, before the seed, and not again on play; rolling it again after `pj.seed` makes the start
  position and hence the first trajectory follow the seed.
- Space keyup (menu -> game: score 0, lives 3, ball on paddle), Space keyup again (launch),
  `pj.frames(1)` so the canvas is drawn (nothing is drawn before the first `loop()`).

`done()`: `Game.current.is('menu')` after start. `loseBall()` on the third lost life calls
`lose()`, which moves the state machine to `'menu'` inside the same `update()`, so `done` flips
within the step. `lose()` keeps `score.score` (only `ongame` resets it), so `score()` at that step
is the final score.

## Score

`score.score + 1000 * levels_cleared`. The bonus is the one the task asked for: clearing a level
is worth its bricks already (2550 for level 0), the extra 1000 marks the event itself. A cleared
level is detected in `afterStep` as `level` changing while the state is still `'game'`
(`winLevel -> nextLevel(true) -> setLevel(level + 1)`); the `'game'` guard matters because `lose()`
also calls `setLevel()` (see Quirks). Verified with a ball-tracking policy: the step that clears a
level pays brick + 1000 in one reward, lives go 3 -> 4 -> 5, the ball relaunches on the next step,
the score never decreases across the transition. Both parts are monotone, so the total is.

## Frame

The `#canvas` is 640 x 480 and the game fills it with `rgba(200,200,200,0.5)` over the page's
stone texture (`images/paving.jpg`, the body background), with the court and walls in the middle.
`canvas()` copies the walls-plus-court rectangle only: x 96..544, y 37..415, i.e. 448 x 378, which
the shim then sends at 1:1 (long side 448). The texture is painted under it first, as in the
tetris hook, otherwise the translucent grey lands on JPEG black and the whole frame is dark grey.

The top wall band (28 px) is the game's HUD: current score, lives as small paddles, and a
"HIGH SCORE" label (1000 until the current score passes it, then the current score). It sits
inside the walls, so it stays in the frame; the level number, sound checkbox and instructions are
DOM outside the canvas and are not.

## Quirks

- `runner.storage()` is `window.localStorage` when available, and the game writes `level` (on
  every `setLevel`) and `highscore` (on leaving the game state) there. Pages of one browser
  context share it and it survives reloads, so with the driver's 4 pages per context another
  page's level was read back by `lose() -> onmenu -> setLevel()` and the HUD showed other
  episodes' high scores. Before the fix nearly every random episode ended with a spurious +1000
  (the level "changed" at the moment of losing), and the bench mean depended on page interleaving.
  The hook sets `Game.Runner.localStorage = {}` from a `document`-level `DOMContentLoaded`
  listener, which runs before the game's own; `storage()` then returns that per-page object (the
  game's own fallback for browsers without localStorage). `start()` pushes a `pj.errors` entry if
  the game is still on `window.localStorage`.
- At the done step `info()` reflects the menu reset the game performs on `lose()`: the court is
  rebuilt (`bricks_left` is the full count again) and the game's `lives` is back to 3; the hook
  reports `lives: 0` there, `bricks_left` is left as the game has it.
- The ball is only lost once it leaves the whole 640 x 480 canvas (`p2.y > game.height`), 65 px
  below the court bottom, so a lost ball falls out of the frame and the life is deducted about
  0.3 s later. Nothing can save it once it is below the paddle line.
- Ball speed grows per brick and never resets within a life; `ball.reset` (new life or level) sets
  it back to 210 px/s.
- SoundManager2 is loaded by the page (`sound/soundmanager2-nodebug-jsmin.js`, local) and asks for
  six mp3s at `/sound/breakout/*.mp3`, which the driver's static server answers with 404. It
  initialises fine, never plays (the sound checkbox is off), and adds no errors.
- The random policy scores very differently per level: on levels 8 and 9 the ball gets trapped
  above dense brick fields and rakes several thousand points with no paddle involvement (max seen
  7575 in 575 steps); on level 0 it is a few hundred.

## Numbers

`python -m playjev.bench breakout --pages 8 --steps 200`: **720 to 820 env-steps/s** over 8 pages
on the local workstation (varies with machine load; the in-page cost is 1.2 ms per step, 1.0 ms of it the JPEG), no
errors, 11 episodes ended, mean random score 242.27 (identical across repeated runs).

Random policy over 88 episodes (8 pages x 1500 steps, seeds from the bench RNG): mean score 739,
median 305, max 7575; mean episode length 123 steps (median 97, min 55, max 601), i.e. about 10 s
of game time for three lives.

`python games/breakout/check_det.py`: two pages, four seeds, score, `t`, `info()` and JPEG bytes
identical at every step. Also checked across two separate browser sessions (one of them after a
warm-up episode that set a high score): identical.
