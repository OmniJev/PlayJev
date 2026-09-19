# pacman (daleharvey/pacman, WTFPL)

Entry `index.html`, code `pacman.js` (canvas 342 x 426: a 19 x 22 grid of 18 px blocks plus a 30 px
footer with score, lives and level). `index.html` calls `PACMAN.init(el, "./")` from a
`setTimeout(..., 0)` after a Modernizr check (canvas, localStorage, ogg/mp3 audio; passes in the
Playwright chromium).

## Loop

- `PACMAN.init` builds `Pacman.Map`, `Pacman.User`, four `Pacman.Ghost`s, draws the maze, then
  loads six audio files and only in the final callback (`loaded()`) binds the keyboard and starts
  `timer = setInterval(mainLoop, 1000 / Pacman.FPS)` with `Pacman.FPS = 30` (33.33 ms per tick).
  The audio callback fires on `canplaythrough`, which is real-time media decoding. The hook replaces
  `Pacman.Audio` (on DOMContentLoaded, before `init` runs) with a silent stub whose `load` calls its
  callback at once, so `loaded()` runs synchronously inside `init` on the first `pj.tick`. Audio is
  muted by the shim anyway; no game logic is touched.
- `mainLoop`: `++tick` (unless paused), `drawPills`, then by state: PLAYING runs `mainDraw` (ghost
  moves, user move, redraw, collision test), COUNTDOWN shows "Starting in: 4..1" for 120 ticks then
  `setState(PLAYING)`, EATEN_PAUSE holds 10 ticks after a ghost is eaten, DYING animates 60 ticks then
  `loseLife()` (`lives -= 1`, then `startLevel()` again if lives remain), WAITING after game over.
- Keys: `document.addEventListener("keydown", keyDown, true)` reading `e.keyCode`; arrows set
  `user.due`, N starts a new game, P pauses, S toggles sound. `due` is buffered: the turn is taken at
  the first grid square where that direction is floor (same-plane reversals are immediate), so the
  agent does not need to time the press.
- Randomness: ghost directions (`getRandomDirection` on reset, every move, and at every wall) via
  `Math.random`, all after `start()` so the seed controls them.

## State (closures, exposed by the patch below)

`PACMAN.getState()` (5 WAITING, 6 PAUSE, 7 PLAYING, 8 COUNTDOWN, 9 EATEN_PAUSE, 10 DYING),
`getTick()`, `getLevel()`, `getUser()` (`theScore()`, `getLives()`, `getPosition()`,
`getDirection()`), `getGhosts()` (`getPosition()`, `getDirection()`, `isVunerable()`,
`isDangerous()`), `getMap()` (`block({y,x})`, `height` 22, `width` 19). Positions are in tenths of
a block (Pac-Man starts at x 90, y 120, i.e. column 9, row 12); Pac-Man moves 2 units per tick,
ghosts 2 (1 when edible, 4 while eaten). Directions: UP 3, DOWN 1, LEFT 2, RIGHT 11, NONE 4.
Pellet 10 points, power pill 50, eaten ghosts 50, 100, 150, 200 per pill. 182 pellets + pills clear
the level (the map is reused, `level` increments).

## One step

`stepFrames = 6` (100 ms). `applyAction` dispatches the arrow keydown, then advances the virtual
clock in 8.3 ms slices until the game's own `tick` counter has grown by `round(k / 2)` = 3 main-loop
ticks, then releases. Counting ticks rather than milliseconds keeps every step exactly three ticks
despite float drift between 16.67 ms frames and 33.33 ms ticks (checked: only 3 ever observed).
Pac-Man moves 6 units = 0.6 block per step; a block is 1.67 steps. All four actions matter (a
buffered turn or an immediate reversal). No `noop`: stopping is not a choice in this game other than
by walking into a wall.

## Score / done / start / frame

- `score()` = `user.theScore()`, the game's score. Starts at 0 (`startNewGame -> user.reset`), only
  ever increases.
- `done()` = state is DYING, or lives dropped below the count at start. One life per episode: the
  collision sets DYING inside the tick where it happens, so `done` flips within that step and the
  score at that step is final (nothing scores during the death animation). The game itself would
  continue with 2 more lives after 2 s; the driver reloads instead.
- `start(seed)`: `pj.tick(1)` fires the deferred `init` (state WAITING), press N (`startNewGame`,
  seeded ghost directions), then tick whole main-loop ticks until state is PLAYING (121 ticks, 4.03 s
  of game time). The countdown's last tick only redraws the maze and the game paints the sprites on
  the next tick (which also moves everything and eats the pellet next to the spawn, 10 points the
  agent did nothing for), so the hook paints Pac-Man and the ghosts with the game's own `draw(ctx)`
  routines on the game's canvas instead of ticking: the first observation is complete, the score is
  0 and nothing has moved. Pac-Man starts heading left.
- `afterStep()`: when a level is cleared the game goes WAITING -> COUNTDOWN; the hook ticks through
  that 4 s countdown inside the same step (same routine as `start`) so no step is wasted (lives
  unchanged distinguishes this from the after-death countdown, which `done` already covers).
  Exercised with the teacher below: the observation after the clearing step is already PLAYING on
  the next level with 178 pellets and `level` incremented.
- Frame: the game's own canvas, `#pacman canvas` (342 x 426, scaled to 360 x 448). The 30 px footer
  with the score, remaining lives and level is part of it; the "s" at the bottom left is the game's
  sound indicator. The page's `<h1>` and links are outside the canvas.

## Vendor patch (pacman.js, additive only, `git diff` in games/pacman)

Three return objects gained read-only getters, marked `/* PlayJev: read-only access to closure state */`:
- `Pacman.Ghost`: `"getPosition": function () { return position; }`, `"getDirection": function () { return direction; }`
- `Pacman.User`: the same two getters
- `PACMAN` module: `"getState"`, `"getTick": getTick`, `"getLevel"`, `"getUser"`, `"getGhosts"`, `"getMap"`
16 lines inserted, 3 lines changed only by a trailing comma. No logic or rendering change.

## Numbers (bench, 8 pages, 200 steps, local workstation)

About 500 to 660 env-steps/s depending on machine load (load average 7 to 9 from other jobs while
measuring). Random policy: mean score 82, mean episode about 50 steps (5 s of game time; ghosts are
random walkers released next to the start, so random Pac-Man meets one quickly; the longest random
episode seen was 150 steps with 370 points). Teacher (BFS to the nearest pellet, avoiding squares
within 2 of a dangerous ghost, from `info()` only): 8 episodes, mean 2774, max 6910, mean 566 steps,
9 levels cleared. Determinism (`check_det.py`): PASS, 4 seeds x 2 pages plus a same-page replay,
identical score/done/info at every step; different seeds give different ghost walks.

## Quirks

- Ghosts have no chase logic and no return-to-base after being eaten (upstream TODO); an eaten ghost
  is harmless and fast for 3 s, then dangerous again where it stands.
- Eating a ghost pauses the game for 10 ticks (EATEN_PAUSE): the following step moves nothing but
  still records the pressed direction.
- `localStorage.soundDisabled` is never set, so the footer "s" stays green; sound is stubbed.
- `info().map` uses `#` wall, `.` pellet, `o` power pill, `=` ghost-house block, space for empty.
