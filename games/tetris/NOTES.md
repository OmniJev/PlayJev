# tetris (jakesgordon/javascript-tetris, MIT)

Entry `index.html`. All of the game is one inline `<script>` at the bottom of that file, so every
variable it declares is a global on `window`. Nothing in the vendored files was changed.

## Loop

`run()` runs at parse time: it calls `resize()`, `reset()` and then `frame()`, which does
`update(dt); draw(); stats.update(); requestAnimationFrame(frame)`. `dt` is
`(now - last)/1000` with `now = new Date().getTime()`, so the shim's fake `Date` plus fake
`requestAnimationFrame` drive it exactly: one `pj.frames(1)` is one game frame of 16.667 ms.
Keys: a single `keydown` listener on `document` reading `ev.keyCode`; a keypress pushes one entry
onto the `actions` queue and `update()` shifts exactly one entry per frame.

## State

| global | meaning |
|---|---|
| `playing` | true between `play()` and `lose()` |
| `score` / `vscore` | real score / the displayed one, which crawls up 1 point per frame |
| `rows` | lines cleared this game |
| `step` | seconds of game time per gravity drop: `0.6 - 0.005*rows`, floor `0.1` |
| `dt` | game time accumulated since the last drop |
| `current`, `next` | piece objects `{type, dir, x, y}`; `setCurrentPiece` assigns a *new* object, which is how the hook detects a landed piece |
| `blocks[x][y]` | locked cells, each a piece type object (or empty) |
| `pieces` | the shuffle bag: 4 of each of the 7 types, drawn with `Math.random` |

Score: `score` itself. 10 points per piece that lands, 100/200/400/800 for 1/2/3/4 lines at once.
It only ever increases inside an episode and `reset()` zeroes it before the first step, so it
satisfies the monotone rule as is. `vscore` is the lagging display value and is not used.

## What one step means

One env step = one row of descent, which is how a player thinks about a Tetris move. `applyAction`
taps the key once on `document` and then runs frames until `current.y` has increased or `current`
has been replaced (the piece locked and the next one spawned), capped at `step*60 + 12` frames.
With the starting `step` of 0.6 s a step is about 36 frames (600 ms) of game time; it shortens as
lines are cleared. Games with a queue of one action per frame need the press and the wait in the
same step, otherwise the move and the drop land in different steps.

`stepFrames` is therefore ignored (`applyAction` overrides it) and `pj.json` records
`step_frames: 1` for consistency with the other turn-based game, snake.

## Actions

`left`, `right`, `rotate`, `drop`, `none`.

`none` is a real choice: the piece sinks one row and nothing else happens, which is what you want
whenever the piece is already where you want it. `drop` is the one place where the step is not a
single row: it taps Down every frame until the piece locks, i.e. a hard drop. Implementing `drop`
as "one row, faster" would have made it a duplicate of `none` (this game gives no bonus for soft
dropping, so the board would end up identical and the option would be dead, which the harness
contract forbids). Holding Down in the real game auto-repeats and does exactly this.

`rotate` is clockwise only, which is all the game offers (Up cycles `dir` 0-1-2-3).

## Episode start and end

`start(seed)` taps Space on the start screen, which calls `play()` -> `reset()`. Two things happen
before that tap:

- the piece bag is emptied and `next` is redrawn (`pieces.length = 0; next = randomPiece()`), so
  the whole piece sequence follows the seed. Without this the first two pieces are drawn while the
  page is loading, before `pj.seed`, and every seed would open with the same piece.
- the court texture is fetched once (see Frame).

`done()` is `playing === false` after start. `lose()` is called inside `drop()` when a freshly
spawned piece already overlaps the stack, in the same step that locked the previous piece, so the
flip happens inside that step and `score()` at that step is the final score.

## Frame

`#canvas` only (250x500 at the 680x680 viewport, 25 px blocks); the `#upcoming` preview, the score
text and the FPS box are DOM/second canvas and stay out of the frame.

The court canvas is transparent where empty and the page shows `texture.jpg` through it as a CSS
background, so copying the canvas alone would give coloured blocks on JPEG black. `canvas()`
returns a same-size offscreen canvas: the texture as a repeat pattern first, the game canvas on
top. The texture is loaded once in `start()` via a real (not virtual) timeout of 3 s and falls
back to a flat `#b5ada1` if it does not arrive.

## Quirks

- `#score`, `#rows` and `#canvas` are element ids, so `window.score` etc. are the DOM elements
  until the game's `var` assignments overwrite them. After `start()` they are numbers; `score()`
  still coerces with `Number(...) || 0`.
- The game keeps drawing the stats.js FPS box every frame. It costs a little throughput and shows
  nonsense under a virtual clock, but it is outside the frame and the vendored code is untouched.
- Rotation has no wall kick: next to a wall `rotate` can be a no-op for one step. That is the real
  game's behaviour, not a harness artefact.

## Numbers

`python -m playjev.bench tetris --pages 8 --steps 200`: **859 env-steps/s** over 8 pages, no
errors, 18 episodes ended, random-policy mean score **173** (a random policy never clears a line;
the points come from the 10 per landed piece). Mean random episode length is about 90 steps
(79/105/113/81 for seeds 1/7/42/1234 in `check_det.py`).

`python games/tetris/check_det.py`: two pages, four seeds, identical score, `info()` and JPEG
bytes at every step.

## Re-verified (second agent, same day)

Hook, manifest and `check_det.py` left as they were; nothing violated the contract. Checked against
the source and by probes: `done()` false right after `start()` and true at the locking step that
loses, with `score()` final there and unchanged on a further step; each of the five actions changes
the piece state from the same start (left/right shift x, rotate changes dir, drop locks the piece
for +10 and spawns the next, none sinks one row); 64 consecutive seeds give 64 distinct boards after
six hard drops; `info()` is about 400 bytes. Bench: **848 to 887 env-steps/s** over 8 pages, no
errors, 18 episodes, mean random score 173.33. `check_det.py` passes (4 seeds, score/info/frame
identical). Observation for the shim, not the hook: over seeds 1..64 the first piece is unevenly
distributed (Z 3, S 3, O 14 of 64), which is mulberry32 on consecutive small seeds with only four
warm-up draws; all 64 episodes still differ.
