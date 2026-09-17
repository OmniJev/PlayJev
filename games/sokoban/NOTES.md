# Sokoban (taniarascia/sokoban) with the 155 Microban levels

Vendored upstream: <https://github.com/taniarascia/sokoban>, MIT. Entry `index.html`, code
`script.js`, `Sokoban.js`, `constants.js`, `utils.js` (all ES modules). Levels: David W. Skinner's
Microban set, `levels/microban.txt` (XSB, 155 levels, terms in `levels/CREDITS.md`).

## How the game is driven

There is no game loop and no randomness. The game is a pure turn-based state machine:

* `script.js` adds one `keydown` listener on `document` and switches on `event.key`
  (`ArrowUp/Down/Left/Right` and `w/a/s/d`), calls `sokoban.move(playerCoords, direction)`, then
  `sokoban.render()`.
* `Sokoban.prototype.move` runs synchronously inside that listener and mutates `this.board` in
  place. `render()` repaints every cell of the canvas immediately, also synchronously. Nothing is
  scheduled with `setTimeout`, `setInterval` or `requestAnimationFrame`.

So the board and the canvas are both current the instant the key event returns. The
`pj.frames(1)` in `applyAction` only exists so that anything the page might schedule would run
before the canvas is read; today it advances the virtual clock by 16.7 ms and nothing else.

## Where state lives

Everything is on a single `Sokoban` instance, `window.sokoban` (vendor patch, below):

| what | where |
|---|---|
| board | `sokoban.board[y][x]`, strings `'empty'`, `'wall'`, `'block'`, `'success_block'` (box on a goal), `'void'` (empty goal), `'player'`. **Row major, `[y][x]`.** |
| level map | `sokoban.levelMap[y][x]`, the level's initial state; `isVoid(levelMap[y][x])` tells whether a goal lies under the player or a box. A player who starts on a goal (`+` in XSB) is `'player'` on the board and `'void'` in the map. |
| level | `sokoban.level` (1-based Microban number), `sokoban.levelCount` (155), `sokoban.goalCount` |
| player | not stored; `sokoban.findPlayerCoords()` scans the board and returns `{x, y, above, below, sideLeft, sideRight}` |
| canvas | `sokoban.canvas`, sized per level: `sokoban.cell` px per board cell (75, or less for big levels) |

`info()` exposes `level` (1..155), `levelIndex` (`seed % 155`), `levels`, `cell`, `player: [x, y]`,
`boxes`, `goals`, `walls` as `[x, y]` lists (x is the column, same orientation as the frame),
`boxesOnGoal`, `solved`, `steps`, and `board` as XSB text rows (`#` wall, space floor, `.` goal,
`$` box, `*` box on goal, `@` player, `+` player on a goal). 440 bytes on level 1, 3.2 KB on
level 155.

## Vendor patch

`git diff` inside `games/sokoban/` shows exactly the following. No rendering was changed: at the
original 75 px cell every drawing constant is unchanged, and the palette, insets, stroke widths
and circle radii are the game's own.

**`script.js`** (one line, unchanged from the first version of this hook): after the constructor
call, `window.sokoban = sokoban`, because the instance is a module-local `const` that a classic
script cannot reach.

**`utils.js`**: `generateGameBoard({ level })` builds the board from `MICROBAN[level - 1]`
(imported from `levels/microban.js`) instead of cloning `levelOneMap`, and returns
`{ board, levelMap }`. XSB to board cells: `#` wall, space empty, `.` void, `$` block, `*`
success_block, `@` player, `+` player on the board and void in the level map. Ragged rows are padded
with `empty`: a missing cell is floor outside the walls in XSB, and the game already draws floor
outside as plain floor (the corner cells of `levelOneMap` are `empty` too). Padding with the game's
`void` would add goals, so that was never an option; padding with walls would change the look of the
original level. `countBlocks` (the chain-push helper) is deleted. `levelCount` is exported.

**`Sokoban.js`**:
* `loadLevel(level)` (new, called from the constructor and from `render({ restart: true })`) sets
  `board`, `levelMap`, `goalCount`, the cell size `cell = min(75, floor(900 / max(columns, rows)))`
  and the canvas size `columns * cell` x `rows * cell`. Levels up to 12 x 12 keep the original 75 px
  cells; 28 Microban levels are wider or taller than 12 (the largest, 155 'The Dungeon', is 30 x 17
  and gets 30 px cells, canvas 900 x 510). The header's "Level N" text is updated on restart (outside
  the canvas, never in a frame).
* `paintCell` reads the cell size from `this.cell`; the inset (5), stroke width (10), player radius
  (20) and goal radius (10) are the original numbers times `cell / 75`.
* `render({ restart: true, level })` reloads that level (default: the current one, so the Restart
  button restarts the same level; the original hardcoded level 1). The win test is "boxes on goals
  equals goalCount" instead of the original "no visible goal and exactly 6 rows with a green box",
  which was specific to the original level. The win banner fills `canvas.width x canvas.height`
  instead of the fixed 600 x 675.
* `movePlayer` and `movePlayerAndBoxes` consult `this.levelMap` instead of `levelOneMap`.
* **Standard push rule.** `movePlayerAndBoxes` moves exactly one box, and only when the cell behind it
  is free floor or a free goal (`isTraversible`). A wall or a second box behind it blocks the move.
  The original counted the run of boxes ahead and pushed them all together if the cell after the last
  one was free (chain push); that code is gone.

`constants.js` is untouched (`levelOneMap`, `size` and `multiplier` still exist; `multiplier` is the
75 px cell size, the other two are no longer read). `index.html` and `style.css` are untouched.

**New files**: `levels/microban.txt` (Skinner's file, unmodified, CRLF), `levels/CREDITS.md`,
`levels/build_levels.py` and its output `levels/microban.js` (`export const MICROBAN = [...]`, one
array of XSB row strings per level, a comment with the level title and size before each). The build
script checks each level: only XSB characters, exactly one player, at least as many boxes as goals,
and that the player's reachable area (walking through boxes as if they were floor) never touches
the bounding rectangle. That last check is what lets the game keep indexing `board[y - 1]`,
`board[y + 2]` and so on without bounds checks, as it always did. Regenerate with
`python games/sokoban/levels/build_levels.py`. All 155 levels pass; every level has as many boxes as
goals.

## Level choice

`start(seed)` plays `levelIndex = seed % 155`, Microban level `levelIndex + 1`; `info().level` is the
1-based number that matches the `; N` titles in the txt. The driver's default seeds start at 1, so a
plain `reset()` on 8 pages plays levels 2..9; `check_det.py` and `random_policy.py` use seeds 0..N-1
so level 1 is included. Levels 1..20 are all at most 12 x 8 with 2 to 6 boxes.

## One step

One step is one move: `pj.press` the arrow on `document.body` (bubbles to `document`), then one
virtual frame. `step_frames` is irrelevant and set to 1.

A move into a wall, or a push that is blocked, is legal and leaves the board unchanged (reward 0,
the step is consumed). So a policy can stall for the whole episode; the driver's step cap bounds that.

## Actions

`up`, `down`, `left`, `right`. Each is what a player does; nothing else is bound in the game apart
from `w/a/s/d` aliases and the Restart button. No `noop`: standing still is never a choice in
Sokoban (a blocked move already acts as one and is not advertised).

## Score and done

The game has no score, no death and no timer, so both are defined here.

`score()` = (running max over the episode of boxes on goals) minus (boxes on goals at level start)
plus 100 on the step that completes the level. It starts at 0, is monotone (the running max means
pushing a box off a goal does not lower it) and never negative. Solving a level scores
(goals minus boxes that started on goals) + 100, so 101 on level 1 (one of its two boxes starts on a
goal), 106 on level 7.

`done()` = level solved (every goal holds a box; this coincides with the game's own `isWin`). The
hook has no step cap of its own: `pj.json` sets `max_steps: 200` and the driver marks the episode
`done` and `truncated` there. The game does not auto-advance (there is no next level), it paints a
black "A Winner is You!" banner over the canvas; `afterStep()` repaints the solved board with the
game's own `paintCell` so the terminal frame is the play area.

### The 200-step cap against the optimal move counts

Move-optimal solutions (BFS in `check_det.py`) for Microban 1..20 need 16 to 107 moves. Over the
whole set, the same solver run offline on the XSB rows (4 M state cap, no browser, about 4 minutes)
solved 141 of 155 levels, median optimum 85 moves. 12 of those need more than 200 moves: 84 (201),
98 (269), 106 (205), 108 (238), 114 (227), 124 (245), 134 (244), 140 (290), 143 (212), 152 (233),
154 'Take the long way home.' (429, one box down a long corridor) and 155 'The Dungeon' (282). The 14
it gave up on (93, 99, 105, 111, 112, 117, 122, 123, 139, 144, 145, 146, 150, 153; 5 to 16 boxes) are
the hardest of the set. So with `max_steps: 200` at least 12 levels (8 percent) cannot be solved by
any policy, and the ceiling on the solve rate is at most 143 of 155. A cap of 300 makes 140 of the 141
BFS-solved levels reachable, 500 makes all of them. The cap is the driver's setting and was
specified as 200; left at 200, flagged for the owner of `pj.json`.

## Frame

`canvas()` returns the game's own `<canvas>`, scaled by the shim so the long side is 448 px (level 1,
6 x 7 cells, gives 384 x 448; level 155 gives 448 x 254). It holds only the board painted by
`paintCell` with the palette in `constants.js`: grey walls, sand floor, yellow boxes, green boxes on
goals, pink goal dots, blue player. Floor outside the walls (ragged-row padding included) is sand
too, as in the original level. The header (title, "Level N", Restart) is outside the canvas.

## Numbers

* `python -m playjev.bench sokoban --pages 8 --steps 200`: no errors, about **1000 env-steps/s**
  during the run (971 to 1033 at the progress lines of two runs); the final figure, **672 and 859
  env-steps/s** in those two runs, includes the 8 page reloads of the capped episodes in the last
  step. 8 episodes ended (all at the 200-step cap, mean episode length 200), mean random score 0.38
  over those 8.
* `python games/sokoban/random_policy.py` (200 episodes, seeds 0..199, every level at least once):
  mean score **1.315**, solve rate **1.0 percent** (2 of 200, both level 44 'Duh!', a one-move level
  that seeds 43 and 198 both map to), mean episode length **198.1 steps**, 48 of 200 episodes put at
  least one new box on a goal.
* `python games/sokoban/check_det.py`: PASS. (a) BFS solves all 20 of Microban 1..20 (optimal moves:
  1: 33, 2: 16, 3: 41, 4: 23, 5: 25, 6: 107, 7: 26, 8: 97, 9: 30, 10: 89, 11: 78, 12: 49, 13: 52,
  14: 51, 15: 37, 16: 100, 17: 25, 18: 71, 19: 41, 20: 50; each solve under 0.05 s) and each replay
  matches the solver's predicted player and boxes after every move, `done` is false before the last
  move and true on it, the final score is goals filled + 100, and the final frame is the board.
  (b) The chain push is refused on Microban 4 (`# .**$@#`, first move left changes nothing).
  (c) Two pages, four seeds (levels 2, 8, 25, 124), 200 random steps each, plus one same-page replay:
  identical score, `done` and `info()` at every step.

## Quirks worth knowing

* `board` is `[y][x]`; `info()` and the frame use `[x, y]`. Easy to get backwards.
* `findPlayerCoords()` indexes `board[y - 1]` and `board[y + 1]` without bounds checks; safe
  because `build_levels.py` verifies every level is enclosed.
* The solver's dead-square pruning (a box on a non-goal square from which no goal can be reached is
  never pushed there) keeps BFS move-optimal; it only drops states with no path to the target.
* After a win, any further key press repaints the black banner; the driver resets on `done`, so
  this never shows up in a rollout.
* The Restart button and `render({ restart: true })` restart the current level; the hook's `start()`
  passes the level explicitly, so it works with or without the driver's page reload.
* `constants.js` still exports `size` (600 x 675) and `levelOneMap`; nothing reads them any more.
