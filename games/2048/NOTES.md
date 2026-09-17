# 2048 (gabrielecirulli/2048)

Vendored upstream: <https://github.com/gabrielecirulli/2048>, MIT. Entry `index.html`.

## How the game is driven

There is no game loop. The game is a pure turn-based state machine:

* `js/keyboard_input_manager.js` adds one `keydown` listener on `document` and maps
  `event.which` (37/38/39/40, plus vim `hjkl` and `wasd`) to a direction, then emits `move`.
* `GameManager.prototype.move` (in `js/game_manager.js`) runs **synchronously inside that
  listener**: it slides and merges tiles, updates `this.score`, inserts one random tile, and sets
  `this.over` when no move is left.
* The only deferred work is rendering: `HTMLActuator.prototype.actuate` wraps its DOM writes in
  `requestAnimationFrame`, and `addTile` queues a second nested rAF for the slide position. Under
  the shim's virtual clock that means the DOM is one to two `pj.frames(1)` behind the model.

So the grid is correct the instant the key event returns; `pj.frames(2)` in `applyAction` exists
only to keep the visible DOM in sync with the model (the frame itself is drawn from the model).

## Where state lives

Everything is on a single `GameManager` instance:

| what | where |
|---|---|
| board | `gm.grid.cells[x][y]` -> `Tile{x, y, value}` or `null`. **Column major.** |
| score | `gm.score` (sum of merged tile values, starts at 0) |
| loss | `gm.over` |
| win | `gm.won` (set when a 2048 tile is created) |
| grid size | `gm.size` = 4 |

`info()` flips the grid to row major, so `info.grid[row][col]`, `0` for an empty cell.

## Reaching the instance: no vendor patch

`js/application.js` is

```js
window.requestAnimationFrame(function () {
  new GameManager(4, KeyboardInputManager, HTMLActuator, LocalStorageManager);
});
```

The instance is never stored anywhere. The brief suggested a one-line vendor patch, but the shim's
virtual `requestAnimationFrame` makes a clean no-patch route possible: at `DOMContentLoaded` all
game scripts have run but **that rAF callback has not fired yet**, because nothing has called
`pj.tick`. So the hook wraps a prototype method the constructor calls on itself:

```js
const setup = GameManager.prototype.setup;
GameManager.prototype.setup = function () { gm = this; return setup.apply(this, arguments); };
```

The first `pj.frames(1)` inside `start()` runs the pending rAF, the constructor calls `setup()`,
and the hook has the instance. **The vendored game is unmodified.**

## Persistence

`LocalStorageManager` writes the whole game state to `localStorage` after every move and restores
it in `setup()`. Two problems for us: a reload would resume a half-played game, and Playwright
pages in one browser context share an origin, so eight parallel pages would fight over one
`gameState` key. The hook forces

```js
LocalStorageManager.prototype.localStorageSupported = function () { return false; };
```

which makes the game fall back to `window.fakeStorage`, the in-memory object that
`local_storage_manager.js` already ships. That object is recreated on every page load, so each
page is isolated and every load starts from an empty grid. `start()` additionally calls
`gm.restart()`, so a fresh episode is guaranteed even if the page was already played.

## One step

One step is one move: press the arrow on `document.body` (it bubbles to `document`), then
advance two virtual frames. No real time passes in the game between decisions, the clock is only
there for the renderer, so `step_frames` is irrelevant and set to 1.

### A move that changes nothing is legal and is a genuine no-op

In `GameManager.prototype.move`, if no tile ended up in a different cell then `moved` stays false
and the function returns without adding a random tile, without touching the score, and without
calling `actuate`. **The state after such a step is bit-identical to the state before it.** This
matters for a teacher and for training:

* It is not an error and it is not rejected; the step is consumed and `reward` is 0.
* It cannot end the game, so `done()` can never flip on a no-op step.
* A policy that only ever picks a blocked direction will loop forever, so an episode cap is a
  good idea on the training side (the hook does not impose one).
* A teacher can detect the legal moves from `info().grid` alone: a direction is a no-op iff no
  tile can slide or merge that way.

## Actions

`up`, `down`, `left`, `right`. All four are always pressable; each is the real thing a player
does. No `noop` action: doing nothing is not a choice in 2048 (and, as above, a blocked direction
already gives the model a way to pass, which we do not want to advertise as an option).

## Score and done

`score()` is `gm.score`, the game's own score: the sum of the values of all tiles created by
merging. It starts at 0, only ever increases, and is exactly the quantity a 2048 player
maximises, so no substitute was needed.

`done()` is `gm.over || gm.won`. The episode therefore ends the moment a 2048 tile appears; the
game's own "Keep going" button is never pressed. (`GameManager.isGameTerminated()` uses
`over || (won && !keepPlaying)`, which is the same thing here since `keepPlaying` stays false.)

## Frame

The 4x4 grid painted onto a 500x500 offscreen canvas, matching `.game-container` exactly:
15 px padding, 121.25 px pitch, 107 px tiles. Colours are not hard coded: `tileStyle(value)`
creates a throwaway `div.tile.tile-<value> > div.tile-inner`, reads `getComputedStyle` for
background, text colour, weight and size, then removes it and caches the result. So the frame
uses the real `style/main.css` palette (`#eee4da` for 2, `#edc22e` for 2048, `#3c3a32` for
`tile-super` above 2048) and the real font sizes (55 px, 45 px at 128+, 35 px at 1024+).

The frame is drawn from `gm.grid`, not from the DOM, so it is never one rAF stale and shows no
merge or slide animation mid-flight. Page chrome (title, score boxes, "New Game", the how-to-play
text, and the win/lose overlay) is outside the frame by construction.

`start()` awaits `document.fonts.load('bold 55px "Clear Sans"')` and `document.fonts.ready` so
that the bundled webfont is in before the first frame; without it, early frames could race the
font load and differ between pages.

Viewport is 700x900 on purpose: `style/main.css` has `@media screen and (max-width: 520px)` that
switches the board to a 280 px mobile layout with different fonts. A narrower viewport would
change the geometry the hook assumes.

## Numbers

* `python -m playjev.bench 2048 --pages 8 --steps 200`: **986 env-steps/s** over 8 pages, no
  errors, 7 episodes ended, mean random-policy score **1365**.
* Random-policy episode length is roughly 60 to 160 steps (seed 7 died at 67 steps with 380
  points, several others were still alive at 120).
* `python games/2048/check_det.py`: PASS. Two pages, four seeds, 120 steps each, plus one
  same-page replay: identical score, `done` and `info()` at every step.

## Quirks worth knowing

* `grid.cells` is `[x][y]`, not `[row][col]`. Easy to get backwards.
* `HTMLActuator` renders merged tiles by stacking three DOM nodes (the merged tile plus its two
  parents), so counting `.tile` elements is not the same as counting tiles. Read `gm.grid`.
* `Math.random` is used for both the tile value (10% chance of a 4) and its position, and is
  seeded by `pj.start` before the game is constructed, so the two opening tiles already depend on
  the seed.
* The `r` key restarts the game and the mapped `wasd`/`hjkl` keys are extra aliases for the
  arrows. Nothing in the hook presses them, but a future hook that uses letter keys should avoid
  `r`, `w`, `a`, `s`, `d`, `h`, `j`, `k`, `l`.
