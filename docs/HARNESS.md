# PlayJev game harness: how a game becomes an environment

Read this before touching any `games/<id>/`. The Snake hook (`games/snake/pj_hook.js`) is the
reference implementation; copy its shape.

## Layout

```
games/_shared/pj_shim.js   shared shim, injected first into every page (do not edit per game)
games/<id>/                vendored upstream game, unmodified where possible (keep its LICENSE)
games/<id>/pj.json         manifest: id, title, entry, upstream, license, viewport, frame, step_frames
games/<id>/pj_hook.js      per-game hook, injected after the shim, before the game's own scripts
playjev/env.py             Playwright driver: GamePage (one page), VecGame (n pages), GameServer
playjev/bench.py           random-policy throughput check: python -m playjev.bench <id> --pages 8 --steps 200
```

Both scripts are injected with Playwright `add_init_script`, so they run before any game script
and again after every reload. The driver resets an episode by reloading the page and calling
`pj.start(seed)`, so hooks do not need an in-page restart.

## What the shim gives you (`window.pj`)

- Virtual clock. `setTimeout`, `setInterval`, `requestAnimationFrame`, `Date`, `performance.now`
  are all faked. The game does not advance unless you call `pj.tick(ms)` or `pj.frames(k)`
  (k frames of 16.667 ms). rAF callbacks fire at frame boundaries, timers at their due time, in order.
- Seeded `Math.random` (`pj.seed(s)`, called by `pj.start`). Anything random the game does after
  `start` is reproducible per seed. Randomness at page load happens before the seed and is
  still deterministic (fixed initial state) but identical across seeds.
- Keys: `pj.keyDown(name, target)`, `pj.keyUp(name, target)`, `pj.press(name, target)`, `pj.releaseAll()`.
  Names: `ArrowLeft/Right/Up/Down`, `Space`, `Enter`, `Escape`, `Shift`, `Control`, `Alt`, letters `a`..`z`,
  digits `0`..`9`. Events carry `key`, `code`, `keyCode`, `which` and bubble, so games that check any of
  those work. Default target is `pj.keyTarget || document.activeElement || document.body`; events dispatched
  there bubble to `document` and `window`. If the game listens on a canvas or a specific element, pass it
  as `target` or set `pj.keyTarget` in `start()`.
- Audio is muted, CSS transitions and animations are disabled.
- `pj.frame()` returns a JPEG data URL of `hook.canvas()` scaled so the long side is 448 px.

## The hook contract (`pj.register({...})`)

Register once the game's globals exist (usually on `DOMContentLoaded` or `load`). Listener order matters: init
scripts run before any page script, so a `document`-level `DOMContentLoaded` listener added by the hook fires before
the game's own (jQuery's ready handlers are document-level too); a `window`-level listener fires after all
document-level ones. If you must wrap a prototype or capture an instance before the game boots, listen on `document`.
Third-party requests (CDNs, fonts, analytics) are aborted by the driver, so a page that needs jQuery from a CDN must
either have a local copy or the hook provides a stand-in.

| field | required | meaning |
|---|---|---|
| `id` | yes | game id, same as the directory |
| `actions` | yes | ordered list of `{name, description, keys}`. `name` is a short verb phrase, `description` one plain sentence a player would understand, `keys` the key names held for the default step. Include `noop` (`keys: []`) if doing nothing is a real choice in the game. |
| `start(seed)` | yes | dismiss menus, start a fresh game, leave the page in a state where the first `step` acts. May be `async`. Set `pj.keyTarget` here if needed. |
| `score()` | yes | number, monotone within an episode, the thing we maximise (points, distance, lines, tiles merged). Never negative. |
| `done()` | yes | boolean, true once the episode is over (death, game over, level complete, win). Must be false right after `start`. |
| `canvas()` | one of these | the game's `<canvas>` element (or an offscreen canvas you paint the DOM into, like Snake). Fast path, about 2 ms. |
| `element()` | one of these | DOM element to screenshot when there is no canvas. Slow path, 30 ms. Prefer `canvas()`. |
| `stepFrames` | no | default frames per step when the driver passes none (default 4). Turn-based games ignore it. |
| `applyAction(i, k)` | no | override the default "hold keys for k frames, release". Turn-based games press once and then advance until the game has processed the move (see Snake). |
| `afterStep()` | no | cleanup after each step. |
| `info()` | no | small JSON object with the internal state a teacher policy needs (grid, positions, velocities). Never shown to the model. Keep it under a few KB. |

Semantics that must hold, because training and evaluation assume them:

1. `pj.step(i)` is deterministic given the seed and the action history.
2. `done()` flips within the step that ends the game, and `score()` at that step is the final score.
3. `score()` is the episode's score so far, not a delta (the shim computes `reward = score - previous`).
4. Every action in `actions` does something in the game (no dead options), and the model never sees the game's name, only the frame and the option list.
5. The frame shows the play area only, no toolbars, menus, high-score tables or help text around it.

## Performance traps seen so far

- WebGL in headless Chromium serialises every page through one software GPU process and adds a readback per frame
  grab: Space Invaders (Phaser, `Phaser.AUTO`) ran at 29 env-steps/s until the hook forced `Phaser.CANVAS` before
  boot (490 after, identical output). Any engine that picks WebGL should be switched to its canvas renderer in the hook.
- Page screenshots cost about 30 ms; `canvas()` costs 1 to 2 ms. DOM games paint their state into an offscreen canvas.
- Resets (page reload) cost 0.3 to 1.5 s; games with very short random episodes are reload-bound in the bench, which
  is fine, the trained policy's episodes are long.
- Third-party scripts (analytics, CDN libraries) delay load by about a second per reset and are aborted by the driver
  anyway; remove the tag or ship a local copy.

## Checklist before you call a game done

- `python -m playjev.bench <id> --pages 8 --steps 200` runs with no `errors` lines, reports a
  sensible random-policy score, and ends at least one episode (if random play never dies within
  200 steps, note the typical episode length instead).
- Look at the saved frames in `runs/bench/<id>/`: the game visibly progresses across steps, the
  frame is the play area, colours look like the real game.
- Two runs with the same seed and the same action sequence give the same scores.
- Write down in `games/<id>/NOTES.md`: how the loop is driven (rAF, setInterval, setTimeout),
  which globals hold state, what one step means in game time, known quirks.
