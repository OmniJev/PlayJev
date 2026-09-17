# Task: tetris and breakout hooks (both by Jake Gordon)

1. `tetris` (games/tetris/, vendored jakesgordon/javascript-tetris, entry `index.html`, game code inline). A previous
   agent finished pj_hook.js, pj.json, NOTES.md, check_det.py: verify by running the bench and check_det, look at the
   frames, fix anything that violates the contract, otherwise leave it. Spec it was built to: loop rAF with `new Date`
   deltas; keydown on document (left/right/up=rotate/down=drop, space=start). Actions: left, right, rotate, drop, none
   (none = let the piece fall one step). One step = press the key then advance until the piece moved down once or a new
   piece spawned. Score: the game's score; expose rows and the board in info(). done: `playing` false after lose().
   Frame: main `#canvas` only.

2. `breakout` (games/breakout/, vendored jakesgordon/javascript-breakout, entry `index.html`; code `breakout.js`,
   `game.js`, `levels.js`; libs in `packaged/`). Nothing done yet. Loop: Game.Runner with setTimeout/rAF (check
   `game.js`), `new Date` timing; keys arrows/space via keyCode. Actions: left, right, stay, plus launch if the ball
   needs a key (or launch inside start(), with a note). stepFrames 4 to 6. Score: the game's score plus a
   level-completion bonus; expose lives, level, ball position and velocity, paddle x in info(). done: lives exhausted
   or level completed if it does not auto-advance. start: from the menu into a running game, done() false right after.
   Frame: the playfield canvas only (the page draws a stone background around it).
