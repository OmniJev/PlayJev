# Task: sokoban with 155 Microban levels and standard push rules

Follow docs/AGENT_BRIEF.md. The sokoban hook in games/sokoban/ works (see NOTES.md, check_det.py) but the vendored
game has one level and a non-standard chain-push rule. Both make it useless as a training environment. Fix by a
documented vendor patch; keep the rendering exactly as it is.

1. Levels. `games/sokoban/levels/microban.txt` holds David W. Skinner's Microban set (155 levels, XSB format:
   `#` wall, ` ` floor, `.` goal, `$` box, `*` box on goal, `@` player, `+` player on goal; each level preceded by a
   `; N` title line; credit and terms in `levels/CREDITS.md`). Patch the game so the board comes from a parsed XSB
   level instead of `levelOneMap`: convert XSB cells to the game's board strings (`wall`, `empty`, `block`,
   `success_block`, `void`, `player`; a player on a goal needs the goal remembered as underlying `void` the way
   `levelOneMap` is used today), size the canvas to the level (75 px per cell, max Microban size fits well under
   1000 px; if a level is larger than 12 x 12 shrink the cell size for that level so the canvas stays at most 900 px
   on the long side), and pad ragged rows with wall or void as appropriate. Cells outside the walls that are floor in
   XSB ("floor outside") should be drawn as empty floor. Load the level list at page load (inline the levels as a JS
   array in a new file `levels/microban.js` generated from the txt by a small script you also commit, so no fetch is
   needed) and pick `level = seed % 155` in `start(seed)`. Expose `level` in info().

2. Rules. Standard Sokoban: a push moves exactly one box, and only if the cell beyond it is free (floor or goal, not
   a wall, not another box). Patch `Sokoban.prototype.move` accordingly and remove the chain-push behaviour. Update
   check_det.py: it currently asserts the chain-push rule and replays a 34-move solution of the old level 1; replace
   that with (a) a BFS solver in Python over info() (state = player plus boxes) that solves at least the first 20
   Microban levels and replays each solution through the env, asserting `done` flips on the last move with the
   completion bonus, and (b) the two-page determinism check. Keep the solver small; Microban levels are tiny.

3. Score and done as before: running max of boxes on goals (minus the count at start) plus 100 on solving; done on
   solving or the driver cap (`max_steps` 200 in pj.json is enforced by the driver now; remove the hook's own cap if
   it duplicates this, or keep them equal).

4. Bench, frames, NOTES.md updated (what changed in the vendor code, level parsing rules, solver results: how many of
   the first 20 levels solved and their optimal lengths). Random-policy mean score and solve rate over 200 episodes.

Ownership: games/sokoban/ only. Budget 75 minutes.
