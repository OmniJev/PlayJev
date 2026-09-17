# Task: 2048 and sokoban hooks (turn-based)

1. `2048` (games/2048/, vendored gabrielecirulli/2048, entry `index.html`, code `js/`). A previous agent reported it
   complete (pj_hook.js, pj.json, NOTES.md, check_det.py): verify with the bench and check_det, look at the frames
   (tiles must look like the real game: tile colours from CSS, numbers drawn), fix contract violations, else leave it.
   Spec: DOM game, GameManager created as a local in js/application.js (a minimal vendor patch exposing it on window is
   acceptable, documented); keydown on document; actions up/down/left/right; one step = press then tick one or two
   frames; a no-op move is legal and leaves state unchanged (say so in NOTES); score = gameManager.score;
   done = over || won; start() must begin a fresh game despite LocalStorageManager; info() carries the 4x4 grid.

2. `sokoban` (games/sokoban/, vendored taniarascia/sokoban, entry `index.html`, code `script.js`, `Sokoban.js`,
   `constants.js`, `utils.js`). A previous agent left a draft pj_hook.js, pj.json and a modification to `script.js`
   (check `git diff script.js` inside games/sokoban; keep it only if minimal and needed, document it). Canvas game,
   keydown on document with `.key`. Actions up/down/left/right; one step = one move plus a frame or two. Score: running
   max of boxes on goals plus a large bonus at level completion; expose level index, player, boxes, goals in info() for a
   BFS teacher. done: level solved (stop at the completion moment even if the game auto-advances) or a step cap constant.
   start(seed): level = seed modulo number of levels. Frame: the game canvas only.
