# Task: mario hook

Game: `mario` (games/mario/, vendored robertkleffner/mariohtml5 = Infinite Mario in HTML5; entry `main.html`;
engine in `Enjine/`, game code in `code/`; `mario.min.js`/`enjine.min.js` bundles exist, check which one main.html
loads and hook the same globals). A previous agent left a partial `pj_hook.js` and `pj.json` here: read them, keep
what works, finish the job.

Known: loop uses setInterval, timing `new Date`, ~190 Math.random calls (level generation is random, so seeds give
different levels), keys via keyCode with onkeydown/keyup (find Enjine.Keyboard for the element and the key map;
Infinite Mario uses arrows plus S jump and A run). Title screen says "Press S to Start"; there is a map state between
title and level, so `start()` must walk TitleState -> MapState -> LevelState with key presses and pj.tick between.

Actions (7, mirror the Jev Mario demo): noop, left, right, jump, right+jump, right+run, right+run+jump. Hold keys for
stepFrames (try 6). Score: max horizontal progress of Mario in the level (running max of world x) plus a bonus on level
completion; expose coins, x, y, death and win flags in info(). done(): Mario died or level completed; flip on the first
step of death if easy, otherwise note the delay. Frame: the game canvas (320x240 scaled by CSS; use the canvas element).
