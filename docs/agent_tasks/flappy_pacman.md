# Task: flappy and pacman hooks

1. `flappy` (games/flappy/, vendored nebez/floppybird, entry `index.html`, code `js/main.js`, jQuery, DOM sprites
   positioned with CSS transforms). Nothing done yet. If it is DOM, paint the play area (sky, pipes, bird, land) into an
   offscreen canvas the way Snake does, reading positions from the DOM and using the sprite images from `assets/`, so
   the frame looks like the real game. Loop: setInterval for the game loop and pipes, setTimeout for state changes,
   `new Date` timing; keydown space on document. Actions: flap, wait. One step about 5 frames (check the loop constant).
   Score: `score` (pipes passed); expose bird y, velocity, next pipe x and gap in info(). done: the death/splash state.
   start: from the splash press Space and tick until `currentstate` is playing; must be deterministic.

2. `pacman` (games/pacman/, vendored daleharvey/pacman, entry `index.html`, code `pacman.js`, canvas). Nothing done
   yet. Loop: one setInterval main loop; keydown on document with keyCode (arrows; N new game, P pause). Actions: up,
   down, left, right. One step about 100 ms of game time or one movement tick; pick from the timing constants and say
   so. Score: the game's score; expose lives, Pac-Man position and direction, ghost positions, remaining pellets in
   info(). done: one life per episode (lives decrease or game over), say so in NOTES. start: press N, tick through the
   ready countdown until Pac-Man can move. Frame: the game canvas (its bottom score bar is fine).
