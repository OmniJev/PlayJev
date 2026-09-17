# snake (patorjk/JavaScript-Snake, MIT)

- Loop: setTimeout chain inside `SNAKE.Snake.go()`, one move per `snakeSpeed` ms (80 ms default, drops by 5 ms
  per food to a floor of 25). The DOM game paints divs; no canvas.
- State: `mySnakeBoard.grid` (24 x 25 with the fullscreen 520x520 viewport): 0 empty, 1 snake or edge wall, -1 food.
  Head element id `snake-snakehead-alive` (class `snake-snakebody-dead` after death). Board state 0 = dialog shown,
  1 = ready, 2 = moving.
- start(): the welcome dialog closes on a Space keyup dispatched on window.
- One step = one snake move: dispatch the arrow keydown/keyup on the board container, then tick in 5 ms slices until
  the head cell changes or the board state returns to 0. A 180 degree turn is ignored by the game (the snake keeps
  going), so that action wastes a move without dying.
- Score = maximum length reached in the episode (the grid drops the tail cell on the fatal move, hence the running max).
  done = board state back to 0 after start.
- Frame: painted from the grid onto an offscreen canvas with the theme's computed colours (field blue, body yellow,
  head white, food red, dead head grey). About 2 ms per frame versus 30 ms for a screenshot.
- Bench: 8 pages, 360 env-steps/s on local-workstation. Random policy: mean length 1.1, dies within a few moves.
  Teacher (BFS to food, largest reachable region fallback): mean 120 to 130, max 186, about 600 moves per episode.
- Determinism: same seed and action sequence gives identical head traces across pages (checked in bench dev).
