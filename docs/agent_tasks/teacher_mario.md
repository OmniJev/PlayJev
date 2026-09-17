# Task: mario teacher

Read docs/TEACHER_BRIEF.md first. Game `mario` (Infinite Mario HTML5): info() has Mario's x, y, velocities, on-ground
state, a 21x13 tile window around Mario, nearby sprites with positions and types, coins, death and win flags; read
NOTES.md for the exact encoding of tiles (which values are solid, which are gaps) and sprite types (enemies vs
items). Actions: noop, left, right, jump, right+jump, right+run, right+run+jump; one step = 100 ms of game time.

Algorithm: rule based, right-going. Default action right+run. Look ahead in the tile window along Mario's row and the
rows below: if there is a gap (no solid tile below the path) within the next 2 to 4 tiles, or a wall/pipe/step at
Mario's height directly ahead, or an enemy within about 3 tiles ahead at or below Mario's height, choose
right+run+jump (or right+jump when the obstacle is close and speed would overshoot). While airborne over a gap keep
the jump held (right+run+jump). If Mario is stuck against a wall (x unchanged for 5 steps while pressing right),
switch to jump then right+run+jump. Use left only to back off when an enemy is falling onto Mario or a jump is
mistimed (rare). Soft targets: 0.8 on the chosen action, 0.2 spread over the other right-going actions, zero on
noop/left except in the back-off case.

Expected level: the random policy dies after about 90 steps with 770 px of progress and never wins. The teacher
should complete a good fraction of levels (Infinite Mario levels are short); report win rate and mean progress over
32 episodes on seeds 1 to 32. If the win rate is below 50 percent after tuning, say so with the failure modes (what
kills Mario most: gaps, which enemy, walls) so we can decide whether to add a PPO agent later.
