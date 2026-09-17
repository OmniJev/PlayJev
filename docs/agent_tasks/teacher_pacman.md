# Task: pacman teacher

Read docs/TEACHER_BRIEF.md first. Game `pacman` (daleharvey): info() has Pac-Man position and direction, ghost
positions (and, check NOTES.md, whether they are edible / eaten), remaining pellets, lives, level, and the map;
one step = 3 main-loop ticks (Pac-Man moves 0.6 block). Actions up/down/left/right; the game buffers a turn until it
is possible, so an action is a desired direction.

Algorithm: BFS over the maze graph from Pac-Man's cell to the nearest pellet (power pellets and edible ghosts count
as targets with a bonus), with ghost avoidance: cells within distance d of a non-edible ghost (d about 2 to 3,
tuned) get a large cost or are blocked, and if every path is blocked pick the direction that maximises distance to the
nearest ghost. Use the maze layout from info() (the hook agent verified a BFS teacher reaching mean 2774, max 6910,
9 levels cleared in 8 episodes; start from that behaviour and improve the ghost logic). Soft targets: 0.9 on the BFS
direction, 0.1 spread over other directions that do not walk into a wall or a ghost, zero on those that do.

Expected level: mean score above 2500 (random is about 90), several levels cleared per episode. Report mean/max
score, levels cleared, mean episode length, cost per step (BFS on a 28x31 maze is cheap; cache the adjacency).
