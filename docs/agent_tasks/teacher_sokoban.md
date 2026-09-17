# Task: sokoban teacher

Read docs/TEACHER_BRIEF.md first. Game `sokoban` after the Microban patch (games/sokoban/NOTES.md): info() has
player, boxes, goals, walls as [x, y] lists, the XSB board, the level index; standard one-box push rules; level =
seed % 155; done on solve or the 200-step cap.

Algorithm: at episode start (and whenever the observed state is not the next state of the current plan) run a solver
over (player position, frozen set of boxes): BFS over pushes with a player-reachability BFS inside each node (the
classic Sokoban search), with simple dead-square pruning (a box in a corner that is not a goal, a box against a wall
with no goal along that wall). Microban levels are small; games/sokoban/check_det.py already has a BFS the levels
agent wrote for the first 20 levels, reuse or generalise it. Cap the search at a node budget (say 200k states);
if it fails, fall back to a greedy move toward the nearest box that can be pushed toward a goal. Emit the next move
of the plan each step. Soft targets: 1.0 on the planned move when a plan exists (there is exactly one right move);
under the greedy fallback 0.7 on the chosen move, the rest spread over moves that do not push a box into a dead square.

Expected level: solve rate over 155 levels (one episode each, seeds 0..154) and mean moves; report the levels the
solver cannot solve within budget and the per-level solve time distribution (the first plan may take longer than the
per-step budget; that is acceptable once per episode if it stays under about 1 s on average, say so in TEACHER.md).
