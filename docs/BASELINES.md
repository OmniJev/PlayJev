# Baselines (local workstation, 8 pages, 8 episodes per policy, episode cap 1500 steps, 2026-09-18)

Random and teacher policies played through the same harness the model will use. Scores are the hooks' own
score definitions (see each `games/<id>/NOTES.md`). A length of 1500 means the cap ended the episode.

| game | actions | env-steps/s | random score (len) | teacher score (len) | teacher |
|---|---|---|---|---|---|
| mario | noop, left, right, jump, right jump, right run, right run jump | 251 | 525.8 (50) | 3790 (124) | rule-based right runner with physics rollouts |
| snake | up, down, left, right | 243 | 1.0 (2) | 108 (444) | BFS to food, largest region fallback |
| tetris | left, right, rotate, drop, none | 837 | 163.8 (69) | 15204 (1500) | Dellacherie placement search |
| 2048 | up, down, left, right | 1046 | 888.5 (123) | 20338 (998) | expectimax depth 2 (+1 ply when crowded) |
| flappy | flap, wait | 630 | 0.0 (90) | 84 (1500) | exact physics, depth-first flap search |
| pacman | up, down, left, right | 621 | 103.8 (47) | 7501 (1500) | ghost occupancy propagation + safe BFS |
| breakout | left, right, stay | 790 | 479.4 (99) | 16069 (1500) | ball-flight simulation + bounce scoring |
| invaders | left, right, noop | 517 | 207.5 (195) | 400 (200) | 28-step dodge-and-aim DP |
| racer | left, right, faster, slower, left faster, right faster | 386 | 232.5 (1500) | 6712 (1402) | lateral target with curve and traffic lookahead |
| sokoban | up, down, left, right | 1047 | 13.1 (438) | 102 (43) | A* over pushes with deadlock pruning |
