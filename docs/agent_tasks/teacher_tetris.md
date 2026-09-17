# Task: tetris teacher

Read docs/TEACHER_BRIEF.md first. Game `tetris` (jakesgordon): info() has the board, the current piece (type,
rotation, x, y) and the next piece; check NOTES.md for the exact fields and coordinates. Actions: left, right, rotate,
drop, none; one step = one key then the piece falls one row (see NOTES.md for the exact step semantics).

Algorithm: when a new piece appears, enumerate all (rotation, column) placements by dropping the piece on the board,
score each resulting board with Dellacherie's features (landing height, eroded piece cells, row transitions, column
transitions, holes, cumulative wells) with his published weights (-4.5, 3.4, -3.2, -9.3, -7.9, -3.4), pick the best,
and store a plan of key presses (rotations first, then horizontal moves, then drop). Each step emit the next key of
the plan; if the piece is already at the target rotation and column, emit `drop`. Use `none` only if the plan is
empty and dropping is unsafe (should be rare). Recompute the plan if the piece observed in info() no longer matches
the plan (a rotate blocked by a wall, for example). Soft targets: 0.9 on the planned key, 0.1 spread over other keys
that keep the placement reachable, zero on keys that make the target unreachable.

Expected level: Dellacherie's evaluator clears hundreds of lines on a standard 10x20 board; here the episode is
bounded by the driver cap, so report lines per episode and score at the cap, versus random (mean score about 160).
