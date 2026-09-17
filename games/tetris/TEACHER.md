# tetris teacher (`playjev/teachers/tetris.py`, `TetrisTeacher`)

## Algorithm

One-piece placement search scored with Dellacherie's evaluator, run on the step graph the harness
actually plays.

1. When a piece appears (or the observed piece is not where the current plan expects it, which happens
   after an epsilon-random action or a blocked key), search from the observed piece state `(x, y, dir)`.
   The search is a breadth-first walk over piece states with exactly the game's step semantics: one key
   (`rotate`, `left`, `right`, `none`) followed by one row of gravity; a piece that cannot sink after its
   key locks where it is; `drop` locks the piece straight down. Blocked moves and blocked or no-op
   rotations (the O piece) are not expanded, they waste a step. Every reachable placement therefore comes
   with the shortest key sequence that reaches it, and slides under overhangs or a sideways lock on the
   stack are found when they exist. From a spawn on a low stack this visits about 600 states.
2. Each distinct placement (deduplicated by cell set, so I 0/2 and S/Z 0/2 count once) is applied to the
   board, lines are cleared exactly as `removeLines()` does it (rows 19 down to 1 with rechecks, row 0 is
   never tested), and the result is scored with Dellacherie's six features and his weights:
   landing height -4.5, eroded piece cells +3.4, row transitions -3.2, column transitions -9.3,
   holes -7.9, cumulative wells -3.4. Definitions follow Thiery and Scherrer's MDPTetris: landing height
   is the height of the piece's centre row before clearing; eroded cells is lines cleared times the
   piece's cells in those lines; row transitions are counted over the rows of the stack with the walls
   filled; column transitions with the floor filled and the row above the stack empty; holes are empty
   cells with a filled cell above them; wells add, for every empty cell whose two side neighbours are
   filled (walls count), the run of empty cells from it downwards.
   One addition for this game: the next piece spawns at `dir 0, y 0, x = round(uniform(0, 10 - size))`
   (the two end columns carry half weight), so a placement is further penalised by 10000 times the
   probability that the known next piece cannot spawn over it.
3. The best placement's key sequence is the plan; one key per step is emitted while the observed piece
   matches the expected state, otherwise step 1 runs again. Ties in the evaluator (within 1e-6) are kept:
   the plan follows the tied placement whose first key has the lowest action index, so the argmax of the
   soft target is the planned key.

Targets: 0.9 on the planned key, split evenly over the first keys of tied placements; 0.1 split over the
other keys after which the chosen placement is still reachable (computed from the same search graph by a
backward pass, no extra search); zero on keys that lose the placement or waste the step (blocked move,
blocked or no-op rotation, `drop` from the wrong column). When no other key keeps the placement, the
planned key gets 1.0. In the smoke shard 1923 of 2000 records have more than one non-zero entry, mean max
probability 0.90.

Hyperparameters: the six Dellacherie weights above, death penalty 10000, tie tolerance 1e-6, expansion
order rotate, left, right, none (so at equal plan length rotations come first). No lookahead beyond the
next-piece spawn check.

## Numbers

`python -m playjev.teacher_eval tetris --pages 8 --episodes 16` (2000-step cap, seeds 1 to 24):

| policy | episodes | score mean | median | max | episode length mean | capped | lines per episode |
|---|---|---|---|---|---|---|---|
| teacher | 16 | 19538 | 20290 | 21290 | 1927 | 15 of 16 | mean 155, 156 to 168 in the 15 capped episodes |
| random (`python -m playjev.play tetris --policy random`, 16 episodes) | 16 | 162 | 165 | 220 | 65 | 0 | 0 |

The 15 capped episodes score 19810 to 21290 with 156 to 168 lines in 2000 steps, about 400 pieces per
episode (about 5 steps per piece). Clears are mostly singles: in the smoke shard the per-step rewards
count 74 singles, 11 doubles, 2 triples and no tetris (Dellacherie's evaluator does not save wells for
tetrises). One episode (seed 2) died at step 831 with 56 lines and 7370 points. A competent
human clears a few lines per minute of the real game and rarely passes 100 lines; the teacher clears 160
in what would be 20 minutes of game time and the only thing bounding it in 15 of 16 episodes is the cap.
Dellacherie's evaluator with free placement on a 10x20 board averages hundreds of thousands of lines;
the gap to that is the mobility constraint discussed below.

Throughput with the teacher in the loop: 800 to 860 env-steps/s over 8 pages, the same as random play.
`python -m playjev.collect tetris --steps 2000 --shard smoke --epsilon 0.1`: clean, 614 records/s.
Teacher action mix in that shard: left 558, rotate 576, right 522, drop 333, none 11.

## Data collection: fallback labels and epsilon

Fallback labels in the smoke shard (`--epsilon 0.1`, 2000 records): 0 of 2000 are uniform or otherwise
outside the placement search. The teacher has two fallbacks, a uniform vector when info() carries no
piece or grid and a one-hot `drop` when the piece already overlaps the stack, and neither state occurs
in collection because the game ends the episode itself in the step that spawns an overlapping piece. 77
of 2000 labels (3.9 percent) are one-hot; those come from the search: the planned key is the only key
after which the chosen placement is still reachable (a piece one column from a wall it must reach, for
example), which is a sharp label, not a degenerate one.

Recommended random-action rate, written to `pj.json` as `{"collect": {"epsilon": 0.02}}`: 0.02.
Measured with `teacher_eval --epsilon x` (16 episodes, 2000-step cap):

| epsilon | score mean | episode length mean | capped |
|---|---|---|---|
| 0 | 19538 | 1927 | 15 of 16 |
| 0.02 | 18841 | 1925 | 12 of 16 |
| 0.05 | 11532 | 1269 | 3 of 16 |
| 0.1 | 4948 | 635 | 0 of 16 |

A random key in this game is one fifth of the time a hard drop of the piece wherever it happens to be,
and the resulting holes cascade faster than the greedy evaluator repairs them, so at 0.1 every episode
dies around step 600 and a shard is dominated by dying boards; at 0.02 the episodes still run to the cap
with the teacher's line rate while one key in fifty is off-plan (the teacher replans from the new state,
so the labels stay the teacher's).

`giveup()` is left at the default False: Tetris has no stalling state, the game ends by itself within a
few steps once the stack reaches the spawn rows, so there is nothing to cut short.

## Cost per step

Measured in-loop over 32000 steps (the 16-episode eval): mean 0.37 ms per `act()`, median 0.02 ms,
maximum 14.4 ms. The cost is concentrated in the search at each new piece (about 1 to 2 ms on a low
stack, up to about 14 ms in the rare case of a replan on a tall board with many reachable states); the
steps that replay a plan cost a dictionary lookup. Pure Python, no dependencies.

## Known failure modes

- Mobility. A piece moves at most one column per row of fall, spawns at a random column and has no wall
  kick, so on a tall stack most placements are unreachable and the evaluator picks the best of what is
  left. This is where the one death came from: the evaluator's known trade of a hole for fewer row
  transitions (it does this on purpose, a hole costs 7.9 and two row transitions 6.4) is harmless with
  free placement but on a tall stack the holes cannot be repaired before the stack reaches the spawn rows.
- Greedy. One-ply over the current piece; the next piece is used only to price a blocked spawn. A
  two-ply search over the next piece would cost about 30 ms per piece (7 ms per step), over the budget,
  so it was not done.
- The predicted piece state was checked against the game's info() on every step of the eval (949 checks
  in a 2-page probe, 0 mismatches), so the step model is exact; any future change to `applyAction` in the
  hook (for example a wall kick or a different lock rule) would need the transition rules in `_plan` updated.

## info() request

None. `grid`, `piece {kind, x, y, dir}`, `next`, `cols` and `rows_high` are everything the teacher needs.
