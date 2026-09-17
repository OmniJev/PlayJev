# 2048 teacher (`playjev/teachers/2048.py`, class `G2048Teacher`)

## Algorithm

Expectimax over the game's exact transition model, read from `info().grid`.

1. Root: the four moves. A move that leaves the board unchanged is dropped (zero mass; the game treats it as a
   consumed no-op step, see NOTES.md).
2. Chance node: every empty cell, a 2 with probability 0.9 and a 4 with probability 0.1, uniform over cells
   (exactly what `GameManager.addRandomTile` does). No sampling, the node is enumerated.
3. Player node: best of the four moves by the value below.
4. Leaf: heuristic of the board after the second player move (before the next spawn). Depth 2 is
   move, spawn, move, heuristic. When the board has 4 or fewer empty cells the search gets one extra ply
   (move, spawn, move, spawn, move, heuristic), because that is where the branching is small and where a
   depth-2 search dies: it cannot see that a merge chain two moves away is the only way out.

Heuristic: per line of four tile exponents, summed over the 4 rows and the 4 columns (the nneonneo weights):

| term | weight | meaning |
|---|---|---|
| empty | +270 per empty cell | room to move |
| merges | +700 per available merge (adjacent equal tiles, counted along the line) | smoothness |
| monotonicity | -47 * min(drops going left, drops going right), drop = rank_a**4 - rank_b**4 | the line should be sorted one way or the other |
| sum | -11 * sum(rank**3.5) | many mid-sized tiles are bad; rewards merging up |
| floor | +200000 per line | a board with no legal move is worth 0, far below any live board |

Monotone rows and columns together push the largest tile into a corner and keep a sorted chain along an edge;
no explicit corner or snake matrix was needed on top of that (the frames in `data/2048/smoke` show the corner
strategy emerging by itself).

Implementation: the board is one 64-bit Python int (16 nibbles of tile exponents, row r in bits [16r, 16r+16),
column c in nibble c of that row, so `grid[r][c]` is nibble 4r+c). Four 65536-entry lists built at import (0.15 s):
row moved left, row moved right, and the same two spread into column layout for up and down after a bit-twiddled
transpose; a fifth list holds the heuristic of every row. A move is four lookups, a leaf is one transpose plus
eight lookups. Everything is deterministic: same board, same distribution.

## Soft targets

`p_i = softmax(v_i / T)` over the moves that change the board, `T = 1200` in heuristic units; moves that do not
change the board get exactly 0; a move whose value is 0 (every spawn leaves no legal reply, a certain loss) gets 0
unless all moves are like that, in which case the legal moves split evenly so the target is still a distribution.
The terminal board (over or won) returns uniform.

Calibration: T was chosen so that the best move gets about 0.8 on a typical board. On 48 offline games (44k
decisions) with the adaptive depth: T=1000 gives mean 0.80 / median 0.85, T=1200 gives mean 0.78 / median 0.80,
T=1500 gives mean 0.75 / median 0.75. In the browser env the smoke shard has mean top probability 0.785 and 1.45
zero-mass moves per record. Ties split exactly (the opening board with two 2s in one column gives up and down
0.2486 each).

Known property of a fixed T: the heuristic's scale grows with the tile ranks (the sum and monotonicity terms are
rank**3.5 and rank**4), so value gaps between moves grow from a few hundred in the opening to 30k or more once a
512 or 1024 is on the board. Early targets are soft (top probability 0.5 to 0.8), late targets are close to
one-hot. The late labels are still right (the runner-up move in those positions typically breaks the corner
chain), but if softer late-game targets are wanted, divide the values by the board's max rank before the softmax
(one line in `act()`), and recalibrate T.

## Numbers

Random policy, browser env (`python -m playjev.play 2048 --policy random --pages 8 --episodes 32`):
score mean 1055, median 1006, max 2424, episode length mean 138 (NOTES.md bench: 1365 over 7 episodes).

Teacher, browser env (`python -m playjev.teacher_eval 2048 --pages 8 --episodes 16 --max-steps 2500`):
**score mean 20906, median 20480, max 24120, episode length mean 999, capped 0**. Throughput with the teacher in
the loop 349 env-steps/s over 8 pages (986 for the random bench). Note the episode ends the moment the 2048 tile
appears (`done()` is `over || won`), so the score is capped at roughly 20k to 25k and the mean is governed by the
fraction of episodes that reach 2048; the classic "keep going" scores (50k+) are not reachable here.

Teacher, browser env, 32 episodes with tile bookkeeping (scratch script, same loop as teacher_eval):
score mean 19938, median 20306, max 24120, min 7424, episode length mean 965, capped 0.
Tiles reached: 512 in 32/32, 1024 in 31/32, 2048 in 29/32 episodes (0.91; the 29 are the `won` endings, the
other 3 are `over`). Mean top probability 0.779, median 0.801.

Offline simulator (same spawn rule, 48 games, seeds 1 to 48), the ablation that fixed the depth policy:

| search | mean | median | min | reach 1024 | reach 2048 | ms/step |
|---|---|---|---|---|---|---|
| depth 2 everywhere (`DEEP_EMPTY=-1`) | 17735 | 20256 | 5332 | 0.90 | 0.60 | 0.19 |
| +1 ply at <=3 empties | 19916 | 20480 | 9348 | 0.98 | 0.90 | 1.5 |
| **+1 ply at <=4 empties (default)** | 20554 | 20486 | 15616 | 1.00 | 0.94 | 2.7 |

(ms/step measured with 12 games in parallel on a loaded 20-core box, so these are upper bounds; see below for the
in-loop numbers.)

Reference points: a random policy scores about 1000 and never passes 256. A competent human reaches 2048 in a
fair share of games, roughly a third to a half; the teacher reaches it in about 94% of episodes. Expectimax with
this heuristic at depth 3 and beyond (nneonneo's C++ program) reaches 2048 in essentially every game and 4096 in
most; the episode cap at 2048 makes anything beyond that irrelevant here.

## Cost per step

Measured inside the browser eval loop (32 episodes, 30.9k decisions, Python 3.12, one process while the machine
had a load of about 6): **mean 1.85 ms, median 1.28 ms, p95 5.4 ms, max 14.8 ms per decision**. Depth-2 boards
cost 0.2 to 0.5 ms, boards with 4 or fewer empties searched at depth 3 cost 3 to 15 ms.
Pure depth 2 would be 0.2 ms; the extra ply on crowded boards costs about 2 ms on average and is what buys the
2048 rate above. Set `DEEP_EMPTY = -1` in the module (or pass `deep_empty=-1`) to fall back to pure depth 2 if
collection throughput matters more than the teacher's ceiling. The 8 teachers run sequentially in the driver
loop, so the wall-clock cost per round of 8 pages is 8x the per-step figure.

## Hyperparameters

`TEMPERATURE = 1200`, `DEPTH = 2`, `DEEP_EMPTY = 4`, heuristic weights as in the table above (module constants
`EMPTY_W, MERGES_W, MONO_W, MONO_P, SUM_W, SUM_P, LOST`).

## Collection: label quality and the random-action rate

Degenerate labels in the smoke shard (`data/2048/smoke`, 2000 records, epsilon 0.1): **0 fallback labels**. No
record has the terminal uniform (the collector never asks for a decision on a finished board) and none comes from
the certain-loss even split (offline, that branch fires about 3 times per 40k decisions, always on the last moves
of a lost game). 5/2000 records (0.25%) are exact ties, all on symmetric boards: one four-way uniform at
`frames/0000004.jpg` (two 2s on a diagonal, every move gives a mirror image) and four two-way splits like the
opening in `frames/0000000.jpg`. 299/2000 (15%) are one-hot because the other legal moves are 10T or more
behind. 157/2000 taken actions differ from the teacher's (the 0.1 epsilon of the smoke run, minus coincidences).

`pj.json` sets `collect.epsilon = 0.02`. Random moves hurt 2048 far more than most games because one move in the
wrong direction late in the game drags the big tile out of its corner and the teacher needs dozens of moves to
rebuild, so with a random move every 10 steps the collected episodes almost never contain a 1024 or 2048 board.
Offline, 48 games per rate, label always the teacher's, action random with probability epsilon:

| epsilon | score mean | median | episode length | reach 1024 | reach 2048 |
|---|---|---|---|---|---|
| 0 | 20554 | 20486 | 995 | 1.00 | 0.94 |
| 0.01 | 18149 | 20264 | 933 | 0.98 | 0.60 |
| **0.02** | 15994 | 18308 | 846 | 0.83 | 0.48 |
| 0.03 | 14655 | 15656 | 806 | 0.75 | 0.31 |
| 0.05 | 12647 | 12040 | 725 | 0.56 | 0.23 |
| 0.1 | 8716 | 7354 | 559 | 0.31 | 0.02 |
| 0.2 | 6317 | 5578 | 435 | 0.19 | 0.00 |

0.02 keeps half the episodes reaching the 2048 board the student must learn to finish, while each 850-step
episode still carries about 17 off-path perturbations (a quarter of them the corner-breaking kind) and the
teacher's recovery from each. About 40% of random actions are no-ops on a crowded board; those just duplicate a
frame. If late-game coverage turns out to matter more, mix in an epsilon-0 shard rather than lowering the rate.

`giveup()` is not implemented: a board with no legal move flips `over` inside the step, the teacher never
chooses a no-op, and a lost position is never "unrecoverable but alive" for long, so the game ends on its own.

## Known failure modes

- About 6% of episodes still die before 2048, typically with 1024 + 512 + 256 on the board when the spawns
  refuse to cooperate for several moves; a deeper search (depth 4 at <=3 empties) would cut this further at a
  cost of about 10 ms on those boards.
- The heuristic has no explicit corner term, so on a board where the largest tile has already been dislodged
  from its corner (only happens after epsilon-random actions during collection) the teacher rebuilds around
  the new position instead of forcing it back, which is the right call for the value but looks unusual.
- Late-game targets are nearly one-hot (see above).
- The teacher exploits `over`/`won` only for the terminal uniform; everything else comes from the grid.

## info() requests

None. `grid[row][col]` with tile values is exactly what the search needs; `over`/`won` are used for the terminal
case, `score` and `empty` are not needed (empty is recomputed from the grid).

## Checks run

- Moves and transpose verified against a naive list implementation on 20k random boards (0 mismatches).
- `python -m playjev.collect 2048 --steps 2000 --shard smoke --epsilon 0.1`: 2000 records, no errors, 294/s
  while another eval was running. Frames inspected: `frames/0000000.jpg` (opening, `left` is a no-op and gets 0,
  up/down tie, right 0.50), `frames/0001915.jpg` (256/128/32/8 chain in the right column, `down` 1.0, `left`
  would tear the chain out of the corner), `frames/0001996.jpg` (512 top-right, `down` gets 0 because it pulls
  the 512 out of the corner, `right` 0.62 keeps the top row, `up` 0.28, `left` 0.10).
