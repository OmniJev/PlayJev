# Sokoban teacher (`playjev/teachers/sokoban.py`, class `SokobanTeacher`)

## Algorithm

Plan the whole level once, then emit the plan one move per step.

1. **Static analysis per level** (cached across episodes and teacher instances, keyed by `info().level`): walls,
   goals and floor from the XSB `board`; cells as integers `y * S + x` with `S = width + 1` (a padding column so a
   shift by one never wraps into the next row); box sets as one Python int bitmask. Dead squares are found by
   pulling every goal backwards over the floor (a box on cell `b` pushed along `d` lands on `c = b + d` with the
   player on `c - 2d`); the same BFS gives every floor cell its minimum push distance to the nearest goal.
2. **Exact search**: A* over states (boxes bitmask, player cell). An edge is one push: a player-reachability BFS
   inside the node gives the walk distance to each pushing position, the edge costs walk + 1 moves, so the
   plan is **move-optimal** (the 20 levels `check_det.py` solves come out at exactly its move counts). Heuristic:
   sum over boxes of the push distance to the nearest goal, admissible because every box has to end on a goal
   (Microban has as many boxes as goals; the teacher drops the heuristic if that ever fails). Pruning: pushes onto
   dead squares, and the classic frozen-box deadlock (a box blocked on both axes by walls, two dead squares, or
   other frozen boxes, while off a goal; checked for the pushed box and the boxes next to it).
3. **If A* runs out of budget**: weighted A* (`f = g + 4h`, same states, same pruning), then greedy best-first over
   (boxes, normalised player position) ordered by the heuristic alone. Both find a valid but not move-optimal
   plan.
4. **If everything fails**: per-step greedy fallback. Walk to the nearest box that can be pushed one square closer
   to a goal without deadlocking (ties by the resulting distance), push it; if no such push exists, the nearest
   pushable box at all.

Plan following is checked every step against the observed (player, boxes). A wasted move (state unchanged, e.g. an
epsilon-random move into a wall) re-emits the same move; a random walking step (boxes unchanged) only recomputes
the walk to the next push (one BFS, no search); a random push re-plans from the new state. Plans are cached per
(level, player, boxes) across episodes (`_PLANS`, capped at 4000 entries), so a level replayed later costs nothing.

Budgets (module constants): `ASTAR_SECONDS = 1.5` (or 150k expansions), `WASTAR_SECONDS = 1.0` with
`WASTAR_WEIGHT = 4`, `GREEDY_SECONDS = 1.5`; after a failure in the current episode any further re-plan (only when
the boxes moved) gets `RETRY_SECONDS = 0.15` per phase, so fallback episodes do not burn seconds on every push.

## Soft targets

- Plan being followed: **1.0 on the planned move** (the task's convention: there is exactly one right move; ties
  between equally optimal plans are not split, the plan picks one).
- Greedy fallback: **0.7 on the chosen move**, 0.3 spread evenly over the other moves that are not wasted (not into
  a wall, not a blocked push) and do not push a box onto a dead square. If no other move qualifies the chosen one
  gets 1.0; if no candidate push exists, uniform over non-wasted moves.
- `solved` state (never seen in practice, the driver resets on done): uniform.

## giveup() and the collection default

`giveup()` returns True once the board is **provably** unsolvable from the last observed state: an off-goal box on a
dead square, a frozen off-goal box (the recursive frozen test, with path-local assumptions so it is sound), or a
search that exhausted the whole pruned state space without a solution. All three are permanent, so the flag holds
for every later state of the episode. It is never set by a budget failure (an unsolved hard level keeps playing the
fallback), and never when the level has more boxes than goals (then a dead box is not fatal; Microban has none).
The collector polls it after each step, so exactly one fallback label is recorded between a deadlocking random push
and the early end of the episode.

`games/sokoban/pj.json` carries `"collect": {"epsilon": 0.03}`, the random-action rate `playjev.collect` uses when
no `--epsilon` is given. Measured shares of non-plan labels in a 2000-record smoke shard (8 pages, seeds from 1000):

| setting | plan labels (1.0) | fallback (0.7) | uniform | episodes ended by giveup |
|---|---|---|---|---|
| `--epsilon 0.1`, no giveup | 1172 (58.6 %) | 161 (8.1 %) | 667 (33.4 %) | (not available) |
| **`epsilon 0.03` from pj.json, giveup on** | **1978 (98.9 %)** | 22 (1.1 %) | 0 | 5 of 25 episodes started |

At 0.03 the 22 non-plan labels are 18 steps of level 93 (a search-failure level, in fallback for the whole episode,
cut off by the record limit) and 4 single labels from the act() between a deadlocking random push and the giveup.

## Numbers

Horizon: `pj.json max_steps: 500` (driver-enforced; it was 200 when the task was written and was raised to 500).

### Offline solver, all 155 levels from `levels/microban.txt`, plans verified by an independent simulator

`python runs/probe/sokoban/offline155.py` (single process, machine load about 7):

First run (A* 1.5 s then greedy 1.5 s, before the weighted A* phase was added): **147 of 155 levels solved**,
130 by A* (move-optimal), 17 by greedy; solve time mean 0.449 s, median 0.027 s, p90 1.83 s, max 3.2 s; moves of
the solved plans mean 128, median 92, max 602 (3 plans over 500: levels 99, 108, 117). Unsolved: 93, 105, 122,
139, 144, 145, 146, 153. On the 20 levels `check_det.py` solves, the A* move counts equal its optimal counts exactly.

Final code (three phases: A* 1.5 s, weighted A* 1.0 s, greedy 1.5 s; path-local frozen test; exhaustion proof):
**148 of 155**, 132 by A* (move-optimal), 2 by weighted A* (145 in 47 moves, 126 in 87), 14 by greedy; solve time
mean 0.573 s, median 0.023 s, p90 2.82 s, max 4.15 s; moves mean 125, median 88.5; 3 greedy plans over 500 moves
(99, 108, 117). Unsolved: 93, 105, 122, 139, 144, 146, 153. No search claimed any level unsolvable, no plan failed
the simulator. Greedy plans (98, 99, 108, 109, 111 to 114, 117, 123, 134, 138, 143, 150) run 1.3 to 2.5 times the
optimum. Which of the borderline levels (99, 109, 140) land inside a budget varies with machine load.

### Browser env, seeds 0..154 (every level once), 8 pages, argmax of the teacher

`python runs/probe/sokoban/eval155.py --policy teacher` versus `--policy random` (same seeds, same cap):

| policy | episodes | score mean | median | max | solve rate | episode length mean | capped |
|---|---|---|---|---|---|---|---|
| random | 155 | 1.01 | 0 | 101 | 1/155 = 0.6 % (level 44, a one-move level) | 496.8 | 154 |
| **teacher** | 155 | **96.75** | **103** | 112 | **145/155 = 93.5 %** | 141.3 | 10 |

Teacher, solved episodes: moves mean 116.6, median 88, max 472. Unsolved (all at the 500 cap, boxes on goal /
goals in brackets): 93 (5/8), 99 (2/4), 105 (4/8), 108 (1/4), 117 (4/5), 122 (0/5), 139 (0/6), 144 (11/16),
146 (6/12), 153 (0/10). Levels 108 and 117 were solved offline by the greedy phase but with 602 and 516 moves,
over the cap; the other eight are the search failures. No `errors` lines. Throughput with the teacher in the loop
137 env-steps/s over 8 pages (565 for random), dominated by the first-plan searches on the hard tail; on levels
1 to 92 it ran at 230 to 290.

First plan per episode (the one search that may exceed the per-step budget): **mean 586 ms, median 32 ms,
p90 2.84 s, max 4.16 s; 27 of 155 episodes over 1 s** (all in levels 93 to 155). The task allowed about 1 s on
average once per episode; the mean is under that, the tail is the 4 s budget on the unsolved levels.

### Standard deliverable

`python -m playjev.teacher_eval sokoban --pages 8 --episodes 16` (seeds 1..16, levels 2..17): **score mean 102.38,
median 102, max 106, episode length mean 48, capped 0**, 553 env-steps/s, all 16 solved with move-optimal plans.

Reference points: the random policy solves about 1 percent of episodes (only level 44 'Duh!', a one-move level;
`random_policy.py` in NOTES.md: 2 of 200) and scores 1.3 on average. A human who knows Sokoban solves every
Microban level, the set was designed as an introduction, but rarely move-optimally; the teacher's plans on the 130
A*-solved levels are optimal in moves.

## Cost per step

Measured inside the 155-level browser eval (21,902 `act()` calls, Python 3.12, machine load about 7):
**median 0.005 ms, p95 0.05 ms, p99 0.6 ms; mean over all calls 4.62 ms**, which is entirely the first-plan
searches (mean 0.052 ms over the 21,801 calls that were not a search). Following a plan is a tuple comparison
and a list index; a rewalk after a random walking step is one BFS (under 0.1 ms); a re-plan after a random push
is a full search (tens of ms on most levels, up to 4 s on the hard tail, once per such push, and instant when
the push created a detectable deadlock). Static analysis per level costs about 1 ms and is cached.

## Known failure modes

- **Unsolved within budget**: the levels listed above (5 to 16 boxes, the hardest of the set). On those the teacher
  runs the greedy fallback for the whole episode; it fills some goals (partial score) and never solves them.
  More budget helps only slowly in pure Python; a stronger lower bound (minimum-cost matching of boxes to goals)
  or macro pushes (tunnels) would be the next step.
- **Weighted A* / greedy plans are long**: the plans found by the non-optimal phases are 1.3 to 2.5 times the
  optimum, and a few exceed the 500-step cap (the episode ends with most boxes placed but unsolved).
- **Plan targets are one-hot**: when two moves are equally good (two optimal plans) the target still puts 1.0 on
  one of them; a model that learned the other one is not wrong. The task asked for 1.0 on the planned move.
- **Fallback is myopic**: it can push a box into a freeze deadlock two moves ahead that the one-push check does not
  see; the level is then unsolvable for the rest of the episode.
- Re-planning after a random push in collection costs a full search (up to 4 s on a hard level, less than 50 ms
  on most); `--epsilon 0.1` triggers this on roughly 2 percent of steps (a quarter of random actions are pushes).

## info() requests

None. `board` (XSB rows) alone would suffice; `player`, `boxes`, `level` and `solved` are used as given, `goals`
and `walls` are re-derived from the board. It would be convenient if `board` rows were padded to equal length,
but the teacher pads them itself.

## Checks run

- `runs/probe/sokoban/offline155.py`: every plan replayed through an independent simulator with the standard
  rules (no wasted moves, solved exactly on the last move) before it counted as solved.
- `runs/probe/sokoban/eval155.py --policy teacher` and `--policy random`: the numbers above; assertions that every
  returned distribution sums to 1 with no NaN.
- `python -m playjev.collect sokoban --steps 2000 --shard smoke --epsilon 0.1`: 2000 records in 7 s, 302/s, no
  errors, 15 episodes started (levels 71 to 85), 7 finished. Labels: 1172 one-hot plan moves, 161 fallback
  (0.7), 667 uniform over non-wasted moves. The last two groups come from episodes that an epsilon-random push
  deadlocked (frames inspected: `0000900.jpg`, level 75, four boxes frozen in a 2x2 block, uniform over the three
  non-wall moves; `0001990.jpg`, level 84, two boxes stacked in a one-wide corridor, 0.7 down toward the only
  pushable box, 0.3 up, walls get 0). `0000000.jpg` (level 71, opening): `up` 1.0, the player stands between two
  boxes and the plan walks around the top corridor to push from the other side.
- **Collection note**: without giveup a dead episode runs to the 500-step cap while a solved one takes about 90
  moves, so at `--epsilon 0.1` about 41 percent of the smoke records were fallback labels from deadlocked boards (a
  quarter of random actions are pushes, and Microban corridors turn many of them into corner or freeze deadlocks).
  With the pj.json default 0.03 and `giveup()` the share is 1.1 percent (table above).
- `python -m playjev.collect sokoban --steps 2000 --shard smoke` (pj.json epsilon 0.03, giveup on): 2000 records in
  15 s, no errors, 25 episodes started (levels 71 to 95), 12 solved, 5 ended by giveup, the rest still running at
  the record limit; 38 records where the taken action differs from the teacher's.
- All 155 start positions pass the deadlock pre-check (none flagged dead); the offline probe checks that no search
  ever claims a level "provably unsolvable" (all Microban levels are solvable), so the exhaustion proof cannot be
  firing wrongly there.
