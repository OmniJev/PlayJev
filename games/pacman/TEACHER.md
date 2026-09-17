# pacman teacher (`playjev/teachers/pacman.py`, `PacmanTeacher`)

## Algorithm

Ghosts in this game have no chase logic. Each one is a random walker with a rule that is exact and cheap to
model: at every grid square it draws a perpendicular direction at random and turns if that square is floor, so it
always turns at a 4-way, goes straight or turns 50/50 at a T, turns at a corner, and never reverses. It moves one
block per 5 ticks (2 units), the same as Pac-Man; 10 ticks per block while edible, 3 while just eaten.
Collision is Euclidean distance under one block, checked every tick. All of this is in `info()`: positions to
a tenth of a block, directions, `vulnerable` and `eaten` flags, the tick counter, the map with pellets.

1. **Ghost occupancy over time.** For every ghost that is dangerous within the next 45 ticks, propagate its
   probability distribution over (cell, direction) one block at a time with the rule above (worst case over the
   coin flips: any cell with positive mass counts as occupied). Edible and just-eaten ghosts are propagated at their
   own speed and only become risk from the tick they turn dangerous again (see timers below). Two ghosts that could
   be at cell c at times t1, t2 give the entries (c, t1), (c, t2).
2. **Collision windows.** Pac-Man arriving at cell c at time t moving in direction d is unsafe if a ghost can be at c
   within 5 ticks of t (either order), except a ghost that left c exactly 5 ticks earlier in direction d (one block
   ahead, same speed: it never closes). Two movers at right angles have a closest approach of k/sqrt2 when they are
   k blocks from the common square, so k < 1.42 collides: a ghost that crossed c 6 or 7 ticks earlier and departed
   across d is unsafe, and leaving c in direction d_out while a ghost will arrive at c 6 or 7 ticks later across
   d_out is unsafe (`depart` check, applied at every expansion and to the square Pac-Man has just left).
3. **Planning.** For each of the four keys: work out what the key does at step granularity (a turn at the next grid
   square; a reversal is immediate, and if the square behind is reached inside this 3-tick step the held key carries
   Pac-Man straight through it, so the first free choice is one square further). BFS from there over cells that are
   safe at Pac-Man's own arrival time. Value = ticks to the nearest pellet or power pill, or to an edible ghost when
   `remaining edible ticks >= arrival + 12` and it is within 10 blocks, with a bonus of value/5 ticks (ghosts are worth
   50/100/150/200 in a row, so the fourth one is worth an 8-block detour). Reversals cost 4 extra ticks (hysteresis:
   without it the plan flipped every step and Pac-Man dithered into ghosts). A key only counts if its safe region
   reaches past the 45-tick horizon; otherwise the nearest pellet may sit in a corridor whose both exits a ghost
   reaches first (this single rule took the mean from 3.7k to above 10k).
4. **Soft target.** 0.9 on the best key(s) (ties split), 0.1 spread over the other keys that are pressable and
   safe, zero on keys that hit a wall, stop Pac-Man, or lead into possible ghost occupancy. When no key has a safe
   route to a pellet, all mass goes to the deepest safe region; when every first step is exposed, to the key that
   meets the ghost latest (continuing straight on ties). Same 0.9/0.1 convention as Snake.

Timers are reconstructed from `tick` (they are not in `info()`): pills dropping by one during a step means a pill
was eaten at the earliest at the step's first tick, so ghosts are edible until that tick + 241 and dangerous again
where they stand; a ghost's `eaten` flag flipping on means it is harmless until that tick + 91. Both use the earliest
tick of the step (3 ticks per step), the conservative side. Ghost value counting (`n_eaten` since the last pill)
comes from the same flag flips.

## Hyperparameters

| name | value | meaning |
|---|---|---|
| `HORIZON` | 45 ticks | ghost propagation depth (9 blocks), also the escape depth a key must reach |
| `W_SAME` | 5 ticks | same-cell window either way (one block) |
| `W_PERP` | 7 ticks | same-cell window when the two movers are at right angles (1.42 blocks) |
| `W_ADJ` | 2.5 ticks | neighbouring-cell window: a ghost crossing the next square at right angles 0 to 2.5 ticks before Pac-Man reaches the cell (mostly overlaps the departure rule, kept as belt and braces) |
| `REVERSE_PENALTY` | 4 ticks | added to the value of a reversal |
| `CHASE_MAX`, `CHASE_SLACK` | 50, 12 ticks | chase an edible ghost at most 10 blocks away, keep 12 ticks of edible time |
| `PILL_TICKS`, `EATEN_TICKS` | 241, 91 | from pacman.js (`secondsAgo > 8` and `> 3` at 30 FPS, checked after the move) |

## Numbers

Random policy (`python -m playjev.play pacman --policy random --pages 8 --episodes 16`): score mean 113, median 60,
max 360, episode length mean 49 steps.

Teacher, `python -m playjev.teacher_eval pacman --pages 8 --episodes 16` (2000-step cap, seeds 1 to 24):

| policy | episodes | score mean | median | max | episode length mean | capped |
|---|---|---|---|---|---|---|
| random | 16 | 113 | 60 | 360 | 49 steps | 0 |
| teacher | 16 | 9029 | 10060 | 10940 | 1761 steps | 12 of 16 at 2000 steps |

Twelve of sixteen episodes ran into the 2000-step cap, so the eval numbers are cap-limited. A development probe
with a 3000-step cap (9 episodes) gave score mean 10463, median 13810, max 15140, mean length 2131 steps, 32 levels
cleared (3.6 per episode), 5 of 9 episodes capped at level 6 with 13.8k to 15.1k points. A level is 1980 points of
pellets plus up to 2000 from ghosts, so about 2400 points per level means most edible ghosts get eaten. The hook
agent's earlier BFS teacher (avoid squares within 2 of a ghost) had mean 2774, max 6910, 9 levels in 8 episodes.
The game grants an extra life at 10,000 points, its own notion of a good run; a casual player clears one or two
levels of this version.

Cost: `act()` mean 0.24 ms, p95 0.48 ms, max about 9 ms (the first call builds the maze tables), measured inside
the probe on this machine with other jobs running; the env runs at 620 to 880 steps/s with 8 pages either way.

Smoke shard, `python -m playjev.collect pacman --steps 2000 --shard smoke --epsilon 0.1`: 2000 records in 5 s, 11
episodes, 4 deaths, no errors, no giveups. Frames 0, 700 and 1500 checked against their labels: the start state
gets left/right 0.5 each (symmetric), a Pac-Man in the eaten-out bottom-left with pellets above gets up 0.9, a
Pac-Man three blocks left of a blue (edible) ghost gets right 0.9. Label modes on the same policy and seeds:
99.9% of decisions had a safe route to a pellet (`target`), 0.1% used the deepest-safe-region fallback, none used
the exposed fallback. Shapes: 0.9/0.1 69%, 0.45/0.45/0.1 13%, 0.9/0.05/0.05 7%, single 1.0 5%, the rest ties.

Collection epsilon: 0.1, written to `pj.json` as `"collect": {"epsilon": 0.1}`. Reasoning: the teacher rarely
dies on its own, so without random actions the shard would hold almost no near-ghost states; at 0.1 the smoke shard
still lost only 4 lives in 2000 decisions while every label stays the teacher's, and a random press in this game is
cheap (a buffered turn that is usually not possible yet, or a reversal the next step undoes).

`giveup()` is not implemented: a death ends the episode within the step (`done` flips in the collision tick), the
level clear countdown is skipped by the hook, and there is no unrecoverable-but-alive state, so the base class
default (False) is right.

## Known failure modes

- The model is worst case over the ghosts' coin flips: any cell a ghost could reach is blocked, so with three ghosts
  around Pac-Man it sometimes waits in a safe pocket instead of taking a 1/8 risk, which costs pellets per step, not
  lives. Deaths that remain come from states where every option is exposed (two ghosts closing a corridor from both
  ends beyond the 45-tick horizon, or a ghost turning dangerous again next to Pac-Man); the fallback then picks the
  latest encounter, which is a guess about coin flips it cannot see.
- Timing is exact for dangerous ghosts (positions are even tenths). Just-eaten ghosts move 4 units with grid
  snapping, modelled as 3 ticks per block; the tick they turn dangerous again is known to within the 3-tick step.
- The BFS values the nearest pellet only, so late in a level it may cross the maze for a single pellet.
- The `eaten_pause` state (game frozen for 10 ticks after eating a ghost) is planned like any other step; the
  positions do not move, so the plan simply repeats.

## Requests for info()

None needed. Everything used is already there. If a hook update is convenient, per-ghost `eatable`/`eaten` start
ticks would remove the 3-tick reconstruction uncertainty, but the conservative reconstruction works.
