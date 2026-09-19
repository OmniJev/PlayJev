# invaders teacher (`playjev/teachers/invaders.py`, `InvadersTeacher`)

## Algorithm

The ship fires by itself every 300 ms, so the only decision is where the ship stands. Every step the teacher
runs a small dynamic program over the next `HORIZON = 28` steps (2.3 s, longer than any enemy shot needs to
reach the ship). The state is (step k, ship x on a 16.67 px grid; one grid unit is one step of travel at
200 px/s), the transitions are left / noop / right, and positions beyond the walls (x in [14, 786]) are cut.
The cost is lexicographic:

1. **Hits.** For every enemy shot in flight (`enemyShots`, `[x, y, vx, vy]`, y is the bullet's bottom edge)
   the frames in which its 9x9 body overlaps the ship's y band (ship 28x21 centred on y = 500) are enumerated,
   and for each such frame the grid positions and moves that put the ship within `(28 + 9) / 2 + 4` px of the
   bullet's x are marked as a hit. The 4 px margin covers the one-frame key lag (the ship's velocity changes in
   the frame after the key event) and the rounding of info() to integers. A hit costs `BIG = 1e4`, so the DP
   first minimises hits and only then cares about aiming.
2. **Aiming bonus** (subtracted). A shot fired at time t hits the lowest alien of column c if the ship is at
   `48 c + blockX(t + tau_c)`, where `blockX` is the alien block's triangle wave (x 100 to 200 at 50 px/s,
   direction tracked from the previous step's `alienBlockX`) and `tau_c = (440 - 50 row) / 500` s is the
   bullet's flight time to that row (bullet centre from y = 490 at 500 px/s). Each future step k gets a bonus
   `w_k * f(d)` with d the distance to that lead-corrected x, `f = 1` within `AIM_TOL = 8` px and falling
   linearly to 0 over `AIM_RAMP = 60` px, plus a `-0.001 d` slope so far-away targets still pull. `w_k = 1.0`
   when the auto-fire timer goes off during step k, `0.3` otherwise, so dodges are planned into the gaps
   between shots. The timer phase is read from the newest of my bullets (`myShots`, fired from y = 508 at
   500 px/s, so `t_fired = now - (508 - y) / 500`) and remembered per episode; the timer restarts after a lost
   life, which the next observed bullet corrects.

**Target column**: the column whose lead-corrected x is nearest to the ship, with `HYSTERESIS = 40` px in
favour of the previous target so the ship does not dither between two columns. Aliens that one of my bullets
in flight is already going to hit (predicted with the same block motion) are discounted, so the ship moves on
to the next column as soon as the current one's fate is sealed rather than after the kill lands.

**Soft targets**: 0.8 on the best first move(s) (ties split evenly), 0.2 spread over first moves with the same
hit count whose cost is within `ACCEPT_SLACK = 1.0` (one shot's worth of aiming bonus) of the best, zero on
moves that take a hit the best move avoids or that clearly waste the step. The task sheet asks for 0.8 instead
of Snake's 0.9 because here two moves are often nearly equivalent (stand still now and catch up next step, or
the reverse). Ties among all three moves (nothing to shoot, nothing incoming) come out as 1/3 each.

Cost per step: **0.95 ms mean, 1.4 ms p95** (pure Python, one page, measured over 884 act() calls), about 6 ms
max apart from a 20 ms first-call warm-up. The DP touches about H^2 = 800 cells with three transitions each.

## Numbers (local workstation, 8 pages)

`python -m playjev.teacher_eval invaders --pages 8 --episodes 16`:

| policy | episodes | score mean / median / max | episode length mean | capped |
|---|---|---|---|---|
| teacher (this file) | 16 | **400 / 400 / 400** | 183 steps | 0 |
| random (`playjev.play --policy random`) | 16 | 215 / 215 / 290 | 214 steps (dies) | 0 |
| greedy probe from NOTES.md (steer under nearest lowest alien, sidestep shots) | | 400 | 330 to 340 steps, 1 to 2 lives left | |

400 is the game's maximum (40 aliens x 10). Every teacher episode clears the wave; the theoretical floor is
40 shots x 3.6 steps = 144 steps if every shot kills, so 183 steps means about 79 percent of shots kill (the
rest are spent moving between columns and dodging). A competent human clears this wave with a life or two
lost; the published game has no leaderboard.

Per episode (`runs/probe/invaders/probe.py --pages 8 --episodes 16 --seed0 9`, greedy argmax, seeds 9 to 24):

| episode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| score | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 | 400 |
| aliens killed | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 | 40 |
| steps to clear | 163 | 170 | 174 | 175 | 180 | 185 | 192 | 203 | 174 | 170 | 177 | 177 | 174 | 177 | 170 | 192 |
| hits taken (of 30) | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 |
| lives lost | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Clear rate 16/16, steps to clear mean 178 (min 163, max 203), hits taken mean 0.3 per episode (5 in 16 episodes,
the ship can absorb 30), lives lost 0. A separate 4-episode run on seeds 1 to 8 gave 170 / 178 / 181 / 221 steps
with 1 / 0 / 0 / 2 hits. act() cost over these 3160 calls: 0.97 ms mean, 1.6 ms p95.

`python -m playjev.collect invaders --steps 2000 --shard smoke --epsilon 0.1`: 2000 records in 14 s on 8 pages (145 records/s, the env is the bottleneck), no `errors` lines, 8 episodes
finished inside the shard, all at score 400 even with 10 percent random actions. Label mix over the 2000 records:
noop best 43 percent, left 29 percent, right 28 percent; the most common target vectors are (0.1, 0.1, 0.8),
(0.8, 0.1, 0.1), (0.1, 0.8, 0.1), then one-sided ones like (0.8, 0.0, 0.2) and (0.0, 1.0, 0.0) where a move
walks into a shot. Frames checked by eye: `frames/0000331.jpg` (seed 1003, step 41) has a shot straight above
the ship and two more to its lower left, target (0, 1, 0) = right is the only safe move; `frames/0000328.jpg`
(seed 1000, step 41) has the ship under the column it just hit with every shot to its right, target
(0.1, 0.1, 0.8) = hold; `frames/0000329.jpg` (seed 1001, step 41) has no threat and the lead-corrected column
slightly left, target (0.8, 0.1, 0.1).

## Data collection: epsilon and label quality

`games/invaders/pj.json` carries `"collect": {"epsilon": 0.1}`, the default the collector uses without
`--epsilon`. Reasoning: at epsilon 0.1 the teacher still clears every episode (all 8 episodes that finished
inside the smoke shard scored 400), so the random moves add off-column positions and near-miss shots to the
state distribution without producing lost-life or game-over states the trained policy should never reach; a
higher rate mostly adds frames of the ship wandering under empty sky. For reference, `teacher_eval --epsilon 0.3` over 8 episodes still scores
400 in every episode (mean length 208 steps versus 183 without noise), so the teacher tolerates a higher rate; 0.1
keeps about 90 percent of the frames on the trajectories the policy is meant to reproduce.

Label quality in the smoke shard (2000 records, epsilon 0.1): fallback labels (the uniform 1/3 vector returned
outside the Play state or with no alien alive) 0 of 2000; one-hot labels 175 (8.8 percent), all of them states
with exactly one move that avoids a hit, so they are informative rather than degenerate; exact two-way ties
1 (0.05 percent). `giveup()` is left at the base class default (False): the wave never descends and the ship
never stops firing, so no state is unrecoverable, and the game ends itself on the third lost life or the last
alien.

## Known failure modes

- The hit model is conservative by 4 px on each side, so in a dense field the DP occasionally sidesteps a shot
  that would have missed; this costs aiming time, never health.
- Block direction is inferred from consecutive `alienBlockX` values. On the very first step of an episode it
  is guessed from position (block x < 150 means moving right, which is always the case after start()); at a
  reversal the guess is one step late, so the lead is off by about 8 px for one step (still inside the 19 px
  hit tolerance).
- The auto-fire phase is unknown until the first of my bullets is visible (the first shot goes off 300 ms after
  Play starts, about step 3) and for up to 4 steps after a lost life while the timer restarts; those steps use a
  flat weight of 0.6.
- Unavoidable hits (shots converging so that no path survives the horizon) still take the fewest-hit path. The
  5 hits taken in the 16-episode run have not been attributed yet (unavoidable geometry versus the hit model's
  approximations, which are the 4 px margin, the grid of 16.67 px positions and the linear ship motion inside
  a step); with 30 hits per episode to spend, none of them cost a life.
- The bonus is computed for the current target column only; the plan does not look ahead to the next column.

## info() requests

None. Everything needed is already in info(): `playerX`, `alienBlockX`, `aliens`, `enemyShots` (with
velocities), `myShots`, `health`, `lives`, `state`, plus `obs["t"]` from the shim. A `blockDir` field (sign of
the block tween's velocity) would remove the one-step guess at reversals, but it is not needed.
