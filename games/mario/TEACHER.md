# mario teacher (`playjev/teachers/mario.py`, `MarioTeacher`)

## Algorithm

Rule based, right-going. The default action is `right run`. Jumps are taken over gaps, steps, pipes and
enemies, Mario waits (`noop`) in front of a pipe whose piranha plant is out and backs off (`left`) only when
nothing else survives.

The rules are checked with a forward model instead of tile thresholds. The model is a Python copy of
`Character.Move` / `SubMove` from `code/character.js`: run (1.2 px/tick) and walk (0.6) acceleration with
inertia 0.89, the held jump (`JumpTime` 7 ticks at -1.9 px/tick each, 66.5 px rise when held 8 ticks),
gravity (`Ya*0.85+3`), tile collision against the 21x13 window from `info()` (`#` and `?` block, `^` blocks
from above only), wall sliding, the wall jump, and stomping. The hook's `applyAction` rule (one tick with the
jump key up when Mario is grounded or sliding with the key still held) is part of the step model, so a
"step" in the model is 3 or 4 ticks exactly as in the env. Nearby sprites are simulated with the physics of
`enemy.js`: walkers at 1.75 px/tick that turn at walls and fall off ledges (red koopas turn at cliffs),
winged ones hopping (`Ya=-10`, gravity 0.6), piranha plants that rise 59 px once they have rested 40 ticks
and Mario is more than 24 px away. Sprite velocities come from matching `sprites[{kind,dx,dy}]` between
consecutive steps; an unmatched sprite is assumed to walk towards Mario.

Each step, fixed key sequences ("plans") are rolled out for at most 30 ticks (10 steps) or until Mario has
passed a target 160 px ahead and landed inside the known tile window:

`RR` run | `RRJ1..3` run+jump held 1, 2, 3 steps | `RR1J..RR3J` run 1 to 3 steps then a full jump |
`R` walk | `RJ1..3` walk+jump | `J3` jump in place then run | `WAIT` noop | `BACK` left.

Plans are ranked by (reached the target alive, cleared the first hazard that kills or stops `RR`, alive,
distance reached, list order). `RR` is tried first and, when it succeeds, only the label alternatives are
also rolled out, so in open terrain the cost is a few rollouts. The first action of the best plan is the
teacher's action. A stall guard forces `right run jump` after 8 steps without x progress while pushing right.

Hidden state (`JumpTime`, `Sliding`, `MayJump`, facing) is not in `info()`. The teacher keeps a mirror by
replaying its previous action through the model from the previous observation and adopting the replayed
hidden variables when the replay lands on the observed `x, y, ya, onGround` (within 0.6 px). Our own action
is tried first, then the other six (collect may have taken an epsilon-random action). Mirror hit rate:
95 to 100 percent of steps per episode; misses are stomp bounces and the first step of an episode, where the hidden state
is inferred from `xa`, `ya` and the tiles.

Soft targets: 0.8 on the chosen action, 0.2 spread evenly over the other right-going actions whose one-step
plan does not die in the model, zero on the rest; `noop` and `left` only get mass when chosen (then 0.8, or
1.0 if no right-going alternative survives). Convention from the task brief (0.8 / 0.2 instead of Snake's
0.9 / 0.1) because in Mario several right-going actions are usually acceptable.

## Hyperparameters

`T_MAX = 30` ticks per rollout, `TARGET_DX = 160` px, hazard clearance 12 px, sprite match radius
`12*dt+8` px, piranha trigger 24 px / 40 ticks, winged detection `vy < -2.5` px/tick, stall guard 8 steps,
reactive continuation: the plain-run continuation of a plan hops when a walker is within 40 px ahead at foot
level (`REACTIVE_CONT`), mirror tolerance 0.6 px / 0.3 px on `ya`.

## Numbers (this machine, headless chromium, 8 pages)

Random policy, `python -m playjev.play mario --policy random --pages 8 --episodes 32 --seed0 1`:
32 episodes, score mean 792, median 548, max 3691, episode length mean 91 steps, no win.

Teacher, `python -m playjev.teacher_eval mario --pages 8 --episodes 32`: 32 episodes, score mean 5064,
median 5264, max 5487, episode length mean 170 steps (17 s of game time), capped 0, 175 env-steps/s with
the teacher in the loop (258 for the random policy in `bench`).

Teacher on exactly seeds 1 to 32 (one episode each, argmax): 27 wins of 32 (84 percent), score mean 4870,
median 5286, max 5490, episode length mean 165 steps; the 5 deaths at x = 2169, 1419, 2267, 3626, 3458
(mean progress 2552 px). A win is 4200 to 4520 px
of progress plus the 1000 bonus, 155 to 210 steps. For reference, a competent human finishes these
short levels most of the time in about the same number of steps; the random policy dies after 91 steps
and 770 px on average and never finishes.

Cost per step: `act()` mean 1.0 to 1.5 ms (pure Python, no dependencies), median 0.6 ms, p95 2.5 ms,
max 4.7 ms on steps where `RR` fails and all 14 plans are rolled out.

## Data collection

`python -m playjev.collect mario --steps 2000 --shard smoke --epsilon 0.1`: clean, 143 records/s, 19 episodes
started, 8 ended (6 wins, 2 deaths at 2303 and 2972 px). Labels: `right run` 40 percent, `right run jump`
27, `right` 21, `jump` 6, `right jump` 5, `noop` 0.5, `left` 0.2; total mass on `noop`+`left` 0.7 percent.
Degenerate labels: 12.8 percent of records are one-hot (no right-going alternative survives in the model,
mostly mid-air steps where the trajectory is committed), 3.5 percent are one-hot `right run`; the no-tiles
fallback (`right run` 1.0 when `info()` has no window) never fired. Three frames checked against their
labels: mid-air before a raised pipe, label `right run jump` 0.8; standing with a goomba walking in from a
lower ledge, label `noop` 1.0 (waiting for it to arrive and be stomped or hopped); descending between two
pillars towards a platform, label `right run` 0.8 / `right run jump` 0.2. All three read correctly.

Recommended `epsilon = 0.1` (written to `pj.json` as `"collect": {"epsilon": 0.1}`): a random action costs
one step (30 px, or a shortened jump) and the mirror recovers the hidden state afterwards, so 0.1 gives
off-policy states (mistimed jumps, standing next to enemies) at a cost of a few percent more deaths per
episode, while a higher rate would start cutting jumps mid-air often enough to fall into pits.

`giveup()` is left at the base default (False): Mario has no unrecoverable state short of the 200 s level
timer; the only stall the teacher ever showed (waiting for a piranha) resolves by itself.

## Known failure modes

1. Enemies arriving during a committed jump. A full-speed jump covers about 175 px and sprites spawn at the
   camera edge 176 px ahead, so a walker can appear after the jump starts and meet Mario at the landing.
   Mid-air only horizontal speed can be modulated. This is 4 of the 5 deaths on seeds 1 to 32 (koopa or
   goomba at `dx` 3 to 7 px on landing). A fix would prefer shorter jumps near the visibility edge or treat
   the last 40 px of a landing zone as occupied.
2. Falling enemies. The vertical velocity of a tracked sprite is the average over the last step, so a
   walker that stepped off a ledge during that step is modelled a few px too slow; in one death the model
   predicted a stomp and the game had the koopa 8 px lower (a hit).
3. Terrain beyond the tile window (12 tiles ahead) is unknown. A plan that leaves the window airborne is
   not a success, but it still ranks above plans that die in view, so Mario occasionally flies into a pit
   just beyond the window (1 of 5 deaths, seed 20).
4. Piranha pipes taller than 2 tiles: the model makes Mario run to the pipe and wait under it until the
   plant is down (the plant does not rise while Mario is within 24 px), then jump from standing. Works but
   costs 10 to 30 steps.
5. The `spiky` sprite kind in `info()` is a piranha plant (see below); a real spiky would be handled the
   same way (unstompable), which is right.

## info() notes and requests (no change strictly needed)

- `spriteKind` in `pj_hook.js` tests `instanceof Mario.Enemy` before `instanceof Mario.FlowerEnemy`, and
  `FlowerEnemy` inherits from `Enemy` with `Type = Spiky`, so piranha plants are reported as `spiky`, never
  as `piranha`. The teacher treats both alike. Reordering the two tests would make the name honest.
- Tile window rows above the level (abs row < 0) and below it (abs row >= 15) are drawn as `#`; the game
  treats them as empty (Mario dies below row 15). The teacher recomputes this from `originY`; a `.` there
  would be clearer.
- Nice to have, not needed: sprite `xa`, `ya` and `winged` per sprite, and Mario's `jumpTime`, `sliding`,
  `mayJump`. The teacher reconstructs all of them (velocities from consecutive steps, Mario's hidden state
  from the mirror replay), so this is a convenience for future teachers, not a blocker.
