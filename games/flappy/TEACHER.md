# flappy teacher (`playjev/teachers/flappy.py`, `FlappyTeacher`)

## Algorithm

Exact physics lookahead with a depth-first search over flap/wait sequences.

1. **Model.** One step is 5 physics ticks (`v += 0.25; y += v`), a flap sets `v = -4.6` before the first tick of
   the step (so a flap step always moves the bird -19.25 px), the pipe advances 2.222 px per tick from the x in
   `info()`. The collision test is the game's own: the rotated bounding rect of the 34 x 24 sprite (rotation
   `min(9v, 90)` degrees), width shrunk by `8 sin(|rot|/90)`, height `(24 + rotated height) / 2`, pipe box from
   `x - 2` to `x + 50`, gap `gapTop .. gapTop + 90`, ground at `y_raw_bottom >= 420`, ceiling clamps `y` to 0
   without killing. Checked against the game over 32 000 steps: 0 mismatches in `birdY` and `velocity` (the only
   deviation is the first step of an episode, which runs 4 ticks because the 5th timer lands 2 µs past the step
   boundary on the shim's clock; harmless, the first pipe is 89 steps away).
2. **Search.** From the current state, depth-first over the two actions per step until the bird has cleared the
   third upcoming pipe (`PIPES_AHEAD = 3`, cap `MAX_H = 50` steps). Children are ordered by a hover heuristic
   (`prefer`): the target is the next pipe's gap centre once its left edge is inside the frame (`x < 320`), else
   mid-screen (210); flap when the centre passes its closest approach to `target + OFF` (OFF = 16) while
   descending, or when one more wait would drop it past `target + OFF + SLACK` (SLACK = 10). The first surviving
   sequence found is therefore the heuristic's trajectory with the fewest corrections. Doomed states are
   memoised by (depth, y, v, pipe index); y lives on a 0.25 px lattice so the keys are exact. A leaf at the
   horizon must also survive a rollout of the heuristic through the following pipe, which makes the search
   prefer pipe exits that leave the next gap comfortably reachable.
3. **Two tiers.** Tier 1 demands `MARGIN_SAFE = 2` px of clearance from pipe edges and ground and uses the
   rollout leaf; if nothing survives, tier 2 demands `MARGIN_MIN = 0.3` px (layout rounds `top`/`left` to
   1/64 px, so zero clearance is not safe) with a plain interval-reachability leaf.

Targets: 0.9 on the first action of the surviving sequence, 0.1 on the other action when it also has a surviving
continuation in either tier, 0.0 when both tiers exhaust its subtree (certain collision within the horizon: for
example a flap while the bird is in the upper half of the gap with a pipe within 6 steps, since the 39.5 px lift
cannot be undone by free fall in time). If both actions are doomed, the one that survives longer gets 1.0 (0.5
each on a tie). Past `NODE_BUDGET = 8000` search nodes the heuristic's action is used with 0.9 / 0.1 (never hit
in evaluation; the maximum seen was 3785 nodes).

Why search rather than the predictive rule alone: the gap is 90 px and the bird's box is 24 to 32 px tall, a
flap lifts the bird 39.5 px over the next 4 steps and decisions come every 5 ticks, so the flap point inside a
pipe has to land in a 22 px window; with 170 px jumps between consecutive gap centres and 9 steps between pipes,
the approach velocity decides whether any flap timing works, which a margin on a predicted y does not capture.

## Numbers

`python -m playjev.teacher_eval flappy --pages 8 --episodes 16` (2000-step cap, seeds 1 to 16):

| policy | episodes | score mean | median | max | episode length mean | capped |
|---|---|---|---|---|---|---|
| teacher (eps 0) | 16 | 114.00 | 114 | 114 | 2000 | 16 of 16 |
| random (`playjev.play --policy random`) | 16 | 0.00 | 0 | 0 | 90 | 0 |
| hook agent's one-line rule (NOTES.md) | | 2 to 3 | | | | |

114 pipes is the cap (the first pipe reaches the bird at step 90, one pipe every 16.8 steps). The teacher never
died in 16 x 2000 steps, so its ceiling is above the cap; the reference game's medals stop at 40 (platinum) and
a competent human scores in the tens. Cause of death at eps 0: none. With the collector's perturbation
(eps 0.1, random action taken 10 % of the time, labels unchanged, seeds 100 to 115): score mean 1.25, median 0.5,
max 5, length mean 114 steps, deaths 13 at the top pipe and 3 at the bottom pipe (the ceiling never kills in
this game). See the failure modes below.

Earlier version (two-pipe horizon, interval leaf, no margin) for the record: 13 of 16 capped, deaths at 48
pipes (top pipe, model clearance 0.001 px, lost to layout rounding) and at 106 pipes (left pipe B high and
slow, and pipe C, 100 px lower, needed the bird 1 px below the furthest free-fall point). The three-pipe
horizon, the rollout leaf and the margins fixed all three deaths.

`python -m playjev.collect flappy --steps 2000 --shard smoke --epsilon 0.1`: 2000 records in 7 s, 23 episodes,
labels (flap, wait): (0.1, 0.9) 1457, (0.0, 1.0) 276, (0.9, 0.1) 204, (0.5, 0.5) 33, (1.0, 0.0) 30. Frames
checked: bird nose-down below the gap centre with the pipe 6 steps away, label flap 0.9 (frame 0000656); bird
high in the gap between two pipes, label wait 1.0 / flap 0.0 (0000768); bird rising at y = 127 towards a gap
top of 103, label flap 1.0 / wait 0.0 (0000664), verified by exhaustive search: after a wait the bird's only
way through is a flap inside the pipe that then clips the upper pipe.

## Collection epsilon and label quality

Smoke shard (eps 0.1, 2000 records): degenerate labels (0.5, 0.5), which the teacher gives when both actions
are proven fatal and survive equally long, 33 of 2000 (1.65 %); node-budget fallbacks (heuristic action with
0.9 / 0.1) 0 of 2000 (0 of 32 000 at eps 0; one in 3656 steps at eps 0.02, in an already doomed state). Every
other label comes from the exact search.

`pj.json` now carries `"collect": {"epsilon": 0.02}`, the default the collector uses without `--epsilon`.
Reasoning: a random action is fatal about one time in three here, so eps 0.1 ends episodes after 114 steps on
average with 1.25 pipes (seeds 100 to 115) and the shard is mostly the pipe-free first 90 steps, while eps 0.02
still perturbs the bird about once per pipe cycle and lets episodes reach 163 steps and 4.25 pipes on average
(max 10, seeds 200 to 215). Mixing in eps 0 shards (2000-step episodes, 114 pipes) covers the long-run states.

`giveup()` returns True after an `act()` in which both tiers proved every flap/wait sequence collides within
the horizon (the bird dies within a few steps whatever it does), so the collector ends the episode instead of
recording the remaining degenerate labels. 23 of 3656 steps at eps 0.02, 0 at eps 0.

## Cost per step

Pure Python. Mean 0.66 ms per `act()` (median 0.40, p99 5.0, max 33.6 ms) at eps 0 over 32 000 steps; 0.60 ms
mean at eps 0.1. Search nodes mean 133, p99 594, max 3785 (budget 8000). The environment runs at about 500
env-steps/s with 8 pages, so the teacher is under 1/3 of the step time even at the p99.

## Failure modes

- The teacher itself does not die within 2000 steps on 16 seeds. An untested tail remains: the exact horizon is
  three pipes plus a heuristic rollout through the fourth, so a sequence of four or more gaps that alternate
  between the extremes (80 and 249) could in principle force a state the rollout accepts but the exact search
  later rejects. Not observed.
- Random perturbations are lethal in this game: about 70 % of states have one certainly fatal action, so a
  random action kills roughly one time in three and eps 0.1 gives episodes of about 114 steps (score 1 to 5).
  For DAgger-style coverage use a much smaller epsilon for flappy (0.02 or less), or randomise only among
  actions the teacher gives nonzero probability (the 0.0 labels are exact within the horizon).
- Node budget fallback (heuristic action, 0.9 / 0.1) would only be reached in a deeply doomed state; never seen.
- The first step of an episode runs 4 ticks rather than 5; the model assumes 5. Irrelevant for play.

## info() requests

None. `birdY`, `velocity`, `pipes[] {x, gapTop}` and `nextPipe` are everything the model needs; `pipes` includes
already-passed pipes still on screen, filtered by `x >= nextPipe.x`.
