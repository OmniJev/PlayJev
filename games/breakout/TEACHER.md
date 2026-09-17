# breakout teacher (`playjev/teachers/breakout.py`, `BreakoutTeacher`)

## Algorithm

The ball's flight does not depend on the paddle until it comes back to the paddle line, and the game is
deterministic, so the teacher simulates the ball's exact path instead of extrapolating it: walls, every
unhit brick (a run of one letter in a row of `info().bricks` is one brick, as in `Court.reset`), the game's
own intercept rules (rectangles expanded by the ball radius 0.3 chunk; the side line facing the ball is
tested first, then the top or bottom line; the nearest hit wins; a side hit flips dx, a face hit flips dy),
the speed increment per brick (`speed += 10 * (1 - speed / maxspeed)`, which only changes timing). One step
of the simulation is one straight segment to the next wall, brick or the paddle line; rows are scanned in
travel order and the scan stops as soon as no farther row can beat the best hit, so a segment costs a
handful of brick tests.

Each env step:

1. Follow the ball to the paddle line `paddle.y - r` (moving down). This gives the landing x, the incoming
   direction, the seconds until impact and hence `K = floor(t / 83 ms)`, the number of whole steps before
   the impact step. A parked ball (the hook launches it this step at (1, -1) from the paddle centre) is
   simulated from that launch state.
2. Candidate paddle centres are the positions reachable by the impact step, one whole move (23.3 px) per
   step: `c_now + k * 23.3` for `|k| <= K`, clamped to the court, kept when the landing x is within
   `w/2 + r - 3 px` of the centre (the true catch window is `w/2 + r`, since the ball is caught iff its centre
   is over the paddle top, expanded by the radius, when it reaches the paddle line).
3. Each candidate is scored by the bounce it produces. The game sets the outgoing direction to
   `normalize(offset / (w/2), -dy_in)` where offset is the hit point relative to the paddle centre, so the
   centre position is the aim. The score is bricks per second over the excursion (simulated until the ball
   is back on the paddle line, cap 60 segments) plus the best follow-up shot (one level of look-ahead over
   the positions reachable by the return), with `-1` when the ball would come back farther than the paddle
   can travel in the time (a guaranteed miss), `10 + rate` when the excursion clears the level, and a tiny
   preference for angled shots when nothing is hittable. The chosen centre is cached per landing
   (`bricks_left`, landing x, incoming dy) so the plan is stable across the descent.
4. Action: move toward the chosen centre if it is more than half a move away, else stay. At the impact
   step (`K = 0`) the three actions are scored directly with the paddle position at impact; the paddle moves
   in whole frames applied before the ball's collision check, so the partial move is `ceil(t * 60)` frames
   of 4.7 px and a moving paddle must keep one frame of margin inside the catch window.

Soft targets: 0.9 on the chosen action, 0.1 on stay when moving, 0.05 on each move when staying, ties split
evenly; a move after which no catchable position is reachable in the remaining steps gets 0, as does a move
at the impact step that lets the ball past. A lost ball (already below the paddle line) or a flight longer
than 80 segments falls back to tracking the ball's x with the same 0.9 / 0.1 shape. `lives == 0` returns
uniform.

Hyperparameters (module constants): `CATCH_MARGIN 3 px`, `FRAME_PX 4` (corner grazes within one frame follow
the game's side-first rule), `MAX_SEG_FLIGHT 80`, `MAX_SEG_AIM 60`, `LOOKAHEAD 1`, `P_BEST 0.9`.

## Numbers

Simulator check against recorded trajectories (`info()` at every step, 17 paddle hits, 387 predictions made
1 to 30 steps before the hit, through brick bounces): landing x error mean 0.05 px, max 0.26 px; the
predicted impact time lands within the predicted step every time.

`python -m playjev.teacher_eval breakout --pages 8 --episodes 16` (seeds 1 to 16, cap 2000 steps):

```
[breakout] teacher eps=0.0: 16 episodes, score mean 22555.31 median 23557.5 max 27835, episode length mean 2000, capped 16
```

Every episode ran to the 2000-step cap (167 s of game time), none ended by losing all lives. Without the
plan hysteresis (previous run of the same seeds) the mean was 21531.88, median 22182.5, max 28185; with the
greedy score and no look-ahead 19398.75 / 19580 / 25170.

Per episode (own driver with lives and levels, same seeds, the run without hysteresis): 25 levels cleared in
16 episodes (1.56 per episode; 9 episodes cleared 2, 7 cleared 1), 0 lives lost in 32,000 steps, per-episode
scores 14460 to 28185 (seed 1 to 16 start on level `seed % 10`, so all ten layouts are covered). The earlier
greedy version lost 1 life in 32,000 steps and cleared 22 levels; the look-ahead version before the impact-step
frame fix lost 2, both by a last-frame move at the impact step (see failure modes).

Random policy (`games/breakout/NOTES.md`, 88 episodes): mean 739, median 305, max 7575, episode length
mean 123 steps (three lives in about 10 s of game time). A competent human clears the first level (30 bricks)
in about a minute; the teacher takes 500 to 650 steps (42 to 54 s) for it and 700 to 1300 steps for the
denser layouts, with the ball speed at its 1.5x cap for most of the level. All 16 teacher episodes end at the
2000-step cap, so the score is set by pace, not by survival.

`python -m playjev.collect breakout --steps 2000 --shard smoke --epsilon 0.1`: 2000 records, 8 episodes, no `errors` lines, total reward 22920. Label
mix 70% stay, 15% left, 15% right; probability vectors are (0.05, 0.05, 0.9) 1349 times, (0, 0.9, 0.1) 299,
(0.9, 0, 0.1) 295, the one-move-zeroed impact-step vectors (0.1, 0, 0.9) / (0, 0.1, 0.9) 49, ties 4. Frames
0000480 / 0000488 (level 0, ball descending almost vertically just under the wall, paddle a third of a width
to its right: left 0.9, then the paddle is under it) and 0000481 (level 3, ball descending left of the paddle
that sits against the right wall, 30 steps to impact: stay 0.9, the target position is the one reachable
later) read correctly against the picture; the left-then-right flip between 480 and 488 (two catchable
targets with near-equal scores, decided by rounding noise) is what the hysteresis removes (5 direct
left-right flips in 400 steps afterwards, all after a brick hit changed the landing or at the impact step).

Cost: `act()` mean 0.32 ms per step (0.14 ms without the look-ahead), max 11 ms on a replan
in a dense layout (level 8, 250 bricks, five candidates times their follow-ups); replans happen once per
landing, so the env throughput with 8 pages is unchanged within noise (700 to 900 env-steps/s on local-workstation, the
in-page cost dominates). Pure Python, no dependencies.

## Collection defaults

`games/breakout/pj.json` carries `"collect": {"epsilon": 0.1}`, read by `playjev.collect` when `--epsilon` is not
given. Reasoning: a random action displaces the paddle by one 23 px move, which the plan absorbs within a step
or two while the ball is still in flight, so 0.1 puts the paddle off its target in about one frame in ten (the
states a trained model produces when it errs) at no cost in lives (0 lost in the 2000-record smoke shard) and
without drifting the label mix away from the teacher's own play.

Label kinds in the smoke shard (replayed with the same seeds and rng, each label classified by the branch that
produced it): 1882 planned (94.1%), 61 impact-step (3.0%, the three actions scored directly), 57 fallback
(2.85%, all of the "trapped" kind: the flight simulation reached its 80-segment cap while the ball was raking
bricks above, so the label tracks the ball's x until the landing becomes visible), 0 lost-ball fallbacks,
0 unreachable, 0 uniform; 5 tie labels (0.25%). No `giveup()` is implemented: a lost ball is deducted by the
game about four steps after it passes the paddle line and the next ball launches on the following step, so
the game never sits in an unrecoverable state.

## Known failure modes

- Pace, not survival, bounds the score: a level needs one excursion per brick or two (about 25 steps each,
  the geometric minimum for bricks 170 px above the paddle is about 22), so a 2000-step episode clears one
  to two levels plus most of the next. Multi-brick excursions (the ball behind the wall) are preferred by
  the score when the simulation sees them, but the teacher does not deliberately dig a channel.
- Lost lives: none in the final 32,000-step run. The only case seen during development was a last-frame
  aim adjustment at the impact step where the game's frame-quantized paddle motion put the edge 2 px
  short; the frame-rounded partial move plus a one-frame margin at `K = 0` removed it. A brick hit low in the court that
  sends the ball to the far side with less time than the paddle needs would also be unrecoverable, but the
  look-ahead scores such shots negative before they are chosen.
- The game prefers a side hit when a per-frame segment crosses both a side line and a face line of one
  expanded brick (a corner graze within about 4 px); the simulator applies that rule with a fixed 4 px
  window rather than the actual frame boundaries, so a graze can bounce the other way. The plan is
  recomputed every step, and the paddle has more than a second to correct, so this has not caused a miss.
- With epsilon-random actions the cached plan is invalidated when the chosen centre is no longer reachable
  and recomputed from the current position.

## info() requests

None. Everything used (court box and chunk, ball position, unit direction, speed and moving flag, paddle x
and width, the brick rows, `bricks_left`, `lives`) is already in `info()`. The ball radius (0.3 chunk), paddle
speed (20 chunks/s), launch speed (15 chunks/s) and max speed (1.5x) are game constants copied from
`breakout.js`.
