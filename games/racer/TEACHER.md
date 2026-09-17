# racer teacher (`playjev/teachers/racer.py`, `RacerTeacher`)

The score is distance in road segments and a lap (6705 segments plus the start offset, about 6713) ends the
episode, so every policy that laps gets the same score; the teacher's quality is the lap length in steps. A lap with
no slowdown at all takes about 1372 steps (5 s of acceleration, then 112 s at 12000 units/s, 83.3 ms per step).

## Algorithm

Rule based, with a one-step model of the game's `update()` (`games/racer/v4.final.html`) to check each action.

1. Lateral target `x*`: a grid over [-0.8, 0.8] in 0.05 steps, minimising
   `W_CURVE (x - x_curve)^2 + W_MOVE |x - playerX| + car costs`, where `x_curve = 0.2 * c_eff` hugs the inside of the
   strongest curve among now / 10 / 30 / 60 segments ahead (`curve`, `curveAhead`). The game pushes `playerX` by
   `dx * sp * curve * 0.3` per update against a steering authority of `dx = 2 * dt * sp`; at top speed on a curve of
   4 the drift (0.2 per step) beats the steering (0.167), so the car has to be at the inside edge before a hold
   starts and lets itself drift across (a 200 segment hold moves it from +0.8 to about -0.5). Grid points that are
   not reachable within 12 steps at the current lateral rate (steering minus drift, direction dependent) are skipped.
   Each car in `carsAhead` is a hazard at `tau = dz / (closing speed)` steps: our position when we reach it (moving
   toward the candidate at the lateral rate) must be at least `need` from the car's x, `need` = 0.303 for a car that
   could be a semi (speed <= 6000) and 0.270 otherwise (the game's `overlap(..., 0.8)` with sprite widths); a miss
   costs `W_HIT = 10`, a gap inside a 0.12 margin `W_NEAR = 2`, both weighted 1.0 up to 4 steps and linearly to 0 at
   16 steps. A car within 1.5 steps cannot be crossed laterally. Cars are tracked between steps by their rounded speed
   (near unique among 200 cars) to extrapolate lateral drift up to 3 steps, since cars swerve around slower traffic.
2. Steering: simulate the step for `left faster` / `faster` / `right faster` and take the one landing closest to
   `x*` (dead band 0.06, then straight).
3. Speed: accelerate. Coast (`left` / `right`, which steer) when a 12-step simulation at the current speed with
   steering toward the inside target still leaves |x| > 0.95: at 83% of top speed the drift on a curve of 4 equals
   the steering. Braking is never used for the road: a brake step steers nothing and costs 0.2 of lateral drift,
   more than the speed it sheds saves. Off road (|x| > 0.95): steer back and accelerate (steering per unit time is
   proportional to speed, so slowing down does not help; the roadside sprites pin speed to 2400 while overlapped).
   When every accelerating steer collides with a car in the simulated step, take the coast action that avoids it (we
   only collide when faster than the car), and if none does, `slower` (post-collision speed is `car.speed^2 / speed`,
   so arriving slower hurts less).

The simulator follows the game's order (cars move, then the player, collision test against the player's segment from
before its move, sprite widths from `common.js`) and flags a hit over 6 updates while reporting the state after 5,
because one env step of 5 rAF frames is 4 to 6 physics updates (the game's 1/60 accumulator fed by 16/17 ms `Date`
ticks). `playerZ` = 1000 / tan(50 deg) = 839.1 is hard coded.

Soft targets: 0.8 on the chosen action, 0.2 spread over the other accelerating actions that are acceptable (no
predicted collision, not off road, |x| after the step <= 0.92, not more than 0.3 further from `x*` than the chosen
one), zero elsewhere. `slower`, `left`, `right` carry mass only when chosen (the task sheet's 0.8 / 0.2 convention).
If nothing else is acceptable the chosen action gets 1.0.

Deviation from the task sheet: `slower` is not used when off road (see 3; measured: braking off road on a curve took
the car from -1.0 to -1.5 and into the roadside sprites). The coast actions are used for speed management on the
strong curves, where the sheet had no rule.

Hyperparameters (module constants): `X_LIMIT 0.8, GRID_STEP 0.05, K_CURVE 0.2, REACH_STEPS 12, ROAD_STEPS 12,
ROAD_EDGE 0.95, W_CURVE 1.0, W_MOVE 0.3, W_HIT 10, W_NEAR 2, MARGIN 0.12, HORIZON 16, URGENT 4, ALONGSIDE 1.5,
DEADBAND 0.06, P_MAIN 0.8`. Tried and rejected: `ROAD_EDGE 0.85` (1407 vs 1402 steps over the same 8 seeds, more
collisions); a 0.15 floor on far-car urgency (made distant cars veto good targets; caused a 130-step pile-up behind one
slow car); unpruned targets (planned for unreachable positions on curve holds).

## Numbers (local-workstation, 8 pages, this session)

`python -m playjev.teacher_eval racer --pages 8 --episodes 16 --max-steps 2400`: 16/16 laps, score mean 6711.61,
median 6711.65, max 6714.6, episode length mean 1412 steps (best 1380, worst 1524), capped 0.

Probe over seeds 1 to 8 (`runs/probe/racer/probe.py --policy teacher`): 8/8 laps, steps mean 1402 (min 1380, max
1428), mean speed 11500 of 12000, off-road 0.27% of steps, 1.25 collisions per lap (speed drop > 1300 in one step),
brake steps 2 per lap, coast steps 0 to 21 per lap. Lap time 114.0 to 119.0 s of game time.

Random policy (`python -m playjev.play racer --policy random --pages 8 --episodes 16 --max-steps 2400`): 16
episodes, score mean 387.8 segments (5.8% of a lap), median 381.2, max 474.8, every episode capped at 2400 steps,
never laps; the car sits near zero speed because `slower` brakes five times harder than `faster` accelerates. The
teacher covers 17.3x the distance in 0.59x the steps (29x the distance per step).

Human reference: the game has no published scores; a careful human lap on this build is about 120 to 130 s (the
development `drive` policy in `check_det.py` laps in 124 to 130 s). The teacher's 114 to 119 s is within 2 to 6% of
the physical minimum of 112 s plus acceleration.

Cost: 0.08 to 0.15 ms per `act()` (pure Python, 6 simulated actions x 6 updates, a 33 point target grid, a 60
update road check). The env runs at 360 to 390 env-steps/s on 8 pages with the teacher in the loop, the same as the
random policy.

## Smoke shard

`python -m playjev.collect racer --steps 2000 --shard smoke --epsilon 0.1`: 2000 records in 11 s, no errors, no
giveups. Labels: argmax `faster` 59.5%, `right faster` 24.3%, `left faster` 16.1%, `right` 0.15%, `slower` 0.05%;
one-hot labels (no acceptable alternative, the fallback) 1.2%; no tie splits; no NaN, every row sums to 1; mean
max probability 0.80; total mass on `slower` 0.0005. Three frames checked against their labels (`frames/0000960` to
`0000962`): straight road with traffic ahead left of centre gets `right faster` 0.8 / `faster` 0.2; slightly right of
centre with a car alongside on the far left gets `left faster` 0.8 / `faster` 0.2; centred with a pink car close
ahead at x = +0.6 gets `faster` 0.8 with 0.1 on each steer, passing it on the left. All three read correctly.

Collection epsilon: `pj.json` `collect.epsilon = 0.1`. Reason: pure teacher play keeps the car near the inside target
at top speed and almost never off road or slow, so the student would not see recovery states; one random action in
ten moves x by up to 0.17 or drops speed by 1000 and costs about 50 steps per lap, enough to visit off-centre,
slow and occasionally off-road states while the teacher label stays the recovery action. Higher rates start to
wreck laps (a random `slower` every 60 steps already costs 4% of the lap).

## Known failure modes

- A slow car in the outside lane during the hold or leave of a strong curve while we have already drifted to the
  outside: at top speed the lateral rate toward the inside is zero, the edge blocks the other side, so the car is hit
  (12000 -> about 1000, then 45 steps to recover). This is most of the remaining 1.25 collisions per lap. Coasting
  earlier on the holds would give room but costs about as many steps as it saves on average.
- Cars swerving late (their dodge rate is `1/i * dspeed / maxSpeed` per update, up to 0.2 per step) into our lane
  within the last 1 to 2 steps; the one-step tracker sees the motion only after one step.
- Cars whose sprite is unknown: speed <= 6000 is treated as a possible semi (need 0.303), so gaps of 0.27 to 0.30
  next to an ordinary slow car are avoided when they need not be.
- The 60 segment `curveAhead` horizon is 12 steps at top speed; the S-curve sequence (holds of 50 segments) is
  handled, but the target flips as soon as a stronger curve appears at 60 segments, which occasionally moves the car
  to the outside of the current easy curve (harmless: easy curves are holdable at any speed).
- No `giveup()`: the game has no death and every state is recoverable.

## info() requests

None needed. Everything used is present: `playerX, speed, maxSpeed, curve, curveAhead, position, carsAhead (dz, x,
speed), steps`. Two things would make the model exact rather than conservative, if the hook owner wants them: the
car sprite (or its width) in `carsAhead`, and the number of physics updates the last step actually ran (4, 5 or 6).
