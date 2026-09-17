"""Racer teacher (jakesgordon/javascript-racer v4, games/racer/pj_hook.js).

Algorithm, rule based with a one-step physics model of the game's update():
  1. Pick a lateral target x* on a grid over [-0.8, 0.8] by minimising a cost: hug the inside of the coming curve
     (x_curve = 0.2 * curve; the game pushes playerX outward by dx*sp*curve*0.3 per update, which at top speed on a
     curve of 4 exceeds the steering authority, so the car has to be on the inside before a hold starts), stay clear
     of every car we will reach within the horizon (a car is a hazard where our lateral position at the moment we
     reach it, moving toward x* at the speed- and curve-dependent lateral rate, is within its collision width; a car
     already alongside cannot be crossed), prefer small moves.
  2. Steer left / straight / right toward x*: simulate the step for each and take the one that lands closest, with a
     dead band so we drive straight when already there.
  3. Accelerate by default. Coast (left / right, which still steer) when a 12-step simulation at the current speed,
     steering toward the inside target, still leaves the road (strong curve holds: at 83% of top speed the drift on a
     curve of 4 equals the steering; a brake step steers nothing and costs more drift than it saves). Off road:
     steer back and keep accelerating (steering per unit time is proportional to speed, so slowing down does not
     help). If every steering option collides with a car this step, coast or brake to get
     below the car's speed (no collision when slower than the car), and if nothing avoids it, brake anyway
     (post-collision speed is car.speed^2 / speed, so arriving slower hurts less).
Soft targets: 0.8 on the chosen action, 0.2 spread over the other accelerating actions that are still acceptable
(no predicted collision, not off road, not clearly steering the wrong way), zero elsewhere; slower/left/right only
get mass when chosen. Cars are tracked between steps by their (rounded, near unique) speed to estimate lateral
velocity, since cars swerve around slower traffic.

Model notes: one env step is 5 rAF frames but 4 to 6 physics updates (the game's accumulator with 16/17 ms Date
ticks), so the simulator reports the state after 5 updates and flags a hit if any of 6 updates collides. The game
tests collisions against the player's segment from before the position update and the cars' positions after
theirs; the simulator does the same.
"""
from __future__ import annotations
import math
from . import Teacher

MAX_SPEED = 12000.0
DT = 1.0 / 60.0
UPDATES = 5                      # nominal physics updates per env step (pj.json step_frames)
UPDATES_HIT = 6                  # updates checked for collisions (a step can contain six)
CENTRIFUGAL = 0.3
ACCEL, BRAKE, DECEL = MAX_SPEED / 5, -MAX_SPEED, -MAX_SPEED / 5
OFFROAD_DECEL, OFFROAD_LIMIT = -MAX_SPEED / 2, MAX_SPEED / 4
SEG = 200.0
PLAYER_Z = 1000.0 / math.tan(math.radians(50))   # cameraHeight / tan(fov/2) = 839.1
PLAYER_W = 0.3                                    # 80 px * SPRITES.SCALE
NEED_SEMI = (PLAYER_W + 0.4575) * 0.4             # 0.303: collision half-gap against a semi (speed <= maxSpeed/2)
NEED_CAR = (PLAYER_W + 0.375) * 0.4               # 0.270: truck (widest of the fast ones)

# hyperparameters
X_LIMIT = 0.8            # target grid stays inside +-0.8 (the road is +-1, the car centre must not cross it)
GRID_STEP = 0.05
K_CURVE = 0.2            # inside-hugging target x_curve = K_CURVE * effective curve (curve 4 -> 0.8)
CURVE_LOOKAHEAD = (1.0, 1.0, 1.0, 1.0)   # weights for curve now, 10, 30, 60 segments ahead (strongest wins)
REACH_STEPS = 12         # a target must be reachable within this many steps at the current lateral rate
ROAD_STEPS = 12          # speed rule: stay on the road for this many steps with full counter-steer
ROAD_EDGE = 0.95         # |x| the speed rule keeps us inside of
W_CURVE, W_MOVE = 1.0, 0.3
W_HIT, W_NEAR, MARGIN = 10.0, 2.0, 0.12
HORIZON = 16.0           # steps; cars we reach later than this do not enter the target cost
URGENT = 4.0             # steps; full weight up to here, then linear to zero at HORIZON
ALONGSIDE = 1.5          # steps; a car this close cannot be crossed laterally
DEADBAND = 0.06          # drive straight when the straight outcome is this close to x*
P_MAIN = 0.8

ACTION_CTRL = {"left": (-1, 1), "right": (1, 1), "faster": (0, 2), "slower": (0, 0),
               "left faster": (-1, 2), "right faster": (1, 2)}   # (steer, throttle: 2 accel, 1 coast, 0 brake)
ACCELERATING = ("faster", "left faster", "right faster")


class RacerTeacher(Teacher):
    def reset(self):
        self.track = {}      # car speed (int) -> (x, step) from the previous step, for lateral velocity
        self.last = {}       # debug: x*, hazards, outcomes of the last act()

    # ------------------------------------------------------------------ physics model
    @staticmethod
    def _curve_at(knots, s):
        """Curve s segments ahead, piecewise linear through the info() samples at 0, 10, 30, 60."""
        if s <= 0: return knots[0][1]
        for (s0, c0), (s1, c1) in zip(knots, knots[1:]):
            if s <= s1: return c0 + (c1 - c0) * (s - s0) / (s1 - s0)
        return knots[-1][1]

    def _sim(self, x, v, z0, knots, cars, steer, throttle):
        """Env step of update(). cars: [z, x, speed, need, vx_per_update]. Returns (x, v, hit, off_updates) with
        x, v, off after UPDATES updates and hit over UPDATES_HIT updates."""
        hit = False; off = 0; z = z0; out = None
        acc = ACCEL if throttle == 2 else DECEL if throttle == 1 else BRAKE
        for f in range(1, UPDATES_HIT + 1):
            seg = math.floor(z / SEG)                     # player segment before this update's move
            sp = v / MAX_SPEED; dx = DT * 2 * sp
            z += DT * v
            x += steer * dx - dx * sp * self._curve_at(knots, (z - z0) / SEG) * CENTRIFUGAL
            v += acc * DT
            if x < -1 or x > 1:
                off += 1
                if v > OFFROAD_LIMIT: v += OFFROAD_DECEL * DT
            for c in cars:
                cz = c[0] + c[2] * DT * f
                if v > c[2] and math.floor(cz / SEG) == seg and abs(x - (c[1] + c[4] * f)) < c[3]:
                    hit = True; v = c[2] * c[2] / v; z = cz
                    break
            x = max(-3.0, min(3.0, x)); v = max(0.0, min(MAX_SPEED, v))
            if f == UPDATES: out = (x, v, off)
            if hit: break
        if out is None: out = (x, v, off)
        return out[0], out[1], hit, out[2]

    # ------------------------------------------------------------------ target selection
    def _target(self, x, sp, c_eff, c_rate, hazards):
        """hazards: [(tau_steps, car_x_at_tau, need)]. c_eff sets the inside-hugging target, c_rate (the curve we are
        on now) the lateral rates. Returns the reachable grid point with the smallest cost."""
        x_curve = max(-X_LIMIT, min(X_LIMIT, K_CURVE * c_eff))
        dx_step = UPDATES * DT * 2 * sp
        drift = dx_step * sp * c_rate * CENTRIFUGAL         # per step, positive curve pushes toward -x
        rate_pos = max(0.005, dx_step - drift); rate_neg = max(0.005, dx_step + drift)
        best, best_cost = max(-X_LIMIT, min(X_LIMIT, x)), float("inf")
        n = int(round(2 * X_LIMIT / GRID_STEP)) + 1
        for i in range(n):
            xt = -X_LIMIT + i * GRID_STEP
            if (xt > x and xt - x > REACH_STEPS * rate_pos + GRID_STEP / 2) or (xt < x and x - xt > REACH_STEPS * rate_neg + GRID_STEP / 2):
                continue                                   # not reachable in time: do not plan for a fantasy
            cost = W_CURVE * (xt - x_curve) ** 2 + W_MOVE * abs(xt - x)
            for tau, cx, need in hazards:
                urg = 1.0 if tau <= URGENT else (HORIZON - tau) / (HORIZON - URGENT)
                if tau <= ALONGSIDE and min(x, xt) - need < cx < max(x, xt) + need and abs(xt - x) > 1e-9 \
                        and (cx - x) * (xt - x) > 0:
                    cost += W_HIT * urg * 2          # would have to cross a car that is alongside
                    continue
                lim = rate_pos * tau if xt > x else rate_neg * tau
                reach = x + max(-lim, min(lim, xt - x))       # where we are (moving toward xt) when we reach the car
                gap = abs(reach - cx)
                if gap < need: cost += W_HIT * urg * (1 + (need - gap) / need)
                elif gap < need + MARGIN: cost += W_NEAR * urg * (need + MARGIN - gap) / MARGIN
            if cost < best_cost: best, best_cost = xt, cost
        return best

    def _road_throttle(self, x, v, knots):
        """2 (accelerate) if steering toward the inside target at this speed keeps |x| <= ROAD_EDGE for ROAD_STEPS
        steps, else 1 (coast while steering). Braking is never used for this: a brake step has no steering and costs
        more lateral drift than the speed it sheds saves. Already at or past the edge: accelerate and steer back."""
        if abs(x) > ROAD_EDGE: return 2
        xx, vv, z = x, v, 0.0
        for _ in range(ROAD_STEPS * UPDATES):
            sp = vv / MAX_SPEED; dx = DT * 2 * sp
            z += DT * vv
            c = self._curve_at(knots, z / SEG)
            tgt = max(-X_LIMIT, min(X_LIMIT, K_CURVE * c))          # steer toward the inside target, not past it
            steer = 1 if xx < tgt - 0.02 else -1 if xx > tgt + 0.02 else 0
            xx += steer * dx - dx * sp * c * CENTRIFUGAL
            vv = max(0.0, min(MAX_SPEED, vv + ACCEL * DT))
            if abs(xx) > ROAD_EDGE: return 1
        return 2

    # ------------------------------------------------------------------ policy
    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; n = len(self.actions)
        x = float(info["playerX"]); v = float(info["speed"]); sp = v / MAX_SPEED
        step = int(info.get("steps", obs.get("steps", 0)))
        c0 = float(info["curve"]); ca = info.get("curveAhead") or [c0, c0, c0]
        knots = [(0, c0), (10, float(ca[0])), (30, float(ca[1])), (60, float(ca[2]))]
        z0 = float(info["position"]) + PLAYER_Z
        # effective curve for inside hugging: the strongest (weighted) of now / 10 / 30 / 60 segments ahead
        c_eff = max((w * c for w, (_, c) in zip(CURVE_LOOKAHEAD, knots)), key=abs)

        # cars: hazards for the target cost and bodies for the one-step simulation
        cars, hazards, track = [], [], {}
        for c in info.get("carsAhead") or []:
            dz, cx, cv = float(c["dz"]), float(c["x"]), float(c["speed"])
            key = int(cv); vx = 0.0
            prev = self.track.get(key)
            if prev is not None and prev[1] == step - 1:
                vx = max(-0.5, min(0.5, cx - prev[0]))
            track[key] = (cx, step)
            need = NEED_SEMI if cv <= MAX_SPEED / 2 else NEED_CAR
            cars.append([z0 + dz, cx, cv, need, vx / UPDATES])
            closing = v - cv
            if closing <= 0:
                if sp >= 0.999: continue           # cannot catch it
                closing = max(closing + ACCEL * 1.0, 200.0)   # we are accelerating; assume we close within about a second
            tau = max(0.0, dz) / (closing * DT * UPDATES)
            if tau < HORIZON:
                hazards.append((tau, cx + vx * min(tau, 3.0), need))
        self.track = track

        c_rate = 0.5 * (c0 + float(ca[0]))
        x_star = self._target(x, sp, c_eff, c_rate, hazards)

        # steering: which of left / straight / right lands closest to x* (accelerating)
        outcomes = {}
        for name in self.actions:
            nm = name["name"]; st, th = ACTION_CTRL[nm]
            outcomes[nm] = self._sim(x, v, z0, knots, cars, st, th)
        best_name = "faster"
        if abs(outcomes["faster"][0] - x_star) > DEADBAND:
            best_name = min(ACCELERATING, key=lambda a: abs(outcomes[a][0] - x_star))
        chosen = best_name

        # speed rule on curves: coast (steering) or brake when full counter-steer at this speed still leaves the road
        throttle = self._road_throttle(x, v, knots) if sp > 0.5 else 2
        if throttle < 2:
            st = ACTION_CTRL[best_name][0]
            if st == 0: st = 1 if c0 > 0 else -1 if c0 < 0 else (1 if x < 0 else -1)   # steer toward the inside
            chosen = "left" if st < 0 else "right"

        # collision handling: prefer an accelerating steer that does not hit; else coast/brake below the car's speed
        if outcomes[chosen][2]:
            clean = [a for a in (ACCELERATING if throttle == 2 else ("left", "right")) if not outcomes[a][2]]
            if clean:
                chosen = min(clean, key=lambda a: abs(outcomes[a][0] - x_star))
            else:
                st = ACTION_CTRL[best_name][0]
                coast = "left" if st < 0 else "right" if st > 0 else None
                if coast is not None and not outcomes[coast][2]:
                    chosen = coast
                else:
                    chosen = "slower"   # avoids the hit if we drop below the car's speed, otherwise softens it

        probs = [0.0] * n
        probs[self.idx[chosen]] = P_MAIN
        x1c = outcomes[chosen][0]
        ok = [a for a in ACCELERATING if a != chosen and not outcomes[a][2] and outcomes[a][3] == 0
              and abs(outcomes[a][0]) <= 0.92 and abs(outcomes[a][0] - x_star) <= abs(x1c - x_star) + 0.3]
        if ok:
            for a in ok: probs[self.idx[a]] = (1.0 - P_MAIN) / len(ok)
        else:
            probs[self.idx[chosen]] = 1.0
        self.last = {"x_star": x_star, "c_eff": c_eff, "hazards": hazards, "outcomes": outcomes, "chosen": chosen, "throttle": throttle}
        s = sum(probs)
        return [p / s for p in probs]
