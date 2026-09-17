"""Space Invaders teacher (games/invaders, StrykerKKD/SpaceInvaders).

The ship fires by itself every 300 ms, so steering is the whole decision. Each step the teacher runs a
small dynamic program over the next H steps: the state is (step k, ship x on a 16.67 px grid, one grid
unit = one step of travel at 200 px/s), the transitions are left / noop / right. Cost, lexicographic:
  1. enemy-bullet hits, checked frame by frame (5 frames per step) against every enemy shot in flight,
     using the [x, y, vx, vy] from info() and the sprite sizes (ship 28x21 at y=500, shot 9x9);
  2. minus an aiming bonus for being under the target column with the right lead: the shot fired at
     time t hits the lowest alien of column c if the ship is at 48*c + blockX(t + flight_time), where
     blockX is the alien block's triangle wave (100..200 at 50 px/s) and flight_time = (440 - 50 row)/500 s.
     Steps in which the auto-fire timer goes off (phase read from my newest bullet in flight) weigh 1.0,
     the others 0.3, so dodging is planned into the gaps between shots.
The target column is the one whose lead-corrected x is nearest to the ship (40 px hysteresis on the
previous target), skipping columns whose remaining aliens are already claimed by my bullets in flight.
Soft targets: 0.8 on the best first move(s), 0.2 spread over first moves with the same hit count and an
aiming cost within one shot's worth, 0 on moves that take a hit the best move avoids (the task sheet asks
for 0.8 instead of Snake's 0.9 because two moves are often almost equally good here).

info() fields used (games/invaders/pj_hook.js): state, lives, health, playerX, alienBlockX, aliens
([x, y] centres), enemyShots ([x, y, vx, vy]; y is the bullet's bottom edge, anchor (0.5, 1)), myShots
([x, y]). obs["t"] is the virtual clock in ms.
"""
from __future__ import annotations
import math
from collections import Counter
from . import Teacher

FPS = 60.0
FRAMES_PER_STEP = 5
DT = FRAMES_PER_STEP / FPS               # 83.3 ms per step
SHIP_V = 200.0                           # px/s
U = SHIP_V * DT                          # 16.67 px, one step of travel
SHIP_W, SHIP_H, SHIP_Y = 28.0, 21.0, 500.0
X_MIN, X_MAX = SHIP_W / 2, 800.0 - SHIP_W / 2
EB_W, EB_H = 9.0, 9.0                    # enemy bullet sprite, anchor (0.5, 1)
PB_W, PB_H = 6.0, 36.0                   # my bullet sprite, anchor (0.5, 1), fired from (ship.x, 508)
PB_V = 500.0
ALIEN_W, COL_DX, ROW_DY, BLOCK_Y = 32.0, 48.0, 50.0, 50.0
BLOCK_LO, BLOCK_HI, BLOCK_V = 100.0, 200.0, 50.0
FIRE_PERIOD = 0.3
SHOT_Y0 = SHIP_Y + 8                     # my bullet's y (bottom edge) at the moment it is fired

HORIZON = 28                             # steps (2.3 s): every shot that can reach the ship in that time is planned around
HIT_MARGIN = 4.0                         # px added to the ship/shot half-widths: 1 frame of key lag plus rounding in info()
HIT_HW = (SHIP_W + EB_W) / 2 + HIT_MARGIN
AIM_TOL = 8.0                            # px: within this of the lead-corrected column centre the shot cannot miss
AIM_RAMP = 60.0                          # px over which the bonus falls from 1 to 0 outside the tolerance
AIM_W_SHOT, AIM_W_IDLE = 1.0, 0.3        # step weight when the auto-fire timer fires in that step / does not
HYSTERESIS = 40.0                        # px in favour of keeping the previous target column
BIG = 1e4                                # one hit outweighs any aiming bonus over the horizon (lexicographic)
ACCEPT_SLACK = 1.0                       # aiming cost within this of the best, same hits: still an acceptable move
MOVE_EPS = 1e-3                          # tiny preference for standing still when moves tie


def block_x(x0: float, d: int, t: float) -> float:
    """Alien block x after t seconds, starting from x0 moving in direction d (triangle wave 100..200)."""
    p = (x0 - BLOCK_LO) + d * BLOCK_V * t
    span = BLOCK_HI - BLOCK_LO
    p %= 2 * span
    return BLOCK_LO + (p if p <= span else 2 * span - p)


class InvadersTeacher(Teacher):
    def reset(self):
        self.prev_block = None
        self.block_dir = 1
        self.fire_ref = None          # virtual time (s) of one of my shots, phase reference for the 300 ms timer
        self.target_col = None

    # ---- helpers -------------------------------------------------------------------------------------
    def _update_block_dir(self, bx: float):
        if self.prev_block is None:
            self.block_dir = 1 if bx < (BLOCK_LO + BLOCK_HI) / 2 else -1
        elif bx > self.prev_block:
            self.block_dir = 1
        elif bx < self.prev_block:
            self.block_dir = -1
        elif bx >= BLOCK_HI - 1:
            self.block_dir = -1
        elif bx <= BLOCK_LO + 1:
            self.block_dir = 1
        self.prev_block = bx

    def _next_fire(self, now: float, my_shots) -> float | None:
        """Seconds until the auto-fire timer next goes off, or None when the phase is unknown."""
        if my_shots:
            y = max(s[1] for s in my_shots)                 # the newest of my bullets is the lowest one
            self.fire_ref = now - (SHOT_Y0 - y) / PB_V
        if self.fire_ref is None:
            return None
        since = (now - self.fire_ref) % FIRE_PERIOD
        return FIRE_PERIOD - since

    def _pick_target(self, aliens, bx: float, sx: float, my_shots):
        """Column to shoot at and the flight time of a shot to its lowest alien. Aliens already claimed by
        my bullets in flight are discounted, so the ship leaves a column as soon as its fate is sealed."""
        cols: dict[int, list[int]] = {}
        for x, y in aliens:
            c = int(round((x - bx) / COL_DX)); r = int(round((y - BLOCK_Y) / ROW_DY))
            cols.setdefault(c, []).append(r)
        claimed: set[tuple[int, int]] = set()
        for x, y in my_shots:
            top = y - PB_H
            for r in range(3, -1, -1):
                ay = BLOCK_Y + r * ROW_DY
                if y < ay - ALIEN_W / 2:                    # bullet's bottom already above this row: passed it
                    continue
                tau = max(0.0, (top - (ay + ALIEN_W / 2)) / PB_V)               # top edge reaches the alien's bottom
                tau_hi = max(0.0, (y - (ay - ALIEN_W / 2)) / PB_V)              # bottom edge leaves the alien's top
                ax_lo = block_x(bx, self.block_dir, tau); ax_hi = block_x(bx, self.block_dir, tau_hi)
                hit = None
                for c, rows in cols.items():
                    if r in rows and (c, r) not in claimed:
                        if abs(x - (ax_lo + c * COL_DX)) < (ALIEN_W + PB_W) / 2 or abs(x - (ax_hi + c * COL_DX)) < (ALIEN_W + PB_W) / 2:
                            hit = c; break
                if hit is not None:
                    claimed.add((hit, r)); break
        best = None
        for c, rows in cols.items():
            live = [r for r in rows if (c, r) not in claimed]
            if not live:
                continue
            r = max(live)                                    # lowest alien is hit first
            tau = (SHOT_Y0 - PB_H / 2 - (BLOCK_Y + r * ROW_DY)) / PB_V
            xs = c * COL_DX + block_x(bx, self.block_dir, tau)
            cost = abs(xs - sx) - (HYSTERESIS if c == self.target_col else 0.0)
            if best is None or cost < best[0]:
                best = (cost, c, tau)
        if best is None:                                     # everything alive is already claimed: keep shooting the nearest
            for c, rows in cols.items():
                r = max(rows)
                tau = (SHOT_Y0 - PB_H / 2 - (BLOCK_Y + r * ROW_DY)) / PB_V
                xs = c * COL_DX + block_x(bx, self.block_dir, tau)
                cost = abs(xs - sx)
                if best is None or cost < best[0]:
                    best = (cost, c, tau)
        self.target_col = best[1]
        return best[1], best[2]

    def _hits(self, shots, sx: float, H: int) -> Counter:
        """Counter of (k, j, m) -> number of enemy shots that hit the ship during step k (1-based) when it
        starts the step at grid position j and moves m in {-1, 0, 1}. Frame-level check."""
        hits: Counter = Counter()
        y_top, y_bot = SHIP_Y - SHIP_H / 2, SHIP_Y + SHIP_H / 2
        nF = H * FRAMES_PER_STEP
        for x, y, vx, vy in shots:
            if vy <= 0:
                continue
            f_in = math.ceil((y_top - y) / vy * FPS)          # first frame the shot's bottom is below the ship's top
            f_out = math.floor((y_bot + EB_H - y) / vy * FPS)  # last frame the shot's top is above the ship's bottom
            f_in = max(1, f_in); f_out = min(nF, f_out)
            if f_in > f_out:
                continue
            seen = set()
            for F in range(f_in, f_out + 1):
                k = (F - 1) // FRAMES_PER_STEP + 1; f = F - (k - 1) * FRAMES_PER_STEP
                bxF = x + vx * F / FPS
                lo = (bxF - HIT_HW - sx) / U; hi = (bxF + HIT_HW - sx) / U
                for m in (-1, 0, 1):
                    off = m * f / FRAMES_PER_STEP
                    j0 = math.floor(lo - off) + 1; j1 = math.ceil(hi - off) - 1
                    for j in range(max(j0, -(k - 1)), min(j1, k - 1) + 1):
                        key = (k, j, m)
                        if key not in seen:
                            seen.add(key); hits[key] += 1
        return hits

    # ---- policy ----------------------------------------------------------------------------------------
    def act(self, obs: dict) -> list[float]:
        info = obs.get("info") or {}; n = len(self.actions)
        aliens = info.get("aliens") or []
        if info.get("state") != "Play" or not aliens or "playerX" not in info:
            return [1.0 / n] * n
        sx = float(info["playerX"]); bx = float(info.get("alienBlockX", BLOCK_LO))
        now = float(obs.get("t", 0)) / 1000.0
        my_shots = info.get("myShots") or []; shots = info.get("enemyShots") or []
        self._update_block_dir(bx)
        t_fire = self._next_fire(now, my_shots)
        col, tau = self._pick_target(aliens, bx, sx, my_shots)
        H = HORIZON
        hits = self._hits(shots, sx, H)

        # per-step aiming bonus for each reachable grid position
        w = [0.0] * (H + 1)
        for k in range(1, H + 1):
            if t_fire is None:
                w[k] = 0.6
            else:                                              # does a shot go off in (a, b]? shots at t_fire + n * 0.3, n >= 0
                a, b = (k - 1) * DT, k * DT
                n_min = max(0, math.floor((a - t_fire) / FIRE_PERIOD) + 1)
                w[k] = AIM_W_SHOT if t_fire + n_min * FIRE_PERIOD <= b + 1e-9 else AIM_W_IDLE
        xstar = [col * COL_DX + block_x(bx, self.block_dir, k * DT + tau) for k in range(H + 1)]

        def bonus(k, j):
            d = abs(sx + j * U - xstar[k])
            core = 1.0 if d <= AIM_TOL else max(0.0, 1.0 - (d - AIM_TOL) / AIM_RAMP)
            return w[k] * core - 0.001 * d

        # backward DP over the reachable triangle; V[k][j] = cost-to-go at (k, j), j in [-k, k] within the walls
        def in_bounds(j):
            return X_MIN - 1e-6 <= sx + j * U <= X_MAX + 1e-6

        V = {j: -bonus(H, j) for j in range(-H, H + 1) if in_bounds(j)}
        for k in range(H - 1, 0, -1):
            Vk = {}
            for j in range(-k, k + 1):
                if not in_bounds(j):
                    continue
                best = None
                for m in (-1, 0, 1):
                    nxt = V.get(j + m)
                    if nxt is None:
                        continue
                    v = nxt + BIG * hits.get((k + 1, j, m), 0) + (MOVE_EPS if m else 0.0)
                    if best is None or v < best:
                        best = v
                Vk[j] = best - bonus(k, j)
            V = Vk
        vals = {}
        for m in (-1, 0, 1):
            nxt = V.get(m)
            if nxt is None:
                continue
            vals[m] = nxt + BIG * hits.get((1, 0, m), 0) + (MOVE_EPS if m else 0.0)
        if not vals:
            return [1.0 / n] * n
        best = min(vals.values()); best_hits = math.floor((best + BIG / 2) / BIG)
        winners = [m for m, v in vals.items() if v - best < 1e-9]
        others = [m for m, v in vals.items() if m not in winners and math.floor((v + BIG / 2) / BIG) == best_hits and v - best < ACCEPT_SLACK]
        name = {-1: "left", 0: "noop", 1: "right"}
        probs = [0.0] * n
        share = 0.8 if others else 1.0
        for m in winners:
            probs[self.idx[name[m]]] += share / len(winners)
        for m in others:
            probs[self.idx[name[m]]] += 0.2 / len(others)
        s = sum(probs)
        return [p / s for p in probs]
