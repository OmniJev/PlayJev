"""Breakout teacher: simulate the ball's exact path (walls, bricks, the game's own intercept rules) to the
paddle plane, pick the paddle position among those reachable in time whose bounce sends the ball into bricks
fastest and comes back catchable, and move toward it. The ball's flight never depends on the paddle until it
returns, so the landing point is known the moment the ball leaves the paddle.

info() from games/breakout/pj_hook.js: court {left, top, right, bottom, chunk}, ball {x, y, dx, dy, speed,
moving}, paddle {x, y, w}, bricks (25 rows of 30 cells joined by '|', '.' empty, a run of one letter is one
brick), bricks_left, lives, level. One step holds a key for 5 frames (83 ms): the paddle moves 23 px, the ball
17 to 26 px.

Soft targets: 0.9 on the chosen action, 0.1 on stay when moving, 0.05 on each move when staying; zero on a move
after which the ball can no longer be caught in time (a lost life). Ties split evenly."""
from __future__ import annotations
import math
from . import Teacher

FPS = 60
STEP_FRAMES = 5
STEP_T = STEP_FRAMES / FPS              # game seconds per env step
BALL_R_CHUNKS = 0.3                     # Breakout.Defaults.ball.radius
PADDLE_SPEED_CHUNKS = 20                # Breakout.Defaults.paddle.speed (chunks/s)
BALL_SPEED_CHUNKS = 15                  # Breakout.Defaults.ball.speed (chunks/s), reset on each new ball
YCHUNKS = 25
CATCH_MARGIN = 3.0                      # px inside the true catch window, covers info() rounding and step jitter
FRAME_PX = 4.0                          # ball travel per frame: corner grazes within one frame take the game's side-first rule
MAX_SEG_FLIGHT = 80                     # segments simulated for the current flight
MAX_SEG_AIM = 60                        # segments simulated per candidate bounce
LOOKAHEAD = 1                           # follow-up shots evaluated when scoring a bounce (0 = greedy)
P_BEST, P_REST = 0.9, 0.1


class BreakoutTeacher(Teacher):
    def reset(self):
        self._plan_key = None      # (bricks_left, landing x, incoming dy) the cached target belongs to
        self._plan_c = None        # paddle centre we are heading to for that landing
        self._g = None; self._n_bricks = 0

    # ------------------------------------------------------------------ geometry
    def _bricks(self, rows_str: str, left: float, top: float, chunk: float, r: float):
        """Rows of expanded brick rectangles (L, R, T, B, id, points), sorted by row. A run of one letter is one brick."""
        rows = {}
        bid = 0
        for j, line in enumerate(rows_str.split("|")):
            x = 0; n = len(line); lst = []
            while x < n:
                ch = line[x]
                if ch == ".":
                    x += 1; continue
                x1 = x
                while x < n and line[x] == ch: x += 1
                lst.append((left + x1 * chunk - r, left + x * chunk + r, top + j * chunk - r, top + (j + 1) * chunk + r,
                            bid, (YCHUNKS - j) * 5))
                bid += 1
            if lst: rows[j] = lst
        return rows

    def _fly(self, x, y, dx, dy, speed, rows, removed, geo, max_seg):
        """Follow the ball until it reaches the paddle plane moving down. Returns
        (landed, x, dx, dy, t_seconds, hits, points, speed). removed is updated with bricks hit."""
        xmin, xmax, ymin, yplane, maxspeed = geo
        t_total = 0.0; hits = 0; points = 0
        if dy > 0 and y > yplane + 1e-6:
            return (False, x, dx, dy, 0.0, 0, 0, speed)          # already below the paddle line: lost
        for _ in range(max_seg):
            # first wall / plane crossing along the unit ray
            tw = math.inf; wall = None
            if dx < 0:
                t = (xmin - x) / dx
                if t < tw: tw, wall = t, "x"
            elif dx > 0:
                t = (xmax - x) / dx
                if t < tw: tw, wall = t, "x"
            if dy < 0:
                t = (ymin - y) / dy
                if t < tw: tw, wall = t, "y"
            else:
                t = (yplane - y) / dy
                if t < tw: tw, wall = t, "plane"
            if tw < 0: tw = 0.0
            # first brick along the segment, rows scanned in travel order (game rule: side line first, else face line)
            best_t = tw; best = None
            y_end = y + tw * dy
            lo, hi = (y, y_end) if dy > 0 else (y_end, y)
            for j in (sorted(rows) if dy > 0 else sorted(rows, reverse=True)):
                band_t, band_b = rows[j][0][2], rows[j][0][3]
                if band_b < lo or band_t > hi:
                    continue
                if best is not None:
                    # nothing in a farther row can beat a hit that ends before that row's band starts
                    by = y + best_t * dy
                    if (dy > 0 and by <= band_t) or (dy < 0 and by >= band_b):
                        break
                for br in rows[j]:
                    if br[4] in removed: continue
                    L, R, T, B = br[0], br[1], br[2], br[3]
                    ts = tf = None
                    if dx < 0:
                        t = (R - x) / dx
                        if 1e-9 < t < best_t and T <= y + t * dy <= B: ts = t
                    elif dx > 0:
                        t = (L - x) / dx
                        if 1e-9 < t < best_t and T <= y + t * dy <= B: ts = t
                    if dy < 0:
                        t = (B - y) / dy
                        if 1e-9 < t < best_t and L <= x + t * dx <= R: tf = t
                    else:
                        t = (T - y) / dy
                        if 1e-9 < t < best_t and L <= x + t * dx <= R: tf = t
                    if ts is None and tf is None: continue
                    if ts is not None and (tf is None or ts <= tf + FRAME_PX):
                        best_t, best = ts, (br, "x")
                    else:
                        best_t, best = tf, (br, "y")
            if best is not None:
                br, side = best
                x += best_t * dx; y += best_t * dy; t_total += best_t / speed
                removed.add(br[4]); hits += 1; points += br[5]
                speed += 10 * (1 - speed / maxspeed)
                if side == "x": dx = -dx
                else: dy = -dy
                continue
            x += tw * dx; y += tw * dy; t_total += tw / speed
            if wall == "plane":
                return (True, x, dx, dy, t_total, hits, points, speed)
            if wall == "x": dx = -dx
            else: dy = -dy
        return (False, x, dx, dy, t_total, hits, points, speed)

    # ------------------------------------------------------------------ policy
    def _aim(self, x_hit, dy_in, c, speed, rows, removed, depth):
        """Value of catching the ball with the paddle centre at c: bricks per second over this excursion and the best
        follow-up shot (depth 1 look-ahead); negative when the ball would come back where the paddle cannot be in
        time. Returns (value, hits, seconds)."""
        g = self._g
        u = (x_hit - c) / g["half_w"]                  # game: outgoing vx = speed * offset / (w/2), vy flipped
        m = math.hypot(u, dy_in)
        rem = set(removed)
        landed, x2, _, dy2, t2, hits, points, speed2 = self._fly(x_hit, g["yplane"], u / m, -dy_in / m, speed, rows, rem, g["geo"], MAX_SEG_AIM)
        if landed and abs(x2 - c) - g["half_catch"] > g["paddle_speed"] * t2 + 1e-6:
            return -1.0, hits, t2                     # guaranteed miss on the return
        if not landed:                                # still bouncing among bricks after MAX_SEG_AIM segments: excellent
            return hits / max(t2, 0.3) + 1e-4 * points, hits, t2
        if len(rem) >= self._n_bricks:                # clears the level
            return 10.0 + hits / max(t2, 0.3), hits, t2
        if depth > 0:
            # the return arrives at x2 after t2 seconds; the paddle can be at any whole-step position reachable by then
            K = int(t2 // STEP_T); best = None
            for k in range(-K, K + 1):
                c2 = min(g["c_max"], max(g["c_min"], c + k * g["step_px"]))
                if abs(x2 - c2) > g["half_catch"]: continue
                v2, h2, t3 = self._aim(x2, dy2, c2, speed2, rows, rem, depth - 1)
                if best is None or v2 > best[0]: best = (v2, h2, t3)
            if best is None:
                return -0.5, hits, t2                 # no catchable position for the return
            if best[0] < 0:
                return best[0], hits, t2
            return (hits + best[1]) / max(t2 + best[2], 0.3) + 1e-4 * points, hits, t2
        if hits == 0:
            return 0.001 * abs(u), hits, t2           # nothing hit: at least travel sideways to find bricks
        return hits / max(t2, 0.3) + 1e-4 * points, hits, t2

    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; n = len(self.actions)
        L_, R_, S_ = self.idx["left"], self.idx["right"], self.idx["stay"]
        if not info or info.get("lives", 1) <= 0:
            return [1.0 / n] * n
        court, ball, pad = info["court"], info["ball"], info["paddle"]
        chunk = court["chunk"]; r = BALL_R_CHUNKS * chunk
        left, right, top = court["left"], court["right"], court["top"]
        w = pad["w"]; half_w = w / 2
        yplane = pad["y"] - r
        paddle_speed = PADDLE_SPEED_CHUNKS * chunk
        step_px = paddle_speed * STEP_T
        c_min, c_max = left + half_w, right - half_w
        c_now = pad["x"] + half_w
        half_catch = half_w + r - CATCH_MARGIN
        maxspeed = BALL_SPEED_CHUNKS * chunk * 1.5
        geo = (left + r, right - r, top + r, yplane, maxspeed)
        rows = self._bricks(info["bricks"], left, top, chunk, r)
        self._n_bricks = sum(len(v) for v in rows.values())
        self._g = {"half_w": half_w, "yplane": yplane, "half_catch": half_catch, "paddle_speed": paddle_speed,
                   "step_px": step_px, "c_min": c_min, "c_max": c_max, "geo": geo}

        if ball["moving"]:
            bx, by, bdx, bdy, speed = ball["x"], ball["y"], ball["dx"], ball["dy"], ball["speed"]
            m = math.hypot(bdx, bdy) or 1.0; bdx, bdy = bdx / m, bdy / m
        else:  # parked: the hook launches it this step from the paddle centre at (1, -1)
            bx, by, speed = c_now, yplane, BALL_SPEED_CHUNKS * chunk
            bdx, bdy = math.sqrt(0.5), -math.sqrt(0.5)

        removed = set()
        landed, x_land, dx_in, dy_in, t_hit, _, _, speed_at = self._fly(bx, by, bdx, bdy, speed, rows, removed, geo, MAX_SEG_FLIGHT)

        def soft(best_set, ok):
            """0.9 on the best action(s), 0.1 on the acceptable others (stay if moving, both moves if staying)."""
            p = [0.0] * n
            if len(best_set) == n:
                return [1.0 / n] * n
            for a in best_set: p[a] = P_BEST / len(best_set)
            others = [a for a in range(n) if a not in best_set and ok[a]]
            if best_set == {S_}:
                pass
            elif S_ in others:
                others = [S_]
            for a in others: p[a] += P_REST / len(others)
            s = sum(p)
            return [v / s for v in p]

        def toward(target):
            d = target - c_now
            if abs(d) <= step_px / 2: return S_
            return R_ if d > 0 else L_

        if not landed:
            # lost ball (below the paddle line) or trapped for very long: track the ball's x, every action is acceptable
            return soft({toward(bx if by > yplane else x_land)}, [True] * n)

        K = int(t_hit // STEP_T)          # whole steps before the impact step
        deltas = {L_: -step_px, R_: step_px, S_: 0.0}

        if K == 0:
            # impact during this step: the paddle moves in whole frames, each applied before the ball's collision
            # check of that frame, so it covers ceil(t_hit / frame) frames of travel before the ball arrives
            frames = min(STEP_FRAMES, math.ceil(t_hit * FPS - 1e-9))
            frac = paddle_speed * frames / FPS
            scores = {}
            for a, d in deltas.items():
                c_a = min(c_max, max(c_min, c_now + (d and math.copysign(frac, d))))
                margin = half_catch if a == S_ else half_catch - paddle_speed / FPS   # a moving paddle keeps one frame spare
                if abs(x_land - c_a) <= margin:
                    scores[a] = self._aim(x_land, dy_in, c_a, speed_at, rows, removed, LOOKAHEAD)[0]
            if not scores:
                return soft({toward(x_land)}, [True] * n)
            top_v = max(scores.values())
            best = {a for a, v in scores.items() if v >= top_v - 1e-9}
            ok = [a in scores for a in range(n)]
            return soft(best, ok)

        # positions reachable by the impact step, one whole move per step
        cands = sorted({min(c_max, max(c_min, c_now + k * step_px)) for k in range(-K, K + 1)})
        catchable = [c for c in cands if abs(x_land - c) <= half_catch]
        if not catchable:
            return soft({toward(x_land)}, [True] * n)

        # keep the current target while the landing is the same flight (info() rounding moves it by hundredths of a
        # pixel between steps; a brick hit moves it by many pixels and changes bricks_left)
        key = (info.get("bricks_left"), x_land, dy_in)
        c_star = self._plan_c
        same = (self._plan_key is not None and key[0] == self._plan_key[0]
                and abs(key[1] - self._plan_key[1]) < 2.0 and abs(key[2] - self._plan_key[2]) < 0.02)
        if not same or c_star is None or min(abs(c_star - c) for c in catchable) > step_px / 2 + 1e-6:
            best_c, best_v = None, -math.inf
            for c in catchable:
                v = self._aim(x_land, dy_in, c, speed_at, rows, removed, LOOKAHEAD)[0]
                if v > best_v + 1e-9 or (abs(v - best_v) <= 1e-9 and abs(c - c_now) < abs(best_c - c_now)):
                    best_c, best_v = c, v
            c_star = best_c
            self._plan_key, self._plan_c = key, c_star
        c_star = min(catchable, key=lambda c: abs(c - c_star))

        # an action is acceptable if some position catchable in the remaining K-1 steps is still reachable after it
        ok = []
        for a in range(n):
            c_a = min(c_max, max(c_min, c_now + deltas[a]))
            ok.append(any(abs(c_a - c) <= (K - 1) * step_px + 1e-6 for c in catchable))
        best = toward(c_star)
        if not ok[best]:
            ok[best] = True
        return soft({best}, ok)
