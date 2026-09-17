"""Pac-Man teacher (games/pacman, daleharvey/pacman).

Algorithm, in three lines:
1. Ghosts are memoryless random walkers with an exactly known rule (turn at every 4-way, 50/50 at a T, never
   reverse, one block per 5 ticks). Propagate each dangerous ghost's occupancy distribution over (cell, direction)
   for 45 ticks; a cell is unsafe for Pac-Man at time t if a ghost can be at it within 5 ticks of t, or can cross
   it 6 to 7 ticks before or after at right angles to Pac-Man's move (collision radius is one block, and the closest
   approach of two perpendicular movers is k/sqrt2); a ghost moving in Pac-Man's direction one block ahead never
   closes and is exempt.
2. For each pressable direction, BFS from Pac-Man's next grid square over cells that are safe at Pac-Man's own
   arrival time, honouring the step granularity (an in-step reversal carries straight through the cell behind).
   The value is the time to the nearest pellet or power pill, or to an edible ghost when the remaining edible time
   allows (worth 50/100/150/200, so a detour of value/5 ticks is accepted); reversals cost 4 ticks extra. A
   direction only counts if its safe region reaches past the ghost horizon (no walking into a dead-end trap).
3. Soft target: 0.9 on the best direction(s), 0.1 spread over other directions that are pressable and safe, zero
   on walls, stops and moves into a ghost. With no safe route to a pellet, take the deepest safe region; with no
   safe first step, the direction that meets the ghost latest.

info() from games/pacman/pj_hook.js: map rows joined by '|' ('#' wall, '=' ghost house, '.' pellet, 'o' power
pill, ' ' empty), pac {x, y, dir} in block units (tenths of a block resolution), ghosts [{x, y, dir, vulnerable,
eaten}], tick, level, pellets, pills. Row 10 wraps: x = -1 and x = 19 are the same off-screen cell.

Timers are not in info(); they are reconstructed from tick: a pill eaten during a step makes ghosts edible until
(step start tick) + 241, a ghost eaten during a step is harmless (and fast) until (step start tick) + 91; both use
the earliest tick of the step, the conservative side."""
from __future__ import annotations
from collections import deque
from . import Teacher

DIRS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}   # (dx, dy); x = column, y = row
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
PERP = {"up": ("left", "right"), "down": ("left", "right"), "left": ("up", "down"), "right": ("up", "down")}
TUNNEL_Y, RING = 10, 20          # the wrap row; cells 0..19 form a ring, 19 is off-screen
TPB = 5                          # ticks per block, Pac-Man and dangerous ghosts (2 units per tick)
SPEED_PRE = {"vulnerable": 10, "hidden": 3}   # ticks per block while edible (1 unit) / just eaten (4 units)
HORIZON = 45                     # ticks of ghost propagation
W_SAME, W_ADJ = 5, 2.5           # collision windows: same cell (either order), neighbouring cell (ghost first)
W_PERP = 7                       # same cell, ghost and Pac-Man perpendicular: closest approach is k/sqrt2, so k < 1.42 blocks collides
REVERSE_PENALTY = 4              # ticks added to a reversal so the plan does not flip every step
PILL_TICKS, EATEN_TICKS = 241, 91
CHASE_MAX, CHASE_SLACK = 50, 12  # chase an edible ghost at most 10 blocks away, keep 12 ticks before it turns
STEP_TICKS = 3                   # ticks per env step (for the earliest-tick reconstruction)

_MAZES: dict[str, "_Maze"] = {}


def _ring(c, dx, dy):
    x, y = c[0] + dx, c[1] + dy
    if y == TUNNEL_Y and c[1] == TUNNEL_Y:
        x %= RING
    return (x, y)


class _Maze:
    """Static part of the map: floor cells, neighbours, and the ghost transition table."""

    def __init__(self, rows: list[str]):
        floor = {(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch not in "#="}
        floor.add((RING - 1, TUNNEL_Y))
        self.floor = floor
        self.nb = {c: {d: (n if (n := _ring(c, *v)) in floor else None) for d, v in DIRS.items()} for c in floor}
        self.gtrans: dict[tuple, list] = {}
        for c in floor:
            for d in DIRS:
                s = self.nb[c][d]; p1, p2 = PERP[d]; n1, n2 = self.nb[c][p1], self.nb[c][p2]
                if n1 and n2: out = [(0.5, n1, p1), (0.5, n2, p2)]        # 4-way or T facing the wall: always turns
                elif n1 and s: out = [(0.5, n1, p1), (0.5, s, d)]
                elif n2 and s: out = [(0.5, n2, p2), (0.5, s, d)]
                elif n1: out = [(1.0, n1, p1)]
                elif n2: out = [(1.0, n2, p2)]
                elif s: out = [(1.0, s, d)]
                else: out = [(1.0, c, d)]                                  # dead end: not on this map
                self.gtrans[(c, d)] = out


def _maze(mapstr: str) -> _Maze:
    key = mapstr.translate(str.maketrans(".o", "  "))
    m = _MAZES.get(key)
    if m is None:
        m = _MAZES[key] = _Maze(key.split("|"))
    return m


def _locate(x: float, y: float, d: str, tpb: int):
    """(x, y) in blocks -> (cell being approached or occupied, ticks until it is reached, cell being left or None)."""
    px, py = round(x * 10), round(y * 10)
    if py == TUNNEL_Y * 10:
        px %= RING * 10
    if px % 10 == 0 and py % 10 == 0:
        return (px // 10, py // 10), 0, None
    if d in ("left", "right"):
        lo, hi = px // 10, -(-px // 10)
        nxt, prev = (hi, lo) if d == "right" else (lo, hi)
        rem = abs(nxt * 10 - px)
        cell, pc = (nxt % RING if py == TUNNEL_Y * 10 else nxt, py // 10), (prev % RING if py == TUNNEL_Y * 10 else prev, py // 10)
    else:
        lo, hi = py // 10, -(-py // 10)
        nxt, prev = (hi, lo) if d == "down" else (lo, hi)
        rem = abs(nxt * 10 - py)
        cell, pc = (px // 10, nxt), (px // 10, prev)
    return cell, -(-rem * tpb // 10), pc     # ceil(rem / units-per-tick)


class PacmanTeacher(Teacher):
    def reset(self):
        self.last_mode = None
        self.level = None; self.prev_pills = None; self.prev_pellets = None
        self.pill_expiry = None; self.eaten_expiry = [None] * 4; self.prev_eaten = [False] * 4; self.n_eaten = 0

    # ---- timers reconstructed from tick -------------------------------------------------------------------
    def _track(self, info):
        now = info["tick"]; ghosts = info["ghosts"]
        if self.level != info["level"] or (self.prev_pellets is not None and info["pellets"] > self.prev_pellets):
            self.level = info["level"]; self.pill_expiry = None; self.eaten_expiry = [None] * len(ghosts)
            self.prev_eaten = [False] * len(ghosts); self.n_eaten = 0; self.prev_pills = None
        if self.prev_pills is not None and info["pills"] < self.prev_pills:
            self.pill_expiry = now - (STEP_TICKS - 1) + PILL_TICKS; self.n_eaten = 0
        for i, g in enumerate(ghosts):
            if g["eaten"] and not self.prev_eaten[i]:
                self.eaten_expiry[i] = now - (STEP_TICKS - 1) + EATEN_TICKS; self.n_eaten += 1
            elif not g["eaten"]:
                self.eaten_expiry[i] = None
            self.prev_eaten[i] = g["eaten"]
        self.prev_pills = info["pills"]; self.prev_pellets = info["pellets"]

    # ---- ghost occupancy over time --------------------------------------------------------------------------
    def _ghost_risk(self, info, mz: _Maze):
        """same[cell] = [(t, p, ghost dir)], adj[cell] = [(t, p, ghost cell, ghost dir)] for dangerous occupancy; edible = {cell: (D_rel, value)}."""
        now = info["tick"]; same: dict = {}; adj: dict = {}; edible: dict = {}
        for i, g in enumerate(info["ghosts"]):
            if g["vulnerable"]:
                status = "vulnerable"; D = self.pill_expiry if self.pill_expiry is not None else now
                if g["eaten"] and self.eaten_expiry[i] is not None: D = max(D, self.eaten_expiry[i])
            elif g["eaten"]:
                status = "hidden"; D = self.eaten_expiry[i] if self.eaten_expiry[i] is not None else now
            else:
                status = "dangerous"; D = None
            d = g["dir"] if g["dir"] in DIRS else "left"
            tpb = TPB if status == "dangerous" else SPEED_PRE[status]
            cell, t0, prev = _locate(g["x"], g["y"], d, tpb)
            if cell not in mz.floor:
                continue
            drel = -1e9 if D is None else D - now
            if status == "vulnerable":
                edible[cell] = (drel, 50 * (self.n_eaten + 1))
            if drel > HORIZON:
                continue
            layers = [(t0, {(cell, d): 1.0})]
            if prev is not None and prev in mz.floor:
                layers.insert(0, (t0 - tpb, {(prev, d): 1.0}))
            t, cur = layers[-1]
            while t < HORIZON:
                if t >= drel:
                    s = TPB
                elif t + tpb <= drel:
                    s = tpb
                else:                       # the speed changes inside this block
                    s = (drel - t) + TPB * (1 - (drel - t) / tpb)
                nxt: dict = {}
                for (c, cd), p in cur.items():
                    for q, c2, d2 in mz.gtrans[(c, cd)]:
                        nxt[(c2, d2)] = nxt.get((c2, d2), 0.0) + p * q
                t += s; cur = nxt; layers.append((t, cur))
            for t, dist in layers:
                if t < drel - W_SAME:
                    continue
                for (c, cd), p in dist.items():
                    same.setdefault(c, []).append((t, p, cd))
                    for n in mz.nb[c].values():
                        if n is not None:
                            adj.setdefault(n, []).append((t, p, c, cd))
        return same, adj, edible

    # ---- planning ---------------------------------------------------------------------------------------------
    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; n = len(self.actions)
        self._track(info)
        mz = _maze(info["map"])
        rows = info["map"].split("|")
        targets = {(x, y) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch in ".o"}
        same, adj, edible = self._ghost_risk(info, mz)
        nb = mz.nb

        def risk(c, t, d):
            """Occupancy mass that can collide with Pac-Man arriving at c at time t moving in d (worst case over
            the ghosts' coin flips). Same cell within 5 ticks either way, unless the ghost left one block ahead in
            our direction (it never closes); a ghost that crossed c perpendicular to us 6 or 7 ticks earlier
            (closest approach k/sqrt2 < 1). Neighbouring cell x: a ghost crossing x perpendicular to the c-x axis
            0 to 2.5 ticks before we reach c; ghosts moving along that axis are covered by the same-cell tests."""
            r = 0.0
            for tk, p, gd in same.get(c, ()):
                dt = t - tk
                if -W_SAME <= dt <= W_SAME:
                    if not (dt >= W_SAME - 0.5 and gd == d): r += p
                elif W_SAME < dt <= W_PERP:          # it left c 1.2 to 1.4 blocks ago: across our line?
                    for q, _c2, d2 in mz.gtrans[(c, gd)]:
                        if d2 != d and d2 != OPP[d]: r += p * q
            for tk, p, x, gd in adj.get(c, ()):
                if 0 <= t - tk <= W_ADJ and nb[c][gd] != x and nb[c][OPP[gd]] != x:
                    r += p
            return r

        def depart(c, t, d_out):
            """Leaving c at time t in d_out while a ghost reaches c 6 or 7 ticks later moving across d_out."""
            r = 0.0
            for tk, p, gd in same.get(c, ()):
                if W_SAME < tk - t <= W_PERP and gd != d_out and gd != OPP[d_out]:
                    r += p
            return r

        pdir = info["pac"]["dir"]
        if pdir not in DIRS:
            pdir = None
        cell, t0, prev = _locate(info["pac"]["x"], info["pac"]["y"], pdir or "left", TPB)
        # candidate actions: (first cell, arrival tick, direction, cells passed before it)
        cands: dict[str, tuple] = {}
        if t0 == 0 or pdir is None:
            for a in DIRS:
                f = nb.get(cell, {}).get(a)
                if f is not None: cands[a] = (f, TPB, a, [(cell, 0, None, a)], 0)
        else:
            for a in DIRS:
                if a == OPP[pdir]:
                    # a reversal is immediate; if the cell behind is reached inside this step (t0 >= 2) the held key
                    # carries us straight through it, so the first free choice is one cell further (or a stop there)
                    ta = TPB - t0
                    if ta <= STEP_TICKS and nb[prev][a] is not None:
                        cands[a] = (nb[prev][a], ta + TPB, a, [(prev, ta, a, a)], REVERSE_PENALTY)
                    else:
                        cands[a] = (prev, ta, a, [], REVERSE_PENALTY)
                else:
                    f = nb.get(cell, {}).get(a)
                    if f is not None: cands[a] = (f, t0 + TPB, a, [(prev, t0 - TPB, None, pdir), (cell, t0, pdir, a)], 0)
        if not cands:
            return [1.0 / n] * n

        results = {}
        for a, (first, tf, d, pre, pen) in cands.items():
            exposed = None                       # first time at which the path meets possible ghost occupancy
            for c, t, d_in, d_out in pre:
                if exposed is None and ((d_in is not None and risk(c, t, d_in) > 0) or depart(c, t, d_out) > 0): exposed = t
            if exposed is None and risk(first, tf, d) > 0: exposed = tf
            if exposed is not None:
                results[a] = (None, exposed, 0, exposed, False); continue
            seen = {first: tf}; q = deque([(first, tf, d)]); best_p = None; best_g = None; depth = tf; count = 1
            while q:
                c, t, dd = q.popleft()
                if best_p is None and c in targets: best_p = t
                if best_g is None and c in edible:
                    drel, val = edible[c]
                    if t <= CHASE_MAX and drel >= t + CHASE_SLACK: best_g = t - val / 5     # value/25 blocks of detour
                if best_p is not None and depth >= tf + HORIZON and (not edible or t > best_p + 40): break
                for a2, c2 in nb[c].items():
                    if c2 is None or c2 in seen: continue
                    t2 = t + TPB
                    if t2 <= HORIZON + tf and (depart(c, t, a2) > 0 or risk(c2, t2, a2) > 0): continue
                    seen[c2] = t2; q.append((c2, t2, a2)); count += 1
                    if t2 > depth: depth = t2
            eff = min(x for x in (best_p, best_g) if x is not None) if (best_p is not None or best_g is not None) else None
            results[a] = (None if eff is None else eff + pen, depth, count, None, depth >= tf + HORIZON)

        safe = [a for a, r in results.items() if r[3] is None]
        free = [a for a in safe if results[a][4]]                     # safe region reaches past the horizon: no trap
        with_target = [a for a in (free or safe) if results[a][0] is not None]
        probs = [0.0] * n
        self.last_mode = "target" if with_target else "region" if safe else "exposed"   # for shard statistics
        if with_target:
            best = min(results[a][0] for a in with_target)
            winners = [a for a in with_target if results[a][0] <= best + 1e-9]
            others = [a for a in safe if a not in winners]
        elif safe:                               # no safe route to a pellet: the deepest safe region
            key = lambda a: (results[a][1], results[a][2])
            top = max(key(a) for a in safe)
            winners = [a for a in safe if key(a) == top]; others = [a for a in safe if a not in winners]
        else:                                    # every option meets a ghost: the one that postpones it longest
            top = max(results[a][3] for a in results)
            winners = [a for a in results if results[a][3] >= top - 1e-9]
            if len(winners) > 1 and pdir in winners: winners = [pdir]
            others = []
        w_main = 0.9 if others else 1.0
        for a in winners: probs[self.idx[a]] = w_main / len(winners)
        for a in others: probs[self.idx[a]] = 0.1 / len(others)
        s = sum(probs)
        return [p / s for p in probs]
