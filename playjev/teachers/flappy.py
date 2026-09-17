"""Flappy teacher: exact physics lookahead with a depth-first search over flap/wait sequences.

The game (nebez/floppybird) is deterministic and fully described by info(): bird top `birdY`, `velocity`,
the pipes' x and gap top. One step is 5 physics ticks (v += 0.25; y += v per tick); a flap sets v = -4.6
before the first tick of the step, so a flap step always moves the bird exactly -19.25 px and afterwards
the state is a function of y alone. Between decisions the pipe advances 2.222 px per tick.

Policy: simulate both actions tick by tick with the game's own collision box (rotation-dependent bounding
rect, ceiling clamp, pipe box x-2 .. x+50, gap top .. top+90, ground at 420) and search depth first for an
action sequence that survives until the bird has left the third upcoming pipe (at most MAX_H steps).
Children are ordered by a hover heuristic (flap at the closest approach to a point OFF px below the gap
centre, or below mid-screen while no pipe is within VISIBLE_X), so the first surviving sequence found is
the heuristic's own trajectory with the minimal corrections; doomed states are memoised by
(depth, y, v, pipe index). A leaf at the horizon must also pass a rollout of the hover heuristic through
the following pipe, so the search prefers exits that leave the next gap comfortably reachable.

Two tiers: tier 1 shrinks the gap and the ground by MARGIN_SAFE px and uses the rollout leaf; when it finds
nothing, tier 2 uses MARGIN_MIN px (layout rounding only) and a plain reachability leaf.

Targets: 0.9 on the first action of the surviving sequence, 0.1 on the other action when it too has a
surviving continuation (either tier), 0.0 when both tiers exhaust its subtree without one (certain
collision within the horizon). If neither action survives, the one that lives longer gets 1.0 (0.5 each on
a tie). If a search runs past NODE_BUDGET nodes the heuristic action is used with 0.9 / 0.1.
"""
from __future__ import annotations
import math
from . import Teacher

GRAVITY, JUMP, TICKS = 0.25, -4.6, 5
PIPE_TICK = (1000.0 / 7500.0) * (1000.0 / 60.0)    # animPipe: 1000 px per 7500 ms, per 16.667 ms tick = 2.2222 px
PIPE_STEP = PIPE_TICK * TICKS                       # 11.11 px per step
CX = 60 + 34 / 2                                    # bird centre x (left 60, sprite 34 wide)
LAND_TOP, GAP = 420.0, 90.0
FLAP_STEP = 5 * JUMP + 3.75                         # -19.25: bird displacement of a flap step, whatever v was

MID_C = 210.0        # hover centre while no pipe is within VISIBLE_X (fly area 0..420)
OFF = 16.0           # flap at the closest approach to gap centre + OFF (the flap then lifts the bird 39.5 px)
SLACK = 10.0         # lazy guard: flap if a wait would put the centre below the flap point + SLACK
VISIBLE_X = 320.0    # a pipe becomes the target once its left edge is inside the 320 px frame
PIPES_AHEAD = 3      # exact horizon: exit of the third upcoming pipe
MAX_H = 50           # horizon cap, steps
LEAF_MARGIN = 8.0    # shrink of the gap band in the interval reachability check, px
MARGIN_SAFE = 2.0    # tier 1 clearance from pipe edges and ground, px
MARGIN_MIN = 0.3     # tier 2: layout rounds top/left to 1/64 px, so exact-zero clearance is not safe
NODE_BUDGET = 8000   # search nodes per act() before falling back to the heuristic

FLAP, WAIT = 0, 1
_GEOM: dict[int, tuple[float, float, float, float]] = {}


def geom(v: float) -> tuple[float, float, float, float]:
    """Collision box of the bird at velocity v, relative to birdY (sprite top), as in gameloop():
    (half width, top offset, bottom offset, raw bounding-rect bottom offset used for the ground test)."""
    k = round(v * 1000)
    g = _GEOM.get(k)
    if g is None:
        rot = min(v / 10 * 90, 90)                              # degrees, negative when rising
        th = abs(rot) * math.pi / 180
        hrot = 34 * math.sin(th) + 24 * math.cos(th)            # rotated bounding rect height
        bw = 34 - math.sin(abs(rot) / 90) * 8                   # the game's (odd) width shrink
        bh = (24 + hrot) / 2
        g = _GEOM[k] = (bw / 2, 12 - bh / 2, 12 + bh / 2, 12 + hrot / 2)
    return g


def sim_step(y: float, v: float, pipes: list, pi: int, ticks: int, action: int, margin: float = 0.0):
    """Advance one step (5 ticks). pipes: [(x at ticks=0, gapTop)] ascending, pi: index of the next unpassed
    pipe, margin: extra clearance demanded from the gap edges and the ground.
    Returns (y, v, pi, ticks, dead, cause) with cause in {None, 'floor', 'top', 'bottom'}."""
    if action == FLAP:
        v = JUMP
    n = len(pipes)
    for _ in range(TICKS):
        ticks += 1
        v += GRAVITY
        y += v
        hw, to, bo, rbo = geom(v)
        if y + rbo >= LAND_TOP - margin:
            return y, v, pi, ticks, True, "floor"
        clamp = y + to <= 0                                     # ceiling: position = 0, checked before the pipe
        if pi < n:
            x = pipes[pi][0] - PIPE_TICK * ticks
            if CX + hw > x - 2:                                 # inside the pipe's x range
                gt = pipes[pi][1]
                if not (y + to > gt + margin and y + bo < gt + GAP - margin):
                    return y, v, pi, ticks, True, ("top" if y + to <= gt + margin else "bottom")
            if CX - hw > x + 50:                                # passed: score, next pipe
                pi += 1
        if clamp:
            y = 0.0
    return y, v, pi, ticks, False, None


def prefer(y, v, pipes, pi, ticks) -> int:
    """Hover heuristic: target the visible next gap (else mid-screen); flap at the closest approach of the
    centre to target + OFF while descending, or when one more wait would drop it past target + OFF + SLACK."""
    cy = y + 12
    if pi < len(pipes) and pipes[pi][0] - PIPE_TICK * ticks < VISIBLE_X:
        c = pipes[pi][1] + GAP / 2
    else:
        c = MID_C
    p = c + OFF
    y1 = cy + 5 * v + 3.75                                      # centre after one wait step
    if y1 > p + SLACK:
        return FLAP
    if v >= 0 and abs(cy - p) <= abs(y1 - p):
        return FLAP
    return WAIT


def steps_to_exit(x: float) -> int:
    """Steps until a pipe now at x has its right edge behind the bird's box."""
    x_exit = CX - 17 - 50
    return math.ceil(max(0.0, x - x_exit) / PIPE_STEP) + 1


class _Budget(Exception):
    pass


class FlappyTeacher(Teacher):
    def reset(self):
        self._memo: dict = {}
        self._leaf_memo: dict = {}
        self._nodes = 0
        self.last_nodes = 0          # search nodes spent by the last act(), for cost reporting
        self.last_plan: list = []
        self.last_tier = 0

    # ---- horizon and leaf ----
    @staticmethod
    def _horizon(pipes) -> int:
        if not pipes:
            return MAX_H
        j = min(PIPES_AHEAD - 1, len(pipes) - 1)
        return max(1, min(MAX_H, steps_to_exit(pipes[j][0])))

    @staticmethod
    def _reachable(y, v, pipes, pi, ticks) -> bool:
        """Interval check: can the bird be inside the next unpassed gap when it gets there?"""
        if pi >= len(pipes):
            return True
        x = pipes[pi][0] - PIPE_TICK * ticks
        if CX + 17 > x - 2:
            return True                                         # inside the pipe and alive
        d = int((x - 2 - (CX + 17)) // PIPE_STEP)               # full steps before the bird reaches it
        lo = max(0.0, y + FLAP_STEP * d)                        # flap every step
        hi = y + 5 * v * d + 3.75 * d + 6.25 * d * (d - 1) / 2  # free fall
        gt = pipes[pi][1]
        return lo <= gt + GAP - 24 - LEAF_MARGIN and hi >= gt + LEAF_MARGIN

    def _leaf_ok(self, y, v, pipes, pi, ticks, margin, rollout) -> bool:
        if not self._reachable(y, v, pipes, pi, ticks):
            return False
        if not rollout or pi >= len(pipes):
            return True
        key = (round(y * 4), round(v * 1000), pi, ticks)
        r = self._leaf_memo.get(key)
        if r is None:
            # follow the hover heuristic until the next unpassed pipe is behind the bird
            n = steps_to_exit(pipes[pi][0] - PIPE_TICK * ticks)
            r = True
            for _ in range(n):
                a = prefer(y, v, pipes, pi, ticks)
                y, v, pi2, ticks, dead, _ = sim_step(y, v, pipes, pi, ticks, a, margin)
                if dead:
                    r = False
                    break
                if pi2 != pi:
                    break
            self._leaf_memo[key] = r
        return r

    # ---- search ----
    def _search(self, y, v, pipes, forced=None, margin=MARGIN_SAFE, rollout=True):
        """Depth-first search for a surviving action sequence. Returns (found, plan, deepest)."""
        H = self._horizon(pipes)
        memo = self._memo.setdefault((margin, rollout), set())
        plan: list = []
        deepest = [0]

        def rec(depth, y, v, pi, ticks):
            if depth > deepest[0]:
                deepest[0] = depth
            if depth >= H:
                return self._leaf_ok(y, v, pipes, pi, ticks, margin, rollout)
            key = (depth, round(y * 4), round(v * 1000), pi)
            if key in memo:
                return False
            self._nodes += 1
            if self._nodes > NODE_BUDGET:
                raise _Budget
            if depth == 0 and forced is not None:
                order = (forced,)
            else:
                a = prefer(y, v, pipes, pi, ticks)
                order = (a, 1 - a)
            for a in order:
                ny, nv, npi, nt, dead, _ = sim_step(y, v, pipes, pi, ticks, a, margin)
                if dead:
                    continue
                if rec(depth + 1, ny, nv, npi, nt):
                    plan.append(a)
                    return True
            memo.add(key)
            return False

        found = rec(0, y, v, 0, 0)
        plan.reverse()
        return found, plan, deepest[0]

    def _survives(self, y, v, pipes, forced=None):
        """Tier 1 then tier 2. Returns (found, plan, deepest, tier)."""
        found, plan, deep1 = self._search(y, v, pipes, forced, MARGIN_SAFE, True)
        if found:
            return True, plan, deep1, 1
        found, plan, deep2 = self._search(y, v, pipes, forced, MARGIN_MIN, False)
        return found, plan, max(deep1, deep2), 2

    def act(self, obs: dict) -> list[float]:
        info = obs["info"]
        y, v = float(info["birdY"]), float(info["velocity"])
        nxt = info.get("nextPipe")
        pipes = []
        if nxt:
            pipes = sorted((float(p["x"]), float(p["gapTop"])) for p in info.get("pipes", [])
                           if p["x"] >= nxt["x"] - 1e-6)
        self._memo.clear()
        self._leaf_memo.clear()
        self._nodes = 0
        n = len(self.actions)
        probs = [0.0] * n
        i_flap, i_wait = self.idx["flap"], self.idx["wait"]
        index = {FLAP: i_flap, WAIT: i_wait}

        try:
            found, plan, _, tier = self._survives(y, v, pipes)
            if found:
                chosen = plan[0]
                self.last_plan, self.last_tier = plan, tier
                other = 1 - chosen
                found2, _, _, _ = self._survives(y, v, pipes, forced=other)
                probs[index[chosen]] = 0.9 if found2 else 1.0
                probs[index[other]] = 0.1 if found2 else 0.0
            else:
                # both actions die within the horizon: prefer the one that lives longer
                _, _, d_flap, _ = self._survives(y, v, pipes, forced=FLAP)
                _, _, d_wait, _ = self._survives(y, v, pipes, forced=WAIT)
                self.last_plan, self.last_tier = [], 3
                if d_flap == d_wait:
                    probs[i_flap] = probs[i_wait] = 0.5
                else:
                    probs[i_flap if d_flap > d_wait else i_wait] = 1.0
        except _Budget:
            a = prefer(y, v, pipes, 0, 0)
            probs[index[a]] = 0.9
            probs[index[1 - a]] = 0.1
            self.last_plan, self.last_tier = [], 4
        self.last_nodes = self._nodes
        s = sum(probs)
        return [p / s for p in probs]

    def giveup(self) -> bool:
        """True after an act() in which both tiers proved that every flap/wait sequence collides within the
        horizon: the bird dies within a few steps whatever it does, so the collector can end the episode."""
        return self.last_tier == 3
