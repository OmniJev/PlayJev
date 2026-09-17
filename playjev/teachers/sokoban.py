"""Sokoban teacher: plan the whole level with a move-optimal A* over pushes, then emit the plan one move at a
time. The state is (bitmask of boxes, player cell); an edge is one push, reached by a player-reachability BFS
inside the node, costing (walk length + 1) moves, so the plan is optimal in moves, not just in pushes. Pruning:
dead squares (a box on a non-goal square from which no goal can be reached, found by pulling every goal backwards
over the floor) and the classic frozen-box deadlock (a box blocked on both axes by walls, dead squares or other
frozen boxes, while off a goal). Heuristic: sum over boxes of the minimum push distance to any goal (admissible).

If A* runs out of budget (node cap or wall clock), a greedy best-first search over the same states (f = heuristic
only, player position normalised to the lowest reachable cell) looks for any solution. If that fails too, the
teacher falls back to a per-step greedy rule: walk to the nearest box that can be pushed one square closer to a
goal, and push it.

Targets: 1.0 on the planned move while a plan is being followed (there is exactly one right move). Under the
greedy fallback 0.7 on the chosen move, the remaining 0.3 spread over the other moves that neither hit a wall nor
push a box into a dead square or a blocked push; wasted moves get 0.

Plan following is checked every step against the observed state (epsilon-random actions in collection may deviate):
a wasted move (state unchanged) re-emits the same move; a random walk step (boxes unchanged) only re-walks to the
next push; a random push re-plans from the new state.

info() from games/sokoban/pj_hook.js: board (XSB rows), player [x, y], boxes, goals, walls as [x, y] lists,
level (1..155), solved, steps."""
from __future__ import annotations
import heapq
import time
from collections import deque
from . import Teacher

INF = 10 ** 9
# per-solve budgets for the first plan of an episode
ASTAR_NODES = 150_000
ASTAR_SECONDS = 1.5
WASTAR_WEIGHT = 4
WASTAR_SECONDS = 1.0
GREEDY_NODES = 150_000
GREEDY_SECONDS = 1.5
# a re-plan after the search already failed once in this episode (fallback mode) gets a small budget
RETRY_SECONDS = 0.15
FALLBACK_TOP = 0.7
PLAN_CACHE_MAX = 4000

_LEVELS: dict = {}          # level number -> _Level (static analysis, shared across teacher instances)
_PLANS: dict = {}           # (level, player, boxes) -> (moves, states) for solved states
_UNSOLVED: dict = {}        # (level, player, boxes) -> True when the full-budget search failed
_DEAD: dict = {}            # (level, player, boxes) -> True when the state is provably unsolvable


class _Level:
    """Static analysis of one level: walls, goals, floor, dead squares, push distance to the nearest goal."""

    def __init__(self, info):
        rows = info["board"]
        H = len(rows); W = max(len(r) for r in rows)
        S = W + 1  # one padding column so a shift by +-1 never wraps into the next row
        self.S, self.N = S, S * H
        walls, goals, floor = set(), set(), set()
        for y, r in enumerate(rows):
            for x in range(W):
                ch = r[x] if x < len(r) else " "
                c = y * S + x
                if ch == "#":
                    walls.add(c)
                else:
                    floor.add(c)
                if ch in ".*+":
                    goals.add(c)
        self.walls, self.goals, self.floor = walls, frozenset(goals), floor
        self.floor_mask = sum(1 << c for c in floor)
        self.goal_mask = sum(1 << c for c in goals)
        N = self.N
        self.dirs = (-S, S, -1, 1)  # up, down, left, right in cell offsets
        nbrs = [() for _ in range(N)]
        for c in floor:
            nbrs[c] = tuple(c + d for d in self.dirs if 0 <= c + d < N and (c + d) in floor)
        self.nbrs = nbrs
        # push distance from every floor cell to the nearest goal: pull each goal backwards over the floor.
        # A box on `b` pushed along d lands on c = b + d with the player standing on b - d = c - 2d.
        dist = [INF] * N
        dq = deque(goals)
        for g in goals:
            dist[g] = 0
        while dq:
            c = dq.popleft()
            for d in self.dirs:
                b, p = c - d, c - 2 * d
                if 0 <= p < N and b in floor and p in floor and dist[b] == INF:
                    dist[b] = dist[c] + 1
                    dq.append(b)
        self.dist = dist
        self.dead_mask = sum(1 << c for c in floor if dist[c] == INF)

    def cell(self, xy):
        return xy[1] * self.S + xy[0]

    def xy(self, c):
        return (c % self.S, c // self.S)

    def boxes_mask(self, boxes):
        m = 0
        for b in boxes:
            m |= 1 << self.cell(b)
        return m

    # ---- deadlock checks -------------------------------------------------------------------------------------
    def frozen(self, c, bm, assumed):
        """Is the box on cell c immovable in every continuation? The classic recursive test: an axis is blocked by
        a wall on either side, by dead squares on both sides (no solution ever pushes a box there), or by a
        neighbouring box that is itself frozen while c is treated as a wall. `assumed` holds the boxes on the
        current recursion path only (path-local), which keeps the test sound: every box the test relies on is
        blocked on both axes by walls or by other boxes of the same set, so the first of them to move could not."""
        assumed.add(c)
        fl, dead, S = self.floor_mask, self.dead_mask, self.S

        def blocked(d):
            n1, n2 = c - d, c + d
            if not (fl >> n1) & 1 or not (fl >> n2) & 1 or n1 in assumed or n2 in assumed:
                return True
            if (dead >> n1) & 1 and (dead >> n2) & 1:
                return True
            if (bm >> n1) & 1 and self.frozen(n1, bm, assumed):
                return True
            if (bm >> n2) & 1 and self.frozen(n2, bm, assumed):
                return True
            return False

        try:
            return blocked(1) and blocked(S)
        finally:
            assumed.discard(c)

    def push_deadlocks(self, t, bm):
        """After a box landed on t: is t (or a box next to it) frozen while off a goal?"""
        gm = self.goal_mask
        if self.frozen(t, bm, set()) and not (gm >> t) & 1:
            return True
        for d in self.dirs:
            n = t + d
            if (bm >> n) & 1 and not (gm >> n) & 1 and self.frozen(n, bm, set()):
                return True
        return False

    def state_deadlocked(self, bm):
        """Cheap pre-check before a search: any off-goal box on a dead square or frozen. A proof of unsolvability
        when there are as many boxes as goals (every box must end on a goal); otherwise never claimed."""
        if bin(bm).count("1") != len(self.goals):
            return False
        if bm & self.dead_mask & ~self.goal_mask:
            return True
        b = bm
        while b:
            low = b & -b; c = low.bit_length() - 1; b ^= low
            if not (self.goal_mask >> c) & 1 and self.frozen(c, bm, set()):
                return True
        return False

    # ---- searches --------------------------------------------------------------------------------------------
    def heuristic(self, bm):
        h, b, dist = 0, bm, self.dist
        while b:
            low = b & -b; c = low.bit_length() - 1; b ^= low
            h += dist[c]
        return h

    def reach(self, player, free):
        """BFS distances from the player over free cells (dict cell -> moves)."""
        nbrs = self.nbrs
        dist = {player: 0}; dq = deque([player])
        while dq:
            c = dq.popleft(); dc = dist[c] + 1
            for n in nbrs[c]:
                if (free >> n) & 1 and n not in dist:
                    dist[n] = dc; dq.append(n)
        return dist

    def astar(self, player, bm, max_nodes, seconds, w=1):
        """A* over (boxes, player) with push edges, f = g + w * h. With w = 1 the plan is move-optimal; w > 1 is
        weighted A* (faster, plan at most w times longer in the bound, in practice close to optimal). Returns a
        list of (bm, player, box, d) pushes, or None (budget exhausted or unsolvable)."""
        fl, dead, gm, S, dist_goal = self.floor_mask, self.dead_mask, self.goal_mask, self.S, self.dist
        nbrs = self.nbrs
        if len(self.goals) != bin(bm).count("1"):
            use_h = False
        else:
            use_h = True
        h0 = self.heuristic(bm) if use_h else 0
        start = (bm, player)
        best = {start: 0}; prev = {start: None}
        heap = [(w * h0, 0, h0, bm, player)]
        deadline = time.perf_counter() + seconds
        expanded = 0
        while heap:
            f, g, h, cbm, pl = heapq.heappop(heap)
            g = -g
            if g > best.get((cbm, pl), INF):
                continue
            if cbm & gm == gm:
                return self._pushes(prev, (cbm, pl))
            expanded += 1
            if expanded > max_nodes or (expanded & 255) == 0 and time.perf_counter() > deadline:
                return None
            free = fl & ~cbm
            # BFS from the player
            dist = {pl: 0}; dq = deque([pl])
            while dq:
                c = dq.popleft(); dc = dist[c] + 1
                for n in nbrs[c]:
                    if (free >> n) & 1 and n not in dist:
                        dist[n] = dc; dq.append(n)
            b = cbm
            while b:
                low = b & -b; c = low.bit_length() - 1; b ^= low
                for d in (-S, S, -1, 1):
                    p = c - d
                    if p not in dist:
                        continue
                    t = c + d
                    if not (free >> t) & 1 or (dead >> t) & 1:
                        continue
                    nbm = cbm ^ low ^ (1 << t)
                    if self.push_deadlocks(t, nbm):
                        continue
                    ng = g + dist[p] + 1
                    key = (nbm, c)
                    if ng < best.get(key, INF):
                        best[key] = ng
                        prev[key] = ((cbm, pl), pl, c, d)
                        nh = h - dist_goal[c] + dist_goal[t] if use_h else 0
                        heapq.heappush(heap, (ng + w * nh, -ng, nh, nbm, c))
        return False

    def greedy(self, player, bm, max_nodes, seconds):
        """Greedy best-first over (boxes, normalised player) ordered by the heuristic; any solution will do.
        Returns pushes, None (budget) or False (state space exhausted, unsolvable)."""
        fl, dead, gm, S, dist_goal = self.floor_mask, self.dead_mask, self.goal_mask, self.S, self.dist
        nbrs = self.nbrs
        free0 = fl & ~bm
        r0 = self.reach(player, free0)
        start = (bm, min(r0))
        seen = {start}; prev = {start: None}
        # heap entries: (h, -pushes, bm, norm, actual player)
        heap = [(self.heuristic(bm), 0, bm, min(r0), player)]
        deadline = time.perf_counter() + seconds
        expanded = 0
        while heap:
            h, npush, cbm, norm, pl = heapq.heappop(heap)
            if cbm & gm == gm:
                return self._pushes(prev, (cbm, norm))
            expanded += 1
            if expanded > max_nodes or (expanded & 255) == 0 and time.perf_counter() > deadline:
                return None
            free = fl & ~cbm
            dist = {pl: 0}; dq = deque([pl])
            while dq:
                c = dq.popleft(); dc = dist[c] + 1
                for n in nbrs[c]:
                    if (free >> n) & 1 and n not in dist:
                        dist[n] = dc; dq.append(n)
            b = cbm
            while b:
                low = b & -b; c = low.bit_length() - 1; b ^= low
                for d in (-S, S, -1, 1):
                    p = c - d
                    if p not in dist:
                        continue
                    t = c + d
                    if not (free >> t) & 1 or (dead >> t) & 1:
                        continue
                    nbm = cbm ^ low ^ (1 << t)
                    if self.push_deadlocks(t, nbm):
                        continue
                    # normalised player position after the push: lowest cell reachable from c
                    nfree = fl & ~nbm
                    nd = {c: 0}; ndq = deque([c])
                    while ndq:
                        cc = ndq.popleft()
                        for n in nbrs[cc]:
                            if (nfree >> n) & 1 and n not in nd:
                                nd[n] = 1; ndq.append(n)
                    key = (nbm, min(nd))
                    if key in seen:
                        continue
                    seen.add(key)
                    prev[key] = ((cbm, norm), pl, c, d)
                    nh = h - dist_goal[c] + dist_goal[t]
                    heapq.heappush(heap, (nh, npush - 1, nbm, key[1], c))
        return False

    @staticmethod
    def _pushes(prev, key):
        """Walk the parent links back to the start: list of (boxes before, player before, box cell, dir)."""
        out = []
        while prev[key] is not None:
            pkey, pl, c, d = prev[key]
            out.append((pkey[0], pl, c, d))
            key = pkey
        return out[::-1]

    def moves_from_pushes(self, pushes, player, bm):
        """Expand pushes into single moves. Returns (moves as direction offsets, states before each move plus the
        final one as (player, bm))."""
        moves, states = [], [(player, bm)]
        for cbm, pl, c, d in pushes:
            free = self.floor_mask & ~cbm
            for step in self.walk(pl, c - d, free):
                player += step; moves.append(step); states.append((player, cbm))
            # the push itself
            player = c; nbm = cbm ^ (1 << c) ^ (1 << (c + d))
            moves.append(d); states.append((player, nbm)); bm = nbm
        return moves, states

    def walk(self, src, dst, free):
        """Shortest walk from src to dst over free cells as a list of direction offsets (BFS with parents)."""
        if src == dst:
            return []
        nbrs = self.nbrs
        par = {src: None}; dq = deque([src])
        while dq:
            c = dq.popleft()
            if c == dst:
                break
            for n in nbrs[c]:
                if (free >> n) & 1 and n not in par:
                    par[n] = c; dq.append(n)
        if dst not in par:
            return None
        path = []
        c = dst
        while par[c] is not None:
            path.append(c - par[c]); c = par[c]
        return path[::-1]


class SokobanTeacher(Teacher):
    def __init__(self, actions):
        super().__init__(actions)
        self.reset()

    def reset(self):
        self.level = None
        self.plan = None        # list of direction offsets
        self.states = None      # states[i] = (player, bm) before plan[i]; len = len(plan) + 1
        self.i = 0
        self.failed_bm = None   # boxes mask at the last failed full search (fallback mode when not None)
        self.mode = "plan"
        self.last_solve_ms = 0.0
        self.solve_calls = 0
        self.proven_dead = False   # set when the board was shown unsolvable (deadlock, or state space exhausted)

    def giveup(self):
        """True once the board is provably unsolvable from the last observed state: an off-goal box on a dead
        square, a frozen off-goal box, or a search that exhausted the whole state space. Deadlocks are permanent,
        so the flag stays valid for every later state of the episode."""
        return self.proven_dead

    # ---- planning ------------------------------------------------------------------------------------------
    def _level(self, info):
        lv = info.get("level")
        L = _LEVELS.get(lv)
        if L is None:
            L = _LEVELS[lv] = _Level(info)
        return L

    def _solve(self, L, player, bm, full):
        key = (self.level, player, bm)
        cached = _PLANS.get(key)
        if cached is not None:
            return cached
        if key in _UNSOLVED:
            return None             # the full-budget search already failed here (this or an earlier episode)
        t0 = time.perf_counter()
        pushes = None
        if L.state_deadlocked(bm) or key in _DEAD:
            pushes = False                      # provably unsolvable, no search needed
        elif full:
            pushes = L.astar(player, bm, ASTAR_NODES, ASTAR_SECONDS)
            if pushes is None:
                pushes = L.astar(player, bm, ASTAR_NODES, WASTAR_SECONDS, WASTAR_WEIGHT)
            if pushes is None:
                pushes = L.greedy(player, bm, GREEDY_NODES, GREEDY_SECONDS)
        else:
            pushes = L.astar(player, bm, ASTAR_NODES, RETRY_SECONDS, WASTAR_WEIGHT)
            if pushes is None:
                pushes = L.greedy(player, bm, GREEDY_NODES, RETRY_SECONDS)
        self.last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self.solve_calls += 1
        if pushes is False and bin(bm).count("1") == len(L.goals):   # exhausted: no solution exists from here
            self.proven_dead = True
            _DEAD[key] = True
            return None
        if pushes is False:
            pushes = None
        if pushes is None:
            if full:
                _UNSOLVED[key] = True
            return None
        moves, states = L.moves_from_pushes(pushes, player, bm)
        if len(_PLANS) < PLAN_CACHE_MAX:
            _PLANS[key] = (moves, states)
        return moves, states

    def _replan(self, L, player, bm):
        if self.proven_dead:
            self.plan, self.states, self.i = None, None, 0
            self.mode = "fallback"
            return False                # the board is dead for the rest of the episode: no more searches
        if self.failed_bm is not None:
            if self.failed_bm == bm:
                return False            # fallback mode and the boxes have not moved: nothing new to search
            full = False                # fallback mode, boxes moved: retry with the small budget
        else:
            full = True
        res = self._solve(L, player, bm, full)
        if res is None:
            self.plan, self.states, self.i = None, None, 0
            self.failed_bm = bm
            self.mode = "fallback"
            return False
        self.plan, self.states = res
        self.i = 0
        self.mode = "plan"
        self.failed_bm = None
        return True

    def _rewalk(self, L, player, bm):
        """Boxes match the plan's state before move i-1 but the player wandered: walk to the next push."""
        j = self.i - 1 if self.i > 0 else 0
        # find the next push in the plan from j on (first move that changes the boxes)
        while j < len(self.plan) and self.states[j + 1][1] == self.states[j][1]:
            j += 1
        if j >= len(self.plan):
            return False
        push_from = self.states[j][0]
        free = L.floor_mask & ~bm
        w = L.walk(player, push_from, free)
        if w is None:
            return False
        moves = w + self.plan[j:]
        states = [(player, bm)]
        p = player
        for step in w:
            p += step; states.append((p, bm))
        states.extend(self.states[j + 1:])
        self.plan, self.states, self.i = moves, states, 0
        return True

    # ---- fallback -------------------------------------------------------------------------------------------
    def _fallback(self, L, player, bm):
        """Greedy: nearest box that can be pushed one square closer to a goal. Returns (chosen dir, ok dirs)."""
        S = L.S; free = L.floor_mask & ~bm; dead = L.dead_mask; dist_goal = L.dist
        dist = L.reach(player, free)
        ok = []     # moves that are not wasted and do not push a box into a dead square
        for d in L.dirs:
            n = player + d
            if not (L.floor_mask >> n) & 1:
                continue
            if (bm >> n) & 1:
                t = n + d
                if not (free >> t) & 1 or (dead >> t) & 1:
                    continue
            ok.append(d)
        best = None
        b = bm
        while b:
            low = b & -b; c = low.bit_length() - 1; b ^= low
            for d in L.dirs:
                p, t = c - d, c + d
                if p in dist and (free >> t) & 1 and not (dead >> t) & 1 and dist_goal[t] < dist_goal[c]:
                    nbm = bm ^ low ^ (1 << t)
                    if L.push_deadlocks(t, nbm):
                        continue
                    cand = (dist[p], dist_goal[t], c, d)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            # no improving push reachable: head for the nearest pushable box at all
            b = bm
            while b:
                low = b & -b; c = low.bit_length() - 1; b ^= low
                for d in L.dirs:
                    p, t = c - d, c + d
                    if p in dist and (free >> t) & 1 and not (dead >> t) & 1:
                        cand = (dist[p], dist_goal[t], c, d)
                        if best is None or cand < best:
                            best = cand
        if best is None:
            return None, ok
        _, _, c, d = best
        p = c - d
        if p == player:
            return d, ok
        w = L.walk(player, p, free)
        if not w:
            return None, ok
        return w[0], ok

    # ---- act -----------------------------------------------------------------------------------------------
    def act(self, obs):
        info = obs["info"]
        n = len(self.actions)
        if info.get("solved"):
            return [1.0 / n] * n
        L = self._level(info)
        if self.level != info.get("level"):
            self.level = info.get("level")
            self.plan = None; self.failed_bm = None; self.mode = "plan"
        player = L.cell(info["player"]); bm = L.boxes_mask(info["boxes"])
        state = (player, bm)
        names = {-L.S: "up", L.S: "down", -1: "left", 1: "right"}

        if self.plan is not None:
            if self.i < len(self.plan) and self.states[self.i] == state:
                pass
            elif self.i > 0 and self.states[self.i - 1] == state:
                self.i -= 1                                   # the last move was wasted or replaced by a no-op
            elif self.i > 0 and self.states[self.i - 1][1] == bm and self._rewalk(L, player, bm):
                pass                                          # the player wandered; walk back to the plan
            elif self.i < len(self.plan) and self.states[self.i][1] == bm and self._rewalk(L, player, bm):
                pass
            else:
                self.plan = None
        if self.plan is None:
            self._replan(L, player, bm)
        if self.plan is not None and self.i < len(self.plan):
            d = self.plan[self.i]; self.i += 1
            return self.one_hot(n, self.idx[names[d]])

        # fallback: greedy push toward a goal
        d, ok = self._fallback(L, player, bm)
        probs = [0.0] * n
        if d is None:
            if not ok:
                return [1.0 / n] * n
            for o in ok:
                probs[self.idx[names[o]]] = 1.0 / len(ok)
            return probs
        others = [o for o in ok if o != d]
        probs[self.idx[names[d]]] = FALLBACK_TOP if others else 1.0
        for o in others:
            probs[self.idx[names[o]]] = (1.0 - FALLBACK_TOP) / len(others)
        s = sum(probs)
        return [p / s for p in probs]
