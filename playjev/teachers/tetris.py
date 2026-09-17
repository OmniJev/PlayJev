"""Tetris teacher: one-piece placement search scored with Dellacherie's evaluator.

info() from games/tetris/pj_hook.js: the locked board as rows top to bottom joined by '|' ('.' empty, a
letter otherwise), the falling piece {kind, x, y, dir}, the next piece's kind. One env step is one key
(left, right, rotate, none) followed by one row of gravity; a piece that cannot sink after its key locks
where it is; drop is a hard drop that locks in the same step (see games/tetris/NOTES.md).

Algorithm. When a piece appears (or the observed piece stops matching the plan) the teacher searches the
step graph exactly as the game runs it: breadth first over piece states (x, y, dir) from the observed
one, one key per level, so every reachable placement comes with the shortest key sequence that reaches
it, including slides under overhangs and locks by a sideways move on the stack. Each placement is scored
on the board it produces with Dellacherie's six features and weights: landing height -4.5, eroded piece
cells +3.4, row transitions -3.2, column transitions -9.3, holes -7.9, cumulative wells -3.4 (feature
definitions as in Thiery and Scherrer's MDPTetris; walls and floor count as filled, transitions are
counted over the rows of the stack). A placement over which the next piece could not spawn is penalised
by 10000 times the spawn-blocking probability (the game spawns at a random column). The best placement's
key sequence becomes the plan, replayed one key per step while the observed piece is where the plan
expects it.

Targets: 0.9 on the planned key (split when several first keys lead to equally good placements), 0.1
split over the other keys after which the chosen placement is still reachable, zero on keys that lose
the placement or waste the step (blocked move, blocked or no-op rotation)."""
from __future__ import annotations
from collections import deque
from . import Teacher

# 16-bit masks per rotation copied from jakesgordon's index.html; eachblock() reads a 4x4 box from bit
# 0x8000 (col 0, row 0) across then down. size is the spawn range: x = round(random(0, cols - size)).
PIECES = {
    "I": (4, (0x0F00, 0x2222, 0x00F0, 0x4444)),
    "J": (3, (0x44C0, 0x8E00, 0x6440, 0x0E20)),
    "L": (3, (0x4460, 0x0E80, 0xC440, 0x2E00)),
    "O": (2, (0xCC00, 0xCC00, 0xCC00, 0xCC00)),
    "S": (3, (0x06C0, 0x8C40, 0x6C00, 0x4620)),
    "T": (3, (0x0E40, 0x4C40, 0x4E00, 0x4640)),
    "Z": (3, (0x0C60, 0x4C80, 0xC600, 0x2640)),
}
W_LANDING, W_ERODED, W_ROW_TR, W_COL_TR, W_HOLES, W_WELLS = -4.5, 3.4, -3.2, -9.3, -7.9, -3.4
DEATH = 10000.0          # times the probability that the next piece cannot spawn on the resulting board
KEYS = ("rotate", "left", "right", "none")   # expansion order: at equal plan length rotations come first
TIE = 1e-6


def _cells(mask: int) -> list[tuple[int, int]]:
    out, bit, col, row = [], 0x8000, 0, 0
    while bit:
        if mask & bit:
            out.append((col, row))
        bit >>= 1; col += 1
        if col == 4:
            col, row = 0, row + 1
    return out


class Shape:
    """One rotation of one piece: cells relative to the piece origin, per-row masks, extent, signature."""
    __slots__ = ("cells", "minx", "maxx", "miny", "maxy", "rows", "sig")

    def __init__(self, cells):
        self.cells = cells
        xs = [c for c, _ in cells]; ys = [r for _, r in cells]
        self.minx, self.maxx, self.miny, self.maxy = min(xs), max(xs), min(ys), max(ys)
        rows: dict[int, int] = {}
        for c, r in cells:
            rows[r] = rows.get(r, 0) | (1 << c)
        self.rows = tuple(sorted(rows.items()))            # (dy, mask with bit cx set)
        self.sig = tuple(sorted((c - self.minx, r - self.miny) for c, r in cells))


SHAPES = {k: tuple(Shape(_cells(m)) for m in masks) for k, (_, masks) in PIECES.items()}
# rotations with identical cell patterns (I 0/2, S and Z 0/2 and 1/3, all four of O) share a signature id
SIG = {k: tuple(sorted({s.sig for s in shs}).index(s.sig) for s in shs) for k, shs in SHAPES.items()}


def _shift(m: int, x: int) -> int:
    return m << x if x >= 0 else m >> -x


class TetrisTeacher(Teacher):
    def reset(self):
        self.plan = []        # [(expected state before the key, key)], consumed one entry per step
        self.kind = None
        self.target = None    # canonical placement (x, y, signature) the plan is heading for
        self.trans = {}       # state -> [(key, next state or None, placement if the key locks the piece)]
        self.reach = {}       # state -> whether the target is still reachable from that state
        self.drop_key = {}    # state -> placement of a hard drop from that state
        self.first_keys = ()  # keys that start equally good plans, valid at the state the plan was made in
        self.plan_state = None

    # ------------------------------------------------------------------ per step
    def act(self, obs: dict) -> list[float]:
        n = len(self.actions)
        info = obs.get("info") or {}
        piece = info.get("piece")
        if not piece or piece.get("kind") not in SHAPES or not info.get("grid"):
            return [1.0 / n] * n
        nx = int(info.get("cols") or 10); ny = int(info.get("rows_high") or 20)
        rows = [sum(1 << x for x, ch in enumerate(r) if ch != ".") for r in info["grid"].split("|")]
        kind = piece["kind"]; s0 = (int(piece["x"]), int(piece["y"]), int(piece["dir"]) & 3)
        if not self.plan or kind != self.kind or self.plan[0][0] != s0:
            self._plan(rows, kind, s0, info.get("next"), nx, ny)
        if not self.plan:   # piece overlaps the stack: the game is over, nothing to plan
            return self.one_hot(n, self.idx["drop"])
        _, key = self.plan.pop(0)
        winners = list(self.first_keys) if s0 == self.plan_state and key in self.first_keys else [key]
        acceptable = []
        for k, s1, pk in self.trans.get(s0, ()):
            if k in winners:
                continue
            if (s1 is not None and self.reach.get(s1)) or pk == self.target:
                acceptable.append(k)
        if "drop" not in winners and self.drop_key.get(s0) == self.target:
            acceptable.append("drop")
        probs = [0.0] * n
        if acceptable:
            for k in winners: probs[self.idx[k]] += 0.9 / len(winners)
            for k in acceptable: probs[self.idx[k]] += 0.1 / len(acceptable)
        else:
            for k in winners: probs[self.idx[k]] += 1.0 / len(winners)
        s = sum(probs)
        return [p / s for p in probs]

    # ------------------------------------------------------------------ per piece
    def _plan(self, rows, kind, s0, next_kind, nx, ny):
        shapes = SHAPES[kind]; sig = SIG[kind]
        free_cache: dict = {}; land_cache: dict = {}

        def free(x, y, d):
            k = (x, y, d); v = free_cache.get(k)
            if v is None:
                sh = shapes[d]
                if x + sh.minx < 0 or x + sh.maxx >= nx or y + sh.miny < 0 or y + sh.maxy >= ny:
                    v = False
                else:
                    v = True
                    for dy, m in sh.rows:
                        if rows[y + dy] & _shift(m, x):
                            v = False; break
                free_cache[k] = v
            return v

        def land(x, y, d):   # lowest row a straight fall from the free state (x, y, d) reaches
            k = (x, y, d); v = land_cache.get(k)
            if v is None:
                yy = y
                while free(x, yy + 1, d):
                    yy += 1
                v = yy; land_cache[k] = v
            return v

        def pkey(x, y, d):
            sh = shapes[d]; return (x + sh.minx, y + sh.miny, sig[d])

        self.plan = []; self.kind = kind; self.plan_state = s0
        self.trans = trans = {}; self.drop_key = drop_key = {}; self.reach = reach = {}
        if not free(*s0):
            return
        parent = {s0: None}; order = [s0]; q = deque([s0])
        terminals: dict = {}   # placement -> (state, key, lock state) for the shortest plan
        while q:
            s = q.popleft(); x, y, d = s
            out = []
            ly = land(x, y, d); pk = pkey(x, ly, d)
            drop_key[s] = pk
            if pk not in terminals:
                terminals[pk] = (s, "drop", (x, ly, d))
            for key in KEYS:
                if key == "rotate":
                    if kind == "O":
                        continue                       # rotating the O changes nothing: a wasted step
                    x1, d1 = x, (d + 1) & 3
                    if not free(x1, y, d1):
                        continue                       # no wall kick: a blocked rotation is a wasted step
                elif key == "left":
                    x1, d1 = x - 1, d
                    if not free(x1, y, d1):
                        continue
                elif key == "right":
                    x1, d1 = x + 1, d
                    if not free(x1, y, d1):
                        continue
                else:
                    x1, d1 = x, d
                if free(x1, y + 1, d1):
                    s1 = (x1, y + 1, d1)
                    out.append((key, s1, None))
                    if s1 not in parent:
                        parent[s1] = (s, key); order.append(s1); q.append(s1)
                else:                                  # gravity fails right after the key: the piece locks
                    pk1 = pkey(x1, y, d1)
                    out.append((key, None, pk1))
                    if pk1 not in terminals:
                        terminals[pk1] = (s, key, (x1, y, d1))
            trans[s] = out

        # score every placement on the board it produces
        scored = []
        for pk, (s, key, lock) in terminals.items():
            scored.append((self._evaluate(rows, kind, lock, next_kind, nx, ny), pk))
        best = max(sc for sc, _ in scored)
        tied = [pk for sc, pk in scored if sc >= best - TIE]

        def path(pk):
            s, key, _ = terminals[pk]; keys = [key]; states = [s]
            while parent[s] is not None:
                s, k = parent[s]; keys.append(k); states.append(s)
            keys.reverse(); states.reverse()
            return list(zip(states, keys))

        plans = [path(pk) for pk in tied]
        # follow the tied plan whose first key has the lowest action index (so argmax of the soft target is
        # the planned key), shortest plan among those
        plans_pk = sorted(zip(plans, tied), key=lambda t: (self.idx[t[0][0][1]], len(t[0])))
        self.plan, self.target = plans_pk[0][0], plans_pk[0][1]
        self.first_keys = tuple(sorted({p[0][1] for p in plans}, key=self.idx.__getitem__))
        # backward reachability of the chosen placement: every transition goes one row down, so states in
        # reverse BFS order see their successors first
        target = self.target
        for s in reversed(order):
            ok = drop_key[s] == target
            if not ok:
                for key, s1, pk1 in trans[s]:
                    if (s1 is not None and reach.get(s1)) or pk1 == target:
                        ok = True; break
            reach[s] = ok

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def _evaluate(rows, kind, lock, next_kind, nx, ny) -> float:
        x, y, d = lock; sh = SHAPES[kind][d]; full = (1 << nx) - 1
        new = list(rows); pm = [0] * ny
        for dy, m in sh.rows:
            mm = _shift(m, x); new[y + dy] |= mm; pm[y + dy] = mm
        landing = (ny - (y + sh.maxy)) + (sh.maxy - sh.miny) / 2.0    # height of the piece's centre row
        # line clearing exactly as the game does it: rows ny-1 down to 1, rechecking after each removal
        cleared = 0; eroded_cells = 0; yy = ny - 1
        while yy >= 1:
            if new[yy] == full:
                cleared += 1; eroded_cells += pm[yy].bit_count()
                del new[yy]; new.insert(0, 0); del pm[yy]; pm.insert(0, 0)
            else:
                yy -= 1
        eroded = cleared * eroded_cells
        top = 0
        while top < ny and new[top] == 0:
            top += 1
        row_tr = col_tr = holes = wells = 0
        if top < ny:
            wall_mask = (1 << (nx + 1)) - 1; hi = 1 << (nx - 1); covered = 0; prev = None
            for yy in range(top, ny):
                r = new[yy]
                ext = (r << 1) | 1 | (1 << (nx + 1))                     # walls filled
                row_tr += ((ext ^ (ext >> 1)) & wall_mask).bit_count()
                holes += (covered & ~r & full).bit_count()
                covered |= r
                if prev is not None:
                    col_tr += (prev ^ r).bit_count()
                prev = r
            col_tr += new[top].bit_count() + (new[ny - 1] ^ full).bit_count()   # empty above, floor below
            # cumulative wells: for every empty cell with both side neighbours filled (walls count), add the
            # run of empty cells from it downwards
            run = [0] * nx
            for yy in range(ny - 1, top - 1, -1):
                r = new[yy]; empty = ~r & full
                run = [run[i] + 1 if (empty >> i) & 1 else 0 for i in range(nx)]
                well = empty & ((r << 1) | 1) & ((r >> 1) | hi)
                while well:
                    low = well & -well; wells += run[low.bit_length() - 1]; well ^= low
        score = (W_LANDING * landing + W_ERODED * eroded + W_ROW_TR * row_tr + W_COL_TR * col_tr
                 + W_HOLES * holes + W_WELLS * wells)
        # the next piece spawns at dir 0, y 0, x = round(uniform(0, nx - size)): ends carry half weight
        if next_kind in SHAPES:
            size = PIECES[next_kind][0]; m = nx - size; sh0 = SHAPES[next_kind][0]; p_death = 0.0
            for sx in range(m + 1):
                if any(new[dy] & (mm << sx) for dy, mm in sh0.rows):
                    p_death += (0.5 if sx in (0, m) else 1.0) / m
            score -= DEATH * p_death
        return score
