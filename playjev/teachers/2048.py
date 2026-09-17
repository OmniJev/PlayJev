"""2048 teacher: depth-2 expectimax (player move, chance node over every empty cell with a 2 at 0.9 and a 4 at 0.1,
player move) with a table heuristic on the leaves; boards with DEEP_EMPTY or fewer empty cells get one extra ply
(small branching there, and that is where depth 2 blunders). Soft targets: softmax over the four root move values
with a fixed temperature; a move that leaves the board unchanged gets zero mass. See games/2048/TEACHER.md.

info() from games/2048/pj_hook.js: grid[row][col] with tile values (0 = empty), score, over, won, empty.

Board representation: one Python int of 64 bits, 16 nibbles holding tile exponents (0 empty, k for 2**k). Row r
sits in bits [16r, 16r+16), column c of that row in bits [4c, 4c+4), so grid[r][c] is nibble 4r+c. Every row move
is one table lookup per row (65536-entry lists built at import); up and down go through a bit-twiddled transpose
and column-layout tables. The heuristic is also per row, so a leaf costs eight lookups plus one transpose.

Heuristic per line of four exponents (the well known nneonneo weights, summed over the 4 rows and 4 columns):
  +EMPTY 270 per empty cell, +MERGES 700 per available merge (adjacent equal tiles), -MONO 47 * min(sum of
  rank**4 drops going left, going right) so a line prefers to be monotone either way, -SUM 11 * sum(rank**3.5)
  which penalises many mid-sized tiles and so rewards merging, +LOST 200000 per line as a floor so that a dead board
  (value 0) is far below any live board. Monotone rows and columns together push the big tile into a corner."""
from __future__ import annotations
import math
from . import Teacher

EMPTY_W, MERGES_W, MONO_W, MONO_P, SUM_W, SUM_P, LOST = 270.0, 700.0, 47.0, 4.0, 11.0, 3.5, 200000.0
TEMPERATURE = 1200.0  # softmax temperature on root move values (see TEACHER.md for the calibration)
DEPTH = 2             # player moves searched: move, spawn, move, heuristic
DEEP_EMPTY = 4        # boards with this many empty cells or fewer get one extra ply (small branching, most blunders)
ROW_MASK = 0xFFFF
COL_MASK = 0x000F000F000F000F


def _reverse(row: int) -> int:
    return ((row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0x0F00) | ((row << 12) & 0xF000)) & ROW_MASK


def _unpack_col(row: int) -> int:
    """Spread the four nibbles of a row to bits 0, 16, 32, 48 (column 0 of each row)."""
    return (row | (row << 12) | (row << 24) | (row << 36)) & COL_MASK


def _slide_left(cells: list[int]) -> list[int]:
    out = [c for c in cells if c]; res = []; i = 0
    while i < len(out):
        if i + 1 < len(out) and out[i] == out[i + 1]:
            res.append(min(out[i] + 1, 15)); i += 2
        else:
            res.append(out[i]); i += 1
    return res + [0] * (4 - len(res))


def _heur(line: list[int]) -> float:
    s = 0.0; empty = 0; merges = 0; prev = 0; counter = 0
    for rank in line:
        s += rank ** SUM_P
        if rank == 0:
            empty += 1
        else:
            if prev == rank:
                counter += 1
            elif counter > 0:
                merges += 1 + counter; counter = 0
            prev = rank
    if counter > 0:
        merges += 1 + counter
    mono_l = mono_r = 0.0
    for i in range(1, 4):
        a, b = line[i - 1], line[i]
        if a > b:
            mono_l += a ** MONO_P - b ** MONO_P
        else:
            mono_r += b ** MONO_P - a ** MONO_P
    return LOST + EMPTY_W * empty + MERGES_W * merges - MONO_W * min(mono_l, mono_r) - SUM_W * s


def _build_tables():
    left = [0] * 65536; right = [0] * 65536; col_up = [0] * 65536; col_down = [0] * 65536; heur = [0.0] * 65536
    for row in range(65536):
        cells = [(row >> 0) & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF]
        moved = _slide_left(cells)
        l = moved[0] | (moved[1] << 4) | (moved[2] << 8) | (moved[3] << 12)
        left[row] = l
        col_up[row] = _unpack_col(l)  # column c of the transposed board moved toward row 0 (up)
        heur[row] = _heur(cells)
    for row in range(65536):
        r = _reverse(left[_reverse(row)])
        right[row] = r
        col_down[row] = _unpack_col(r)
    return left, right, col_up, col_down, heur


LEFT, RIGHT, COL_UP, COL_DOWN, HEUR = _build_tables()


def transpose(x: int) -> int:
    a1 = x & 0xF0F00F0FF0F00F0F
    a2 = x & 0x0000F0F00000F0F0
    a3 = x & 0x0F0F00000F0F0000
    a = a1 | (a2 << 12) | (a3 >> 12)
    b1 = a & 0xFF00FF0000FF00FF
    b2 = a & 0x00FF00FF00000000
    b3 = a & 0x00000000FF00FF00
    return b1 | (b2 >> 24) | (b3 << 24)


def move_left(b: int, L=LEFT) -> int:
    return L[b & 0xFFFF] | (L[(b >> 16) & 0xFFFF] << 16) | (L[(b >> 32) & 0xFFFF] << 32) | (L[b >> 48] << 48)


def move_right(b: int, R=RIGHT) -> int:
    return R[b & 0xFFFF] | (R[(b >> 16) & 0xFFFF] << 16) | (R[(b >> 32) & 0xFFFF] << 32) | (R[b >> 48] << 48)


def move_up_t(t: int, U=COL_UP) -> int:
    """Move up given the TRANSPOSED board t; returns the result in normal layout."""
    return U[t & 0xFFFF] | (U[(t >> 16) & 0xFFFF] << 4) | (U[(t >> 32) & 0xFFFF] << 8) | (U[t >> 48] << 12)


def move_down_t(t: int, D=COL_DOWN) -> int:
    return D[t & 0xFFFF] | (D[(t >> 16) & 0xFFFF] << 4) | (D[(t >> 32) & 0xFFFF] << 8) | (D[t >> 48] << 12)


def heuristic(b: int, H=HEUR) -> float:
    t = transpose(b)
    return (H[b & 0xFFFF] + H[(b >> 16) & 0xFFFF] + H[(b >> 32) & 0xFFFF] + H[b >> 48]
            + H[t & 0xFFFF] + H[(t >> 16) & 0xFFFF] + H[(t >> 32) & 0xFFFF] + H[t >> 48])


def grid_to_board(grid) -> int:
    b = 0; shift = 0
    for row in grid:
        for v in row:
            if v:
                b |= (v.bit_length() - 1) << shift
            shift += 4
    return b


def _best_after_spawn(b: int, H=HEUR) -> float:
    """Player node one ply below the root: best heuristic over the moves that change the board; 0 if none (dead)."""
    best = 0.0
    t = transpose(b)
    nb = move_left(b)
    if nb != b:
        v = heuristic(nb)
        if v > best: best = v
    nb = move_right(b)
    if nb != b:
        v = heuristic(nb)
        if v > best: best = v
    nb = move_up_t(t)
    if nb != b:
        v = heuristic(nb)
        if v > best: best = v
    nb = move_down_t(t)
    if nb != b:
        v = heuristic(nb)
        if v > best: best = v
    return best


def chance_value(b: int, depth: int = 1) -> float:
    """Expected value over the spawn (uniform empty cell, 2 with 0.9 and 4 with 0.1) of the best reply.
    depth is the number of player moves still to search below the spawn (1 = reply then heuristic)."""
    total = 0.0; n = 0; shift = 0; x = b
    if depth <= 1:
        while shift < 64:
            if not (x & 0xF):
                total += 0.9 * _best_after_spawn(b | (1 << shift)) + 0.1 * _best_after_spawn(b | (2 << shift)); n += 1
            x >>= 4; shift += 4
    else:
        while shift < 64:
            if not (x & 0xF):
                total += 0.9 * player_value(b | (1 << shift), depth) + 0.1 * player_value(b | (2 << shift), depth); n += 1
            x >>= 4; shift += 4
    return total / n if n else 0.0


def player_value(b: int, depth: int) -> float:
    """Best over the moves that change the board of the chance value below; 0 if no move changes it (dead)."""
    best = 0.0
    t = transpose(b)
    for nb in (move_left(b), move_right(b), move_up_t(t), move_down_t(t)):
        if nb != b:
            v = chance_value(nb, depth - 1)
            if v > best: best = v
    return best


def count_empty(b: int) -> int:
    n = 0
    for s in range(0, 64, 4):
        if not (b >> s) & 0xF: n += 1
    return n


def root_values(b: int, depth: int = 2) -> dict[str, float | None]:
    """Value of each root move at the given search depth (player moves); None when the move leaves the board as is."""
    t = transpose(b); out = {}
    for name, nb in (("left", move_left(b)), ("right", move_right(b)), ("up", move_up_t(t)), ("down", move_down_t(t))):
        out[name] = None if nb == b else chance_value(nb, depth - 1)
    return out


class G2048Teacher(Teacher):
    def __init__(self, actions, temperature: float = TEMPERATURE, depth: int = DEPTH, deep_empty: int = DEEP_EMPTY):
        super().__init__(actions); self.temperature = temperature; self.depth = depth; self.deep_empty = deep_empty
        self.last_values: dict | None = None  # for inspection / calibration

    def reset(self):
        self.last_values = None

    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; n = len(self.actions)
        b = grid_to_board(info["grid"])
        depth = self.depth + 1 if count_empty(b) <= self.deep_empty else self.depth
        vals = root_values(b, depth); self.last_values = vals
        legal = {k: v for k, v in vals.items() if v is not None}
        probs = [0.0] * n
        if not legal:  # terminal board (over/won): nothing changes anything, keep a valid distribution
            for k in vals: probs[self.idx[k]] = 1.0 / n
            return probs
        vmax = max(legal.values())
        alive = {k: v for k, v in legal.items() if v > 0.0}  # value 0 means every spawn leaves no move: certain loss
        if not alive:
            alive = legal
        T = self.temperature
        w = {k: math.exp((v - vmax) / T) for k, v in alive.items()}
        s = sum(w.values())
        for k, x in w.items(): probs[self.idx[k]] = x / s
        return probs
