"""Mario teacher (Infinite Mario HTML5, games/mario). Rule based and right-going: the default is
`right run`; jumps are taken over gaps, steps, pipes and enemies; Mario waits (noop) in front of a pipe
whose piranha plant is out, and backs off (left) only when nothing else survives.

The rules are checked with a short forward model of Mario's own physics (a Python copy of
Character.Move / SubMove from code/character.js: run and walk acceleration, the 7-tick held jump,
gravity, tile collision against the 21x13 window from info(), wall sliding and the wall jump, stomping)
plus a simple model of the nearby sprites (walkers at 1.75 px/tick, piranhas that rise 59 px once
Mario is more than 24 px away; walkers fall off ledges, red koopas turn at cliffs, winged ones hop).
A plan is one of a few fixed key sequences (run; run+jump held 1, 2 or 3 steps; run 1 to 3 steps then
jump; walk; walk+jump; jump in place; wait; back off), simulated for at most 30 ticks or until Mario has
passed a target 160 px ahead and landed inside the known tile window. Plans are ranked (reached the
target, cleared the first hazard, alive, distance) and ties go to the earlier plan in the list, so the
teacher runs whenever running is safe.

Hidden state (JumpTime, Sliding, MayJump, facing) is not in info(); the teacher keeps a mirror by
replaying its previous action through the model and adopting the mirror when it lands on the observed
x, y, ya, onGround (it does for nearly every step; when collect() took an epsilon-random action the
other six actions are tried). When nothing matches (a stomp bounce, a hit) the hidden state is inferred
from xa, ya and the tiles.

Soft targets (task spec): 0.8 on the chosen action, 0.2 spread over the other right-going actions whose
one-step plan does not die, zero on the rest. noop / left only get mass when chosen.

info() fields used: x, y, xa, ya, onGround, large, ticks, exitX, tiles{originX, originY, rows},
sprites[{kind, dx, dy}]. Rows of the window above the level (abs row < 0) and below it (abs row >= 15)
are drawn as '#' by the hook but are empty in the game (falling below row 15 kills), so they are
treated as empty here."""
from __future__ import annotations
from . import Teacher

LEVEL_ROWS = 15            # LevelGenerator(320, 15)
CELL = 16
BLOCK_ALL, BLOCK_UPPER = 2, 1
JUMP_YSPEED = -1.9
STOMPABLE = {"goomba", "redKoopa", "greenKoopa", "shell", "bullet"}
# 'spiky' is what the hook reports for a piranha plant (FlowerEnemy inherits Enemy with Type=Spiky and the
# hook tests `instanceof Mario.Enemy` first); real spikies never appear at the difficulties reachable
# from world 1. Both are unstompable, so they are treated alike.
PIRANHA = {"piranha", "spiky"}
HAZARDS = STOMPABLE | PIRANHA
ENEMY_HEIGHT = {"redKoopa": 24, "greenKoopa": 24}       # everything else is 12 px tall
ENEMY_HALF_W = {"piranha": 2, "spiky": 2}                # Enemy.Width; flowers are narrower

# plan name -> (step actions, continuation action). Preference order = list order.
PLANS = [
    ("RR", ["right run"], "right run"),
    ("RRJ1", ["right run jump"], "right run"),
    ("RRJ2", ["right run jump"] * 2, "right run"),
    ("RRJ3", ["right run jump"] * 3, "right run"),
    ("RR1J", ["right run"] + ["right run jump"] * 3, "right run"),      # approach, then jump (pit edges)
    ("RR2J", ["right run"] * 2 + ["right run jump"] * 3, "right run"),
    ("RR3J", ["right run"] * 3 + ["right run jump"] * 3, "right run"),
    ("R", ["right"], "right"),
    ("RJ1", ["right jump"], "right"),
    ("RJ2", ["right jump"] * 2, "right"),
    ("RJ3", ["right jump"] * 3, "right"),
    ("J3", ["jump"] * 3, "right run"),
    ("WAIT", ["noop"], "noop"),
    ("BACK", ["left"], "noop"),
]
PLAN_INDEX = {p[0]: i for i, p in enumerate(PLANS)}
ONE_STEP = {"right": "R", "right jump": "RJ1", "right run": "RR", "right run jump": "RRJ1"}
KEYS = {  # action name -> (left, right, run, jump)
    "noop": (0, 0, 0, 0), "left": (1, 0, 0, 0), "right": (0, 1, 0, 0), "jump": (0, 0, 0, 1),
    "right jump": (0, 1, 0, 1), "right run": (0, 1, 1, 0), "right run jump": (0, 1, 1, 1),
}
RIGHT_GOING = ["right", "right jump", "right run", "right run jump"]
T_MAX = 30           # ticks per rollout (10 steps)
TARGET_DX = 160      # px ahead a plan must reach alive, and then land, to count as a success
REACTIVE_CONT = True   # let the rollout's continuation hop over walkers ahead (off: plain run continuation)
WALL_JUMP_XA = (4.272, 5.34, 6.408, 5.874, 4.806)   # |Xa| one tick into a wall jump, per key combination


class _World:
    """Tile window as a flat behaviour grid plus the tracked sprites, in absolute pixel coordinates."""
    __slots__ = ("ox", "oy", "beh", "right_edge", "enemies", "exit_x")

    def __init__(self, info, tracks):
        t = info["tiles"]; rows = t["rows"]
        self.ox, self.oy = t["originX"] // CELL, t["originY"] // CELL
        beh = []
        for r, row in enumerate(rows):
            ay = self.oy + r
            if ay < 0 or ay >= LEVEL_ROWS:
                beh.extend([0] * len(row)); continue
            for ch in row:
                beh.append(BLOCK_ALL if ch in "#?" else BLOCK_UPPER if ch == "^" else 0)
        self.beh = beh
        self.right_edge = (self.ox + len(rows[0]) - 1) * CELL     # start of the last known column
        self.exit_x = info.get("exitX") or 10 ** 9
        self.enemies = tracks

    def blocking(self, tx, ty, ya):
        c, r = tx - self.ox, ty - self.oy
        if c < 0 or c > 20 or r < 0 or r > 12:
            return False
        b = self.beh[r * 21 + c]
        return b == BLOCK_ALL or (ya > 0 and b == BLOCK_UPPER)

    def pipe_top_y(self, ex, ey):
        """Top surface (px) of the first solid tile at or below a plant's head in its column."""
        tx, ty = int(ex // CELL), int((ey - 12) // CELL)
        for ay in range(ty, ty + 6):
            if self.blocking(tx, ay, 0):
                return ay * CELL
        return None


class _Mario:
    __slots__ = ("X", "Y", "Xa", "Ya", "W", "H", "on_ground", "was_on_ground", "may_jump", "sliding",
                 "jump_time", "xjs", "yjs", "facing", "dead", "won", "world")

    def __init__(self, info, world, hidden, jump_held, facing, large, pushing_right):
        self.X, self.Y, self.Xa, self.Ya = info["x"], info["y"], info["xa"], info["ya"]
        self.W, self.H = 4, 24 if large else 12
        self.on_ground = self.was_on_ground = bool(info["onGround"])
        self.jump_time, self.xjs, self.yjs = 0, 0.0, JUMP_YSPEED
        self.facing = facing; self.dead = self.won = False; self.world = world
        if hidden is not None:      # mirror state replayed from the previous step
            (self.jump_time, self.xjs, self.yjs, self.sliding, self.may_jump, self.facing, self.was_on_ground) = hidden
            return
        # --- no mirror: infer the hidden variables from what info() shows ---
        # remaining held-jump ticks: with S held, ya after a tick is jt*(-1.9)*0.85+3 (jt before decrement)
        if jump_held and not self.on_ground:
            q = (3 - self.Ya) / (0.85 * -JUMP_YSPEED)
            if abs(q - round(q)) < 0.2 and 1 <= round(q) <= 8:
                self.jump_time = int(round(q)) - 1
        # Character.Sliding after a tick: the right key was pushed into a wall that blocks head, middle and
        # feet (SubMove sets it whatever the vertical motion); Xa is 0 after such a collision.
        X, Y, W, H = self.X, self.Y, self.W, self.H
        tx = int((X + W + 1.2) / CELL)
        wall = world.blocking(tx, int((Y - H) / CELL), 0) and world.blocking(tx, int((Y - H // 2) / CELL), 0) \
            and world.blocking(tx, int(Y / CELL), 0)
        self.sliding = (not self.on_ground) and self.Xa == 0 and pushing_right and wall
        self.may_jump = (self.on_ground or self.sliding) and not jump_held
        # a wall jump in progress (JumpTime < 0) keeps pushing Mario away from the wall for 6 ticks
        if not self.on_ground and self.Ya < 3 and self.Xa != 0:
            k = (3 - self.Ya) / (0.85 * 2)
            ax = abs(self.Xa)
            if abs(k - round(k)) < 0.15 and 1 <= round(k) <= 6 and any(abs(ax - v) < 0.12 for v in WALL_JUMP_XA):
                self.xjs = 6 if self.Xa > 0 else -6; self.yjs = -2; self.jump_time = -(int(round(k)) - 1)

    def hidden(self):
        return (self.jump_time, self.xjs, self.yjs, self.sliding, self.may_jump, self.facing, self.was_on_ground)

    def is_blocking(self, x, y, ya):
        tx, ty = int(x / CELL), int(y / CELL)         # JS `| 0` truncates toward zero
        if tx == int(self.X / CELL) and ty == int(self.Y / CELL):
            return False
        return self.world.blocking(tx, ty, ya)

    def sub_move(self, xa, ya):
        while xa > 8:
            if not self.sub_move(8, 0): return False
            xa -= 8
        while xa < -8:
            if not self.sub_move(-8, 0): return False
            xa += 8
        while ya > 8:
            if not self.sub_move(0, 8): return False
            ya -= 8
        while ya < -8:
            if not self.sub_move(0, -8): return False
            ya += 8
        X, Y, W, H, blk = self.X, self.Y, self.W, self.H, self.is_blocking
        collide = False
        if ya > 0:
            if blk(X + xa - W, Y + ya, 0) or blk(X + xa + W, Y + ya, 0) or \
               blk(X + xa - W, Y + ya + 1, ya) or blk(X + xa + W, Y + ya + 1, ya):
                collide = True
        if ya < 0:
            if blk(X + xa, Y + ya - H, ya) or blk(X + xa - W, Y + ya - H, ya) or blk(X + xa + W, Y + ya - H, ya):
                collide = True
        if xa > 0:
            b1 = blk(X + xa + W, Y + ya - H, ya); b2 = blk(X + xa + W, Y + ya - (H // 2), ya); b3 = blk(X + xa + W, Y + ya, ya)
            if b1 or b2 or b3: collide = True
            self.sliding = bool(b1 and b2 and b3)
        if xa < 0:
            b1 = blk(X + xa - W, Y + ya - H, ya); b2 = blk(X + xa - W, Y + ya - (H // 2), ya); b3 = blk(X + xa - W, Y + ya, ya)
            if b1 or b2 or b3: collide = True
            self.sliding = bool(b1 and b2 and b3)
        if collide:
            if xa < 0: self.X = int((X - W) / CELL) * CELL + W; self.Xa = 0
            if xa > 0: self.X = int((X + W) / CELL + 1) * CELL - W - 1; self.Xa = 0
            if ya < 0: self.Y = int((Y - H) / CELL) * CELL + H; self.jump_time = 0; self.Ya = 0
            if ya > 0: self.Y = int((Y - 1) / CELL + 1) * CELL - 1; self.on_ground = True
            return False
        self.X += xa; self.Y += ya
        return True

    def move(self, left, right, run, jump):
        """One game tick of Character.Move (no ducking, no fireballs, no carried shell)."""
        self.was_on_ground = self.on_ground
        side = 1.2 if run else 0.6
        if self.Xa > 2: self.facing = 1
        if self.Xa < -2: self.facing = -1
        if jump or (self.jump_time < 0 and not self.on_ground and not self.sliding):
            if self.jump_time < 0:
                self.Xa = self.xjs; self.Ya = -self.jump_time * self.yjs; self.jump_time += 1
            elif self.on_ground and self.may_jump:
                self.xjs = 0; self.yjs = JUMP_YSPEED; self.jump_time = 7; self.Ya = 7 * JUMP_YSPEED
                self.on_ground = False; self.sliding = False
            elif self.sliding and self.may_jump:
                self.xjs = -self.facing * 6; self.yjs = -2; self.jump_time = -6
                self.Xa = self.xjs; self.Ya = -self.jump_time * self.yjs
                self.on_ground = False; self.sliding = False; self.facing = -self.facing
            elif self.jump_time > 0:
                self.Xa += self.xjs; self.Ya = self.jump_time * self.yjs; self.jump_time -= 1
        else:
            self.jump_time = 0
        if left:
            if self.facing == 1: self.sliding = False
            self.Xa -= side
            if self.jump_time >= 0: self.facing = -1
        if right:
            if self.facing == -1: self.sliding = False
            self.Xa += side
            if self.jump_time >= 0: self.facing = 1
        if (not left and not right) or self.Ya < 0 or self.on_ground:
            self.sliding = False
        self.may_jump = (self.on_ground or self.sliding) and not jump
        if abs(self.Xa) < 0.5: self.Xa = 0
        if self.sliding: self.Ya *= 0.5
        self.on_ground = False
        self.sub_move(self.Xa, 0)
        self.sub_move(0, self.Ya)
        if self.Y > LEVEL_ROWS * CELL + CELL: self.dead = True
        if self.X < 0: self.X = 0; self.Xa = 0
        if self.X > self.world.exit_x: self.won = True
        self.Ya *= 0.85
        self.Xa *= 0.89
        if not self.on_ground: self.Ya += 3

    def stomp(self, ey, eh):
        self.sub_move(0, (ey - eh / 2) - self.Y)
        self.xjs = 0; self.yjs = JUMP_YSPEED; self.jump_time = 8; self.Ya = 8 * JUMP_YSPEED
        self.on_ground = False; self.sliding = False


def _step(m, action, held, ens, mh, on_tick=None):
    """One env step (the hook's applyAction): an extra tick with the jump key up when Mario is grounded or
    sliding with the key still held from the previous step, then 3 ticks with the action's keys."""
    left, right, run, jump = KEYS[action]
    if jump and held and (m.on_ground or m.sliding) and not m.may_jump:
        m.move(left, right, run, 0)
        _enemies_tick(m, ens, mh)
        if on_tick and (on_tick() or m.dead): return bool(jump)
    for _ in range(3):
        m.move(left, right, run, jump)
        _enemies_tick(m, ens, mh)
        if m.dead or (on_tick and on_tick()): break
    return bool(jump)


class _Enemy:
    """A walker (goomba, koopa, moving shell) or a piranha plant, with the physics of enemy.js: 1.75 px/tick
    towards its facing, turning at walls, red koopas turning at cliffs, gravity 2 (drag 0.85) and, for winged
    ones, hops of Ya=-10 with gravity 0.6 (drag 0.95). Plants sit in their pipe and rise (Ya=-8, drag 0.9,
    gravity 0.1) once they have rested 40 ticks and Mario is more than 24 px away."""
    __slots__ = ("kind", "X", "Y", "Xa", "Ya", "W", "H", "alive", "stompable", "plant", "rest_y", "bottom",
                 "winged", "on_ground", "facing", "world", "cliffs")

    def __init__(self, t, world):
        self.kind = t["kind"]; self.X, self.Y = t["ax"], t["ay"]; self.world = world
        self.W = ENEMY_HALF_W.get(self.kind, 4); self.H = ENEMY_HEIGHT.get(self.kind, 12)
        self.alive = True; self.stompable = self.kind in STOMPABLE; self.plant = self.kind in PIRANHA
        self.winged = t.get("winged", False); self.cliffs = self.kind == "redKoopa"
        self.bottom = t["bottom"]; self.Ya = t["vy"]
        if self.plant:
            top = world.pipe_top_y(self.X, self.Y)
            self.rest_y = (top + 24) if top is not None else self.Y
            self.Xa = 0.0; self.facing = 0
        else:
            self.rest_y = None
            self.facing = 1 if t["vx"] > 0 else -1
            self.Xa = self.facing * (11.0 if self.kind == "shell" else 1.75)
        self.on_ground = (not self.plant) and abs(self.Ya) < 0.5 and self._blocked_below()

    def _blk(self, x, y, ya):
        tx, ty = int(x / CELL), int(y / CELL)
        if tx == int(self.X / CELL) and ty == int(self.Y / CELL):
            return False
        return self.world.blocking(tx, ty, ya)

    def _blocked_below(self):
        return self._blk(self.X - self.W, self.Y + 1, 1) or self._blk(self.X + self.W, self.Y + 1, 1)

    def tick(self, mx):
        if self.plant:
            if self.Y >= self.rest_y - 0.01 and self.Ya >= 0:          # sitting in the pipe
                self.Y = self.rest_y; self.bottom += 1
                self.Ya = -8 if (self.bottom > 40 and abs(mx - self.X) > 24) else 0
            self.Y += self.Ya; self.Ya = self.Ya * 0.9 + 0.1
            return
        X, Y, W, H, blk = self.X, self.Y, self.W, self.H, self._blk
        xa = self.Xa
        side = X + xa + (W if xa > 0 else -W)
        blocked = blk(side, Y - H, 0) or blk(side, Y - H // 2, 0) or blk(side, Y, 0)
        if self.cliffs and self.on_ground and not self.world.blocking(int(side / CELL), int(Y / CELL) + 1, 1):
            blocked = True
        if blocked:
            self.facing = -self.facing; self.Xa = -xa
        else:
            self.X += xa
        X = self.X
        self.on_ground = False
        ya = self.Ya
        while ya > 8:
            if blk(X - W, Y + 8, 8) or blk(X + W, Y + 8, 8) or blk(X - W, Y + 9, 8) or blk(X + W, Y + 9, 8):
                self.Y = int((Y - 1) / CELL + 1) * CELL - 1; self.on_ground = True; ya = 0; break
            Y += 8; ya -= 8
        if ya > 0:
            if blk(X - W, Y + ya, 0) or blk(X + W, Y + ya, 0) or blk(X - W, Y + ya + 1, ya) or blk(X + W, Y + ya + 1, ya):
                Y = int((Y - 1) / CELL + 1) * CELL - 1; self.on_ground = True
            else:
                Y += ya
        elif ya < 0:
            if blk(X, Y + ya - H, ya) or blk(X - W, Y + ya - H, ya) or blk(X + W, Y + ya - H, ya):
                self.Ya = 0
            else:
                Y += ya
        self.Y = Y
        self.Ya *= 0.95 if self.winged else 0.85
        if not self.on_ground:
            self.Ya += 0.6 if self.winged else 2
        elif self.winged:
            self.Ya = -10


def _enemies_tick(m, ens, mh):
    mx, my = m.X, m.Y
    for e in ens:
        if not e.alive: continue
        e.tick(mx)
        xd, yd = mx - e.X, my - e.Y
        lim = e.W * 2 + 4
        if -lim < xd < lim and -e.H < yd < mh:
            if e.stompable and m.Ya > 0 and yd <= 0 and (not m.on_ground or not m.was_on_ground):
                m.stomp(e.Y, e.H); e.alive = False
            else:
                m.dead = True; return


class MarioTeacher(Teacher):
    def reset(self):
        self.prev_jump = False       # did the previous step hold the jump key (our argmax, or the matched action)
        self.prev_right = False
        self.prev_ticks = None
        self.tracks = []             # sprites seen last step: dicts with kind, ax, ay, vx, vy, bottom
        self.facing = 1
        self.stall = 0; self.prev_x = None
        self.prev_info = None; self.prev_world = None; self.prev_action = None; self.prev_hidden = None
        self.pp_jump = False; self.pp_right = False
        self.mirror_hits = 0; self.mirror_misses = 0

    # ---- sprite tracking -------------------------------------------------------------------
    def _track(self, info, dt):
        x, y = info["x"], info["y"]; new = []
        for s in info.get("sprites") or []:
            kind = s["kind"]
            if kind not in HAZARDS: continue
            ax, ay = x + s["dx"], y + s["dy"]
            best = None
            for t in self.tracks:
                if t["kind"] != kind or t.get("used"): continue
                d = abs(t["ax"] - ax) + abs(t["ay"] - ay)
                if d <= 12 * dt + 8 and (best is None or d < best[0]): best = (d, t)
            winged = False
            if best:
                t = best[1]; t["used"] = True
                vx, vy = (ax - t["ax"]) / dt, (ay - t["ay"]) / dt
                bottom = t["bottom"] + dt if (kind in PIRANHA and abs(vy) < 0.3) else 0
                winged = t["winged"] or (kind not in PIRANHA and vy < -2.5)     # only winged walkers ever rise
                if abs(vx) < 0.3 and kind not in PIRANHA and kind != "shell": vx = t["vx"]  # blocked this step: keep heading
            else:
                vx = -1.75 if ax > x else 1.75          # spawned walking towards Mario
                vy = 0.0; bottom = 99                   # unknown history: may pop up any time
            if kind in PIRANHA: vx = 0.0
            new.append({"kind": kind, "ax": ax, "ay": ay, "vx": vx, "vy": vy, "bottom": bottom, "winged": winged})
        self.tracks = new
        return new

    def _enemy_copies(self, world):
        return [_Enemy(t, world) for t in world.enemies if not (t["kind"] == "shell" and abs(t["vx"]) < 2)]

    # ---- mirror of the hidden state ---------------------------------------------------------
    def _recover_hidden(self, info, world):
        """Replay the previous step from the previous observation; adopt the model's hidden variables if it
        lands on the observed state. Tries our own action first, then the others (collect() may have taken
        an epsilon-random action). None when nothing matches (stomp bounce, hit, first step)."""
        if self.prev_info is None:
            return None
        pi, pw, large = self.prev_info, self.prev_world, bool(self.prev_info.get("large"))
        for a in [self.prev_action] + [k for k in RIGHT_GOING if k != self.prev_action] + [k for k in KEYS if k not in RIGHT_GOING and k != self.prev_action]:
            m = _Mario(pi, pw, self.prev_hidden, self.pp_jump, self.facing, large, self.pp_right)
            _step(m, a, self.pp_jump, [], m.H)
            if abs(m.X - info["x"]) < 0.6 and abs(m.Y - info["y"]) < 0.6 and abs(m.Ya - info["ya"]) < 0.3 \
                    and m.on_ground == bool(info["onGround"]):
                if a != self.prev_action:       # the env took another action: the key history is that one
                    self.prev_jump = bool(KEYS[a][3]); self.prev_right = bool(KEYS[a][1])
                self.mirror_hits += 1
                return m.hidden()
        self.mirror_misses += 1
        return None

    # ---- one plan --------------------------------------------------------------------------
    def _rollout(self, info, world, steps, cont, hidden, large):
        m = _Mario(info, world, hidden, self.prev_jump, self.facing, large, self.prev_right)
        mh = m.H; ens = self._enemy_copies(world)
        target = info["x"] + TARGET_DX
        state = {"ticks": 0, "x_alive": m.X, "reached": False}

        def on_tick():
            state["ticks"] += 1
            if m.dead: return True
            if m.X > state["x_alive"]: state["x_alive"] = m.X
            if m.won or m.X >= target: state["reached"] = True
            return state["reached"] and (m.on_ground or m.won or m.X >= world.right_edge)

        held = self.prev_jump; si = 0; left_window = False
        while state["ticks"] < T_MAX:
            if si < len(steps):
                act = steps[si]
            else:
                act = cont
                # the continuation re-plans like the teacher would: hop when a walker is close ahead at foot level
                if REACTIVE_CONT and cont in ("right run", "right") and m.on_ground and any(
                        e.alive and not e.plant and 0 < e.X - m.X < 40 and abs(e.Y - m.Y) < 20 for e in ens):
                    act = cont + " jump"
            si += 1
            held = _step(m, act, held, ens, mh, on_tick)
            if m.X >= world.right_edge and not m.on_ground and not m.won: left_window = True
            if m.dead or (state["reached"] and (m.on_ground or m.won or m.X >= world.right_edge)): break
        # landing beyond the known tiles is not a success: the ground there is unknown
        success = state["reached"] and not m.dead and not left_window
        return {"dead": m.dead, "success": success, "x_alive": state["x_alive"], "x_end": m.X}

    # ---- decision ---------------------------------------------------------------------------
    def act(self, obs: dict) -> list[float]:
        info = obs["info"]; n = len(self.actions)
        probs = [0.0] * n
        if not info or not info.get("tiles") or info.get("x") is None:
            probs[self.idx["right run"]] = 1.0; return probs
        ticks = info.get("ticks", 0)
        dt = max(1, min(4, ticks - self.prev_ticks)) if self.prev_ticks is not None else 3
        self.prev_ticks = ticks
        tracks = self._track(info, dt)
        world = _World(info, tracks)
        x, on_ground, large = info["x"], info["onGround"], bool(info.get("large"))
        hidden = self._recover_hidden(info, world)
        if hidden is not None: self.facing = hidden[5]
        elif info["xa"] > 0.5: self.facing = 1
        elif info["xa"] < -0.5: self.facing = -1
        # stall guard: standing still while pushing right for 8 steps means the model is missing
        # something; force a full jump
        if self.prev_x is not None and abs(x - self.prev_x) < 1 and self.prev_right:
            self.stall += 1
        else:
            self.stall = 0
        self.prev_x = x

        results = {}; first_hazard = None
        for name, steps, cont in PLANS:
            r = self._rollout(info, world, steps, cont, hidden, large)
            results[name] = r
            if name == "RR":
                if r["dead"]: first_hazard = r["x_end"]
                elif not r["success"]: first_hazard = r["x_alive"]      # stuck at a wall
                if r["success"]: break                                   # running is safe
        for name in ("RRJ1", "R", "RJ1"):                                # alternatives for the soft target
            if name not in results:
                _, steps, cont = PLANS[PLAN_INDEX[name]]
                results[name] = self._rollout(info, world, steps, cont, hidden, large)

        def rank(item):
            # success first; otherwise a plan that clears the first hazard (even if the rollout dies at a later
            # one: we re-plan after clearing) beats one that does not; then alive, distance, list order
            name, r = item
            cleared = first_hazard is not None and r["x_alive"] > first_hazard + 12
            return (r["success"], cleared, not r["dead"], r["x_alive"], -PLAN_INDEX[name])

        best = max(results.items(), key=rank)[0]
        self.last = (best, first_hazard, results)      # for debugging tools
        chosen = PLANS[PLAN_INDEX[best]][1][0]
        if self.stall >= 8 and on_ground:
            chosen = "right run jump"; self.stall = 0
        probs[self.idx[chosen]] = 0.8
        others = [a for a in RIGHT_GOING if a != chosen and ONE_STEP[a] in results and not results[ONE_STEP[a]]["dead"]]
        if others:
            for a in others: probs[self.idx[a]] += 0.2 / len(others)
        else:
            probs[self.idx[chosen]] = 1.0
        # remember for the next step's mirror replay
        self.pp_jump, self.pp_right = self.prev_jump, self.prev_right
        self.prev_jump = bool(KEYS[chosen][3]); self.prev_right = bool(KEYS[chosen][1])
        self.prev_info, self.prev_world, self.prev_action, self.prev_hidden = info, world, chosen, hidden
        s = sum(probs)
        return [p / s for p in probs]

