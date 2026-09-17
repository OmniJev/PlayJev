"""Determinism and contract check for sokoban (Microban levels, standard push rules).

1. BFS solver over info() (state = player position plus the set of boxes, one move per edge, so the
   solution is move-optimal) for the first 20 Microban levels (seeds 0..19). Each solution is replayed
   through the env: after every move the env's player and boxes must equal the solver's prediction
   (this pins the standard rules: a push moves exactly one box, only onto free floor or a free goal),
   done must be false before the last move and true on it, and the final score must be the number of
   goals the player filled plus the 100 completion bonus.
2. The chain push the vendored game used to allow is refused: on Microban 4 (`# .**$@#`) the first
   move left changes nothing.
3. Two pages, same seed, same random action sequence: score, done and info() agree at every step;
   the same page replayed (reload + reseed) reproduces the trace.
Run from anywhere:  python games/sokoban/check_det.py
"""
import asyncio
import io
import json
import pathlib
import random
import sys
import time
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from PIL import Image, ImageStat  # noqa: E402
from playjev.env import VecGame  # noqa: E402

GAME = "sokoban"
SOLVE_LEVELS = 20          # Microban 1..20, seeds 0..19
DET_SEEDS = [1, 7, 123, 99999]
DET_STEPS = 200
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}   # action index -> (dx, dy)
OUT = pathlib.Path(__file__).resolve().parents[2] / "runs" / "bench" / GAME


def dead_squares(walls, goals, floor):
    """Squares a box can never leave towards a goal (pull every goal backwards over the floor).
    Pushing a box onto one of them can never be part of a solution, so the BFS skips those pushes.
    This keeps the move-optimal answer: it only removes states that have no path to the target."""
    live = set(goals)
    stack = list(goals)
    while stack:
        x, y = stack.pop()
        for dx, dy in DIRS.values():
            box, player = (x - dx, y - dy), (x - 2 * dx, y - 2 * dy)   # box pulled from `box` onto (x, y)
            if box in floor and player in floor and box not in live:
                live.add(box); stack.append(box)
    return floor - live


def solve(info, max_states=3_000_000):
    """BFS over moves from the env's info(). Returns (actions, states) with states[k] the predicted
    (player, boxes) after k moves, or (None, None) when no solution exists within max_states."""
    walls = set(map(tuple, info["walls"]))
    goals = frozenset(map(tuple, info["goals"]))
    width, height = len(info["board"][0]), len(info["board"])
    floor = {(x, y) for y in range(height) for x in range(width) if (x, y) not in walls}
    dead = dead_squares(walls, goals, floor)
    start = (tuple(info["player"]), frozenset(map(tuple, info["boxes"])))
    prev = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        player, boxes = state
        if goals <= boxes:
            actions, states = [], [state]
            while prev[state] is not None:
                state, a = prev[state]
                actions.append(a); states.append(state)
            return actions[::-1], states[::-1]
        for a, (dx, dy) in DIRS.items():
            nxt = (player[0] + dx, player[1] + dy)
            if nxt in walls:
                continue
            new_boxes = boxes
            if nxt in boxes:
                beyond = (nxt[0] + dx, nxt[1] + dy)
                if beyond in walls or beyond in boxes or beyond in dead:
                    continue
                new_boxes = (boxes - {nxt}) | {beyond}
            ns = (nxt, new_boxes)
            if ns not in prev:
                prev[ns] = (state, a); queue.append(ns)
        if len(prev) > max_states:
            return None, None
    return None, None


def env_state(info):
    return tuple(info["player"]), frozenset(map(tuple, info["boxes"]))


async def solve_and_replay(page, level_index):
    o = await page.reset(seed=level_index)
    info = o["info"]
    assert info["levelIndex"] == level_index and info["level"] == level_index + 1, info["level"]
    assert o["done"] is False, "done() must be false right after start()"
    t0 = time.time()
    actions, states = solve(info)
    dt = time.time() - t0
    if actions is None:
        print(f"  level {level_index + 1:3d}: solver gave up ({dt:.1f}s)")
        return False, None
    start_on_goal = info["boxesOnGoal"]
    expected_final = len(info["goals"]) - start_on_goal + 100
    ok, reason = True, ""
    for k, a in enumerate(actions):
        o = await page.step(a)
        if env_state(o["info"]) != states[k + 1]:
            ok, reason = False, f"state mismatch after move {k + 1}: env {env_state(o['info'])} vs solver {states[k + 1]}"
            break
        last = k == len(actions) - 1
        if o["done"] != last:
            ok, reason = False, f"done={o['done']} after move {k + 1} of {len(actions)}"
            break
    if ok and not (o["info"]["solved"] and o["score"] == expected_final):
        ok, reason = False, f"final score {o['score']} (expected {expected_final}), solved={o['info']['solved']}"
    bright = ImageStat.Stat(Image.open(io.BytesIO(o["frame"])).convert("L")).mean[0]
    if ok and bright < 60:
        ok, reason = False, f"final frame is dark (mean {bright:.0f}), the win banner leaked into the frame"
    if level_index == 0:
        OUT.mkdir(parents=True, exist_ok=True); (OUT / "solved.jpg").write_bytes(o["frame"])
    print(f"  level {level_index + 1:3d}: {len(actions):3d} moves, {o['info']['boxesOnGoal']} boxes on goals, "
          f"score {o['score']}, solver {dt:.2f}s -> {'PASS' if ok else 'FAIL ' + reason}")
    return ok, len(actions)


async def rollout(env, page_idx, seed, actions):
    """Play `actions` on one page and return the (score, done, info) trace."""
    p = env.pages[page_idx]
    o = await p.reset(seed=seed)
    assert o["done"] is False, "done() must be false right after start()"
    trace = []
    for a in actions:
        o = await p.step(a)
        trace.append((o["score"], o["done"], json.dumps(o["info"], sort_keys=True)))
        if o["done"]:
            break
    return trace


async def main():
    ok = True
    async with VecGame(GAME, n=2) as env:
        print(f"BFS solutions for Microban 1..{SOLVE_LEVELS}, replayed through the env:")
        lengths = {}
        for idx in range(SOLVE_LEVELS):
            good, n = await solve_and_replay(env.pages[0], idx)
            ok &= good
            if n is not None:
                lengths[idx + 1] = n
        print(f"  solved {len(lengths)}/{SOLVE_LEVELS}; optimal move counts: {lengths}")

        # standard rules: no chain push (Microban 4 starts as `# .**$@#`, player at the right end)
        p = env.pages[0]
        o = await p.reset(seed=3)
        before = env_state(o["info"])
        o = await p.step(LEFT)
        chain_refused = env_state(o["info"]) == before and o["reward"] == 0
        ok &= chain_refused
        print(f"chain push refused on Microban 4: {'PASS' if chain_refused else 'FAIL'} (player {o['info']['player']}, boxes {sorted(o['info']['boxes'])})")

        # determinism: two pages, same seed and actions
        for seed in DET_SEEDS:
            rng = random.Random(seed)
            actions = [rng.randrange(4) for _ in range(DET_STEPS)]
            a = await rollout(env, 0, seed, actions)
            b = await rollout(env, 1, seed, actions)
            same = a == b
            ok &= same
            level = json.loads(a[0][2])["level"]
            print(f"seed {seed} (level {level}): {len(a)} steps, final score {a[-1][0]}, done={a[-1][1]}, "
                  f"identical={'yes' if same else 'NO'}")
            if not same:
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(f"  first divergence at step {i}:\n    A {x}\n    B {y}")
                        break
        # replay on the same page must also reproduce (page reload + reseed)
        rng = random.Random(5)
        actions = [rng.randrange(4) for _ in range(DET_STEPS)]
        a = await rollout(env, 0, 42, actions)
        c = await rollout(env, 0, 42, actions)
        ok &= a == c
        print(f"same page replayed, seed 42: identical={'yes' if a == c else 'NO'}")
    print("DETERMINISM:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
