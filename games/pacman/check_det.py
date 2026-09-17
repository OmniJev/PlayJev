"""Determinism check for pacman.

Two pages, same seed, same action sequence: score, done and info() (Pac-Man, ghosts, pellets, map)
must agree at every step. Run from anywhere:  python games/pacman/check_det.py
"""
import asyncio
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from playjev.env import VecGame  # noqa: E402

GAME = "pacman"
SEEDS = [1, 7, 123, 99999]
STEPS = 150


async def rollout(env, page_idx, seed, actions):
    """Play `actions` on one page and return the (score, done, info) trace."""
    p = env.pages[page_idx]
    await p.reset(seed=seed)
    trace = []
    for a in actions:
        o = await p.step(a)
        if o.get("errors"):
            print(f"  page {page_idx} errors: {o['errors'][:2]}")
        trace.append((o["score"], o["done"], json.dumps(o["info"], sort_keys=True)))
        if o["done"]:
            break
    return trace


async def main():
    ok = True
    async with VecGame(GAME, n=2) as env:
        for seed in SEEDS:
            rng = random.Random(seed)
            actions = [rng.randrange(4) for _ in range(STEPS)]
            a = await rollout(env, 0, seed, actions)
            b = await rollout(env, 1, seed, actions)
            same = a == b
            ok &= same
            print(f"seed {seed}: {len(a)} steps, final score {a[-1][0]}, ended={a[-1][1]}, "
                  f"identical={'yes' if same else 'NO'}")
            if not same:
                for i, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        print(f"  first divergence at step {i}:\n    A {x}\n    B {y}")
                        break
        # replay on the same page must also reproduce (page reload + reseed)
        rng = random.Random(5)
        actions = [rng.randrange(4) for _ in range(STEPS)]
        a = await rollout(env, 0, 42, actions)
        c = await rollout(env, 0, 42, actions)
        ok &= a == c
        print(f"same page replayed, seed 42: identical={'yes' if a == c else 'NO'}")
        # different seeds must give different ghost walks (the seed has to reach Math.random in the game)
        a1 = await rollout(env, 0, 1, actions[:20])
        a2 = await rollout(env, 0, 2, actions[:20])
        g1 = json.loads(a1[-1][2])["ghosts"]; g2 = json.loads(a2[-1][2])["ghosts"]
        differ = g1 != g2
        ok &= differ
        print(f"seeds 1 vs 2 ghost positions after 20 steps differ={'yes' if differ else 'NO'}")
    print("DETERMINISM:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
