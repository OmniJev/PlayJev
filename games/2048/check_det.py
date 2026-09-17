"""Determinism check for 2048.

Two pages, same seed, same action sequence: the score and info() must agree at every step.
Run from anywhere:  python games/2048/check_det.py
"""
import asyncio
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from playjev.env import VecGame  # noqa: E402

GAME = "2048"
SEEDS = [1, 7, 123, 99999]
STEPS = 120


async def rollout(env, page_idx, seed, actions):
    """Play `actions` on one page and return the (score, info) trace."""
    p = env.pages[page_idx]
    await p.reset(seed=seed)
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
        for seed in SEEDS:
            rng = random.Random(seed)
            actions = [rng.randrange(4) for _ in range(STEPS)]
            a = await rollout(env, 0, seed, actions)
            b = await rollout(env, 1, seed, actions)
            same = a == b
            ok &= same
            print(f"seed {seed}: {len(a)} steps, final score {a[-1][0]}, "
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
    print("DETERMINISM:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
