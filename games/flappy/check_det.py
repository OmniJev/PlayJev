"""Determinism check for flappy.

Two pages, same seed, same action sequence: score, done and info() must agree at every step.
The action sequence is a biased random policy (flap with p=0.45) so the bird reaches the pipes.
Run from anywhere:  python games/flappy/check_det.py
"""
import asyncio
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from playjev.env import VecGame  # noqa: E402

GAME = "flappy"
SEEDS = [1, 7, 123, 99999]
STEPS = 160


def policy_actions(seed, n):
    rng = random.Random(seed)
    return [0 if rng.random() < 0.45 else 1 for _ in range(n)]


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
            actions = policy_actions(seed, STEPS)
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
        actions = policy_actions(5, STEPS)
        a = await rollout(env, 0, 42, actions)
        c = await rollout(env, 0, 42, actions)
        ok &= a == c
        print(f"same page replayed, seed 42: identical={'yes' if a == c else 'NO'}")
        # different seeds must give different pipe gaps (otherwise the seed is not reaching the game)
        a1 = await rollout(env, 0, 1, actions[:40])
        a2 = await rollout(env, 0, 2, actions[:40])
        gaps = [json.loads(t[2])["nextPipe"] for t in (a1[-1], a2[-1])]
        differ = gaps[0] != gaps[1]
        ok &= differ
        print(f"seeds 1 vs 2 first pipe gap: {gaps[0] and gaps[0]['gapTop']} vs {gaps[1] and gaps[1]['gapTop']}, "
              f"differ={'yes' if differ else 'NO'}")
    print("DETERMINISM:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
