"""Determinism check for tetris: two pages, same seed, same action sequence.

Compares score, done, info() and the encoded frame at every step, for several seeds.
Run:  source .venv/bin/activate && python games/tetris/check_det.py
"""
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.env import VecGame  # noqa: E402

GAME = "tetris"
SEEDS = (1, 7, 42, 1234)
STEPS = 150


async def main() -> int:
    bad = []
    async with VecGame(GAME, n=2) as env:
        n_act = None
        for seed in SEEDS:
            obs = await env.reset([seed, seed])
            n_act = len(env.actions)
            rng = random.Random(seed)
            steps = 0
            for s in range(STEPS):
                a = rng.randrange(n_act)
                obs = await env.step([a, a])
                steps += 1
                o0, o1 = obs
                for key in ("score", "done", "t"):
                    if o0[key] != o1[key]:
                        bad.append(f"seed {seed} step {s}: {key} {o0[key]} != {o1[key]}")
                if json.dumps(o0.get("info"), sort_keys=True) != json.dumps(o1.get("info"), sort_keys=True):
                    bad.append(f"seed {seed} step {s}: info differs")
                if o0.get("frame") != o1.get("frame"):
                    bad.append(f"seed {seed} step {s}: frame bytes differ")
                if o0.get("errors") or o1.get("errors"):
                    bad.append(f"seed {seed} step {s}: errors {o0.get('errors')} {o1.get('errors')}")
                if o0["done"]:
                    break
            print(f"seed {seed}: {steps} steps, score {obs[0]['score']}, done={obs[0]['done']}, "
                  f"lines={obs[0]['info']['lines']}")
    if bad:
        print(f"MISMATCH ({len(bad)}):")
        for line in bad[:10]:
            print("  " + line)
        return 1
    print(f"[{GAME}] deterministic: {len(SEEDS)} seeds, score/info/frame identical at every step")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
