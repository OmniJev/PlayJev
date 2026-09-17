"""Determinism check for the mario hook.

Two pages, same seed, same action sequence: scores and info() must agree at every step, and the
frames must be byte-identical. Run from the repo root:  python games/mario/check_det.py [--seeds 1 2 3] [--steps 300]
"""
import argparse, asyncio, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.env import VecGame  # noqa: E402


async def main(a):
    bad = 0
    async with VecGame("mario", n=2) as env:
        for seed in a.seeds:
            obs = await env.reset(seeds=[seed, seed])
            rng = random.Random(seed)
            n_act = len(env.actions); steps = 0; ended = 0
            for s in range(a.steps):
                act = rng.randrange(n_act)
                o0, o1 = await env.step([act, act]); steps += 1
                same = (o0["score"] == o1["score"] and o0["done"] == o1["done"]
                        and json.dumps(o0.get("info"), sort_keys=True) == json.dumps(o1.get("info"), sort_keys=True)
                        and o0["frame"] == o1["frame"])
                if not same:
                    bad += 1
                    print(f"seed {seed} step {s}: MISMATCH score {o0['score']} vs {o1['score']} done {o0['done']} vs {o1['done']}")
                    print("  info0", json.dumps(o0.get("info"), sort_keys=True)[:300])
                    print("  info1", json.dumps(o1.get("info"), sort_keys=True)[:300])
                    break
                if o0.get("errors") or o1.get("errors"):
                    print(f"seed {seed} step {s}: errors {o0.get('errors')} {o1.get('errors')}")
                if o0["done"]:
                    ended += 1
                    # start a second episode on the same seed and keep comparing
                    obs = await env.reset(seeds=[seed, seed])
                    same = (obs[0]["score"] == obs[1]["score"] and obs[0]["frame"] == obs[1]["frame"]
                            and json.dumps(obs[0].get("info"), sort_keys=True) == json.dumps(obs[1].get("info"), sort_keys=True))
                    if not same:
                        bad += 1; print(f"seed {seed}: MISMATCH right after reset"); break
            print(f"seed {seed}: {steps} steps compared, {ended} episode ends, final score {o0['score']}, "
                  f"x={o0['info']['x']} dead={o0['info']['dead']} won={o0['info']['won']}: {'OK' if not bad else 'FAIL'}")
    print("determinism check:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 11, 42])
    p.add_argument("--steps", type=int, default=300)
    sys.exit(asyncio.run(main(p.parse_args())))
