"""Determinism check for racer: two pages, same seed, same action sequence, must agree on score and
info() at every step.  python games/racer/check_det.py [--steps 600] [--seed 7] [--policy random|drive]
`random` draws actions uniformly (the car barely moves); `drive` accelerates and steers toward the
road centre with seeded noise, so traffic, collisions and the lap wrap are all exercised.
Also reports whether a different seed gives a different trajectory (the hook rebuilds traffic and
scenery with the seeded Math.random)."""
import argparse, asyncio, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.env import VecGame  # noqa: E402

LEFT, RIGHT, FASTER, SLOWER, LEFT_FASTER, RIGHT_FASTER = range(6)


def drive(info, rng):
    x = info["playerX"]
    target = max(-0.6, min(0.6, 0.08 * info["curveAhead"][0])) + rng.uniform(-0.3, 0.3)
    if rng.random() < 0.05:
        return SLOWER
    if x < target - 0.08:
        return RIGHT_FASTER
    if x > target + 0.08:
        return LEFT_FASTER
    return FASTER


async def main(a):
    async with VecGame("racer", n=3) as env:
        obs = await env.reset(seeds=[a.seed, a.seed, a.seed + 1])
        rng = random.Random(a.seed)
        n_act = len(env.actions)
        mismatch = None
        differs_from_other_seed = False
        finished = None
        for s in range(a.steps):
            act = rng.randrange(n_act) if a.policy == "random" else drive(obs[0]["info"], rng)
            obs = await env.step([act, act, act])
            for i, o in enumerate(obs):
                if o.get("errors"):
                    print(f"  page{i} step{s} errors: {o['errors'][:2]}")
            a0, a1, a2 = (json.dumps({"score": o["score"], "done": o["done"], "info": o.get("info")}, sort_keys=True) for o in obs)
            if a0 != a1 and mismatch is None:
                mismatch = (s, a0[:300], a1[:300])
            if a0 != a2:
                differs_from_other_seed = True
            if obs[0]["done"] and obs[1]["done"] and finished is None:
                finished = s
                break
        print(f"[racer] seed {a.seed}, policy {a.policy}: {s + 1} steps, final scores {[o['score'] for o in obs[:2]]}, "
              f"done={[o['done'] for o in obs[:2]]}, lastLapTime={obs[0]['info']['lastLapTime']}"
              + (f", episode ended at step {finished}" if finished is not None else ""))
        if mismatch:
            print(f"[racer] DETERMINISM FAILED at step {mismatch[0]}:\n  {mismatch[1]}\n  {mismatch[2]}")
            sys.exit(1)
        print(f"[racer] determinism OK (score and info identical on both pages at every step); "
              f"seed {a.seed + 1} trajectory {'differs' if differs_from_other_seed else 'IDENTICAL (seed has no effect!)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--policy", choices=["random", "drive"], default="random")
    asyncio.run(main(p.parse_args()))
