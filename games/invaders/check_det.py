"""Determinism check for invaders: two pages, same seed, same random action sequence, must agree on
score and info() at every step.  python games/invaders/check_det.py [--steps 300] [--seed 7]
Also reports whether a different seed gives a different trajectory (the hook re-sows Phaser's RNG)."""
import argparse, asyncio, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.env import VecGame  # noqa: E402


async def main(a):
    async with VecGame("invaders", n=3) as env:
        obs = await env.reset(seeds=[a.seed, a.seed, a.seed + 1])
        rng = random.Random(a.seed)
        n_act = len(env.actions)
        mismatch = None
        differs_from_other_seed = False
        finished = None
        for s in range(a.steps):
            act = rng.randrange(n_act)
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
        print(f"[invaders] seed {a.seed}: {s + 1} steps, final scores {[o['score'] for o in obs[:2]]}, "
              f"done={[o['done'] for o in obs[:2]]}" + (f", episode ended at step {finished}" if finished is not None else ""))
        if mismatch:
            print(f"[invaders] DETERMINISM FAILED at step {mismatch[0]}:\n  {mismatch[1]}\n  {mismatch[2]}")
            sys.exit(1)
        print(f"[invaders] determinism OK (score and info identical on both pages at every step); "
              f"seed {a.seed + 1} trajectory {'differs' if differs_from_other_seed else 'IDENTICAL (seed has no effect!)'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    asyncio.run(main(p.parse_args()))
