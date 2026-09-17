"""Random-policy baseline for sokoban: N episodes (seeds 0..N-1, so every Microban level is played at least
once for N >= 155), uniform random moves, episodes end on a solve or at the driver cap (pj.json max_steps).
Reports mean score, solve rate and mean episode length.
Run from anywhere:  python games/sokoban/random_policy.py [--episodes 200] [--pages 8]
"""
import argparse
import asyncio
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from playjev.env import VecGame  # noqa: E402


async def main(a):
    rng = random.Random(0)
    seeds = list(range(a.episodes))
    results = []  # (seed, level, score, steps, solved)
    async with VecGame("sokoban", n=a.pages) as env:
        await env.reset(seeds[:a.pages]); n_act = len(env.actions)
        active = {i: (seeds[i], 0) for i in range(min(a.pages, len(seeds)))}
        next_seed = len(active); reported = 0
        while active:
            idx = sorted(active)
            obs = await asyncio.gather(*(env.pages[i].step(rng.randrange(n_act)) for i in idx))
            for i, o in zip(idx, obs):
                seed, steps = active[i]; steps += 1
                if o.get("errors"):
                    print(f"  page{i} errors: {o['errors'][:2]}")
                if o["done"]:
                    results.append((seed, o["info"]["level"], o["score"], steps, bool(o["info"]["solved"])))
                    if next_seed < len(seeds):
                        active[i] = (seeds[next_seed], 0); next_seed += 1
                        await env.pages[i].reset(seed=seeds[next_seed - 1])
                    else:
                        del active[i]
                else:
                    active[i] = (seed, steps)
            if len(results) // 40 > reported:
                reported = len(results) // 40; print(f"  {len(results)}/{len(seeds)} episodes done", flush=True)
    n = len(results)
    solved = [r for r in results if r[4]]
    print(f"random policy, {n} episodes (seeds 0..{n - 1}), cap {env.spec.get('max_steps')} steps:")
    print(f"  mean score {sum(r[2] for r in results) / n:.3f}, solve rate {len(solved) / n:.3f} ({len(solved)} solved: "
          f"levels {sorted(r[1] for r in solved)}), mean episode length {sum(r[3] for r in results) / n:.1f} steps, "
          f"episodes with any box newly placed on a goal: {sum(1 for r in results if r[2] > 0)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--episodes", type=int, default=200); p.add_argument("--pages", type=int, default=8)
    asyncio.run(main(p.parse_args()))
