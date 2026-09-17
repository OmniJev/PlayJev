"""Random-policy throughput check: python -m playjev.bench snake --pages 8 --steps 200"""
import argparse, asyncio, random, time
from pathlib import Path
from .env import VecGame, ROOT


async def main(a):
    out = ROOT / "runs" / "bench" / a.game; out.mkdir(parents=True, exist_ok=True)
    async with VecGame(a.game, n=a.pages) as env:
        t0 = time.time(); obs = await env.reset(); t_reset = time.time() - t0
        print(f"[{a.game}] actions: {[x['name'] for x in env.actions]}; reset {a.pages} pages in {t_reset:.2f}s")
        rng = random.Random(0); n_act = len(env.actions); done_count = 0; ep_scores = []; ep_lens = []; steps = [0] * a.pages
        t0 = time.time(); total = 0
        for s in range(a.steps):
            acts = [rng.randrange(n_act) for _ in range(a.pages)]
            obs = await env.step(acts); total += a.pages
            for i, o in enumerate(obs):
                if o.get("errors"): print(f"  page{i} errors: {o['errors'][:2]}")
                steps[i] += 1
                if o["done"]:
                    done_count += 1; ep_scores.append(o["score"]); ep_lens.append(steps[i]); steps[i] = 0
                    await env.pages[i].reset(seed=rng.randrange(10**6))
            if s in (0, 5, 20, 60, a.steps - 1):
                (out / f"p0_step{s:04d}.jpg").write_bytes(obs[0]["frame"])
            if (s + 1) % max(1, a.steps // 5) == 0:
                dt = time.time() - t0
                print(f"  step {s+1}/{a.steps}: {total/dt:.0f} env-steps/s, score p0={obs[0]['score']}, t={obs[0]['t']:.0f}ms, episodes done={done_count}")
        dt = time.time() - t0
        print(f"[{a.game}] {total/dt:.0f} env-steps/s over {a.pages} pages; {done_count} episodes ended, mean random score {sum(ep_scores)/max(1,len(ep_scores)):.2f}, "
              f"mean episode length {sum(ep_lens)/max(1,len(ep_lens)):.0f} steps; frames in {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("game"); p.add_argument("--pages", type=int, default=8); p.add_argument("--steps", type=int, default=200)
    asyncio.run(main(p.parse_args()))
