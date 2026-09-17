"""Play a game with its teacher; report the teacher ceiling.
python -m playjev.teacher_eval snake --pages 8 --episodes 16 --max-steps 2000"""
import argparse, asyncio, random, statistics, time
from .env import VecGame, ROOT
from .teachers import make_teacher


async def main(a):
    async with VecGame(a.game, n=a.pages) as env:
        obs = await env.reset(list(range(1, a.pages + 1)))
        teachers = [make_teacher(a.game, env.actions) for _ in range(a.pages)]
        for t in teachers: t.reset()
        rng = random.Random(0); scores, lengths = [], []; steps = [0] * a.pages; t0 = time.time(); total = 0
        seed = a.pages
        while len(scores) < a.episodes:
            acts = []
            for i, o in enumerate(obs):
                p = teachers[i].act(o)
                acts.append(max(range(len(p)), key=p.__getitem__) if rng.random() >= a.epsilon else rng.randrange(len(p)))
            obs = await env.step(acts); total += a.pages
            for i, o in enumerate(obs):
                steps[i] += 1
                if o.get("errors"): print("errors:", o["errors"][:2])
                if o["done"] or steps[i] >= a.max_steps:
                    scores.append(o["score"]); lengths.append(steps[i]); steps[i] = 0; seed += 1
                    obs[i] = await env.pages[i].reset(seed); teachers[i].reset()
                    if len(scores) % max(1, a.episodes // 4) == 0:
                        print(f"  {len(scores)}/{a.episodes} episodes, last score {o['score']} in {lengths[-1]} steps, {total/(time.time()-t0):.0f} env-steps/s")
        out = ROOT / "runs" / "teacher" / a.game; out.mkdir(parents=True, exist_ok=True)
        (out / "last.jpg").write_bytes(obs[0]["frame"])
        print(f"[{a.game}] teacher eps={a.epsilon}: {len(scores)} episodes, score mean {statistics.mean(scores):.2f} median {statistics.median(scores)} max {max(scores)}, "
              f"episode length mean {statistics.mean(lengths):.0f}, capped {sum(1 for l in lengths if l >= a.max_steps)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("game"); p.add_argument("--pages", type=int, default=8); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000); p.add_argument("--epsilon", type=float, default=0.0)
    asyncio.run(main(p.parse_args()))
