"""Collect (frame, teacher target) pairs by playing with the teacher, optionally with epsilon-random
actions for state coverage (DAgger-style: the label is always the teacher's, the action taken may not be).

--actor local --ckpt DIR (or --actor server --url URL) is the DAgger round proper: the trained model plays
(argmax, with the same no-op rule as playjev.play, plus epsilon-random moves), and every frame it visits is
labelled by the teacher. Records carry "actor" so shards can be told apart.

Output layout, one shard per call:
  data/<game>/<shard>/frames/<000000>.jpg     448 px long side JPEG, the state the decision was made in
  data/<game>/<shard>/records.jsonl           one line per decision (see RECORD_FIELDS)

python -m playjev.collect snake --steps 4000 --pages 8 --epsilon 0.1 --shard s0
python -m playjev.collect snake --steps 20000 --pages 8 --epsilon 0.05 --shard dagger1 --actor local --ckpt ckpt/sft_all1/final
"""
import argparse, asyncio, json, random, time
from pathlib import Path
from .env import VecGame, ROOT
from .teachers import make_teacher

RECORD_FIELDS = ["game", "shard", "seed", "episode", "step", "frame", "prev_frame", "actions",
                 "teacher_probs", "teacher_action", "taken_action", "score", "reward", "done", "t", "actor"]


async def main(a):
    out = ROOT / "data" / a.game / a.shard; frames_dir = out / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    rec = open(out / "records.jsonl", "w"); n_written = 0; t0 = time.time()
    async with VecGame(a.game, n=a.pages) as env:
        seeds = [a.seed0 + i for i in range(a.pages)]; next_seed = a.seed0 + a.pages
        obs = await env.reset(seeds)
        names = [x["name"] for x in env.actions]
        teachers = [make_teacher(a.game, env.actions) for _ in range(a.pages)]
        for t in teachers: t.reset()
        rng = random.Random(a.seed0); ep = [0] * a.pages; step = [0] * a.pages; prev_frame = [None] * a.pages; n_giveup = 0
        if a.epsilon is None:  # per-game default from pj.json {"collect": {"epsilon": x}}, else 0.1
            a.epsilon = float(env.spec.get("collect", {}).get("epsilon", 0.1))
        actor = None  # DAgger: the model acts, the teacher labels
        if a.actor == "local":
            from .play import LocalPolicy
            actor = LocalPolicy(a.ckpt, env.actions, a.pages, device=a.device, two_frame=a.two_frame, stack=a.stack)
        elif a.actor == "server":
            from .play import ServerPolicy
            actor = ServerPolicy(a.url, env.actions, a.pages)
        banned = [set() for _ in range(a.pages)]  # the no-op rule of playjev.play: never repeat a move that changed nothing
        print(f"[{a.game}] epsilon {a.epsilon} actor {a.actor}")
        while n_written < a.steps:
            acts, targets = [], []
            model_probs = actor.decide([o["frame"] for o in obs], env.actions, [o.get("info") for o in obs]) if actor else None
            for i, o in enumerate(obs):
                p = teachers[i].act(o); best = max(range(len(p)), key=p.__getitem__)
                if model_probs is not None:
                    mp = model_probs[i]; allowed = [j for j in range(len(mp)) if j not in banned[i]] or list(range(len(mp)))
                    chosen = max(allowed, key=mp.__getitem__)
                else:
                    chosen = best
                taken = rng.randrange(len(p)) if rng.random() < a.epsilon else chosen
                acts.append(taken); targets.append((p, best))
            # save the frame the decision was made in, before stepping
            fids = []
            for i, o in enumerate(obs):
                fid = f"{n_written + i:07d}"; (frames_dir / f"{fid}.jpg").write_bytes(o["frame"]); fids.append(fid)
            new_obs = await env.step(acts)
            for i, (o, no) in enumerate(zip(obs, new_obs)):
                p, best = targets[i]
                rec.write(json.dumps({"game": a.game, "shard": a.shard, "seed": seeds[i], "episode": ep[i], "step": step[i],
                                      "frame": f"frames/{fids[i]}.jpg", "prev_frame": prev_frame[i], "actions": names,
                                      "teacher_probs": [round(x, 4) for x in p], "teacher_action": best, "taken_action": acts[i],
                                      "score": o["score"], "reward": no["reward"], "done": no["done"], "t": o["t"], "actor": a.actor}) + "\n")
                prev_frame[i] = f"frames/{fids[i]}.jpg"; step[i] += 1
                if actor is not None:
                    if not no["done"] and no["score"] == o["score"] and no.get("frame") == o.get("frame"): banned[i].add(acts[i])
                    else: banned[i].clear()
            n_written += len(obs); obs = new_obs
            for i, o in enumerate(obs):
                if o.get("errors"): print("errors:", o["errors"][:2])
                if o["done"] or step[i] >= a.max_steps or teachers[i].giveup():
                    n_giveup += int(not o["done"] and step[i] < a.max_steps)
                    seeds[i] = next_seed; next_seed += 1; ep[i] += 1; step[i] = 0; prev_frame[i] = None
                    obs[i] = await env.pages[i].reset(seeds[i]); teachers[i].reset(); banned[i].clear()
                    if actor is not None and hasattr(actor, "reset"): actor.reset(i)
            if n_written % max(a.pages, (a.steps // 10) // a.pages * a.pages) == 0:
                print(f"  {n_written}/{a.steps} records, {n_written/(time.time()-t0):.0f}/s")
    rec.close()
    print(f"[{a.game}] wrote {n_written} records to {out} in {time.time()-t0:.0f}s; episodes ended early by teacher giveup: {n_giveup}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("game"); p.add_argument("--steps", type=int, default=4000); p.add_argument("--pages", type=int, default=8)
    p.add_argument("--epsilon", type=float, default=None, help="random-action rate; default from pj.json collect.epsilon or 0.1"); p.add_argument("--shard", default="s0"); p.add_argument("--seed0", type=int, default=1000); p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--actor", default="teacher", choices=["teacher", "local", "server"], help="who plays: the teacher (default) or the trained model (DAgger; the teacher still labels)")
    p.add_argument("--ckpt", default=None, help="--actor local: checkpoint directory"); p.add_argument("--device", default="cuda:0"); p.add_argument("--two-frame", action="store_true")
    p.add_argument("--stack", default="temporal", choices=["temporal", "separate"])
    p.add_argument("--url", default="http://127.0.0.1:18732/v1/systemone", help="--actor server: PlayJev server")
    asyncio.run(main(p.parse_args()))
