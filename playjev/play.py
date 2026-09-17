"""Closed-loop evaluation: a policy plays a game in VecGame; report score per episode.

Policies expose  decide(frames: list[bytes], options: list[dict], infos: list[dict]) -> list[list[float]]
(one probability vector per page). Built in: random, teacher, server (HTTP, PlayJev/OpenJev image state), local (in-process PlayJevModel, --ckpt).

python -m playjev.play snake --policy teacher --pages 8 --episodes 16
python -m playjev.play snake --policy server --url http://127.0.0.1:18731/v1/systemone --episodes 8
python -m playjev.play snake --policy local --ckpt /path/to/ckpt --episodes 16
"""
import argparse, asyncio, base64, json, random, statistics, time, urllib.request
from .env import VecGame, ROOT
from .teachers import make_teacher

INSTRUCTIONS = "Which move should the player make next?"  # frozen with the prompt; no game name anywhere


class RandomPolicy:
    def __init__(self, actions, seed=0): self.n, self.rng = len(actions), random.Random(seed)
    def decide(self, frames, options, infos):  # one-hot at a random index, so argmax is a uniform random action
        out = []
        for _ in frames:
            p = [0.0] * self.n; p[self.rng.randrange(self.n)] = 1.0; out.append(p)
        return out


class TeacherPolicy:
    def __init__(self, game_id, actions, n): self.t = [make_teacher(game_id, actions) for _ in range(n)]
    def reset(self, i): self.t[i].reset()
    def decide(self, frames, options, infos): return [self.t[i].act({"info": info}) for i, info in enumerate(infos)]


class ServerPolicy:
    """POST /v1/systemone with an image state: {"state": {"frames": [dataURL]}, "questions": {"q": Choice}}."""
    def __init__(self, url, actions, model="playjev-latest"):
        self.url, self.model = url, model
        self.criteria = {a["name"]: a["description"] for a in actions}
    def decide(self, frames, options, infos):
        out = []
        for f in frames:
            body = {"model": self.model, "state": {"frames": ["data:image/jpeg;base64," + base64.b64encode(f).decode()]},
                    "questions": {"q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": self.criteria}}}
            req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            ans = json.loads(urllib.request.urlopen(req, timeout=60).read())["answers"]["q"]["probabilities"]
            out.append([ans[a["name"]] for a in options])
        return out


class LocalPolicy:
    """In-process PlayJevModel (playjev/model.py) on a checkpoint directory or HF id: frames in, probabilities over
    the options out, one batched forward per step. Two-frame mode keeps the previous frame per page."""
    def __init__(self, ckpt, actions, n, device="cuda:0", template="plain", two_frame=False, stack="temporal"):
        from .model import PlayJevModel
        self.m = PlayJevModel(ckpt, device=device, template=template).load()
        self.options = [{"name": a["name"], "description": a["description"]} for a in actions]
        self.two_frame, self.stack, self.prev = two_frame, stack, [None] * n
    def reset(self, i): self.prev[i] = None
    def decide(self, frames, options, infos):
        if self.two_frame:
            items = [(self.prev[i] or f, f) for i, f in enumerate(frames)]; self.prev = list(frames)
            dec = self.m.decide(items, self.options, frames_per_state=2, stack=self.stack, batch_size=len(frames))
        else:
            dec = self.m.decide(list(frames), self.options, batch_size=len(frames))
        return [d.probs for d in dec]


def argmax(p): return max(range(len(p)), key=p.__getitem__)


async def main(a):
    async with VecGame(a.game, n=a.pages) as env:
        obs = await env.reset(list(range(a.seed0, a.seed0 + a.pages)))
        acts_meta = env.actions
        pol = {"random": lambda: RandomPolicy(acts_meta), "teacher": lambda: TeacherPolicy(a.game, acts_meta, a.pages),
               "server": lambda: ServerPolicy(a.url, acts_meta),
               "local": lambda: LocalPolicy(a.ckpt, acts_meta, a.pages, device=a.device, two_frame=a.two_frame)}[a.policy]()
        if hasattr(pol, "reset"):
            for i in range(a.pages): pol.reset(i)
        scores, lengths, confs = [], [], []; steps = [0] * a.pages; seed = a.seed0 + a.pages; t0 = time.time(); total = 0
        rng = random.Random(0)
        while len(scores) < a.episodes:
            probs = pol.decide([o["frame"] for o in obs], acts_meta, [o.get("info") for o in obs])
            acts = [argmax(p) if not a.sample else rng.choices(range(len(p)), p)[0] for p in probs]
            K = len(acts_meta); confs.extend((max(p) - 1 / K) / (1 - 1 / K) for p in probs)
            obs = await env.step(acts); total += a.pages
            for i, o in enumerate(obs):
                steps[i] += 1
                if o["done"] or steps[i] >= a.max_steps:
                    scores.append(o["score"]); lengths.append(steps[i]); steps[i] = 0; seed += 1
                    obs[i] = await env.pages[i].reset(seed)
                    if hasattr(pol, "reset"): pol.reset(i)
        dt = time.time() - t0
        res = {"game": a.game, "policy": a.policy, "episodes": len(scores), "score_mean": statistics.mean(scores), "score_median": statistics.median(scores),
               "score_max": max(scores), "len_mean": statistics.mean(lengths), "capped": sum(l >= a.max_steps for l in lengths),
               "conf_mean": statistics.mean(confs), "steps_per_s": total / dt}
        out = ROOT / "runs" / "play"; out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.game}_{a.policy}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("game"); p.add_argument("--policy", default="random", choices=["random", "teacher", "server", "local"])
    p.add_argument("--ckpt", help="local policy: checkpoint directory or HF id for PlayJevModel"); p.add_argument("--device", default="cuda:0"); p.add_argument("--two-frame", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:18731/v1/systemone"); p.add_argument("--pages", type=int, default=8); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000); p.add_argument("--seed0", type=int, default=5000); p.add_argument("--sample", action="store_true", help="sample actions from the policy instead of argmax")
    asyncio.run(main(p.parse_args()))
