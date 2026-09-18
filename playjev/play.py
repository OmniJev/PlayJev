"""Closed-loop evaluation: a policy plays a game in VecGame; report score per episode.

Policies expose  decide(frames: list[bytes], options: list[dict], infos: list[dict]) -> list[list[float]]
(one probability vector per page). Built in: random, teacher, server (HTTP, PlayJev/OpenJev image state), local (in-process PlayJevModel, --ckpt).

python -m playjev.play snake --policy teacher --pages 8 --episodes 16
python -m playjev.play snake --policy server --url http://127.0.0.1:18731/v1/systemone --episodes 8
python -m playjev.play snake --policy local --ckpt /path/to/ckpt --episodes 16

Execution: argmax by default (--sample to sample); a move that changed nothing (same frame and score) is not repeated
on the same observation, the next-best move is taken (--no-skip-noop for plain execution); --delay 1 applies each
decision one step late.
"""
import argparse, asyncio, base64, json, random, statistics, time, urllib.request
from pathlib import Path
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
    """POST /v1/systemone with an image state: {"state": {"frames": [previous, current]}, "questions": {"q": Choice}}.
    The previous frame of each page goes along once there is one; a single-frame server uses the last frame."""
    def __init__(self, url, actions, n=1, model="playjev-latest"):
        self.url, self.model, self.prev = url, model, [None] * n
        self.criteria = {a["name"]: a["description"] for a in actions}
    def reset(self, i): self.prev[i] = None
    def decide(self, frames, options, infos):
        out = []
        for i, f in enumerate(frames):
            data = [("data:image/jpeg;base64," + base64.b64encode(x).decode()) for x in ([self.prev[i]] if self.prev[i] else []) + [f]]
            self.prev[i] = f
            body = {"model": self.model, "state": {"frames": data},
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


class HandoverPolicy:
    """System One with a System Two behind it: the model decides, and whenever its Jev confidence is below `tau` the
    decision is handed to the teacher (which reads the game's internal state). tau 0 never hands over, tau above 1
    always does. `handed` counts the steps handed over; `last` marks which pages were handed over on the last call."""
    def __init__(self, inner, game_id, actions, n, tau, random_rate=None, seed=0):
        self.inner, self.t, self.tau, self.K = inner, [make_teacher(game_id, actions) for _ in range(n)], tau, len(actions)
        self.handed = 0; self.total = 0; self.last = [False] * n
        self.random_rate, self.rng = random_rate, random.Random(seed)  # control: hand over a random share of the steps instead
    def reset(self, i):
        self.t[i].reset()
        if hasattr(self.inner, "reset"): self.inner.reset(i)
    def decide(self, frames, options, infos):
        probs = self.inner.decide(frames, options, infos); out = []
        for i, p in enumerate(probs):
            conf = (max(p) - 1 / self.K) / (1 - 1 / self.K)
            self.last[i] = (self.rng.random() < self.random_rate) if self.random_rate is not None else conf < self.tau
            self.total += 1
            if self.last[i]:
                self.handed += 1; out.append(self.t[i].act({"info": infos[i]}))
            else:
                out.append(p)
        return out


async def main(a):
    async with VecGame(a.game, n=a.pages) as env:
        obs = await env.reset(list(range(a.seed0, a.seed0 + a.pages)))
        acts_meta = env.actions
        pol = {"random": lambda: RandomPolicy(acts_meta), "teacher": lambda: TeacherPolicy(a.game, acts_meta, a.pages),
               "server": lambda: ServerPolicy(a.url, acts_meta, a.pages),
               "local": lambda: LocalPolicy(a.ckpt, acts_meta, a.pages, device=a.device, two_frame=a.two_frame)}[a.policy]()
        if a.handover is not None or a.handover_random is not None:
            pol = HandoverPolicy(pol, a.game, acts_meta, a.pages, a.handover or 0.0, a.handover_random)
        if hasattr(pol, "reset"):
            for i in range(a.pages): pol.reset(i)
        scores, lengths, confs = [], [], []; steps = [0] * a.pages; seed = a.seed0 + a.pages; t0 = time.time(); total = 0
        rng = random.Random(0)
        # --delay 1: the action decided from frame k is applied at step k+1 (real-time play, inference overlaps the
        # current step). The first step of an episode applies the current decision. Turn-based games behave the same.
        pending = [None] * a.pages
        # Executor rule (default, --no-skip-noop turns it off): a move that left the observation unchanged (same
        # frame bytes, same score, not done) is not repeated on that same observation; the executor takes the best
        # move outside the banned set instead. Games where a blocked move is a genuine no-op (2048, sokoban)
        # otherwise trap a deterministic policy for the rest of the episode. The ban clears as soon as the
        # observation changes. The policy is not consulted differently: its probabilities are recorded as given,
        # only the executed index changes; the result JSON counts noop_steps and skipped_steps.
        banned = [set() for _ in range(a.pages)]; noop_steps = 0; skipped_steps = 0
        def pick(i, p, decided):
            if not banned[i] or len(banned[i]) >= len(p) or decided not in banned[i]:
                return decided, False
            allowed = [j for j in range(len(p)) if j not in banned[i]]
            return max(allowed, key=p.__getitem__), True
        page_seed = list(range(a.seed0, a.seed0 + a.pages)); traces = [[] for _ in range(a.pages)]  # for --record
        rec_dir = None
        if a.record:
            rec_dir = Path(a.record) / a.game; rec_dir.mkdir(parents=True, exist_ok=True)
        while len(scores) < a.episodes:
            probs = pol.decide([o["frame"] for o in obs], acts_meta, [o.get("info") for o in obs])
            decided = [argmax(p) if not a.sample else rng.choices(range(len(p)), p)[0] for p in probs]
            if a.delay:
                acts = [pending[i] if pending[i] is not None else decided[i] for i in range(a.pages)]
                pending = list(decided)
            else:
                acts = decided
            if a.skip_noop:
                picked = [pick(i, probs[i], acts[i]) for i in range(a.pages)]
                acts = [x for x, _ in picked]; skipped_steps += sum(sk for _, sk in picked)
            K = len(acts_meta); confs.extend((max(p) - 1 / K) / (1 - 1 / K) for p in probs)
            prev_obs = obs
            obs = await env.step(acts); total += a.pages
            for i, o in enumerate(obs):
                steps[i] += 1
                if a.skip_noop:
                    same = (not o["done"] and o["score"] == prev_obs[i]["score"] and o.get("frame") == prev_obs[i].get("frame"))
                    if same: banned[i].add(acts[i]); noop_steps += 1
                    else: banned[i].clear()
                if rec_dir is not None:
                    st = {"a": acts[i], "p": [round(x, 4) for x in probs[i]], "score": o["score"]}
                    if isinstance(pol, HandoverPolicy) and pol.last[i]: st["h"] = 1  # this step was decided by System Two
                    traces[i].append(st)
                if o["done"] or steps[i] >= a.max_steps:
                    truncated = bool(o.get("truncated")) or steps[i] >= a.max_steps
                    scores.append(o["score"]); lengths.append(steps[i]); steps[i] = 0
                    if rec_dir is not None:  # replay file in the docs/DEMO.md format
                        rec = {"game": a.game, "policy": a.policy_name or a.policy, "seed": page_seed[i], "actions": [x["name"] for x in acts_meta],
                               "frames_per_step": env.spec.get("step_frames"), "steps": traces[i], "final_score": o["score"], "truncated": truncated}
                        (rec_dir / f"{rec['policy']}_{page_seed[i]}.json").write_text(json.dumps(rec, separators=(",", ":")))
                        traces[i] = []
                    seed += 1; page_seed[i] = seed; pending[i] = None; banned[i].clear()
                    obs[i] = await env.pages[i].reset(seed)
                    if hasattr(pol, "reset"): pol.reset(i)
        dt = time.time() - t0
        res = {"game": a.game, "policy": a.policy, "delay": a.delay, "skip_noop": bool(a.skip_noop), "episodes": len(scores), "score_mean": statistics.mean(scores), "score_median": statistics.median(scores),
               "score_max": max(scores), "len_mean": statistics.mean(lengths), "capped": sum(l >= a.max_steps for l in lengths),
               "conf_mean": statistics.mean(confs), "steps_per_s": total / dt, "steps": total}
        if a.skip_noop:
            res["noop_steps"] = noop_steps; res["skipped_steps"] = skipped_steps  # unchanged observations seen; executed moves changed by the rule
        if isinstance(pol, HandoverPolicy):
            res["handover_tau"] = a.handover; res["handover_random"] = a.handover_random
            res["handover_rate"] = pol.handed / max(1, pol.total); res["handed_steps"] = pol.handed
        out = ROOT / "runs" / "play"; out.mkdir(parents=True, exist_ok=True)
        tag = f"{'_delay' + str(a.delay) if a.delay else ''}{'' if a.skip_noop else '_noskip'}{'_handover' + str(a.handover) if a.handover is not None else ''}{'_hrandom' + str(a.handover_random) if a.handover_random is not None else ''}"
        (out / f"{a.game}_{a.policy}{tag}.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("game"); p.add_argument("--policy", default="random", choices=["random", "teacher", "server", "local"])
    p.add_argument("--ckpt", help="local policy: checkpoint directory or HF id for PlayJevModel"); p.add_argument("--device", default="cuda:0"); p.add_argument("--two-frame", action="store_true")
    p.add_argument("--url", default="http://127.0.0.1:18731/v1/systemone"); p.add_argument("--pages", type=int, default=8); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--record", default=None, help="directory: write one replay JSON per finished episode (docs/DEMO.md format)")
    p.add_argument("--policy-name", dest="policy_name", default=None, help="label stored in replay files (default: --policy)")
    p.add_argument("--delay", type=int, default=0, choices=[0, 1], help="1: apply each decision one step late (real-time latency model)")
    p.add_argument("--max-steps", type=int, default=2000); p.add_argument("--seed0", type=int, default=5000); p.add_argument("--sample", action="store_true", help="sample actions from the policy instead of argmax")
    p.add_argument("--no-skip-noop", dest="skip_noop", action="store_false", help="plain execution: a move that left the observation unchanged may be repeated (2048 and sokoban can then loop to the cap)")
    p.add_argument("--handover", type=float, default=None, help="System Two: hand the decision to the teacher when the model's Jev confidence is below this (0 never, 1.01 always)")
    p.add_argument("--handover-random", dest="handover_random", type=float, default=None, help="control: hand over this share of the steps at random instead of by confidence")
    p.set_defaults(skip_noop=True)
    asyncio.run(main(p.parse_args()))
