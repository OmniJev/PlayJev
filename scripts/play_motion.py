"""Closed loop with motion composites: playjev.play with a policy that folds the last k frames into one image.

The policy keeps a per-page history of the observations it was given and composes them exactly the way
scripts/make_motion_shard.py composed the training frames, so train and test see the same kind of picture.
Everything else (seeds, cap, argmax, the no-op rule, the result JSON) is playjev.play unchanged.

    python scripts/play_motion.py breakout --ckpt ckpt/m_ghost/final --motion ghost --episodes 16 --out runs/play/x.json
"""
import argparse, asyncio, io, sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from playjev import play
from playjev.motion import K, MODES, compose

MODE = "ghost"
KK = K


class MotionPolicy(play.LocalPolicy):
    def __init__(self, ckpt, actions, n, device="cuda:0", template="plain", two_frame=False, stack="temporal",
                 history=0):
        super().__init__(ckpt, actions, n, device=device, template=template, two_frame=False, stack=stack,
                         history=history)
        self.hist = [deque(maxlen=KK) for _ in range(n)]

    def reset(self, i):
        super().reset(i); self.hist[i].clear()

    def decide(self, frames, options, infos):
        imgs = []
        for i, f in enumerate(frames):
            self.hist[i].append(Image.open(io.BytesIO(f)).convert("RGB"))
            imgs.append(compose(list(self.hist[i]), MODE))
        return [d.probs for d in self.m.decide(imgs, self.options, batch_size=len(frames),
                                               state_text=self._state_text(len(frames)))]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("game"); p.add_argument("--ckpt", required=True); p.add_argument("--motion", default="ghost", choices=list(MODES))
    p.add_argument("--k", type=int, default=K); p.add_argument("--device", default="cuda:0")
    p.add_argument("--pages", type=int, default=8); p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=2000); p.add_argument("--seed0", type=int, default=5000)
    p.add_argument("--out", default=None); p.add_argument("--record", default=None)
    p.add_argument("--policy-name", dest="policy_name", default=None)
    p.add_argument("--delay", type=int, default=0, choices=[0, 1]); p.add_argument("--sample", action="store_true")
    p.add_argument("--history", type=int, default=0, metavar="K", help="also put the K moves this page has played into the state (match the checkpoint's training --history)")
    p.add_argument("--no-skip-noop", dest="skip_noop", action="store_false"); p.set_defaults(skip_noop=True)
    a = p.parse_args()
    MODE, KK = a.motion, a.k
    play.LocalPolicy = MotionPolicy
    a.policy = "local"; a.two_frame = False; a.stack = "temporal"; a.url = None
    a.handover = None; a.handover_random = None; a.handover_invert = False
    a.policy_name = a.policy_name or f"local_{a.motion}" + (f"_h{a.history}" if a.history else "")
    asyncio.run(play.main(a))
