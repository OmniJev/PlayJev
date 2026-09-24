"""Rebuild a shard's frames from its records.

The released training data is records only: for every decision, the episode seed, the step, the move that
was actually taken and the teacher's target. The games are deterministic (the shim's virtual clock and seeded
Math.random), so starting each episode from pj.start(seed) and replaying its taken moves visits the same
states and draws the same frames, including the DAgger rounds whose moves came from a model. Every step is
checked against the recorded score and virtual time, so a replay that drifts is reported, not written, and
every frame against the record's frame_md5, so you know how many are byte for byte the ones we trained on.

python -m playjev.rebuild data/snake/clone_a            # write frames/ next to records.jsonl
python -m playjev.rebuild data                          # every shard under data/
python -m playjev.rebuild data/snake/clone_a --check    # compare with the frames already there
"""
import argparse, asyncio, hashlib, io, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .env import VecGame


def shard_dirs(path: Path) -> list[Path]:
    if (path / "records.jsonl").exists():
        return [path]
    return sorted(p.parent for p in path.rglob("records.jsonl"))


def episodes(shard: Path) -> tuple[str, list[list[dict]]]:
    eps = defaultdict(list)
    with open(shard / "records.jsonl") as f:
        for line in f:
            r = json.loads(line); eps[r["seed"]].append(r)
    out = []
    for seed, rs in eps.items():
        rs.sort(key=lambda r: r["step"])
        if [r["step"] for r in rs] != list(range(len(rs))):
            raise SystemExit(f"{shard}: seed {seed} does not run from step 0 without gaps")
        out.append(rs)
    return out[0][0]["game"], out


def pixel_diff(a: bytes, b: bytes) -> int:
    x = np.asarray(Image.open(io.BytesIO(a)).convert("RGB"), dtype=np.int16)
    y = np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.int16)
    return 255 if x.shape != y.shape else int(np.abs(x - y).max())


async def replay(page, rs: list[dict], shard: Path, check: bool, stats: dict):
    for attempt in range(3):  # a page now and then fails to leave the start screen; that is a race, start again
        try:
            obs = await page.reset(rs[0]["seed"]); break
        except Exception as e:
            if attempt == 2: raise
            print(f"  seed {rs[0]['seed']}: reset failed ({str(e)[:80]}), retrying", flush=True)
    for i, r in enumerate(rs):
        if obs["score"] != r["score"] or abs(obs["t"] - r["t"]) > 1e-3:
            stats["drift"] += 1
            print(f"  {shard.name} seed {r['seed']} step {r['step']}: replay has score {obs['score']} t {obs['t']}, "
                  f"record has {r['score']} t {r['t']}; the rest of this episode is skipped", flush=True)
            return
        dst = shard / r["frame"]
        if "frame_md5" in r:
            stats["md5_ok" if hashlib.md5(obs["frame"]).hexdigest() == r["frame_md5"] else "md5_bad"] += 1
        if check:
            old = dst.read_bytes()
            if old == obs["frame"]: stats["same"] += 1
            else: stats["differ"] += 1; stats["max_px"] = max(stats["max_px"], pixel_diff(old, obs["frame"]))
        else:
            dst.write_bytes(obs["frame"])
        stats["frames"] += 1
        if i + 1 < len(rs):
            obs = await page.step(r["taken_action"])


async def rebuild(shard: Path, pages: int, check: bool) -> dict:
    game, eps = episodes(shard)
    (shard / "frames").mkdir(exist_ok=True)
    total = sum(len(rs) for rs in eps)
    stats = {"frames": 0, "same": 0, "differ": 0, "max_px": 0, "drift": 0, "md5_ok": 0, "md5_bad": 0}
    queue = sorted(eps, key=len, reverse=True)  # longest first, so no page is left with a long tail
    t0, step = time.time(), max(1, total // 10); next_report = step
    async with VecGame(game, n=min(pages, len(eps))) as env:
        async def worker(page):
            nonlocal next_report
            while queue:
                await replay(page, queue.pop(0), shard, check, stats)
                if stats["frames"] >= next_report:
                    print(f"  {shard}: {stats['frames']}/{total} frames, {stats['frames'] / (time.time() - t0):.0f}/s", flush=True)
                    next_report += step
        await asyncio.gather(*(worker(p) for p in env.pages))
    line = f"[{shard}] {len(eps)} episodes, {stats['frames']}/{total} frames in {time.time() - t0:.0f} s"
    if stats["md5_ok"] or stats["md5_bad"]:
        line += f"; {stats['md5_ok']} match the released frame_md5"
    if check:
        line += f"; byte-identical {stats['same']}, different {stats['differ']} (max pixel difference {stats['max_px']})"
    print(line + (f"; {stats['drift']} episodes drifted" if stats["drift"] else ""), flush=True)
    return stats


async def main(a):
    bad = frames = md5_ok = 0
    for shard in shard_dirs(Path(a.path)):
        s = await rebuild(shard, a.pages, a.check)
        bad += s["drift"] + s["differ"]; frames += s["frames"]; md5_ok += s["md5_ok"]
    print(f"{frames} frames, {md5_ok} byte for byte the frames we trained on")
    if bad:
        raise SystemExit(f"{bad} episodes drifted or frames differed, see above")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path", help="a shard directory (holding records.jsonl) or any directory above shards")
    p.add_argument("--pages", type=int, default=8, help="browser pages replaying episodes in parallel")
    p.add_argument("--check", action="store_true", help="compare with the frames on disk instead of writing")
    asyncio.run(main(p.parse_args()))
