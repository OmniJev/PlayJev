"""Render a motion-composited copy of an existing shard: same labels, same seeds, new frames.

The shard's records already carry (seed, episode, step), so the k most recent frames of an episode are
recoverable without collecting anything again. Each record's frame is replaced by the composite of that
record's frame and its k-1 predecessors inside the same episode (clamped at the episode start), and the
labels, seeds, episode and step fields are copied through unchanged, so the validation split stays the
same set of seeds and the trainer needs no new flag: it reads the new shard with --shards.

    python scripts/make_motion_shard.py breakout --src sft_all1_a --mode ghost --dst m_ghost_a
    python scripts/make_motion_shard.py breakout --src sft_all1_a --mode rgbt  --dst m_rgbt_a
"""
import argparse, json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from playjev.motion import K, MODES, compose

ROOT = Path(__file__).resolve().parents[1]


def job(args):
    src_dir, dst_dir, mode, k, group = args  # group: list of (index, frame_rel) in step order
    hist = []
    for _, rel in group:
        hist.append(Image.open(src_dir / rel).convert("RGB"))
        if len(hist) > k:
            hist.pop(0)
        out = compose(hist, mode)
        dest = dst_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.save(dest, quality=92)
    return len(group)


def main(a):
    src_dir = ROOT / "data" / a.game / a.src
    dst_dir = ROOT / "data" / a.game / a.dst
    recs = [json.loads(l) for l in open(src_dir / "records.jsonl")]
    by_ep = defaultdict(list)
    for i, r in enumerate(recs):
        by_ep[(r["seed"], r["episode"])].append((r["step"], i, r["frame"]))
    groups = []
    for key in sorted(by_ep):
        items = sorted(by_ep[key])
        groups.append([(i, rel) for _, i, rel in items])
    (dst_dir / "frames").mkdir(parents=True, exist_ok=True)
    print(f"{a.game}/{a.src} -> {a.dst}: {len(recs)} records, {len(groups)} episodes, mode {a.mode}, k {a.k}", flush=True)
    t0, done, mark = time.time(), 0, 0
    with Pool(a.workers) as pool:
        for n in pool.imap_unordered(job, [(src_dir, dst_dir, a.mode, a.k, g) for g in groups]):
            done += n
            if done * 100 // max(1, len(recs)) >= mark:
                el = time.time() - t0
                print(f"  {mark:3d}%  {done}/{len(recs)} frames  {el:.0f}s  eta {el / max(1, done) * (len(recs) - done):.0f}s", flush=True)
                mark += 5
    with open(dst_dir / "records.jsonl", "w") as f:
        for r in recs:
            r = dict(r, shard=a.dst, prev_frame=None)
            f.write(json.dumps(r) + "\n")
    print(f"done {len(recs)} frames in {time.time() - t0:.0f}s -> {dst_dir}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("game"); p.add_argument("--src", required=True); p.add_argument("--dst", required=True)
    p.add_argument("--mode", required=True, choices=list(MODES)); p.add_argument("--k", type=int, default=K)
    p.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    main(p.parse_args())
