#!/usr/bin/env python3
"""Replay recorded episodes through the games with the teacher watching: on-policy agreement and calibration.

    python scripts/relabel_replays.py                                  # every runs/replays/<game>/playjev*_<seed>.json
    python scripts/relabel_replays.py --games snake,2048 --policy playjev-0.8b-sft_all1 --pages 8

For every step of a recording the teacher (reading the game's internal state, as in collection) judges the frame the
model saw, and the model's recorded probabilities are scored against that judgement: agreement (the taken move is the
teacher's argmax), agreement with the teacher's top set (ties), ECE over 15 bins and Brier of p_max, plus the
reliability bins the demo page draws. Because the recordings are replayed through the real game, the score at every
step is checked against the recording too (a mismatch means the replay is not deterministic and is reported).

Output: runs/results/<policy>_relabel.json, a list with one entry per game in the shape scripts/build_demo.py reads
({"game", "policy_name", "calibration": {"bins": [...]}, ...}); the closed-loop score rows come from runs/play.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playjev.env import VecGame  # noqa: E402
from playjev.teachers import make_teacher  # noqa: E402

GAMES = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]
BINS = 15


def ece(conf, correct, bins=BINS):
    n = len(conf); total = 0.0; out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = statistics.fmean(correct[i] for i in idx); avg = statistics.fmean(conf[i] for i in idx)
            total += len(idx) / n * abs(acc - avg); out.append({"p": round(avg, 4), "acc": round(acc, 4), "n": len(idx)})
    return total, out


async def replay_one(page, teachers, rec):
    """Returns per-step (pmax, taken == teacher argmax, taken in teacher top set) and the determinism check."""
    obs = await page.reset(rec["seed"])
    if not teachers:  # the action list (with indices) exists once a page has started
        teachers.append(make_teacher(rec["game"], page.actions))
    teacher = teachers[0]; teacher.reset()
    rows, mismatch = [], None
    for i, st in enumerate(rec["steps"]):
        tp = teacher.act(obs)
        best = max(range(len(tp)), key=tp.__getitem__); top = max(tp)
        a = st["a"]; p = st["p"]
        rows.append((max(p), a == best, tp[a] >= top - 1e-6))
        obs = await page.step(a)
        if obs["score"] != st["score"] and mismatch is None:
            mismatch = {"step": i, "replayed": obs["score"], "recorded": st["score"]}
        if obs["done"] and i + 1 < len(rec["steps"]):
            mismatch = mismatch or {"step": i, "replayed": "done", "recorded": "continues"}; break
    return rows, mismatch, obs["score"]


async def run_game(game, files, pages, log):
    async with VecGame(game, n=pages) as env:
        queue = list(files); results = []
        async def worker(page):
            teachers = []  # one per page, created after its first reset
            while queue:
                f = queue.pop(0); rec = json.loads(f.read_text())
                rows, mismatch, final = await replay_one(page, teachers, rec)
                results.append((f.name, rec, rows, mismatch, final))
                log(f"    {f.name}: {len(rows)} steps, final {final} (recorded {rec['final_score']})"
                    + (f", MISMATCH at step {mismatch['step']}: {mismatch}" if mismatch else ""))
        await asyncio.gather(*(worker(p) for p in env.pages))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replays", default=str(ROOT / "runs" / "replays"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "results"))
    ap.add_argument("--policy", default="playjev-0.8b-sft_all1", help="exact policy name of the recordings to relabel")
    ap.add_argument("--games", default=None, help="comma-separated subset")
    ap.add_argument("--pages", type=int, default=8)
    a = ap.parse_args()
    games = a.games.split(",") if a.games else GAMES
    log = lambda s: print(s, flush=True)
    summary = []
    for g in games:
        d = Path(a.replays, g)  # exactly <policy>_<seed>.json (a prefix match would also catch <policy>-delay1_<seed>.json)
        files = sorted(f for f in d.glob(f"{a.policy}_*.json") if f.stem[len(a.policy) + 1:].isdigit()) if d.is_dir() else []
        if not files:
            log(f"[{g}] no recordings for policy {a.policy}*"); continue
        pol = json.loads(files[0].read_text())["policy"]
        log(f"[{g}] {len(files)} recordings of {pol}")
        t0 = time.time()
        results = asyncio.run(run_game(g, files, min(a.pages, len(files)), log))
        conf = [r[0] for _, _, rows, _, _ in results for r in rows]
        agree = [r[1] for _, _, rows, _, _ in results for r in rows]
        top = [r[2] for _, _, rows, _, _ in results for r in rows]
        e, bins = ece(conf, top)
        entry = {"game": g, "policy_name": pol, "episodes": len(results), "steps": len(conf),
                 "agreement": round(statistics.fmean(agree), 4), "agreement_top": round(statistics.fmean(top), 4),
                 "ece": round(e, 4), "brier": round(statistics.fmean((c - float(ok)) ** 2 for c, ok in zip(conf, top)), 4),
                 "pmax_mean": round(statistics.fmean(conf), 4),
                 "mismatches": [{"file": n, **m} for n, _, _, m, _ in results if m],
                 "calibration": {"bins": bins, "correct": "taken move in the teacher's top set", "source": "replays"},
                 "seconds": round(time.time() - t0, 1)}
        summary.append(entry)
        log(f"[{g}] agreement {entry['agreement']:.3f} (top set {entry['agreement_top']:.3f}), ECE {entry['ece']:.3f}, "
            f"Brier {entry['brier']:.3f}, p_max {entry['pmax_mean']:.3f}, {len(conf)} steps, "
            f"{len(entry['mismatches'])} mismatches, {entry['seconds']} s")
    if summary:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        name = summary[0]["policy_name"].replace("/", "_"); path = out / f"{name}_relabel.json"
        if path.is_file():  # a partial run (--games) replaces those games' entries and keeps the rest
            done = {e["game"] for e in summary}
            try:
                summary = [e for e in json.loads(path.read_text()) if e.get("game") not in done] + summary
            except json.JSONDecodeError:
                pass
            summary.sort(key=lambda e: GAMES.index(e["game"]) if e["game"] in GAMES else 99)
        path.write_text(json.dumps(summary, indent=1))
        log(f"wrote {path} ({len(summary)} games)")


if __name__ == "__main__":
    main()
