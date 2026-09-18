#!/usr/bin/env python3
"""Gather the closed-loop numbers of one training run into runs/results/<run>.json for the demo and the README.

    python scripts/collect_results.py --run sft_all1 --ckpt-dir runs/ckpt/sft_all1 --play-dir runs/play

Inputs are playjev.play result files (one JSON object each: game, policy, score_mean, score_median, len_mean,
capped, conf_mean, steps_per_s, ...):
  <ckpt-dir>/play_<game>_local.json           trained model, argmax          -> policy_name playjev-0.8b-<run>
  <ckpt-dir>/play_<game>_local_sampled.json   trained model, sampled actions -> playjev-0.8b-<run>-sampled
  <ckpt-dir>/play_<game>_local_delay<d>.json  trained model, argmax, delay d -> playjev-0.8b-<run>-delay<d>
  <play-dir>/<game>_random.json, <game>_teacher.json   the references on the same seeds -> random, teacher
(the hpc job copies the model results into the checkpoint directory because runs/play/<game>_local.json is
overwritten by each variant). Also prints the table with the teacher-normalised score.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]


def load(p: Path):
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) and "score_mean" in d else None
    except (OSError, json.JSONDecodeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt-dir", required=True, help="directory holding the job's play_<game>_*.json copies")
    ap.add_argument("--play-dir", default=str(ROOT / "runs" / "play"))
    ap.add_argument("--model-name", default=None, help="policy name prefix (default playjev-0.8b-<run>)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    name = a.model_name or f"playjev-0.8b-{a.run}"
    ck, pl = Path(a.ckpt_dir), Path(a.play_dir)
    rows = []
    for g in GAMES:
        for f in sorted(ck.glob(f"play_{g}_local*.json")):
            d = load(f)
            if d is None:
                continue
            suf = f.stem[len(f"play_{g}_local"):]
            m = re.fullmatch(r"(?:_sampled|_delay(\d+)|-delay(\d+))?", suf)
            if m is None:
                continue
            delay = m.group(1) or m.group(2)
            pol = name + ("-sampled" if suf == "_sampled" else f"-delay{delay}" if delay else "")
            rows.append({**d, "policy_name": pol, "run": a.run, "source": f.name})
        for ref in ("random", "teacher"):
            d = load(pl / f"{g}_{ref}.json")
            if d is not None:
                rows.append({**d, "policy_name": ref, "run": a.run, "source": f"{g}_{ref}.json"})
    out = Path(a.out) if a.out else ROOT / "runs" / "results" / f"{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    by = {}
    for r in rows:
        by.setdefault(r["game"], {})[r["policy_name"]] = r
    print(f"{'game':9s} {'random':>8s} {'model':>8s} {'sampled':>8s} {'delay1':>8s} {'teacher':>8s}  vs teacher  capped(model)")
    for g in GAMES:
        p = by.get(g, {})
        if not p:
            continue
        s = lambda k: (f"{p[k]['score_mean']:8.1f}" if k in p else f"{'-':>8s}")
        vs = ""
        if name in p and "random" in p and "teacher" in p and p["teacher"]["score_mean"] != p["random"]["score_mean"]:
            vs = f"{(p[name]['score_mean'] - p['random']['score_mean']) / (p['teacher']['score_mean'] - p['random']['score_mean']):.2f}"
        cap = f"{p[name]['capped']}/{p[name]['episodes']}" if name in p else ""
        print(f"{g:9s} {s('random')} {s(name)} {s(name + '-sampled')} {s(name + '-delay1')} {s('teacher')}  {vs:>10s}  {cap}")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
