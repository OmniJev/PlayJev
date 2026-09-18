#!/usr/bin/env python3
"""Gather the closed-loop numbers of one training run into runs/results/<run>.json for the demo and the README.

    python scripts/collect_results.py --run sft_all1 --ckpt-dir runs/ckpt/sft_all1 --play-dir runs/play

Inputs are playjev.play result files (one JSON object each: game, policy, delay, skip_noop, score_mean, score_median,
len_mean, capped, conf_mean, steps_per_s, ...), copied by the hpc jobs into the checkpoint directory because
runs/play/<game>_local.json is overwritten by each variant:
  <ckpt-dir>/play_<game>_local*.json            trained model -> policy_name playjev-0.8b-<run> plus "-sampled" when the
                                                file name says so, "-delay<d>" when delay > 0, and "-plain" when the
                                                result was played without the no-op rule (skip_noop false or absent)
  <ckpt-dir>/play_<game>_random*.json           random on the same seeds -> random (or random-plain)
  <ckpt-dir>/play_<game>_teacher.json or <play-dir>/<game>_{random,teacher}.json -> the references on the same seeds
Also prints the table with the teacher-normalised score.
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
    def add(d, pol, src):
        rows.append({**d, "policy_name": pol, "run": a.run, "source": src})
    for g in GAMES:
        for f in sorted(ck.glob(f"play_{g}_local*.json")):
            d = load(f)
            if d is None:
                continue
            pol = name + ("-sampled" if "_sampled" in f.stem else "") + (f"-delay{d['delay']}" if d.get("delay") else "") \
                + ("" if d.get("skip_noop") else "-plain")
            add(d, pol, f.name)
        for f in sorted(ck.glob(f"play_{g}_random*.json")):
            d = load(f)
            if d is not None:
                add(d, "random" if d.get("skip_noop") else "random-plain", f.name)
        for ref in ("random", "teacher"):
            d = load(ck / f"play_{g}_{ref}.json") if ref == "teacher" else None
            src = f"play_{g}_{ref}.json"
            if d is None:
                d = load(pl / f"{g}_{ref}.json"); src = f"{g}_{ref}.json"
            if d is None or (ref == "random" and any(r["game"] == g and r["policy_name"].startswith("random") for r in rows)):
                continue
            add(d, ref if (ref == "teacher" or d.get("skip_noop")) else "random-plain", src)
    out = Path(a.out) if a.out else ROOT / "runs" / "results" / f"{a.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    by = {}
    for r in rows:
        by.setdefault(r["game"], {})[r["policy_name"]] = r
    cols = ["random", "random-plain", name, name + "-plain", name + "-sampled-plain", name + "-sampled", name + "-delay1", name + "-delay1-plain", "teacher"]
    cols = [c for c in cols if any(c in p for p in by.values())]
    short = lambda c: c.replace(name, "model").replace("-sampled", "-smp").replace("-delay", "-d").replace("-plain", "-pl")
    print(f"{'game':9s} " + " ".join(f"{short(c):>10s}" for c in cols) + "  vs teacher  capped(model)")
    for g in GAMES:
        p = by.get(g, {})
        if not p:
            continue
        s = lambda k: (f"{p[k]['score_mean']:10.1f}" if k in p else f"{'-':>10s}")
        rnd = p.get("random") or p.get("random-plain"); vs = ""
        if name in p and rnd and "teacher" in p and p["teacher"]["score_mean"] != rnd["score_mean"]:
            vs = f"{(p[name]['score_mean'] - rnd['score_mean']) / (p['teacher']['score_mean'] - rnd['score_mean']):.2f}"
        cap = f"{p[name]['capped']}/{p[name]['episodes']}" if name in p else ""
        print(f"{g:9s} " + " ".join(s(c) for c in cols) + f"  {vs:>10s}  {cap}")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
