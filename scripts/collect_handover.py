#!/usr/bin/env python3
"""Collect a System One / System Two handover sweep into one results file.

    python scripts/collect_handover.py --model dagger1 runs/logs/handover_dagger1_*.log
    python scripts/collect_handover.py --model dagger1 --append runs/logs/hrand_*.log

The sweep and its control write runs/play/<game>_local_handover<tau>.json and <game>_local_hrandom<rate>.json.
Those paths carry no model name, so a second model overwrites the first; the job log does not, and every play run
prints its result as one JSON line. Reading the logs keeps each number attached to the checkpoint that produced it.

Written to runs/results/handover_<model>.json: one row per run with the game, the threshold (tau) or the control
rate, the share of steps actually handed over, and the score. --append merges into an existing file, replacing rows
with the same (game, tau, random) key.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ["game", "handover_tau", "handover_random", "handover_invert", "handover_rate", "handed_steps", "score_mean", "score_median",
        "score_max", "episodes", "capped", "conf_mean", "steps"]


def key(r: dict) -> tuple:
    """Runs recorded before --handover-invert existed carry no such field, which means the same as False."""
    return (r["game"], r.get("handover_tau"), r.get("handover_random"), bool(r.get("handover_invert")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--model", required=True, help="checkpoint name the sweep ran with, e.g. dagger1")
    ap.add_argument("--append", action="store_true", help="merge into the existing file instead of replacing it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / "runs" / "results" / f"handover_{a.model}.json"
    rows: dict[tuple, dict] = {}
    if a.append and out.is_file():
        for r in json.loads(out.read_text()):
            rows[key(r)] = r
    found = 0
    for pat in a.logs:
        for path in sorted(Path().glob(pat)) or [Path(pat)]:
            if not path.is_file():
                print(f"[skip] {path} not found", file=sys.stderr); continue
            for line in path.read_text(errors="replace").splitlines():
                if not line.startswith('{"game"'):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "handover_rate" not in d:
                    continue
                row = {k: d[k] for k in KEEP if k in d}
                row["model"] = a.model
                row["log"] = path.name
                rows[key(row)] = row
                found += 1
    ordered = sorted(rows.values(), key=lambda r: (r["game"], bool(r.get("handover_invert")), r.get("handover_random") is not None, r["handover_rate"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not ordered and out.is_file() and json.loads(out.read_text()):
        raise SystemExit(f"refusing to replace {out} with nothing: the logs matched no runs (check the paths, or "
                         f"quote a glob the shell has already expanded to nothing)")
    out.write_text(json.dumps(ordered, indent=1))
    print(f"{found} runs read, {len(ordered)} rows -> {out}")


if __name__ == "__main__":
    main()
