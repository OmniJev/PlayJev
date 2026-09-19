#!/usr/bin/env python3
"""Collect the transfer runs (fine-tune one held-out game from two initial checkpoints) into one results file.

    rsync -a --include='tr_*/' --include='play_*.json' --include='final_eval.json' --exclude='*' \
        <cluster>:<ckpt root>/ runs/results/transfer/
    python scripts/collect_transfer.py

Each run directory is named tr_<game>_<frames>_<init>[_lr<lr>] and holds final_eval.json (agreement with the
teacher on the 2000-frame validation split of that game, never trained on) and play_<game>_local_delay0.json (the
closed loop on 16 held-out episodes). init "base" starts from Qwen3.5-0.8B-Base, "hold8" from the eight-game model
that never saw this game, "shuf" from the same eight games with their targets dealt out at random.

Written to runs/results/transfer.json: per game the teacher and random reference scores, and per (init, frames) the
agreement, the score and the score as a fraction of the teacher's.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^tr_([a-z0-9]+)_(\d+)_(base|hold8|shuf)(?:_lr(\S+))?$")


def ref(game: str, kind: str) -> float | None:
    p = ROOT / "runs" / "play" / f"{game}_{kind}.json"
    return json.loads(p.read_text())["score_mean"] if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "runs" / "results" / "transfer"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "results" / "transfer.json"))
    a = ap.parse_args()
    games: dict[str, dict] = {}
    for d in sorted(Path(a.dir).iterdir()):
        m = RUN_RE.match(d.name)
        if not d.is_dir() or not m:
            continue
        game, frames, init, lr = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        ev, pl = d / "final_eval.json", d / f"play_{game}_local_delay0.json"
        row = {"init": init, "frames": frames, "run": d.name}
        if lr:
            row["lr"] = lr
        if ev.is_file():
            e = json.loads(ev.read_text()).get(game, {})
            for k in ("agreement", "loss", "ece", "confidence"):
                if k in e:
                    row[k] = round(e[k], 4)
        if pl.is_file():
            p = json.loads(pl.read_text())
            row.update(score=p["score_mean"], score_median=p["score_median"], episodes=p["episodes"],
                       len_mean=p["len_mean"], conf_mean=round(p["conf_mean"], 4))
        games.setdefault(game, {"game": game, "teacher": ref(game, "teacher"), "random": ref(game, "random"),
                                "runs": []})["runs"].append(row)
    for g in games.values():
        t, r = g["teacher"], g["random"]
        for row in g["runs"]:
            if "score" in row and t is not None and r is not None and t != r:
                row["vs_teacher"] = round((row["score"] - r) / (t - r), 3)
        g["runs"].sort(key=lambda x: (x["init"], x.get("lr") or "", x["frames"]))
    out = {"note": "Fine-tune one held-out game from two initial checkpoints on N of its frames: base is "
                   "Qwen3.5-0.8B-Base, hold8 is the eight-game model that never saw this game, shuf is the same "
                   "eight games with their targets dealt out at random (answer format and option text, no game "
                   "skill). Agreement is with the teacher on 2000 validation frames of that game; score is 16 "
                   "held-out episodes; vs teacher is (score - random) / (teacher - random).",
           "games": [games[g] for g in sorted(games)]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{sum(len(g['runs']) for g in out['games'])} runs across {len(out['games'])} games -> {a.out}")


if __name__ == "__main__":
    main()
