#!/usr/bin/env python3
"""Does the model read the option text? Probe a checkpoint on validation frames with edited option lists.

    python scripts/probe_options.py --ckpt ckpt/sft_all1/final --shard sft_all1_a --per-game 300
    python scripts/probe_options.py --ckpt ... --games snake,tetris --out runs/results/probe_options.json

For each game, N validation frames (episode seed % 10 == 0, never trained on) with their teacher targets are scored
under these option lists (the frozen prompt is unchanged, only the lettered options differ):

  baseline    the hook's options, hook order (what play.py uses)
  shuffled    the same options in a fixed random order (order invariance; training permuted every sample)
  renamed     names replaced by neutral words (alpha, bravo, ...), descriptions kept: does the decision follow the text?
  nameonly    descriptions dropped (the option is just its name): how much the descriptions carry
  swapped     descriptions rotated one option along while the names stay: name or description, which one wins?
  dropped     the teacher's best option removed: mass should go to the teacher's second choice, not anywhere
  distractor  one extra option appended ("hold: keep the current move and do nothing new"): it should get ~0 mass

Per variant and game: agreement with the teacher (argmax in the teacher's top set, mapped back to the original
options where the list was edited), mean probability on the distractor / on the removed option's successor, and the
mean total variation from the baseline distribution. JSON to --out, table to stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playjev.model import PlayJevModel  # noqa: E402

GAMES = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]
ACTION_RE = re.compile(r"""name:\s*'([^']+)'\s*,\s*description:\s*'([^']+)'""")
NEUTRAL = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
DISTRACTOR = {"name": "hold", "description": "keep the current move and do nothing new"}


def hook_actions(game: str) -> list[dict]:
    js = (ROOT / "games" / game / "pj_hook.js").read_text()
    acts = [{"name": n, "description": d} for n, d in ACTION_RE.findall(js)]
    if not acts:
        raise SystemExit(f"no options found in games/{game}/pj_hook.js")
    return acts


def load_val(game: str, shard: str, n: int, seed: int) -> list[dict]:
    path = ROOT / "data" / game / shard / "records.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(l) for l in path.open()]
    rows = [r for r in rows if r["seed"] % 10 == 0]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def tv(p, q):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", default="sft_all1_a")
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--per-game", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs" / "results" / "probe_options.json"))
    a = ap.parse_args()
    model = PlayJevModel(a.ckpt, device=a.device).load()
    results = {"ckpt": a.ckpt, "shard": a.shard, "per_game": a.per_game, "games": {}}
    for g in a.games.split(","):
        rows = load_val(g, a.shard, a.per_game, a.seed)
        if not rows:
            print(f"[{g}] no validation records in data/{g}/{a.shard}, skipped", flush=True); continue
        base = hook_actions(g); K = len(base)
        frames = [(ROOT / "data" / g / a.shard / r["frame"]).read_bytes() for r in rows]
        targets = [r["teacher_probs"] for r in rows]
        top = [[j for j, x in enumerate(t) if x >= max(t) - 1e-6] for t in targets]
        perm = list(range(K)); random.Random(a.seed + 1).shuffle(perm)
        variants = {
            "baseline": (base, list(range(K))),
            "shuffled": ([base[j] for j in perm], perm),
            "renamed": ([{"name": NEUTRAL[j], "description": o["description"]} for j, o in enumerate(base)], list(range(K))),
            "nameonly": ([{"name": o["name"], "description": o["name"]} for o in base], list(range(K))),
            "swapped": ([{"name": o["name"], "description": base[(j + 1) % K]["description"]} for j, o in enumerate(base)], list(range(K))),
            "distractor": (base + [DISTRACTOR], list(range(K)) + [None]),
        }
        out = {}; t0 = time.time(); base_probs = None
        for name, (opts, mapping) in variants.items():
            dec = model.decide(frames, opts, batch_size=a.batch)
            probs = [d.probs for d in dec]
            # map the variant's slots back onto the original option indices (None = the distractor slot)
            mapped = []
            for p in probs:
                q = [0.0] * K; extra = 0.0
                for slot, orig in enumerate(mapping):
                    if orig is None: extra += p[slot]
                    else: q[orig] += p[slot]
                mapped.append((q, extra))
            if name == "baseline":
                base_probs = [q for q, _ in mapped]
            agree = statistics.fmean(max(range(K), key=q.__getitem__) in top[i] for i, (q, _) in enumerate(mapped))
            entry = {"agreement_top": round(agree, 4), "pmax": round(statistics.fmean(max(q) for q, _ in mapped), 4),
                     "tv_from_baseline": round(statistics.fmean(tv(q, base_probs[i]) for i, (q, _) in enumerate(mapped)), 4)}
            if name == "distractor":
                entry["distractor_mass"] = round(statistics.fmean(e for _, e in mapped), 4)
                entry["distractor_argmax_share"] = round(statistics.fmean(e > max(q) for q, e in mapped), 4)
            if name == "swapped":
                # follows the name: argmax stays where the baseline argmax was; follows the description: it moves
                # to the slot now carrying that description (one further along)
                bm = [max(range(K), key=base_probs[i].__getitem__) for i in range(len(rows))]
                am = [max(range(K), key=q.__getitem__) for q, _ in mapped]
                entry["follows_name"] = round(statistics.fmean(am[i] == bm[i] for i in range(len(rows))), 4)
                entry["follows_description"] = round(statistics.fmean(am[i] == (bm[i] - 1) % K for i in range(len(rows))), 4)
            out[name] = entry
        # dropped: remove the teacher's best option (the first of its top set) from the list, frame by frame
        # (one forward per distinct removal index, batched by index)
        drop_agree, drop_second = [], []
        by_idx: dict[int, list[int]] = {}
        for i, t in enumerate(top):
            by_idx.setdefault(t[0], []).append(i)
        for idx, members in by_idx.items():
            opts = [o for j, o in enumerate(base) if j != idx]; mapping = [j for j in range(K) if j != idx]
            dec = model.decide([frames[i] for i in members], opts, batch_size=a.batch)
            for i, d in zip(members, dec):
                q = [0.0] * K
                for slot, orig in enumerate(mapping): q[orig] = d.probs[slot]
                am = max(range(K), key=q.__getitem__)
                rest = [j for j in top[i] if j != idx]  # remaining teacher-best options (ties)
                second = max((j for j in range(K) if j != idx), key=targets[i].__getitem__)
                drop_agree.append(am in rest if rest else am == second)
                drop_second.append(am == second)
        out["dropped"] = {"agreement_with_teacher_second": round(statistics.fmean(drop_second), 4),
                          "agreement_with_remaining_top": round(statistics.fmean(drop_agree), 4)}
        results["games"][g] = {"n": len(rows), "K": K, "variants": out, "seconds": round(time.time() - t0, 1)}
        print(f"[{g}] n={len(rows)} K={K} " + " ".join(f"{k}={v['agreement_top']:.3f}" for k, v in out.items() if "agreement_top" in v)
              + f" swapped:name={out['swapped']['follows_name']:.2f}/desc={out['swapped']['follows_description']:.2f}"
              + f" distractor_mass={out['distractor']['distractor_mass']:.3f} dropped->2nd={out['dropped']['agreement_with_teacher_second']:.3f}"
              + f" ({out['baseline']['pmax']:.2f} pmax, {time.time() - t0:.0f} s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
