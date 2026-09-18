#!/usr/bin/env python3
"""Assemble demo/ (the PlayJev GitHub Pages demo, docs/DEMO.md).

    python scripts/build_demo.py                       # replays from runs/replays, model results from runs/results
    python scripts/build_demo.py --replays DIR --results DIR --games snake,2048

What it writes (everything else in demo/ is hand-written source: index.html, demo.css, demo.js, pj_bridge.js):

  demo/games/_shared/pj_shim.js         copy of the training shim
  demo/games/<id>/pj_hook.js            copy of the training hook
  demo/games/<id>/<entry>               copy of the vendored entry page whose <head> starts with the shim, the hook
                                        and the bridge (same order as the driver's add_init_script), external
                                        <script>/<link> tags removed (the page must not load anything off-origin)
  demo/games/<id>/...                   only the vendored files the entry page needs: script/link/img references,
                                        css url() and JS string references walked recursively, RequireJS module names,
                                        plus per-game extras (assets built from string concatenation). Audio is left
                                        out (the training driver aborts audio requests, so the model never had it; the
                                        shim mutes it anyway), as are swf, psd, packaged builds and files over 2 MB.
  demo/games/index.json                 manifest: id, title, entry, viewport, view selector, step timing, actions, prompt
  demo/replays/<game>/<policy>_<seed>.json   copies of the replay files found (docs/DEMO.md format), validated
  demo/replays/<game>/<policy>_<seed>.js     the same wrapped in PJ_DEMO_REPLAY(...) so file:// (no fetch) works
  demo/replays/index.json               per game: the recordings found (policy, seed, steps, final score)
  demo/results.json                     results table: docs/BASELINES.md plus runs/results/*.json for the model
  demo/data.js                          the three JSON files above as window.PJ_DEMO (script tag, works from file://)

Rebuild after new replays land:  python scripts/build_demo.py   (then python scripts/shot_demo.py to check).
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "games"
DEMO = ROOT / "demo"
ORDER = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]  # GAMES.md order

# Per game: the element whose box the tile shows (the game area as the model saw it), cosmetic overlays to hide
# inside the iframe (DOM only, never game state), and assets the reference walk cannot see (built by string
# concatenation in the game's code).
VIEW = {
    "mario": dict(view="#canvas"),
    "snake": dict(view=".snake-playing-field", hide=".snake-toolbar,.snake-gamepad{display:none!important}"),
    "tetris": dict(view="#canvas"),
    "2048": dict(view=".game-container"),
    "flappy": dict(view="#gamescreen", hide="#footer{display:none!important}", extra=["assets/*.png"]),
    "pacman": dict(view="#pacman canvas"),
    "breakout": dict(view="#canvas", hide="#levels,#controls,#instructions{display:none!important}"),
    "invaders": dict(view="canvas"),
    "racer": dict(view="#canvas", hide="#hud{display:none!important}", extra=["images/background.png", "images/sprites.png"]),
    "sokoban": dict(view="canvas"),
}
# Audio stays out on purpose: the training driver aborts every audio request (playjev/env.py), so the pages the model
# saw had no sound files either; the shim mutes audio and no hook waits for it. Missing audio only shows as 404s in the console.
AUDIO_EXT = {".mp3", ".ogg", ".wav", ".m4a", ".mid", ".midi", ".oga", ".flac"}
SKIP_EXT = AUDIO_EXT | {".swf", ".psd", ".zip", ".scss", ".md", ".py", ".patch"}
MAX_FILE_BYTES = 2_000_000
SKIP_DIRS = {"packaged", ".git", "__pycache__", "node_modules"}
REF_EXT = r"js|css|html|png|jpe?g|gif|svg|ico|ttf|woff2?|eot|otf|json|webp|bmp"
REF_RE = re.compile(r"(?<![\w:/\\-])((?:\.\.?/)*[\w][\w.\-/ ]*?\.(?:" + REF_EXT + r"))(?![\w/])", re.I)
MODULE_RE = re.compile(r"""['"]((?:[\w-]+/)+[\w.-]+)['"]""")  # RequireJS names such as 'state/Load', 'lib/phaser.min'
EXTERNAL_TAG_RE = re.compile(r"""<(script|link)\b[^>]*?\b(?:src|href)\s*=\s*["']https?://[^"']+["'][^>]*>(?:\s*</script>)?""", re.I)
ACTION_RE = re.compile(r"""name:\s*'([^']+)'\s*,\s*description:\s*'([^']+)'""")

SYSTEM_PROMPT = ("Apply the question to the state. Choose exactly one of the listed options. "
                 "Respond with only its uppercase letter, with no explanation or reasoning.")
INSTRUCTIONS = "Which move should the player make next?"
FRAME_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LINKS = {"repo": "https://github.com/OmniJev/PlayJev", "openjev": "https://github.com/OmniJev/openJev",
         "awesome": "https://omnijev.github.io/awesome-jev/"}


def log(msg: str) -> None:
    print(msg, flush=True)


def relpath(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def plain_prompt(actions: list[dict]) -> str:
    """Same text as playjev.model.build_plain_prompt (kept in sync by the MODEL_NOTES check below)."""
    lines = [f"{L}. {a['name']}: {a['description']}" for L, a in zip(LETTERS, actions)]
    letters = ", ".join(LETTERS[: len(actions)])
    return (f"{SYSTEM_PROMPT}\n\n<state>\n{FRAME_PLACEHOLDER}\n</state>\n\nQuestion: {INSTRUCTIONS}\n\nOptions:\n"
            + "\n".join(lines) + f"\n\nAnswer with one letter: {letters}.\nAnswer:")


def hook_actions(hook_js: str) -> list[dict]:
    acts = [{"name": n, "description": d} for n, d in ACTION_RE.findall(hook_js)]
    if not acts:
        raise SystemExit("no `name: '...', description: '...'` pairs found in a hook")
    return acts


# ----------------------------------------------------------------------------------------------------- assets
class Copier:
    def __init__(self, src_root: Path, dst_root: Path, require_base: str | None):
        self.src_root, self.dst_root, self.require_base = src_root, dst_root, require_base
        self.copied: set[Path] = set()
        self.skipped: list[str] = []

    def _candidates(self, ref: str, from_dir: Path) -> list[Path]:
        ref = ref.split("?")[0].split("#")[0].strip()
        if not ref or "://" in ref or ref.startswith(("data:", "//", "javascript:")):
            return []
        if ref.startswith("/"):
            return [self.src_root / ref.lstrip("/")]
        return [from_dir / ref, self.src_root / ref]

    def copy_file(self, src: Path) -> bool:
        try:
            src = src.resolve()
            rel = src.relative_to(self.src_root.resolve())
        except (ValueError, OSError):
            return False
        if not src.is_file() or src in self.copied:
            return src in self.copied
        if any(part in SKIP_DIRS for part in rel.parts) or src.suffix.lower() in SKIP_EXT or src.stat().st_size > MAX_FILE_BYTES:
            self.skipped.append(str(rel)); return False
        dst = self.dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.copied.add(src)
        if src.suffix.lower() in (".js", ".css", ".html", ".htm"):
            self.walk(src)
        return True

    def walk(self, f: Path) -> None:
        text = f.read_text(errors="replace")
        here = f.parent
        for ref in REF_RE.findall(text):
            for c in self._candidates(ref, here):
                if self.copy_file(c):
                    break
        if self.require_base is not None and f.suffix.lower() == ".js":
            for name in MODULE_RE.findall(text):
                self.copy_file(self.src_root / self.require_base / (name + ".js"))


def build_game(game_id: str, out_games: Path) -> dict:
    src = GAMES_DIR / game_id
    spec = json.loads((src / "pj.json").read_text())
    entry_rel = spec["entry"]
    entry_src = src / entry_rel
    if not entry_src.is_file():
        raise SystemExit(f"{game_id}: entry {entry_rel} missing")
    dst = out_games / game_id
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    html = entry_src.read_text()
    m = re.search(r"data-main\s*=\s*[\"']([^\"']+)[\"']", html)
    require_base = str(Path(m.group(1)).parent) if m else None
    cp = Copier(src, dst, require_base)

    # Entry page: drop off-origin tags, then put shim + hook + bridge at the very start of <head>.
    externals = EXTERNAL_TAG_RE.findall(html)
    for tag in EXTERNAL_TAG_RE.finditer(html):
        log(f"  {game_id}: removed external tag {tag.group(0)[:90]}")
    html = EXTERNAL_TAG_RE.sub("", html)
    entry_dir = (dst / entry_rel).parent
    rel = lambda target: os.path.relpath(target, entry_dir).replace(os.sep, "/")  # noqa: E731
    inject = (f'<script src="{rel(out_games / "_shared" / "pj_shim.js")}"></script>'
              f'<script src="{rel(dst / "pj_hook.js")}"></script>'
              f'<script src="{rel(DEMO / "pj_bridge.js")}"></script>')
    if not re.search(r"<head[^>]*>", html, re.I):
        raise SystemExit(f"{game_id}: entry page has no <head>")
    html = re.sub(r"(<head[^>]*>)", lambda mm: mm.group(1) + inject, html, count=1, flags=re.I)
    entry_dst = dst / entry_rel
    entry_dst.parent.mkdir(parents=True, exist_ok=True)
    entry_dst.write_text(html)
    cp.copied.add(entry_src.resolve())
    cp.walk(entry_src)                                   # references relative to the original entry
    if m:                                                # RequireJS entry module: data-main has no extension
        cp.copy_file(src / (m.group(1) + ".js"))
    shutil.copy2(src / "pj_hook.js", dst / "pj_hook.js")
    cp.copied.add((src / "pj_hook.js").resolve())
    cp.walk(src / "pj_hook.js")                          # hooks load textures and sprites too
    for pat in VIEW.get(game_id, {}).get("extra", []):
        hits = sorted(src.glob(pat))
        if not hits:
            log(f"  {game_id}: extra pattern {pat} matched nothing")
        for h in hits:
            cp.copy_file(h)
    for keep in ("LICENSE", "LICENSE.md", "LICENSE.txt", "UPSTREAM"):
        if (src / keep).is_file():
            shutil.copy2(src / keep, dst / keep)

    hook_js = (src / "pj_hook.js").read_text()
    actions = hook_actions(hook_js)
    step_frames = int(spec.get("step_frames") or 1)
    step_ms = 150.0 if step_frames <= 1 else round(step_frames * 1000 / 60, 2)
    n_bytes = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    log(f"  {game_id}: {len(cp.copied)} files, {n_bytes/1e6:.1f} MB, {len(actions)} actions, step {step_ms} ms"
        + (f", left out {len(cp.skipped)}: {', '.join(cp.skipped[:4])}" if cp.skipped else ""))
    return {
        "id": game_id, "title": spec.get("title", game_id), "upstream": spec.get("upstream"), "license": spec.get("license"),
        "entry": f"games/{game_id}/{entry_rel}", "viewport": spec["viewport"], "view": VIEW.get(game_id, {}).get("view"),
        "hide": VIEW.get(game_id, {}).get("hide"), "step_frames": step_frames, "step_ms": step_ms,
        "max_steps": spec.get("max_steps"), "actions": actions, "prompt": plain_prompt(actions),
        "externals_removed": len(externals),
        # ES module scripts are blocked from file:// in Chromium (CORS on a null origin); the page says so on that tile.
        "modules": bool(re.search(r"""<script[^>]+type\s*=\s*["']module["']""", html, re.I)),
    }


# ---------------------------------------------------------------------------------------------------- replays
def policy_rank(name: str) -> tuple:
    n = name.lower()
    return (0 if n.startswith("playjev") else 1 if n == "teacher" else 2 if n == "random" else 3, n)


def build_replays(replay_dirs: list[Path], games: dict[str, dict], out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    index: dict[str, list] = {g: [] for g in games}
    for rdir in replay_dirs:
        for f in sorted(rdir.glob("*/*.json")):
            game = f.parent.name
            if game not in games:
                log(f"  replay {f}: game {game} not in this build, skipped"); continue
            try:
                rec = json.loads(f.read_text())
            except json.JSONDecodeError as e:
                log(f"  replay {f}: bad JSON ({e}), skipped"); continue
            missing = [k for k in ("game", "policy", "seed", "actions", "steps", "final_score") if k not in rec]
            if missing:
                log(f"  replay {f}: missing {missing}, skipped"); continue
            names = [a["name"] for a in games[game]["actions"]]
            if list(rec["actions"]) != names:
                log(f"  replay {f}: action list {rec['actions']} differs from the hook's {names}, skipped"); continue
            if rec["game"] != game:
                log(f"  replay {f}: file says game {rec['game']}, folder says {game}, skipped"); continue
            bad = [i for i, s in enumerate(rec["steps"]) if not (0 <= int(s["a"]) < len(names)) or len(s["p"]) != len(names)]
            if bad:
                log(f"  replay {f}: {len(bad)} steps with a bad action or probability vector, skipped"); continue
            if any(e["policy"] == rec["policy"] and e["seed"] == rec["seed"] for e in index[game]):
                log(f"  replay {f}: duplicate {rec['policy']}_{rec['seed']}, first one kept"); continue
            name = f"{rec['policy']}_{rec['seed']}"
            (out / game).mkdir(exist_ok=True)
            body = json.dumps(rec, separators=(",", ":"))
            (out / game / f"{name}.json").write_text(body)
            (out / game / f"{name}.js").write_text(f"PJ_DEMO_REPLAY({body});\n")
            index[game].append({"policy": rec["policy"], "seed": rec["seed"], "file": f"{game}/{name}.json",
                                "steps": len(rec["steps"]), "final_score": rec["final_score"],
                                "truncated": bool(rec.get("truncated", False)),
                                "frames_per_step": rec.get("frames_per_step")})
    for g, lst in index.items():
        lst.sort(key=lambda e: (policy_rank(e["policy"]), e["seed"]))
        pols = sorted({e["policy"] for e in lst}, key=policy_rank)
        log(f"  {g}: {len(lst)} recordings" + (f" ({', '.join(pols)})" if pols else ""))
    return index


# ---------------------------------------------------------------------------------------------------- results
def parse_baselines(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.is_file():
        log(f"  {path} missing, results table will have no baselines"); return rows
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("game", "---") or set(cells[0]) <= {"-"}:
            continue
        def score_len(s):
            m = re.match(r"([-\d.]+)\s*\((\d+)\)", s)
            return {"score": float(m.group(1)), "len": int(m.group(2))} if m else None
        try:
            rows[cells[0]] = {"actions": cells[1], "env_steps_per_s": float(cells[2]), "random": score_len(cells[3]),
                              "teacher": dict(score_len(cells[4]) or {}, what=cells[5])}
        except ValueError:
            continue
    return rows


def norm_bins(cal) -> list[dict] | None:
    """Accept bins as [[p, acc, n], ...] or [{p|conf|p_mean, acc|agreement|accuracy, n|count}, ...]."""
    if isinstance(cal, dict):
        cal = cal.get("bins", cal.get("reliability"))
    if not isinstance(cal, list) or not cal:
        return None
    out = []
    for b in cal:
        if isinstance(b, dict):
            p = b.get("p", b.get("conf", b.get("p_mean", b.get("confidence"))))
            acc = b.get("acc", b.get("agreement", b.get("accuracy")))
            n = b.get("n", b.get("count", 0))
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            p, acc, n = b[0], b[1], (b[2] if len(b) > 2 else 0)
        else:
            return None
        if p is None or acc is None:
            return None
        out.append({"p": float(p), "acc": float(acc), "n": int(n or 0)})
    return out


def read_model_results(results_dir: Path) -> tuple[dict, dict, dict]:
    """runs/results/*.json in the playjev.play result shape (game, policy, score_mean, len_mean, steps_per_s), one
    dict or a list of dicts per file; a `calibration` key or a separate *calib*.json {game, bins} gives the
    reliability diagram. Rows named random/teacher are kept apart (same-seed references for the model rows); a
    row whose policy name ends in "-sampled" or "-delay<n>" is a variant and stays out of the table."""
    model, calib, ref = {}, {}, {}
    if not results_dir.is_dir():
        return model, calib, ref
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            log(f"  {f}: bad JSON ({e})"); continue
        for d in (data if isinstance(data, list) else [data]):
            if not isinstance(d, dict) or "game" not in d:
                continue
            g = d["game"]
            bins = norm_bins(d.get("calibration")) or (norm_bins(d) if "bins" in d else None)
            if bins:
                calib[g] = {"bins": bins, "policy": d.get("policy_name") or d.get("policy"), "source": f.name,
                            "agreement": d.get("agreement_top", d.get("agreement")), "ece": d.get("ece")}
            pol = d.get("policy_name") or d.get("policy")
            if pol is None or ("score_mean" not in d and "score" not in d):
                continue
            row = {"policy": pol, "score": d.get("score_mean", d.get("score")), "len": d.get("len_mean", d.get("len")),
                   "decisions_per_s": d.get("decisions_per_s", d.get("steps_per_s")), "episodes": d.get("episodes"),
                   "conf": d.get("conf_mean"), "capped": d.get("capped"), "source": f.name}
            if pol in ("random", "teacher"):
                ref.setdefault(g, {})[pol] = row
            elif re.search(r"-(sampled|delay\d+)$", str(pol)):
                continue
            elif g not in model or (str(pol).startswith("playjev") and not str(model[g]["policy"]).startswith("playjev")):
                model[g] = row
    return model, calib, ref


def build_results(games: dict[str, dict], results_dir: Path) -> dict:
    """One row per game. Random and teacher come from the results directory when the model's run produced them on
    the same seeds; otherwise from docs/BASELINES.md (eight episodes on local-workstation). vs_teacher is (model - random) /
    (teacher - random): 0 is random play, 1 is the teacher."""
    base = parse_baselines(ROOT / "docs" / "BASELINES.md")
    model, calib, ref = read_model_results(results_dir)
    rows, same_seed = [], 0
    for g, spec in games.items():
        b, r = base.get(g, {}), ref.get(g, {})
        rnd = r.get("random") or b.get("random"); tea = r.get("teacher") or (dict(b["teacher"]) if b.get("teacher") else None)
        same_seed += bool(r.get("random") and r.get("teacher"))
        m = model.get(g); vs = None
        if m and rnd and tea and tea.get("score") is not None and tea["score"] != rnd["score"]:
            vs = (m["score"] - rnd["score"]) / (tea["score"] - rnd["score"])
        rows.append({"game": g, "title": spec["title"], "upstream": spec["upstream"], "k": len(spec["actions"]),
                     "env_steps_per_s": b.get("env_steps_per_s"), "random": rnd, "teacher": tea,
                     "model": m, "vs_teacher": None if vs is None else round(vs, 3), "calibration": calib.get(g),
                     "same_seed": bool(r.get("random") and r.get("teacher"))})
    n_model = sum(1 for r in rows if r["model"])
    eps = {r["model"]["episodes"] for r in rows if r["model"] and r["model"].get("episodes")}
    note = ("Mean score per policy through the same harness, episodes capped at 1500 steps. "
            + (f"{'/'.join(str(e) for e in sorted(eps))} held-out episodes per game, random and teacher on the same seeds as the model. "
               if same_seed == len(rows) and eps else "Random and teacher: eight episodes on the development machine (docs/BASELINES.md). ")
            + "vs teacher is (model - random) / (teacher - random): 0 is random play, 1 is the teacher.")
    log(f"  results: baselines for {sum(1 for r in rows if r['random'])} games ({same_seed} same-seed), model rows for "
        f"{n_model}, calibration for {sum(1 for r in rows if r['calibration'])}")
    return {"generated": dt.datetime.now().isoformat(timespec="seconds"), "baselines": "docs/BASELINES.md",
            "results_dir": relpath(results_dir), "note": note, "rows": rows}


def check_prompt_against_notes(games: dict[str, dict]) -> None:
    notes = ROOT / "docs" / "MODEL_NOTES.md"
    if "snake" not in games or not notes.is_file():
        return
    text = notes.read_text()
    m = re.search(r"## 2\..*?```\n(.*?)```", text, re.S)
    if not m:
        log("  MODEL_NOTES section 2 prompt block not found; prompt not cross-checked"); return
    if m.group(1).rstrip("\n") == games["snake"]["prompt"]:
        log("  prompt: rendered snake prompt matches docs/MODEL_NOTES.md section 2 exactly")
    else:
        log("  WARNING: rendered snake prompt differs from docs/MODEL_NOTES.md section 2; check plain_prompt()")


# ------------------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", action="append", help="replay directory (repeatable; default runs/replays)")
    ap.add_argument("--results", default=str(ROOT / "runs" / "results"), help="model results directory")
    ap.add_argument("--games", default=",".join(ORDER), help="comma-separated game ids")
    ap.add_argument("--out", default=str(DEMO), help="demo directory (default demo/)")
    a = ap.parse_args()
    out = Path(a.out)
    replay_dirs = [Path(p) for p in (a.replays or [str(ROOT / "runs" / "replays")])]
    ids = [g for g in a.games.split(",") if g]
    for g in ids:
        if not (GAMES_DIR / g / "pj.json").is_file():
            raise SystemExit(f"unknown game {g}")
    for f in ("index.html", "demo.css", "demo.js", "pj_bridge.js"):
        if not (out / f).is_file():
            log(f"  note: {out / f} is missing (hand-written source)")

    log("games")
    out_games = out / "games"
    (out_games / "_shared").mkdir(parents=True, exist_ok=True)
    shutil.copy2(GAMES_DIR / "_shared" / "pj_shim.js", out_games / "_shared" / "pj_shim.js")
    games = {g: build_game(g, out_games) for g in ids}
    for stale in out_games.iterdir():
        if stale.is_dir() and stale.name != "_shared" and stale.name not in games:
            shutil.rmtree(stale); log(f"  removed stale demo/games/{stale.name}")
    (out_games / "index.json").write_text(json.dumps({"games": list(games.values())}, indent=1))
    check_prompt_against_notes(games)

    log("replays")
    index = build_replays(replay_dirs, games, out / "replays")
    policies = sorted({e["policy"] for lst in index.values() for e in lst}, key=policy_rank)
    replays_doc = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "sources": [relpath(p) for p in replay_dirs],
                   "policies": policies, "games": index}
    (out / "replays" / "index.json").write_text(json.dumps(replays_doc, indent=1))

    log("results")
    results = build_results(games, Path(a.results))
    (out / "results.json").write_text(json.dumps(results, indent=1))

    data = {"generated": replays_doc["generated"], "games": list(games.values()), "replays": index, "policies": policies,
            "results": results, "links": LINKS}
    (out / "data.js").write_text("window.PJ_DEMO = " + json.dumps(data, separators=(",", ":")) + ";\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "shots" not in p.parts)
    log(f"done: demo/ is {total/1e6:.1f} MB without shots/; {sum(len(v) for v in index.values())} recordings, "
        f"policies {policies or ['none']}")


if __name__ == "__main__":
    main()
