"""Run bench, random and (if registered) teacher evaluation for every game; write runs/summary.md.
python scripts/bench_all.py --episodes 8"""
import argparse, asyncio, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from playjev.teachers import _REGISTRY  # noqa: E402

GAMES = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    return r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()[-300:]


def main(a):
    rows = []
    for g in GAMES:
        spec = json.loads((ROOT / "games" / g / "pj.json").read_text())
        row = {"game": g, "upstream": spec.get("upstream"), "license": spec.get("license")}
        line = run([sys.executable, "-m", "playjev.bench", g, "--pages", "8", "--steps", str(a.steps)])
        row["bench"] = line
        rnd = json.loads(run([sys.executable, "-m", "playjev.play", g, "--policy", "random", "--pages", "8", "--episodes", str(a.episodes), "--max-steps", str(a.max_steps)]))
        row["random"] = rnd
        if g in _REGISTRY:
            row["teacher"] = json.loads(run([sys.executable, "-m", "playjev.play", g, "--policy", "teacher", "--pages", "8", "--episodes", str(a.episodes), "--max-steps", str(a.max_steps)]))
        rows.append(row); print(g, "done:", row["bench"][:90])
    out = ROOT / "runs" / "summary.md"
    with open(out, "w") as f:
        f.write("| game | upstream | env-steps/s | random score (len) | teacher score (len) |\n|---|---|---|---|---|\n")
        for r in rows:
            sps = r["bench"].split("env-steps/s")[0].split()[-1] if "env-steps/s" in r["bench"] else "?"
            t = r.get("teacher"); ts = f"{t['score_mean']:.0f} ({t['len_mean']:.0f})" if t else "-"
            f.write(f"| {r['game']} | {r['upstream']} | {sps} | {r['random']['score_mean']:.1f} ({r['random']['len_mean']:.0f}) | {ts} |\n")
    (ROOT / "runs" / "summary.json").write_text(json.dumps(rows, indent=1))
    print(out.read_text())


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=100); p.add_argument("--episodes", type=int, default=8); p.add_argument("--max-steps", type=int, default=1500)
    main(p.parse_args())
