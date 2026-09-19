"""README charts: System Two handover, calibration, and the one-step-late collapse.

Light and dark versions of each, written to docs/assets/. Numbers come from runs/results/
(handover/sft_all1_*.json, *_hrandom*.json, sft_all1.json, playjev-0.8b-dagger2_relabel.json),
which are the same runs the README tables report.

    python scripts/readme_charts.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "runs" / "results"
OUT = ROOT / "docs" / "assets"

GAMES = ["mario", "snake", "tetris", "2048", "flappy", "pacman", "breakout", "invaders", "racer", "sokoban"]
NAME = {"mario": "Infinite Mario", "snake": "Snake", "tetris": "Tetris", "2048": "2048",
        "flappy": "Floppy Bird", "pacman": "Pacman", "breakout": "Breakout",
        "invaders": "Space Invaders", "racer": "Racer", "sokoban": "Sokoban"}
# random and teacher on the same 16 held-out seeds (README results table)
BASE = {"mario": (613, 4229), "snake": (1.0, 114), "tetris": (162, 15288), "2048": (1021, 19593),
        "flappy": (0.0, 84.0), "pacman": (113, 7026), "breakout": (496, 16547),
        "invaders": (215, 400), "racer": (238, 6712), "sokoban": (6.6, 102.2)}


class Theme:
    def __init__(self, dark):
        self.dark = dark
        self.bg = "#0c0d0f" if dark else "#ffffff"
        self.fg = "#e9eaec" if dark else "#16181c"
        self.muted = "#8f97a2" if dark else "#6d747e"
        self.grid = "#24272c" if dark else "#ececeb"
        self.c1 = "#3987e5" if dark else "#2a78d6"   # blue
        self.c2 = "#d95926" if dark else "#eb6834"   # orange
        self.ramp = ("#184f95", "#2a78d6", "#86b6ef") if dark else ("#86b6ef", "#3987e5", "#1c5cab")

    def apply(self, fig, axes):
        fig.patch.set_facecolor(self.bg)
        for ax in axes:
            ax.set_facecolor(self.bg)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(self.grid)
            ax.tick_params(colors=self.muted, labelsize=10, length=3)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_color(self.muted)

    def save(self, fig, stem):
        p = OUT / (f"{stem}-dark.png" if self.dark else f"{stem}.png")
        fig.savefig(p, facecolor=self.bg, bbox_inches="tight", pad_inches=0.22)
        plt.close(fig)
        return p


def _handover_rows(game):
    """(share handed over, score) for the confidence rule and for the random control."""
    conf, rand = [], []
    for f in sorted((RES / "handover").glob(f"sft_all1_{game}_handover*.json")):
        d = json.load(open(f))
        if d["handover_rate"] < 0.99:            # 1.01 is the teacher alone
            conf.append((d["handover_rate"], d["score_mean"]))
    for f in sorted((RES / "handover").glob(f"{game}_local_hrandom*.json")):
        d = json.load(open(f))
        rand.append((d["handover_rate"], d["score_mean"]))
    alone = [c for c in conf if c[0] == 0.0]
    return sorted(conf), sorted(alone + rand)


def handover(dark=False, games=("breakout", "tetris", "pacman", "snake")):
    """Score against the share of steps the teacher was asked to decide."""
    t = Theme(dark)
    fig, axes = plt.subplots(1, len(games), figsize=(13.6, 3.3), dpi=112, sharey=True)
    t.apply(fig, axes)
    for ax, g in zip(axes, games):
        teacher = BASE[g][1]
        conf, rand = _handover_rows(g)
        ax.axhline(1.0, color=t.muted, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.plot([x for x, _ in rand], [y / teacher for _, y in rand], color=t.c2, lw=2,
                marker="o", ms=5, mec=t.bg, mew=1.4, zorder=3)
        ax.plot([x for x, _ in conf], [y / teacher for _, y in conf], color=t.c1, lw=2,
                marker="o", ms=5, mec=t.bg, mew=1.4, zorder=4)
        ax.set_title(f"{NAME[g]}    teacher {teacher:,.0f}", color=t.fg, fontsize=11.5, loc="left", pad=10)
        ax.set_xlim(-0.03, 0.72)
        ax.set_ylim(-0.04, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticks([0, 0.25, 0.5])
        ax.set_xticklabels(["0", "25%", "50%"])
        ax.grid(axis="y", color=t.grid, lw=1, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticklabels(["0", "", "half the teacher", "", "teacher"])   # shared axis: set once
    fig.text(0.008, 1.02, "steps picked by confidence", color=t.c1, fontsize=12, fontweight="bold")
    fig.text(0.20, 1.02, "the same number picked at random", color=t.c2, fontsize=12, fontweight="bold")
    fig.text(0.5, -0.04, "share of steps handed to the teacher", color=t.muted, fontsize=10.5, ha="center")
    return t.save(fig, "handover")


def calibration(dark=False, run="playjev-0.8b-dagger2_relabel.json"):
    """How often the model's move matched the teacher, against the probability it gave that move."""
    t = Theme(dark)
    rows = {r["game"]: r for r in json.load(open(RES / run))}
    fig, axes = plt.subplots(2, 5, figsize=(13.6, 5.4), dpi=112, sharex=True, sharey=True)
    flat = list(axes.flat)
    t.apply(fig, flat)
    for ax, g in zip(flat, GAMES):
        bins = rows[g]["calibration"]["bins"]
        total = sum(b["n"] for b in bins)
        ax.plot([0, 1], [0, 1], color=t.muted, lw=1.2, ls=(0, (4, 3)), alpha=.5, zorder=1)
        ax.plot([b["p"] for b in bins], [b["acc"] for b in bins], color=t.c1, lw=1.8, zorder=3)
        ax.scatter([b["p"] for b in bins], [b["acc"] for b in bins],
                   s=[12 + 190 * (b["n"] / total) ** 0.5 for b in bins],
                   color=t.c1, edgecolor=t.bg, linewidth=1.2, zorder=4)
        ax.set_title(NAME[g], color=t.fg, fontsize=11.5, loc="left", pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 0.5, 1.0])
        ax.grid(color=t.grid, lw=1, zorder=0)
        ax.set_axisbelow(True)
    flat[0].set_ylabel("matched the teacher", color=t.muted, fontsize=10.5)
    flat[5].set_ylabel("matched the teacher", color=t.muted, fontsize=10.5)
    fig.text(0.5, 0.0, "probability the model gave the move it took", color=t.muted, fontsize=10.5, ha="center")
    fig.tight_layout(h_pad=2.4, w_pad=1.6)
    return t.save(fig, "calibration")


def realtime(dark=False, run="sft_all1.json"):
    """Teacher-normalised score with the decision on time and one step late."""
    t = Theme(dark)
    rows = json.load(open(RES / run))
    def score(game, delay):
        # the guarded protocol run: skip-noop executor, delay 0 and 1 on the same seeds
        want = f"play_{game}_local_delay{delay}.json"
        return next(r["score_mean"] for r in rows if r.get("source") == want)
    def frac(game, delay):
        rnd, tea = BASE[game]
        return max(0.0, (score(game, delay) - rnd) / (tea - rnd))

    order = sorted(GAMES, key=lambda g: -frac(g, 0))
    fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=112)
    t.apply(fig, [ax])
    h = 0.34
    for i, g in enumerate(order):
        y = len(order) - i
        for d, col, off in ((0, t.c1, h / 2 + 0.02), (1, t.c2, -h / 2 - 0.02)):
            v = frac(g, d)
            ax.barh(y + off, v, height=h, color=col, zorder=3)
            ax.text(v + 0.012, y + off, f"{v:.2f}", va="center", ha="left",
                    color=t.fg, fontsize=10, zorder=4)
        ax.text(-0.015, y, NAME[g], va="center", ha="right", color=t.fg, fontsize=11)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.3, len(order) + 0.9)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["random", "", "half", "", "teacher"])
    ax.grid(axis="x", color=t.grid, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.0, len(order) + 0.62, "on time", color=t.c1, fontsize=11.5, fontweight="bold")
    ax.text(0.14, len(order) + 0.62, "one step late", color=t.c2, fontsize=11.5, fontweight="bold")
    return t.save(fig, "realtime")


if __name__ == "__main__":
    for dark in (False, True):
        for fn in (handover, calibration, realtime):
            print("  ", fn(dark))
