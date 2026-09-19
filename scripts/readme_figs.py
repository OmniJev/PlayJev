"""README figures: the ten-game board and the vs-teacher chart (light and dark).

Inputs: docs/assets/games/<game>.png from scripts/hero_shots.py, and the numbers below, which are
the closed-loop results in README.md (16 held-out episodes per game, argmax, 1500-step cap).
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
SHOTS = OUT / "games"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# game id, display name, score in the replay that the shot comes from, (sft_all1, dagger1, dagger2)
GAMES = [
    ("invaders", "Space Invaders", "200",    (1.00, 1.00, 1.00)),
    ("racer",    "Racer",          "2308",   (0.92, 1.00, 1.00)),
    ("sokoban",  "Sokoban",        "2",      (0.54, 1.00, 1.00)),
    ("snake",    "Snake",          "125",    (0.68, 0.95, 0.79)),
    ("pacman",   "Pacman",         "4460",   (0.13, 0.45, 0.52)),
    ("mario",    "Infinite Mario", "3152",   (0.15, 0.15, 0.32)),
    ("tetris",   "Tetris",         "9300",   (0.06, 0.09, 0.30)),
    ("flappy",   "Floppy Bird",    "27",     (0.11, 0.11, 0.16)),
    ("breakout", "Breakout",       "7965",   (0.01, 0.07, 0.14)),
    ("2048",     "2048",           "7204",   (0.12, 0.06, 0.13)),
]
# fractional crop, to keep page furniture (theme pickers, how-to-play copy) out of the board
CROP = {"snake": (0, .105, 1, 1), "2048": (0, 0, 1, .80)}
BOARD_ORDER = ["tetris", "snake", "pacman", "racer", "invaders", "sokoban", "mario", "flappy", "breakout", "2048"]


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def edge_colour(im: Image.Image) -> tuple:
    """Most common colour on the frame's border, for letterboxing."""
    im = im.convert("RGB")
    w, h = im.size
    px = im.load()
    counts = {}
    for x in range(0, w, max(1, w // 120)):
        for y in (0, h - 1):
            counts[px[x, y]] = counts.get(px[x, y], 0) + 1
    for y in range(0, h, max(1, h // 120)):
        for x in (0, w - 1):
            counts[px[x, y]] = counts.get(px[x, y], 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def board(cell=(384, 288), strip=38, gut=8, cols=5):
    cw, ch = cell
    meta = {g[0]: g for g in GAMES}
    rows = (len(BOARD_ORDER) + cols - 1) // cols
    W = cols * cw + (cols - 1) * gut
    H = rows * (ch + strip) + (rows - 1) * gut
    canvas = Image.new("RGB", (W, H), "#f6f6f4")
    d = ImageDraw.Draw(canvas)
    f = font(21)
    fb = font(21, bold=True)
    for i, gid in enumerate(BOARD_ORDER):
        _, name, score, _ = meta[gid]
        src = Image.open(SHOTS / f"{gid}.png").convert("RGB")
        l, t, r, b = CROP.get(gid, (0, 0, 1, 1))
        src = src.crop((int(src.width * l), int(src.height * t), int(src.width * r), int(src.height * b)))
        bg = edge_colour(src)
        src.thumbnail((cw, ch), Image.LANCZOS)
        tile = Image.new("RGB", (cw, ch), bg)
        tile.paste(src, ((cw - src.width) // 2, (ch - src.height) // 2))
        x = (i % cols) * (cw + gut)
        y = (i // cols) * (ch + strip + gut)
        canvas.paste(tile, (x, y))
        d.rectangle([x, y + ch, x + cw - 1, y + ch + strip - 1], fill="#16181c")
        d.text((x + 12, y + ch + strip // 2), name.lower(), font=fb, fill="#ffffff", anchor="lm")
        d.text((x + cw - 12, y + ch + strip // 2), score, font=f, fill="#9aa2ad", anchor="rm")
    canvas.save(OUT / "board.png", optimize=True)
    return canvas.size


def thumbs(size=(220, 150)):
    """Small letterboxed shots for the roster table."""
    out = OUT / "thumbs"
    out.mkdir(exist_ok=True)
    tw, th = size
    for gid, *_ in GAMES:
        src = Image.open(SHOTS / f"{gid}.png").convert("RGB")
        l, t, r, b = CROP.get(gid, (0, 0, 1, 1))
        src = src.crop((int(src.width * l), int(src.height * t), int(src.width * r), int(src.height * b)))
        bg = edge_colour(src)
        src.thumbnail(size, Image.LANCZOS)
        tile = Image.new("RGB", (tw, th), bg)
        tile.paste(src, ((tw - src.width) // 2, (th - src.height) // 2))
        tile.save(out / f"{gid}.png", optimize=True)
    return len(GAMES), size


def social(W=1280, H=640, cols=5, cell=(224, 166), gut=12, pad=56):
    """GitHub social preview card (Settings > General > Social preview)."""
    cw, ch = cell
    meta = {g[0]: g for g in GAMES}
    im = Image.new("RGB", (W, H), "#0c0d0f")
    d = ImageDraw.Draw(im)
    d.text((pad, 128), "playjev", font=font(74, bold=True), fill="#ffffff", anchor="ls")
    d.text((pad, 182), "a 0.8B model playing ten browser games from pixels, one forward pass per move",
           font=font(27), fill="#9aa2ad", anchor="ls")
    top = H - pad - 2 * ch - gut
    for i, gid in enumerate(BOARD_ORDER):
        src = Image.open(SHOTS / f"{gid}.png").convert("RGB")
        l, t, r, b = CROP.get(gid, (0, 0, 1, 1))
        src = src.crop((int(src.width * l), int(src.height * t), int(src.width * r), int(src.height * b)))
        bg = edge_colour(src)
        src.thumbnail(cell, Image.LANCZOS)
        tile = Image.new("RGB", cell, bg)
        tile.paste(src, ((cw - src.width) // 2, (ch - src.height) // 2))
        im.paste(tile, (pad + (i % cols) * (cw + gut), top + (i // cols) * (ch + gut)))
    im.save(OUT / "social.png", optimize=True)
    return im.size


def chart(dark=False):
    bg, fg, muted, track = ("#0c0d0f", "#e9eaec", "#8f97a2", "#24272c") if dark else \
                           ("#ffffff", "#16181c", "#6d747e", "#ececeb")
    c1, c2, c3 = ("#184f95", "#2a78d6", "#86b6ef") if dark else ("#86b6ef", "#3987e5", "#1c5cab")
    pad, lab, right = 34, 236, 92
    barh, gap, rowgap = 15, 5, 26
    rowh = 3 * barh + 2 * gap + rowgap
    head = 64
    W = 1500
    x0 = pad + lab
    x1 = W - pad - right
    H = head + len(GAMES) * rowh + 22
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f, fb, fs = font(21), font(21, bold=True), font(19)

    # legend
    lx = x0
    for name, col in (("cloning", c1), ("version 1", c2), ("version 2", c3)):
        d.rectangle([lx, 22, lx + 26, 22 + 13], fill=col)
        d.text((lx + 34, 28), name, font=fb, fill=fg, anchor="lm")
        lx += 34 + int(d.textlength(name, font=fb)) + 30
    d.text((x1, 28), "1.00 = the teacher's score on the same seeds", font=f, fill=muted, anchor="rm")

    for i, (gid, name, _, vals) in enumerate(GAMES):
        top = head + i * rowh
        d.text((x0 - 18, top + (3 * barh + 2 * gap) / 2), name, font=f, fill=fg, anchor="rm")
        for j, (v, col) in enumerate(zip(vals, (c1, c2, c3))):
            y = top + j * (barh + gap)
            d.rectangle([x0, y, x1, y + barh - 1], fill=track)
            w = max(2, int((x1 - x0) * max(0.0, min(1.0, v))))
            d.rectangle([x0, y, x0 + w, y + barh - 1], fill=col)
        d.text((x1 + 16, top + (3 * barh + 2 * gap) / 2), f"{vals[2]:.2f}", font=fb, fill=fg, anchor="lm")
    d.line([x0, head - 12, x0, H - 26], fill=track, width=2)
    d.line([x1, head - 12, x1, H - 26], fill=track, width=2)
    d.text((x0, H - 16), "0", font=fs, fill=muted, anchor="lm")
    d.text((x1, H - 16), "teacher", font=fs, fill=muted, anchor="rm")
    im.save(OUT / ("chart-dark.png" if dark else "chart.png"), optimize=True)
    return im.size


if __name__ == "__main__":
    print("  board.png     ", board())
    print("  thumbs/       ", thumbs())
    print("  social.png    ", social())
    print("  chart.png     ", chart(False))
    print("  chart-dark.png", chart(True))
