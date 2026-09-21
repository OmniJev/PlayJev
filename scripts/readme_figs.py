"""README figures: the ten-game board, the vs-teacher chart and the general-ability chart (light and dark).

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

# game id, display name, score in the replay that the shot comes from, (cloning, version 1, version 2, version 3)
# vs teacher, ordered by version 3
GAMES = [
    ("invaders", "Space Invaders", "230",    (1.00, 1.00, 1.00, 1.00)),
    ("racer",    "Racer",          "2270",   (0.92, 1.00, 1.00, 1.00)),
    ("sokoban",  "Sokoban",        "2",      (0.54, 1.00, 1.00, 1.00)),
    ("snake",    "Snake",          "151",    (0.68, 0.95, 0.79, 0.90)),
    ("pacman",   "Pacman",         "4350",   (0.13, 0.45, 0.52, 0.56)),
    ("tetris",   "Tetris",         "11890",   (0.06, 0.09, 0.30, 0.37)),
    ("mario",    "Infinite Mario", "3038",   (0.15, 0.15, 0.32, 0.32)),
    ("2048",     "2048",           "9504",   (0.12, 0.06, 0.13, 0.21)),
    ("flappy",   "Floppy Bird",    "45",     (0.11, 0.11, 0.16, 0.18)),
    ("breakout", "Breakout",       "7125",   (0.01, 0.07, 0.14, 0.14)),
]
# general ability, accuracy on 200 held-out questions each, every question asked in both option orders and the
# two runs averaged (scripts/general/eval_general.py, eval_text.py): the base model, version 2, version 3, chance
GENERAL = [
    ("MMBench dev", {"base": 0.66, "version 2": 0.37, "version 3": 0.74, "chance": 0.40}),
    ("MMLU test",   {"base": 0.33, "version 2": 0.22, "version 3": 0.32, "chance": 0.25}),
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


# where the crop sits on a frame taller than the thumbnail: 0 keeps the top, 1 the bottom
THUMB_Y = {"tetris": .82, "flappy": .16, "pacman": .54, "snake": .5}
# a tighter crop for the roster thumbnail alone, where the board is a small part of the page
THUMB_CROP = {"2048": (.02, .27, .98, .89)}


def thumbs(size=(240, 180)):
    """Roster thumbnails, filled edge to edge. The frame is scaled until its short side covers the box and
    the long side is cropped, so a tall game is zoomed into instead of sitting between two black bars."""
    out = OUT / "thumbs"
    out.mkdir(exist_ok=True)
    tw, th = size
    for gid, *_ in GAMES:
        src = Image.open(SHOTS / f"{gid}.png").convert("RGB")
        l, t, r, b = THUMB_CROP.get(gid) or CROP.get(gid, (0, 0, 1, 1))
        src = src.crop((int(src.width * l), int(src.height * t), int(src.width * r), int(src.height * b)))
        k = max(tw / src.width, th / src.height)
        src = src.resize((max(tw, round(src.width * k)), max(th, round(src.height * k))), Image.LANCZOS)
        x = round((src.width - tw) * .5)
        y = round((src.height - th) * THUMB_Y.get(gid, .5))
        src.crop((x, y, x + tw, y + th)).save(out / f"{gid}.png", optimize=True)
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


VERSIONS = ("cloning", "version 1", "version 2", "version 3")


def palette(dark):
    """Page colours and one blue ramp, light to dark on the light page and the reverse on the dark one, so the
    newest version is always the one that stands out most against the surface."""
    bg, fg, muted, track = ("#0c0d0f", "#e9eaec", "#8f97a2", "#24272c") if dark else \
                           ("#ffffff", "#16181c", "#6d747e", "#ececeb")
    ramp = ("#184f95", "#2a78d6", "#6da7ec", "#b7d3f6") if dark else ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
    accent = "#d95926" if dark else "#eb6834"
    return bg, fg, muted, track, ramp, accent


def chart(dark=False):
    bg, fg, muted, track, ramp, _ = palette(dark)
    n = len(VERSIONS)
    pad, lab, right = 34, 236, 92
    barh, gap, rowgap = 13, 4, 24
    rowh = n * barh + (n - 1) * gap + rowgap
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
    for name, col in zip(VERSIONS, ramp):
        d.rectangle([lx, 22, lx + 26, 22 + 13], fill=col)
        d.text((lx + 34, 28), name, font=fb, fill=fg, anchor="lm")
        lx += 34 + int(d.textlength(name, font=fb)) + 30
    d.text((x1, 28), "1.00 = the teacher on the same seeds", font=f, fill=muted, anchor="rm")

    mid = (n * barh + (n - 1) * gap) / 2
    for i, (gid, name, _, vals) in enumerate(GAMES):
        top = head + i * rowh
        d.text((x0 - 18, top + mid), name, font=f, fill=fg, anchor="rm")
        for j, (v, col) in enumerate(zip(vals, ramp)):
            y = top + j * (barh + gap)
            d.rectangle([x0, y, x1, y + barh - 1], fill=track)
            w = max(2, int((x1 - x0) * max(0.0, min(1.0, v))))
            d.rectangle([x0, y, x0 + w, y + barh - 1], fill=col)
        d.text((x1 + 16, top + mid), f"{vals[-1]:.2f}", font=fb, fill=fg, anchor="lm")
    d.line([x0, head - 12, x0, H - 26], fill=track, width=2)
    d.line([x1, head - 12, x1, H - 26], fill=track, width=2)
    d.text((x0, H - 16), "0", font=fs, fill=muted, anchor="lm")
    d.text((x1, H - 16), "teacher", font=fs, fill=muted, anchor="rm")
    im.save(OUT / ("chart-dark.png" if dark else "chart.png"), optimize=True)
    return im.size


def general(dark=False):
    """Two panels, one per held-out set: the published version 2 and version 3 as bars in the same blues as the
    chart above, the base model as a marker on the same axis, chance as a dotted line."""
    bg, fg, muted, track, ramp, accent = palette(dark)
    cols = {"version 2": ramp[2], "version 3": ramp[3]}
    W, H = 1500, 234
    pad, head, lab, right, gutter = 34, 70, 132, 78, 70
    barh, gap = 32, 12
    pw = (W - 2 * pad - gutter) // 2
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f, fb, fs, ft = font(21), font(21, bold=True), font(19), font(24, bold=True)

    lx = pad
    for name, col in cols.items():
        d.rectangle([lx, 22, lx + 26, 22 + 13], fill=col)
        d.text((lx + 34, 28), name, font=fb, fill=fg, anchor="lm")
        lx += 34 + int(d.textlength(name, font=fb)) + 30
    d.line([lx, 28, lx + 26, 28], fill=accent, width=3)
    d.text((lx + 34, 28), "base model", font=fb, fill=fg, anchor="lm")
    lx += 34 + int(d.textlength("base model", font=fb)) + 30
    for x in range(lx, lx + 26, 6):
        d.line([x, 28, x + 2, 28], fill=muted, width=3)
    d.text((lx + 34, 28), "chance", font=fb, fill=fg, anchor="lm")
    d.text((W - pad, 28), "accuracy on 200 held-out questions, asked in both option orders", font=f, fill=muted, anchor="rm")

    for k, (title, v) in enumerate(GENERAL):
        px = pad + k * (pw + gutter)
        x0, x1 = px + lab, px + pw - right
        top = head + 34
        d.text((px, head + 4), title, font=ft, fill=fg, anchor="lm")
        ys = []
        for j, name in enumerate(cols):
            y = top + j * (barh + gap)
            ys.append(y)
            d.text((x0 - 16, y + barh / 2), name, font=f, fill=fg, anchor="rm")
            d.rectangle([x0, y, x1, y + barh - 1], fill=track)
            w = max(2, int((x1 - x0) * v[name]))
            d.rectangle([x0, y, x0 + w, y + barh - 1], fill=cols[name])
            d.text((x1 + 16, y + barh / 2), f"{v[name]:.2f}", font=fb, fill=fg, anchor="lm")
        y_top, y_bot = ys[0] - 8, ys[-1] + barh + 7
        # chance, dotted
        xc = x0 + int((x1 - x0) * v["chance"])
        for y in range(y_top, y_bot, 8):
            d.line([xc, y, xc, y + 3], fill=muted, width=2)
        d.text((xc, y_bot + 14), f"chance {v['chance']:.2f}", font=fs, fill=muted, anchor="mm")
        # the base model, solid accent
        xb = x0 + int((x1 - x0) * v["base"])
        d.line([xb, y_top - 4, xb, y_bot + 4], fill=accent, width=3)
        d.text((xb, y_top - 16), f"base model {v['base']:.2f}", font=fb, fill=accent, anchor="mm")
        d.line([x0, y_bot, x0, y_top], fill=track, width=2)
        d.text((x0, y_bot + 14), "0", font=fs, fill=muted, anchor="mm")
        d.text((x1, y_bot + 14), "1", font=fs, fill=muted, anchor="mm")
    im.save(OUT / ("general-dark.png" if dark else "general.png"), optimize=True)
    return im.size


if __name__ == "__main__":
    print("  board.png     ", board())
    print("  thumbs/       ", thumbs())
    print("  social.png    ", social())
    print("  chart.png     ", chart(False))
    print("  chart-dark.png", chart(True))
    print("  general.png     ", general(False))
    print("  general-dark.png", general(True))
