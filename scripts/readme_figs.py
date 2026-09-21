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
    ("MMBench dev", {"base model": 0.66, "version 2": 0.37, "version 3": 0.74, "chance": 0.40}),
    ("MMLU test",   {"base model": 0.33, "version 2": 0.22, "version 3": 0.32, "chance": 0.25}),
]
# the replay sweep: four runs of 1500 steps from the base model, the same budget, a growing share of the batches
# drawn from general image and text questions (games only = games0aug, then mix10, mix20, mix30). Game agreement is
# validation agreement with the teachers; MMBench and MMLU as above. Base model: MMBench 0.66, MMLU 0.33.
REPLAY = {
    "share": [0, 10, 20, 30],
    "MMBench dev": [0.48, 0.65, 0.78, 0.83],
    "MMLU test": [0.29, 0.42, 0.44, 0.47],
    "game agreement": [0.430, 0.431, 0.586, 0.547],
}
REPLAY_BASE = {"MMBench dev": 0.66, "MMLU test": 0.33}
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
    """Two panels, one per held-out set: three bars, the base model, the published version 2 and version 3, with
    chance as a dotted line. The two versions wear the same blues as the chart above; the base model is the
    warm reference colour, the same one the replay chart uses for it."""
    bg, fg, muted, track, ramp, accent = palette(dark)
    cols = {"base model": accent, "version 2": ramp[2], "version 3": ramp[3]}
    W, H = 1500, 600
    pad_l, pad_r, gutter = 70, 150, 160
    pw = (W - pad_l - pad_r - gutter) // 2
    top, bottom = 120, H - 92          # plot area
    barw, gap = 124, 46
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f, fb, fs, ft, fv = font(24), font(24, bold=True), font(21), font(30, bold=True), font(34, bold=True)

    def y_of(v):
        return bottom - int((bottom - top) * v)

    d.text((W / 2, H - 22), "accuracy on 200 held-out questions, each asked in both option orders", font=fs, fill=muted, anchor="mm")
    for k, (title, v) in enumerate(GENERAL):
        px = pad_l + k * (pw + gutter)
        d.text((px, 48), title, font=ft, fill=fg, anchor="lm")
        # hairline grid at quarter steps, the baseline a little stronger
        for g, lab in ((0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1")):
            d.line([px, y_of(g), px + pw, y_of(g)], fill=track, width=2)
            d.text((px - 14, y_of(g)), lab, font=fs, fill=muted, anchor="rm")
        d.line([px, bottom, px + pw, bottom], fill=muted, width=2)
        d.text((px - 14, bottom), "0", font=fs, fill=muted, anchor="rm")
        n = len(cols)
        x0 = px + (pw - (n * barw + (n - 1) * gap)) // 2
        for j, (name, col) in enumerate(cols.items()):
            x = x0 + j * (barw + gap)
            y = y_of(v[name])
            d.rectangle([x, y, x + barw - 1, bottom - 1], fill=col)
            d.text((x + barw / 2, y - 22), f"{v[name]:.2f}", font=fv, fill=fg, anchor="mm")
            d.text((x + barw / 2, bottom + 26), name, font=fb if name == "version 3" else f, fill=fg, anchor="mm")
        # chance, dotted across the panel, named just outside it
        yc = y_of(v["chance"])
        for xx in range(px, px + pw, 12):
            d.line([xx, yc, xx + 5, yc], fill=muted, width=3)
        d.text((px + pw + 12, yc), f"chance {v['chance']:.2f}", font=fs, fill=muted, anchor="lm")
    im.save(OUT / ("general-dark.png" if dark else "general.png"), optimize=True)
    return im.size


def replay(dark=False):
    """The replay sweep as three lines over the share of general batches: what the games keep and what comes
    back off them. Dotted lines mark the base model on the two held-out sets."""
    bg, fg, muted, track, ramp, accent = palette(dark)
    aqua = "#199e70" if dark else "#1baf7a"
    cols = {"MMBench dev": ramp[2], "MMLU test": accent, "game agreement": aqua}
    W, H = 1500, 640
    left, right, top, bottom = 120, 1110, 70, H - 110
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    f, fb, fs = font(24), font(24, bold=True), font(21)
    xs = REPLAY["share"]

    def x_of(v):
        return left + int((right - left) * v / xs[-1])

    def y_of(v):
        return bottom - int((bottom - top) * v)

    for g in (0.25, 0.5, 0.75, 1.0):
        d.line([left, y_of(g), right, y_of(g)], fill=track, width=2)
        d.text((left - 18, y_of(g)), f"{g:.2f}".rstrip("0").rstrip(".") if g != 1.0 else "1", font=fs, fill=muted, anchor="rm")
    d.line([left, bottom, right, bottom], fill=muted, width=2)
    d.text((left - 18, bottom), "0", font=fs, fill=muted, anchor="rm")
    for v, lab in zip(xs, ("games only", "10 %", "20 %", "30 %")):
        d.line([x_of(v), bottom, x_of(v), bottom + 8], fill=muted, width=2)
        d.text((x_of(v), bottom + 30), lab, font=f, fill=fg, anchor="mm")
    d.text(((left + right) / 2, H - 28), "share of the training batches drawn from general image and text questions",
           font=fs, fill=muted, anchor="mm")

    # the base model, dotted, in the colour of the set it belongs to; named mid-way, on the side the series
    # leaves free
    for name, v in REPLAY_BASE.items():
        y = y_of(v)
        for xx in range(left, right, 14):
            d.line([xx, y, xx + 6, y], fill=cols[name], width=3)
        below = name.startswith("MMBench")
        d.text((x_of(15), y + 17 if below else y - 17), f"base model, {name.split()[0]}", font=fs, fill=muted, anchor="mm")

    for name, col in cols.items():
        ys = REPLAY[name]
        pts = [(x_of(x), y_of(y)) for x, y in zip(xs, ys)]
        d.line(pts, fill=col, width=5, joint="curve")
        for (x, y) in pts:
            d.ellipse([x - 11, y - 11, x + 11, y + 11], fill=bg)
            d.ellipse([x - 8, y - 8, x + 8, y + 8], fill=col)
        # the series named at its right end, in ink, with the line's colour as a swatch
        x, y = pts[-1]
        d.line([x + 22, y, x + 48, y], fill=col, width=5)
        d.text((x + 58, y), f"{name}  {ys[-1]:.2f}", font=fb, fill=fg, anchor="lm")
    im.save(OUT / ("replay-dark.png" if dark else "replay.png"), optimize=True)
    return im.size


if __name__ == "__main__":
    print("  board.png     ", board())
    print("  thumbs/       ", thumbs())
    print("  social.png    ", social())
    print("  chart.png     ", chart(False))
    print("  chart-dark.png", chart(True))
    print("  general.png     ", general(False))
    print("  general-dark.png", general(True))
    print("  replay.png       ", replay(False))
    print("  replay-dark.png  ", replay(True))
