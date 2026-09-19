"""Animated version of the README board: ten recorded dagger2 episodes playing at once.

Each tile replays the best demo/replays/<game>/playjev-0.8b-dagger2_*.json through playjev.env,
exactly the way scripts/hero_shots.py replays it, and screenshots the game area every few steps.
The frames go into the 5x2 grid scripts/readme_figs.py uses for docs/assets/board.png (same order,
same CROP, same letterboxing, same black label strip), except that the score on the right of each
strip is the live score at that point of the episode, so it counts up while the board plays.

Episodes differ in length, so every game gets its own stride: the shots are spread evenly over
usable_steps, the hero_shots CUT fraction of the episode (it ends the clip before the death frame
where that matters), which gives every tile the same frame count over a representative stretch.
A game with fewer usable steps than frames is captured every step and its tile holds the last
picture once the episode is over.

    .venv/bin/python scripts/board_gif.py            # docs/assets/board.gif
    .venv/bin/python scripts/board_gif.py --reuse    # recompose from the cached tiles, no browser
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playjev.env import GameServer, GamePage, load_spec, ROOT  # noqa: E402
from readme_figs import BOARD_ORDER, CROP, GAMES, edge_colour, font  # noqa: E402

# how far into each episode the clip runs, as in scripts/hero_shots.py
CUT = {"snake": .92, "tetris": .88, "2048": .95, "mario": .70, "pacman": .55,
       "breakout": .80, "flappy": .90, "invaders": .55, "racer": .35, "sokoban": .80}
POLICY = "playjev-0.8b-dagger2"
CACHE = Path(tempfile.gettempdir()) / "playjev-board-gif"
CANVAS_BBOX_JS = (
    "() => { let b = null, best = 0;"
    "  for (const c of document.querySelectorAll('canvas')) {"
    "    const r = c.getBoundingClientRect();"
    "    if (r.width * r.height > best) { best = r.width * r.height;"
    "      b = {x: r.x, y: r.y, width: r.width, height: r.height}; } }"
    "  return b; }")


def best_replay(game: str, policy: str = POLICY) -> dict:
    files = sorted((ROOT / "demo" / "replays" / game).glob(f"{policy}_*.json"))
    if not files:
        raise SystemExit(f"{game}: no {policy} replay")
    return max((json.loads(f.read_text()) for f in files), key=lambda r: r.get("final_score", 0))


def layout(width: int, cols: int = 5, gut: int = 4):
    """Tile geometry for a board `width` px wide, in the proportions of readme_figs.board()."""
    cw = (width - (cols - 1) * gut) // cols
    ch = round(cw * 288 / 384)          # readme_figs cell is 384x288
    strip = max(18, round(cw * 0.115))  # a touch taller than a straight scale, so the text stays readable
    return cw, ch, strip, gut, cols


def to_tile(png: bytes, gid: str, cell, bg=None):
    """Crop, letterbox and scale one screenshot the way readme_figs.board() does."""
    cw, ch = cell
    src = Image.open(io.BytesIO(png)).convert("RGB")
    l, t, r, b = CROP.get(gid, (0, 0, 1, 1))
    src = src.crop((int(src.width * l), int(src.height * t), int(src.width * r), int(src.height * b)))
    if bg is None:
        bg = edge_colour(src)
    src.thumbnail((cw, ch), Image.LANCZOS)
    tile = Image.new("RGB", (cw, ch), bg)
    tile.paste(src, ((cw - src.width) // 2, (ch - src.height) // 2))
    return tile, bg


async def capture(gp: GamePage, gid: str, frames: int, cell) -> tuple[np.ndarray, list]:
    """Replay one episode into `frames` tiles (padded by holding the last one) plus their scores."""
    rep = best_replay(gid)
    obs = await gp.reset(rep["seed"])
    steps = rep["steps"]
    usable = max(1, int(len(steps) * CUT.get(gid, .8)))
    # spread the shots evenly over the whole usable stretch, so a long episode is not clipped to its
    # first seconds and a short one still gets every step it has
    want = ({round(j * (usable - 1) / max(1, frames - 1)) for j in range(frames)} if usable >= frames
            else set(range(usable)))
    tiles, scores, box, bg = [], [], None, None
    for i, s in enumerate(steps[:usable]):
        obs = await gp.step(s["a"], rep.get("frames_per_step"))
        if i in want:
            if box is None:  # frozen after the first shot: every tile of a game must frame the same way
                box = obs.get("bbox") or await gp.page.evaluate(CANVAS_BBOX_JS)
            png = await gp.page.screenshot(clip=box, type="png") if box else await gp.page.screenshot(type="png")
            tile, bg = to_tile(png, gid, cell, bg)
            tiles.append(np.asarray(tile))
            scores.append(obs.get("score"))
            if len(tiles) == frames:
                break
        if obs.get("done"):
            break
    live = len(tiles)
    while len(tiles) < frames:  # out of episode: hold the last picture and the last score
        tiles.append(tiles[-1])
        scores.append(scores[-1])
    return np.stack(tiles), scores, live, usable / max(1, live), len(steps)


async def replay_all(games, frames, cell, scale, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    server = GameServer()
    t0 = time.time()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-gpu", "--mute-audio", "--disable-dev-shm-usage",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        for i, gid in enumerate(games, 1):
            ctx = await browser.new_context(device_scale_factor=scale)
            gp = GamePage(ctx, load_spec(gid), server.base_url)
            await gp.open()
            tiles, scores, live, stride, n = await capture(gp, gid, frames, cell)
            await ctx.close()
            np.savez_compressed(cache / f"{gid}.npz", tiles=tiles, scores=np.array(scores, dtype=float))
            print(f"[{i:2d}/{len(games)}] {gid:9s} {live}/{frames} live frames, every {stride:.1f} "
                  f"of {n} steps, score {scores[live - 1]}, {time.time() - t0:.0f}s", flush=True)
        await browser.close()
    server.close()


def load_cache(games, cache: Path):
    out = {}
    for gid in games:
        f = cache / f"{gid}.npz"
        if not f.exists():
            raise SystemExit(f"{gid}: no cached tiles at {f}, run without --reuse")
        z = np.load(f)
        out[gid] = (z["tiles"], z["scores"])
    return out


def fmt_score(v) -> str:
    """Whole numbers only, as in readme_figs: racer's score ticks in tenths and would jitter."""
    if v is None or v != v:
        return ""
    return f"{round(v)}"


def compose(shots: dict, frames: int, width: int) -> list[Image.Image]:
    """The 5x2 board, one RGB image per frame, with the label strip of readme_figs.board()."""
    cw, ch, strip, gut, cols = layout(width)
    names = {g[0]: g[1] for g in GAMES}
    rows = (len(BOARD_ORDER) + cols - 1) // cols
    W = cols * cw + (cols - 1) * gut
    H = rows * (ch + strip) + (rows - 1) * gut
    fb, f = font(round(strip * 0.60), bold=True), font(round(strip * 0.55))
    pad = max(6, round(cw * 0.04))
    out = []
    for k in range(frames):
        canvas = Image.new("RGB", (W, H), "#f6f6f4")
        d = ImageDraw.Draw(canvas)
        for i, gid in enumerate(BOARD_ORDER):
            tiles, scores = shots[gid]
            x, y = (i % cols) * (cw + gut), (i // cols) * (ch + strip + gut)
            canvas.paste(Image.fromarray(tiles[k]), (x, y))
            d.rectangle([x, y + ch, x + cw - 1, y + ch + strip - 1], fill="#16181c")
            d.text((x + pad, y + ch + strip / 2), names[gid].lower(), font=fb, fill="#ffffff", anchor="lm")
            d.text((x + cw - pad, y + ch + strip / 2), fmt_score(scores[k]), font=f, fill="#9aa2ad", anchor="rm")
        out.append(canvas)
    return out


def save_gif(boards: list[Image.Image], out: Path, fps: float, colors: int):
    """One adaptive palette for the whole clip, then only the pixels that changed get written.

    Unchanged pixels go to a spare palette index marked transparent and the frame is left on screen
    (disposal 1), which is the same trick gifsicle -O2 plays, and it is what keeps the file small:
    the label strips, the letterbox bars and the static parts of each game cost almost nothing.
    """
    colors = max(2, min(255, colors))  # one of the 256 slots is kept back for the transparent index
    step = max(1, len(boards) // 12)
    sample = boards[::step]
    montage = Image.new("RGB", (boards[0].width, boards[0].height * len(sample)))
    for i, b in enumerate(sample):
        montage.paste(b, (0, i * boards[0].height))
    pal = montage.convert("P", palette=Image.ADAPTIVE, colors=colors)
    used = len(pal.getpalette()) // 3
    trans = used
    palette = pal.getpalette() + [0, 0, 0] * (256 - used)

    frames, prev = [], None
    for b in boards:
        idx = np.asarray(b.quantize(palette=pal, dither=Image.Dither.NONE))
        if prev is None:
            keep = idx
        else:
            keep = np.where(idx == prev, trans, idx).astype(np.uint8)
        prev = idx
        im = Image.fromarray(keep, mode="P")
        im.putpalette(palette)
        frames.append(im)

    dur = max(10, int(round(1000 / fps / 10)) * 10)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                   disposal=1, transparency=trans, optimize=False)
    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", "-o", str(out), str(out)], check=True)
    return dur, used


def main(a):
    games = a.games or list(BOARD_ORDER)
    cw, ch, strip, gut, cols = layout(a.width)
    cache = Path(a.cache)
    if not a.reuse:
        asyncio.run(replay_all(games, a.frames, (cw, ch), a.scale, cache))
    shots = load_cache(list(BOARD_ORDER), cache)
    boards = compose(shots, a.frames, a.width)
    out = ROOT / "docs" / "assets" / a.out
    dur, used = save_gif(boards, out, a.fps, a.colors)
    mb = out.stat().st_size / 1e6
    print(f"\n{out}: {boards[0].width}x{boards[0].height}, {len(boards)} frames, "
          f"{1000 / dur:.1f} fps ({dur} ms), {used} colours, loop forever, {mb:.2f} MB")
    if mb >= 5:
        print("over the 5 MB budget: lower --colors, --frames or --width")
    # The same animation as WebP, which GitHub renders in the README and which carries the full colour of the
    # frames at about half the bytes. The GIF stays for the places that still want one.
    if a.webp:
        w = out.with_suffix(".webp")
        boards[0].save(w, format="WEBP", save_all=True, append_images=boards[1:], duration=dur, loop=0,
                       quality=a.quality, method=6, minimize_size=True)
        print(f"{w}: same frames, quality {a.quality}, {w.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("games", nargs="*", help="games to replay (default: all ten)")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=float, default=12.5)
    p.add_argument("--width", type=int, default=996)
    p.add_argument("--colors", type=int, default=192, help="palette size, 255 at most")
    p.add_argument("--scale", type=int, default=1, help="device scale factor for the screenshots")
    p.add_argument("--out", default="board.gif")
    p.add_argument("--cache", default=str(CACHE), help="where the replayed tiles are kept")
    p.add_argument("--reuse", action="store_true", help="recompose from the cache, skip the browser")
    p.add_argument("--webp", action="store_true", default=True, help="also write the animation as WebP")
    p.add_argument("--no-webp", dest="webp", action="store_false")
    p.add_argument("--quality", type=int, default=82, help="WebP quality")
    main(p.parse_args())
