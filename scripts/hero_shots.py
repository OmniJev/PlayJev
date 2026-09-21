"""Frames for the README: replay a recorded version 3 episode and screenshot the game at 2x.

The actions come from demo/replays/<game>/playjev-0.8b-dagger3_*.json, so every picture is a
position the trained model actually reached, not a hand-played one.
"""
import argparse, asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playjev.env import GameServer, GamePage, load_spec, ROOT

GAMES = ["snake", "tetris", "2048", "mario", "pacman", "breakout", "flappy", "invaders", "racer", "sokoban"]
CUT = {"snake": .92, "tetris": .88, "2048": .95, "mario": .70, "pacman": .55,
       "breakout": .80, "flappy": .90, "invaders": .55, "racer": .35, "sokoban": .80}


def best_replay(game: str, policy: str = "playjev-0.8b-dagger3") -> dict:
    files = sorted(Path(f"demo/replays/{game}").glob(f"{policy}_*.json"))
    if not files:
        raise SystemExit(f"{game}: no {policy} replay")
    reps = [json.loads(f.read_text()) for f in files]
    return max(reps, key=lambda r: r.get("final_score", 0))


async def shoot(game: str, out: Path, scale: int = 2) -> str:
    rep = best_replay(game)
    spec = load_spec(game)
    server = GameServer()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-gpu", "--mute-audio", "--disable-dev-shm-usage",
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding"])
        ctx = await browser.new_context(device_scale_factor=scale)
        gp = GamePage(ctx, spec, server.base_url)
        await gp.open()
        obs = await gp.reset(rep["seed"])
        steps = rep["steps"]
        cut = max(1, int(len(steps) * CUT.get(game, .8)))
        for s in steps[:cut]:
            obs = await gp.step(s["a"], rep.get("frames_per_step"))
            if obs.get("done"):
                break
        box = obs.get("bbox") or await gp.page.evaluate(
            "() => { let b = null, best = 0;"
            "  for (const c of document.querySelectorAll('canvas')) {"
            "    const r = c.getBoundingClientRect();"
            "    if (r.width * r.height > best) { best = r.width * r.height;"
            "      b = {x: r.x, y: r.y, width: r.width, height: r.height}; } }"
            "  return b; }")
        png = await gp.page.screenshot(clip=box, type="png") if box else await gp.page.screenshot(type="png")
        out.write_bytes(png)
        await browser.close()
    server.close()
    return f"{game}: step {cut}/{len(steps)}, score {obs.get('score')}, {len(png)//1024} KB"


async def main(a):
    out_dir = ROOT / "docs" / "assets" / "games"; out_dir.mkdir(parents=True, exist_ok=True)
    games = a.games or GAMES
    for i, g in enumerate(games, 1):
        line = await shoot(g, out_dir / f"{g}.png", a.scale)
        print(f"[{i:2d}/{len(games)}] {line}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("games", nargs="*"); p.add_argument("--scale", type=int, default=2)
    asyncio.run(main(p.parse_args()))
