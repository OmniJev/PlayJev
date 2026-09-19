#!/usr/bin/env python3
"""Serve demo/ on a free localhost port, screenshot it with headless Chromium, and verify the replays.

    python scripts/shot_demo.py                 # screenshots (1440 px, 400 px, one tile close-up) + console check
    python scripts/shot_demo.py --check         # also replay every recording at 4x and compare final scores
    python scripts/shot_demo.py --check --speed 8 --games snake,2048 --out runs/probe/demo/shots

Screenshots go to demo/shots/ by default. Exit code 1 if a page error was seen or a replay did not match.
The server is an in-process thread and the browser is closed at the end; nothing is killed by name.
"""
from __future__ import annotations

import argparse
import functools
import json
import socket
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(directory: Path):
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(_Quiet, directory=str(directory)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{port}"


def open_page(browser, url, width, height, scale, log):
    ctx = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=scale)
    page = ctx.new_page()
    page.on("pageerror", lambda e: log.append(("pageerror", "page", str(e)[:300])))
    page.on("console", lambda m: log.append((m.type, (m.location or {}).get("url", "")[-60:], m.text[:300]))
            if m.type in ("error", "warning") else None)
    page.goto(url, wait_until="load")
    return ctx, page


def wait_started(page, timeout_s=90):
    """Every tile has left the loading overlay (game started) or shows an error."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        # sections that only run while scrolled into view have to be woken first, or they read as "not started"
        page.evaluate("() => window.pjDemo && pjDemo.startAll && pjDemo.startAll()")
        n = page.evaluate("() => window.pjDemo ? pjDemo.tiles.filter(t => t.suspended || t.overlay.hidden || t.overlay.classList.contains('err')).length : -1")
        total = page.evaluate("() => window.pjDemo ? pjDemo.tiles.length : 0")
        if n >= total > 0:
            return True
        time.sleep(0.5)
    return False


AUDIO_EXT = (".mp3", ".ogg", ".wav", ".m4a", ".mid", ".midi", ".oga", ".flac")


def report_console(log, what):
    """JS errors and failed resource loads, apart from audio 404s (the build leaves audio out on purpose, like the driver)."""
    js = [x for x in log if x[0] == "pageerror" or (x[0] == "error" and not x[2].startswith("Failed to load resource"))]
    missing = [x for x in log if x[0] == "error" and x[2].startswith("Failed to load resource")]
    audio = [x for x in missing if x[1].split("?")[0].lower().endswith(AUDIO_EXT)]
    other = [x for x in missing if x not in audio]
    warns = [x for x in log if x[0] == "warning"]
    print(f"console ({what}): {len(js)} JS errors, {len(other)} missing files, {len(audio)} missing audio files (expected), {len(warns)} warnings")
    for x in js[:30]:
        print("  JS ERROR", x[1], x[2])
    for x in other[:30]:
        print("  MISSING ", x[1], x[2][:80])
    for x in warns[:10]:
        print("  warn    ", x[1], x[2][:120])
    return len(js) + len(other)


def tile_states(page):
    return page.evaluate("""() => pjDemo.tiles.map(t => ({game: t.game.id, policy: t.policy, overlay: t.overlay.hidden ? '' : t.overlay.textContent,
        err: t.overlay.classList.contains('err'), step: t.stepEl.textContent, score: t.scoreEl.textContent, warn: t.warnEl.textContent, errors: t.errors.length}))""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEMO / "shots"))
    ap.add_argument("--check", action="store_true", help="replay every recording and compare the final scores")
    ap.add_argument("--speed", type=float, default=4, help="playback speed for --check (default 4x)")
    ap.add_argument("--games", default=None, help="comma-separated subset for --check")
    ap.add_argument("--policies", default=None, help="comma-separated policy subset for --check")
    ap.add_argument("--closeup", default=None, help="game id for the tile close-up (default: first tile with a recording)")
    ap.add_argument("--settle", type=float, default=6.0, help="seconds of playback before the desktop screenshot")
    ap.add_argument("--no-shots", action="store_true")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    httpd, base = serve(DEMO)
    url = base + "/index.html"
    print(f"serving {DEMO} at {base}")
    failures = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not a.headed, args=["--no-sandbox", "--disable-gpu", "--mute-audio", "--autoplay-policy=no-user-gesture-required"])
        try:
            if not a.no_shots:
                # ---- desktop
                log = []
                ctx, page = open_page(browser, url, 1440, 1000, 1, log)
                ok = wait_started(page)
                print("desktop: all tiles started" if ok else "desktop: WARNING some tiles did not start")
                time.sleep(a.settle)
                for s in tile_states(page):
                    print(f"  {s['game']:9s} {s['policy'] or '-':22s} {s['step']:22s} {s['score']:12s} {s['overlay'] or ''} {s['warn']}")
                page.screenshot(path=str(out / "desktop-1440.png"), full_page=True)
                print(f"  wrote {out / 'desktop-1440.png'}")
                # close-up of one tile mid-episode, at 2x for legibility
                closeup = a.closeup or page.evaluate("() => (pjDemo.tiles.find(t => t.entries.length) || pjDemo.tiles[0]).game.id")
                page.evaluate("id => { const t = pjDemo.tiles.find(t => t.game.id === id); if (t) t.setSpeed(1); }", closeup)
                ctx.close()
                log2 = []
                ctx2, page2 = open_page(browser, url, 1440, 1000, 2, log2)
                wait_started(page2)
                # wait until the chosen tile is mid-episode (at least 25 steps in), then freeze it for the shot
                deadline = time.time() + 60
                while time.time() < deadline:
                    st = page2.evaluate("id => { const t = pjDemo.tiles.find(t => t.game.id === id); return t ? t.stepEl.textContent : ''; }", closeup)
                    try:
                        n = int(st.split()[1])
                    except (IndexError, ValueError):
                        n = 0
                    if n >= 25:
                        break
                    time.sleep(0.3)
                page2.evaluate("id => { const t = pjDemo.tiles.find(t => t.game.id === id); if (t) t.pause(); }", closeup)
                time.sleep(0.4)
                page2.locator(f'.tile[data-game="{closeup}"]').screenshot(path=str(out / f"tile-{closeup}.png"))
                print(f"  wrote {out / f'tile-{closeup}.png'} ({closeup} at {st})")
                ctx2.close()
                # ---- phone
                log3 = []
                ctx3, page3 = open_page(browser, url, 400, 860, 2, log3)
                ok3 = wait_started(page3)
                print("phone: all tiles started" if ok3 else "phone: WARNING some tiles did not start")
                time.sleep(3)
                page3.screenshot(path=str(out / "phone-400.png"), full_page=True)
                print(f"  wrote {out / 'phone-400.png'}")
                ctx3.close()
                failures += report_console(log + log2 + log3, "three loads")
            if a.check:
                log = []
                ctx, page = open_page(browser, url, 1440, 1000, 1, log)
                wait_started(page)
                opts = {"speed": a.speed}
                if a.policies:
                    opts["policies"] = a.policies.split(",")
                games = a.games.split(",") if a.games else None
                print(f"check: replaying every recording at {a.speed:g}x" + (f" for {games}" if games else ""))
                t0 = time.time()
                js = "async (o) => { const tiles = o.games ? pjDemo.tiles.filter(t => o.games.includes(t.game.id)) : pjDemo.tiles; " \
                     "for (const t of pjDemo.tiles) if (!tiles.includes(t)) { t.gen++; t.pause(); } " \
                     "const res = await Promise.all(tiles.map(t => t.verify(o.speed, o.policies || null))); return res.flat(); }"
                results = page.evaluate(js, dict(opts, games=games), )
                dt = time.time() - t0
                print(f"  {len(results)} recordings in {dt:.0f} s")
                for r in results:
                    ok = r.get("match")
                    flag = "ok  " if ok else "FAIL"
                    extra = f" diverged at step {r['diverged_at']}" if r.get("diverged_at") else ""
                    if r.get("error"):
                        extra += f" error: {r['error']}"
                    if r.get("errors"):
                        extra += f" page errors: {len(r['errors'])}"
                    print(f"  {flag} {r['game']:9s} {r['policy']:22s} seed {r['seed']:<6} steps {r.get('steps_played', '?')}/{r.get('steps_recorded', '?')} "
                          f"score {r.get('final')} recorded {r.get('expected')} done={r.get('done')}{extra}")
                    failures += 0 if ok else 1
                failures += report_console(log, "check run")
                (out / "check.json").write_text(json.dumps(results, indent=1))
                print(f"  wrote {out / 'check.json'}")
                ctx.close()
        finally:
            browser.close()
    httpd.shutdown(); httpd.server_close()
    print("FAILURES:" if failures else "all good:", failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
