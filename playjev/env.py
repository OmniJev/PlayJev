"""Browser game environments for PlayJev.

Each game is a vendored HTML5/JS game under games/<id>/ with a pj.json manifest and a pj_hook.js.
The shared shim (games/_shared/pj_shim.js) gives the page a virtual clock, so the game only
advances inside pj.step(). One GamePage = one Playwright page. VecGame steps many pages
concurrently with asyncio.gather.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import io
import json
import socket
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
GAMES_DIR = ROOT / "games"
SHIM_JS = (GAMES_DIR / "_shared" / "pj_shim.js").read_text()
FRAME_LONG_SIDE = 448
JPEG_QUALITY = 85


def load_spec(game_id: str) -> dict:
    spec = json.loads((GAMES_DIR / game_id / "pj.json").read_text())
    spec["hook_js"] = (GAMES_DIR / game_id / "pj_hook.js").read_text()
    return spec


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a):  # noqa: D401
        pass


class GameServer:
    """Static server for games/ on a free localhost port (file:// taints canvases, http does not)."""

    def __init__(self, directory: Path = GAMES_DIR):
        s = socket.socket(); s.bind(("127.0.0.1", 0)); self.port = s.getsockname()[1]; s.close()
        handler = functools.partial(_QuietHandler, directory=str(directory))
        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True); self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


def _resize_jpeg(data: bytes, long_side: int = FRAME_LONG_SIDE) -> bytes:
    im = Image.open(io.BytesIO(data)).convert("RGB")
    if max(im.size) != long_side:
        s = long_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=JPEG_QUALITY); return buf.getvalue()


class GamePage:
    def __init__(self, context, spec: dict, base_url: str):
        self.ctx, self.spec, self.base_url = context, spec, base_url
        self.url = f"{base_url}/{spec['id']}/{spec['entry']}"
        self.page = None
        self.actions: list[dict] = []

    async def open(self):
        self.page = await self.ctx.new_page()
        await self.page.set_viewport_size(self.spec["viewport"])
        await self.page.add_init_script(SHIM_JS)
        await self.page.add_init_script(self.spec["hook_js"])
        self.page_errors: list[str] = []
        self.page.on("pageerror", lambda e: self.page_errors.append(str(e)[:200]))
        # Vendored pages must not depend on the network (compute nodes are air-gapped): abort everything off-origin.
        # Audio files are aborted too: media elements delay the page's load event until they have decoded data, and
        # on the compute nodes that never happens (flappy's five .ogg sounds hung every page.goto at 30 s), while
        # audio is muted anyway. An aborted source just errors the media element and the game carries on.
        base = self.base_url
        audio = (".mp3", ".ogg", ".wav", ".m4a", ".mid", ".midi", ".oga", ".flac")
        await self.page.route(lambda url: not url.startswith(base) or url.split("?", 1)[0].lower().endswith(audio),
                              lambda route: route.abort())

    async def reset(self, seed: int = 1) -> dict:
        await self.page.goto(self.url, wait_until="load")
        obs = await self.page.evaluate("s => pj.start(s)", seed)
        self.actions = await self.page.evaluate("pj.actions")
        return await self._finish(obs)

    async def step(self, action: int, frames: int | None = None) -> dict:
        obs = await self.page.evaluate("([a, k]) => pj.step(a, k)", [action, frames])
        return await self._finish(obs)

    async def _finish(self, obs: dict) -> dict:
        # Driver-level episode cap (pj.json "max_steps"): games without a natural end, or where a policy can stall.
        cap = self.spec.get("max_steps")
        if cap and not obs.get("done") and obs.get("steps", 0) >= cap:
            obs["done"] = True; obs["truncated"] = True
        if obs.get("frame"):
            obs["frame"] = base64.b64decode(obs["frame"].split(",", 1)[1])
            if self.spec.get("resize_canvas_frames"):
                obs["frame"] = _resize_jpeg(obs["frame"])
        elif obs.get("bbox"):
            raw = await self.page.screenshot(clip=obs["bbox"], type="jpeg", quality=90)
            obs["frame"] = _resize_jpeg(raw)
        if self.page_errors:
            obs.setdefault("errors", []).extend(self.page_errors); self.page_errors.clear()
        return obs

    async def close(self):
        await self.page.close()


class VecGame:
    """n pages of one game, stepped concurrently."""

    def __init__(self, game_id: str, n: int = 8, headless: bool = True, pages_per_context: int = 4):
        self.spec, self.n, self.headless, self.ppc = load_spec(game_id), n, headless, pages_per_context
        self.server = None; self.pages: list[GamePage] = []

    async def __aenter__(self):
        self.server = GameServer()
        self._pw = await async_playwright().start()
        self.browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-gpu", "--mute-audio", "--disable-dev-shm-usage", "--disable-background-timer-throttling",
                  "--disable-renderer-backgrounding", "--autoplay-policy=no-user-gesture-required"])
        self.contexts = [await self.browser.new_context() for _ in range((self.n + self.ppc - 1) // self.ppc)]
        for i in range(self.n):
            gp = GamePage(self.contexts[i // self.ppc], self.spec, self.server.base_url)
            await gp.open(); self.pages.append(gp)
        return self

    async def __aexit__(self, *exc):
        await self.browser.close(); await self._pw.stop(); self.server.close()

    async def reset(self, seeds: list[int] | None = None) -> list[dict]:
        seeds = seeds or list(range(1, self.n + 1))
        return list(await asyncio.gather(*(p.reset(s) for p, s in zip(self.pages, seeds))))

    async def step(self, actions: list[int], frames: int | None = None) -> list[dict]:
        return list(await asyncio.gather(*(p.step(a, frames) for p, a in zip(self.pages, actions))))

    @property
    def actions(self) -> list[dict]:
        return self.pages[0].actions
