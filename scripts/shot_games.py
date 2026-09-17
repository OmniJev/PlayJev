import sys, time, os
from playwright.sync_api import sync_playwright
GAMES = {
 "snake":"JavaScript-Snake/src/index.html", "tetris":"javascript-tetris/index.html",
 "mario":"mariohtml5/main.html", "flappy":"floppybird/index.html", "2048":"2048/index.html",
 "pacman":"pacman/index.html", "invaders":"spaceinvaders/index.html", "pong":"javascript-pong/index.html",
 "trex":"t-rex-runner/index.html", "asteroids":"HTML5-Asteroids/index.html", "breakout":"Gamedev-Canvas-workshop/lesson10.html"}
PORT=open('/tmp/games_http.port').read().strip()
out="/tmp/gshots"; os.makedirs(out, exist_ok=True)
with sync_playwright() as p:
    b=p.chromium.launch(headless=True, args=["--no-sandbox","--disable-gpu","--autoplay-policy=no-user-gesture-required"])
    for k,path in GAMES.items():
        pg=b.new_page(viewport={"width":900,"height":700})
        errs=[]; pg.on("pageerror", lambda e: errs.append(str(e)[:80]))
        try:
            pg.goto(f"http://127.0.0.1:{PORT}/{path}", wait_until="load", timeout=20000)
            pg.wait_for_timeout(1200)
            pg.mouse.click(450,350); pg.keyboard.press("Space"); pg.keyboard.press("Enter")
            pg.wait_for_timeout(1500)
            pg.screenshot(path=f"{out}/{k}.png")
            print(f"{k:10s} ok  errs={errs[:2]}")
        except Exception as e:
            print(f"{k:10s} FAIL {str(e)[:100]}")
        pg.close()
    b.close()
