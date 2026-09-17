# Task: invaders and racer hooks

1. `invaders` (games/invaders/, vendored StrykerKKD/SpaceInvaders, entry `index.html`; `indexOpt.html` also exists,
   use the plain one and say why). A previous agent left a draft pj_hook.js and pj.json (not benched): read, finish,
   verify. Loop: rAF plus setInterval and Date.now, keys via keyCode and `.key`. Actions: left, right, fire, left+fire,
   right+fire, noop. stepFrames about 5. Score: the game's score; expose lives/health, player x, alien count, lowest
   alien row in info(). done: out of lives/health, or all aliens cleared (if a new wave spawns, keep going, say so).
   start: into a running game from the splash. Frame: the game canvas only.

2. `racer` (games/racer/, vendored jakesgordon/javascript-racer, entry `v4.final.html` with `common.js`; the page has
   a settings panel and links, frame must be the game canvas only). Nothing done yet. Loop: rAF with Date.now; keys
   arrows via keyCode on document. Actions: left, right, faster, slower, left+faster, right+faster. stepFrames about 5.
   Score: distance travelled along the track (the game keeps `position` on a track of length `trackLength`; running
   total across laps), expose speed, playerX, current curve, lap time in info(). The game has no death: done() after one
   full lap or after N steps, constant in the hook, documented. The page auto-starts on load; check the settings UI
   needs no interaction.
