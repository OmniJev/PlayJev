# PlayJev game roster (decided 2026-09-18)

Ten browser games, all plain HTML5/JS, cloned shallow into `games/js/`. Picked for: colourful
rendering (owner: 黑白线条的不要), discrete action sets, a numeric score, and a permissive licence.
Screenshots reviewed in headless Chromium before selection.

| # | Game | Repo | Licence | Entry file | Actions (Choice options) |
|---|------|------|---------|------------|--------------------------|
| 1 | Super Mario (Infinite Mario HTML5) | robertkleffner/mariohtml5 | Unlicense (code); Nintendo sprites, see risk note | `mariohtml5/main.html` | noop, left, right, jump, right+jump, right+run, right+run+jump |
| 2 | Snake | patorjk/JavaScript-Snake | MIT | `JavaScript-Snake/src/index.html` | up, down, left, right |
| 3 | Tetris | jakesgordon/javascript-tetris | MIT | `javascript-tetris/index.html` | left, right, rotate, drop, none |
| 4 | 2048 | gabrielecirulli/2048 | MIT | `2048/index.html` | up, down, left, right |
| 5 | Flappy Bird | nebez/floppybird | Apache-2.0 | `floppybird/index.html` | flap, wait |
| 6 | Pac-Man | daleharvey/pacman | WTFPL | `pacman/index.html` | up, down, left, right |
| 7 | Breakout | jakesgordon/javascript-breakout | MIT | `javascript-breakout/index.html` | left, right, stay |
| 8 | Space Invaders | StrykerKKD/SpaceInvaders | MIT | `SpaceInvaders/index.html` | left, right, fire, left+fire, right+fire, noop |
| 9 | Racer (outrun style) | jakesgordon/javascript-racer | MIT | `javascript-racer/v4.final.html` | left, right, faster, slower, left+faster, right+faster |
| 10 | Sokoban | taniarascia/sokoban | MIT | `sokoban/index.html` | up, down, left, right |

Rejected after screenshots: jakesgordon/javascript-pong, dmcinnes/HTML5-Asteroids, wayou/t-rex-runner
(black and white line art), dwmkerr/spaceinvaders and MDN Gamedev-Canvas-workshop breakout (plain
rectangles), MattSkala/html5-bombergirl (needs CreateJS from a CDN, breaks offline),
victorqribeiro/invaderz (page error on load). FullScreenShenanigans/FullScreenMario is DMCA-blocked
on GitHub (Nintendo, 2016), which is the risk note for game 1: the Infinite Mario code is public
domain but the sprites are Nintendo's. Decide before the public demo whether to swap in free art.

Tooling here: `serve_games.sh` (static server on a free localhost port, writes
`/tmp/games_http.port`), `shot_games.py` (Playwright screenshots), venv `.venv` (Python 3.12,
playwright 1.63, chromium headless shell 1243).
