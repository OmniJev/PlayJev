# PlayJev demo page: specification

The demo is what people will see first. It lives on the `gh-pages` branch of OmniJev/PlayJev and needs no
backend: every game tile replays a recorded episode of the trained model with the model's per-step probabilities
baked in, and a "live" switch talks to a running PlayJev server for anyone who has one.

## What the page shows

One page. A short plain-language opening (two or three sentences: an 0.8B open model, ten browser games, one
forward pass per move, probabilities over the moves, no text). Then the ten game tiles in a responsive grid
(five wide on desktop, two on a phone). Each tile:

- the real game canvas, playing (the vendored game with the same shim and hook as in training, driven by the
  recorded action sequence so the picture is the real game, not a video);
- to its right or below it, one horizontal bar per action, labelled with the action name, its length the model's
  probability at the current step, the chosen action highlighted; the Jev confidence as a number;
- a small line under the bars: step counter, score, and "trained model" versus "random" toggle (random replays a
  random-policy episode so the gap is visible);
- controls: play/pause, speed (1x, 2x, 4x), restart, next recorded episode.

Below the grid: a results table (per game: random score, trained model score, teacher score, decisions per second),
one calibration plot per game as small inline SVGs (reliability diagram: predicted probability of the chosen move vs
how often it agreed with the teacher), and a short "how it works" section with the exact prompt the model sees
(image placeholder plus the option list) and a link to the repo, the OpenJev server and the paper when it exists.

Style: follow the no-AI-look rule (plain sentences, no oversized headline with a grey caption under it, no uppercase
eyebrow labels, no em-dashes). Colours come from the cover art palette already used on Awesome-JEV (JEV blue #2F80ED,
coral #F0545C, amber, violet, green); one accent for the chosen action, greys for the rest. The game tiles are the
visual; the chrome stays quiet. Light and dark both work.

## Replay format (produced by `python -m playjev.record`, to be written)

`demo/replays/<game>/<policy>_<seed>.json`:
```
{"game": "snake", "policy": "playjev-0.8b-sft1", "seed": 5003, "actions": ["up","down","left","right"],
 "steps": [{"a": 3, "p": [0.02, 0.05, 0.03, 0.90], "score": 1}, ...], "final_score": 131, "frames_per_step": 1}
```
The page reloads the game with `pj.start(seed)` and replays `steps[i].a` through `pj.step`, so the canvas is the
real game. Determinism of the hooks (checked per game) is what makes this possible; the recorder verifies the
replayed score matches before writing a file.

## Live mode

If a server URL is given (`?server=http://host:port`), the page sends the current canvas frame (JPEG data URL) with
the option list in the OpenJev request shape (`docs/DESIGN.md` 3.1) and uses the returned probabilities instead of
the recording. Latency is shown next to the confidence.

## Build

`demo/` in the repo: `index.html`, `demo.js`, `demo.css`, the ten vendored games copied by a build script (only the
files the entry page needs), `replays/`. A `scripts/build_demo.py` assembles it and `scripts/shot_demo.py` screenshots
it at desktop and phone widths for review before publishing.
