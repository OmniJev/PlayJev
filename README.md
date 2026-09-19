<p align="center">
  <img src="docs/assets/board.webp" alt="Ten browser games, each one being played by PlayJev, with the score it had reached">
</p>

<h1 align="center">PlayJev: A Multimodal JEV-Like Model for Small Games</h1>

<p align="center">
  <a href="https://omnijev.github.io/PlayJev/"><img alt="live demo" src="https://img.shields.io/badge/live%20demo-show%20it-eda100?style=flat-square&labelColor=16181c"></a>
  <a href="https://huggingface.co/OmniJev/PlayJev-0.8B"><img alt="model" src="https://img.shields.io/badge/model-PlayJev--0.8B-2a78d6?style=flat-square&logo=huggingface&logoColor=white&labelColor=16181c"></a>
  <a href="#-the-ten-games"><img alt="ten games" src="https://img.shields.io/badge/games-10-1baf7a?style=flat-square&labelColor=16181c"></a>
  <a href="#-how-a-decision-is-made"><img alt="43 ms per move" src="https://img.shields.io/badge/per%20move-43%20ms-eb6834?style=flat-square&labelColor=16181c"></a>
</p>

<p align="center">
  <img alt="pixels only" src="https://img.shields.io/badge/input-pixels_only-1c5cab?style=flat-square">
  <img alt="863k frames" src="https://img.shields.io/badge/training-863k_frames-3987e5?style=flat-square">
  <a href="#-results"><img alt="0.53 vs teacher" src="https://img.shields.io/badge/vs_teacher-0.53-eda100?style=flat-square"></a>
  <a href="https://github.com/OmniJev/openJev"><img alt="Jev System One" src="https://img.shields.io/badge/contract-Jev_System_One-1baf7a?style=flat-square"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/licence-Apache_2.0-6d747e?style=flat-square"></a>
</p>

<h2 align="center">🎮 &nbsp;<a href="https://omnijev.github.io/PlayJev/">Show the live demo</a>&nbsp; 🎮</h2>

PlayJev is Qwen3.5-0.8B-Base fine-tuned to play ten classic browser games from raw pixels. One frame goes in,
one forward pass runs, one move comes out, 43 ms on an H200. Every picture above is the trained model playing,
each a frame from a recorded held-out episode with the score it had reached by then. The weights are on
[Hugging Face](https://huggingface.co/OmniJev/PlayJev-0.8B).

## 🎮 The Ten Games

Plain HTML5/JS with one hook each (`window.pj`: `start(seed)`, `step(action)`, `frame()`, `score()`, `done()`,
`actions`), and the same page is both training environment and demo tile. Every name opens that game's board on
the demo, with the model playing.

|   | Game | Moves |   | Game | Moves |
|---|---|---:|---|---|---:|
| [<img src="docs/assets/thumbs/tetris.png" width="180">][tetris] | [Tetris][tetris] | 5 | [<img src="docs/assets/thumbs/snake.png" width="180">][snake] | [Snake][snake] | 4 |
| [<img src="docs/assets/thumbs/pacman.png" width="180">][pacman] | [Pacman][pacman] | 4 | [<img src="docs/assets/thumbs/racer.png" width="180">][racer] | [Javascript Racer][racer] | 6 |
| [<img src="docs/assets/thumbs/invaders.png" width="180">][invaders] | [Space Invaders][invaders] | 3 | [<img src="docs/assets/thumbs/sokoban.png" width="180">][sokoban] | [Sokoban][sokoban] | 4 |
| [<img src="docs/assets/thumbs/mario.png" width="180">][mario] | [Infinite Mario][mario] | 7 | [<img src="docs/assets/thumbs/flappy.png" width="180">][flappy] | [Floppy Bird][flappy] | 2 |
| [<img src="docs/assets/thumbs/breakout.png" width="180">][breakout] | [Breakout][breakout] | 3 | [<img src="docs/assets/thumbs/2048.png" width="180">][2048] | [2048][2048] | 4 |

## 🔍 Why Pixels

Every Jev-shaped game demo before this one feeds the model text. TypeSafe's Mario parses emulator RAM into
JSON and says the model gets no screenshots, the open-jev Doom demo writes one line of text per frame, and
VideoGameBench and lmgame-Bench run frontier models with reasoning loops at seconds per move. PlayJev plays
ten games from the frame alone, in real time, and returns a probability the game loop can act on.

## 🧠 How a Decision Is Made

The model never sees a game's name. It sees the current frame and the list of moves, and it answers with one
of them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/decision-flow-dark.svg">
  <img alt="A game frame and an option list go into one forward pass, which returns a probability for every move" src="docs/assets/decision-flow.svg" width="100%">
</picture>

![The demo showing one Mario frame, the probability the model puts on each of the seven moves, and its confidence through the episode](docs/assets/decision.png)

The demo's single game view is the whole model in one picture: the frame on the left is the only input, the
bars are what the forward pass returns, the line below them is how sure it was at every step so far. Moves are
shuffled in every training sample, so position carries no information. One frame per decision; where velocity
matters the vision tower also takes the previous frame, at no extra token cost. The prompt is built in
[playjev/model.py](playjev/model.py), and the demo prints the exact one for every game.

## ▶️ Run It

The model plays a game, one command:

```bash
python -m playjev.play snake --policy local --ckpt OmniJev/PlayJev-0.8B --episodes 1
```

It pulls the weights from Hugging Face, opens Snake in headless Chromium and plays it. Setup once (Python 3.12,
a CUDA GPU, about 3 GB for inference and 17 GB for training at batch 64):

```bash
git clone https://github.com/OmniJev/PlayJev && cd PlayJev
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && playwright install chromium
```

| Command | What it does |
|---|---|
| `playjev.play <game> --policy teacher` (or `random`) | the two reference rows; `--delay 1` decides one step late |
| `playjev.collect <game> --steps 4000 --shard s0` | (frame, teacher target) pairs under `data/<game>/s0` |
| `playjev.train_sft --games <game> --model Qwen/Qwen3.5-0.8B-Base --out ckpt/x` | fine-tune, one epoch |
| `playjev.serve --ckpt <ckpt> --port 18732` | the checkpoint at `/v1/systemone` in the OpenJev request shape; the demo switches every tile to it with `?server=http://127.0.0.1:18732` |
| `python scripts/build_demo.py` | rebuild `demo/` from the games and the recorded runs, then serve it and open `index.html` |

## 📊 Results

One model for all ten games, playing 16 held-out episodes per game, argmax move, episodes capped at 1500
steps. Random play and the teacher run the same seeds through the same harness. A teacher is the per-game
program that plays on the game's internal state, which the model never sees; **vs teacher** is
(model - random) / (teacher - random), so 0 is random play and 1.00 is the teacher.

| Game | Random | PlayJev | Teacher | vs teacher |
|---|---:|---:|---:|---:|
| Space Invaders | 215 | **400** | 400 | 1.00 |
| Racer | 238 | **6707** | 6712 | 1.00 |
| Sokoban | 6.6 | **102.1** | 102.2 | 1.00 |
| Snake | 1.0 | **89.5** | 114 | 0.79 |
| Pacman | 113 | **3702** | 7026 | 0.52 |
| Infinite Mario | 613 | **1764** | 4229 | 0.32 |
| Tetris | 162 | **4718** | 15288 | 0.30 |
| Floppy Bird | 0.0 | **13.8** | 84.0 | 0.16 |
| Breakout | 496 | **2712** | 16547 | 0.14 |
| 2048 | 1021 | **3386** | 19593 | 0.13 |
| **mean** | | | | **0.53** |

Zero-shot the base model puts 0.7 on option A whatever the frame; three games end up at their teacher and the
mean at 0.53. Three things are still open: motion from a single frame (Breakout, Mario), single-step precision
(Floppy Bird, Tetris), and reading tile digits at 448 px (2048). Snake is the one column that goes backwards,
and only the closed loop sees it.

<details>
<summary>📋 &nbsp;Every version, every game</summary>

`cloning` is one epoch of behaviour cloning on 863k teacher-labelled frames. `version 1` and `version 2` are
two DAgger rounds on top of it: the current model plays 40k frames per game, the teachers label every frame it
visited, and the model trains one more epoch on those plus the earlier shards. 4 h 11 min, 3 h 10 min and
4 h 20 min on one H200.

| Game | Random | Cloning | Version 1 | Version 2 | Teacher |
|---|---:|---:|---:|---:|---:|
| Space Invaders | 215 | 400 | 400 | 400 | 400 |
| Racer | 238 | 6211 | 6704 | 6707 | 6712 |
| Sokoban | 6.6 | 57.9 | 102.3 | 102.1 | 102.2 |
| Snake | 1.0 | 77.8 | 107.9 | 89.5 | 114 |
| Pacman | 113 | 1036 | 3209 | 3702 | 7026 |
| Infinite Mario | 613 | 1156 | 1170 | 1764 | 4229 |
| Tetris | 162 | 1034 | 1561 | 4718 | 15288 |
| Floppy Bird | 0.0 | 8.9 | 9.3 | 13.8 | 84.0 |
| Breakout | 496 | 611 | 1552 | 2712 | 16547 |
| 2048 | 1021 | 3174 | 2170 | 3386 | 19593 |
| **mean vs teacher** | | **0.37** | **0.49** | **0.53** | |
</details>

<details>
<summary>🔬 &nbsp;What each round fixed, and the Snake regression</summary>

| What was wrong | Games | How we know | What the rounds did |
|---|---|---|---|
| Covariate shift | Sokoban, Racer, Snake, Pacman | agreement on the teacher's frames 0.95 / 0.84 / 0.99 / 0.87, on their own 0.91 / 0.56 / 0.93 / 0.92, errors where they cost most | round 1 brings three of them to the teacher and triples Pacman, round 2 holds |
| One frame shows no motion | Breakout, Mario | 16 percent agreement on its own play: cloning learned to read the paddle, which sits under the ball on every teacher frame | DAgger 47 percent and 2.5x the score, round 2 doubles both again |
| Single-step precision | Floppy Bird, Tetris | Flappy matches the teacher on 99.8 percent of frames and dies at 9 pipes on the one it misses, Tetris drops a move early | a few hundred such moments in 40k frames, so round 1 barely moves them, round 2 triples Tetris and lifts Flappy to 0.16 |
| Reading tile digits at 448 px | 2048 | 0.48 agreement either way, and the model knows it: confidence 0.25 | still open |

**Snake goes backwards in round 2** and it is real: lower on 12 of the 14 held-out seeds the two rounds share,
paired difference -17.4 (sd 22.5), sign-flip permutation p = 0.0073. Both agreement instruments miss it, because
agreement averages over the frames the model visits and those frames changed: the mean shifts to the easy early
game where a three-segment snake can hardly be wrong. In a fatal-on-mistake game, agreement moves against the
score.

| Snake, round 1 to round 2 | | |
|---|---:|---|
| score | 107.9 → 89.5 | the closed loop, held-out seeds |
| validation agreement | .780 → .775 | flat |
| on-policy agreement | .722 → .824 | rises more than in any other game |
| steps per episode | 455 → 348 | fewer frames, and easier ones |

Two ablations decided the recipe. Mario and Racer are the only games whose camera translates, so their frame
difference is a global shift that the patch convolution mixes into every embedding, while for a fixed camera
the frame difference *is* the object that moved. Snake's label is a BFS arrow, uniform over the four moves and
decided by the board alone, so no option list predicts it and the image pathway has to be used.

| Ablation | What happened |
|---|---|
| second frame merged into the vision tower's temporal patch | nothing outside Breakout |
| second frame passed as its own image | Breakout +.096 validation agreement at every eval point, the largest effect in the ablation, Mario and Racer lose at every point |
| eight-game mix with Snake and Racer held out | two seeds collapse: five games never leave the letter prior, three take the most common move for their option list, the image pathway goes unused |
| same mix with Snake back in | every game reads the frame by step 1500 |
</details>

<details>
<summary>👁️ &nbsp;Does it read the move descriptions?</summary>

Both halves of an option are read, and a fixed classification head could not be moved by editing a sentence.
400 validation frames per game, prompt otherwise unchanged (`scripts/probe_options.py`).

| Edit to the option list | What the model does |
|---|---|
| shuffle the moves | nothing, training permutes them in every sample |
| neutral names (alpha, bravo, ...), descriptions kept | loses 0 to 2 points in six games, 7 in Tetris, 8 in Racer, 13 in Invaders |
| names only, descriptions dropped | loses nothing, even in Snake, 2048, Pacman and Sokoban, whose names are the same four words: which "up" it is comes from the frame |
| descriptions rotated one move along, names left in place | follows the description in Flappy (84%), Sokoban (67) and Racer (47), the name in Breakout (95), Mario (82) and Pacman (80), splits in Snake and Tetris |
| an extra fake move, "hold: keep the current move and do nothing new" | takes 2 to 9% of the mass in seven games, 13 to 18% in the three it is least sure about (2048, Invaders, Breakout) |

The same probes on a model that never learned to look (the eight-game run above) come out the opposite way: it
follows the name 100 percent of the time wherever the two conflict, neutral names collapse it to chance in
Flappy and Breakout, and its agreement is at chance wherever a name alone does not give the move away.
</details>

<details>
<summary>⚙️ &nbsp;The execution rule and the real-time setting</summary>

**Execution rule.** A move that leaves the observation unchanged (a blocked direction in 2048 or Sokoban is a
legal no-op) is not repeated on that observation, and the next most probable move goes instead. Without it a
deterministic policy loops to the step cap and 2048 scores 54. It never fires where the frame changes every
step, and those numbers are identical with and without it.

**Latency.** One step late, frame k deciding step k+1, is what real time costs at 83 to 100 ms per step, and it
takes the reflex games apart. Training on shifted labels buys part of it back where the next decision follows
from the current frame and costs where the next decision depends on what the current move does to the board, so
the shift has to be per game. Even where it helps, the real-time score stays a fraction of the undelayed one,
which is what the second frame is for.

| Cloning model | On time | One step late | Late, `--label-delay 1` |
|---|---:|---:|---:|
| Floppy Bird | 8.9 | 0.2 | **2.3** |
| Sokoban | 57.9 | 6.6 | 6.4 |
| Snake | 77.8 | 11.4 | **19.6** |
| Tetris | 1034 | 248 | **306** |
| Pacman | 1036 | 502 | 236 |
| Breakout | 611 | 394 | **636** |
| 2048 | 3174 | 2191 | **3195** |
| Infinite Mario | 1156 | 820 | 242 |
| Racer | 6211 | 5885 | 5116 |
| Space Invaders | 400 | 400 | 400 |
</details>

<details>
<summary>🎚️ &nbsp;Is the confidence worth anything?</summary>

Low-confidence steps are where the errors are: on its own play Tetris agrees with the teacher on 36 percent of
the steps below confidence 0.5 against 61 overall, Invaders 41 against 97 above 0.9, Racer 34 against 97,
Sokoban 3 against 94. The operational test is a System Two behind the model: below a confidence threshold the
decision goes to the teacher (`playjev.play --handover`), and the control hands the same share over at random
(`--handover-random`). Cloning model, 16 episodes per setting.

| Game | Alone | Handed over | By confidence | At random | Teacher |
|---|---:|---:|---:|---:|---:|
| Breakout | 701 | 16% | **3423** | 1257 | 16547 |
| | | 34% | **15139** | 6192 | |
| Tetris | 1054 | 14% | **2976** | 1610 | 15288 |
| | | 33% | **9109** | 3177 | |
| | | 63% | **15156** | 8473 | |
| Pacman | 1029 | 16% | **1739** | 1594 | 7026 |
| | | 22% | **3573** | 2947 | |
| Snake | 77 | 38% | **90** | 82 | 114 |
| | | 53% | **112** | 92 | |
| Floppy Bird | 8.9 | 3% | **14.4** | | 84 |
| | | 18% | **38.9** | | |
| 2048 | 3009 | 32% | **6322** | | 19593 |
| Racer | 6214 | 11% | **6634** | | 6712 |
| | | 39% | **6712** | | |

A third of Breakout's steps handed over by confidence reaches the teacher's level, the same share at random
gets a third of the way: the confidence picks the steps that matter, which is what a System One is for.
Sokoban is the exception (34 percent handed over, 58 to 71): its failures are boards the model has already
deadlocked, and no later decision repairs them.
</details>

## 🏋️ Training Recipe

| Stage | What it is |
|---|---|
| **Teachers** | One search program per game on the game's internal state: BFS (Snake), expectimax (2048), Dellacherie (Tetris), A* with deadlock pruning (Sokoban), exact physics (Floppy Bird), ghost occupancy (Pacman), ball flight (Breakout), dodge-and-aim DP (Invaders), lookahead steering (Racer), physics rollouts (Mario). Soft target: 0.9 on the best move, 0.1 over acceptable ones, 0 on losing ones. |
| **Collection** | 100k frames per game, 448 px JPEGs, teachers playing with 2 to 30 percent random moves so the data covers recoveries. |
| **Fine-tuning** | Full fine-tuning, one epoch over the ten games mixed, batch 64, lr 2e-5, bf16 autocast on fp32 master weights. |
| **Closed loop** | 16 held-out episodes per game, the same seeds as random play and the teacher. Validation also reports agreement, calibration error and per-position bias. |

Random play and every teacher on the same seeds: [docs/BASELINES.md](docs/BASELINES.md).

## 📈 More Charts

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/chart-dark.png">
  <img alt="Score against the teacher for each game after cloning and after each of the two rounds" src="docs/assets/chart.png">
</picture>

Every game and every version against its teacher. Three games are finished after one round, and the rest
split by what was wrong with them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/handover-dark.png">
  <img alt="Score against the share of steps handed to the teacher, picked by confidence and picked at random" src="docs/assets/handover.png">
</picture>

Hand the hardest steps to a System Two and the score climbs to the teacher's. Hand over the same number of
steps at random and it does not, which is the whole claim about the confidence in one picture.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/calibration-dark.png">
  <img alt="How often the model matched the teacher against the probability it gave the move it took, for all ten games" src="docs/assets/calibration.png">
</picture>

The probability means something in every game. Most of these curves sit above the diagonal, so the model is
usually more right than it claims; 2048, the game it is least sure about, sits on it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/realtime-dark.png">
  <img alt="Teacher-normalised score with the decision on time and one step late, for all ten games" src="docs/assets/realtime.png">
</picture>

One step of latency, which is what real time costs, takes the reflex games apart and leaves the slow ones
alone. Both bars are the cloning model, which is why they sit below the table above.

## 📁 Repository

```
games/<id>/        vendored game, pj.json manifest, pj_hook.js, NOTES.md, TEACHER.md
games/_shared/     pj_shim.js: virtual clock, seeded Math.random, synthetic keys, frame grab
playjev/           env.py (Playwright driver), collect.py, teachers/, model.py, train_sft.py, play.py, serve.py
demo/              the GitHub Pages site; scripts/build_demo.py assembles it from games/ and runs/replays/
docs/              HARNESS.md (the hook contract), BASELINES.md (the reference scores), DEMO.md
```

## 🔗 Related

[OpenJev](https://github.com/OmniJev/openJev), the text-state System One server this model plugs into, and
[Awesome-JEV](https://github.com/OmniJev/awesome-jev), the reading list behind System One models.

## ⚖️ Licence and Credits

Code and trained weights: Apache-2.0.

The ten games are other people's work, vendored under `games/<id>/` with the author's own licence file and a
`vendor.patch` of every line we changed.

- Tetris, [github.com/jakesgordon/javascript-tetris](https://github.com/jakesgordon/javascript-tetris), MIT
- Snake, [github.com/patorjk/JavaScript-Snake](https://github.com/patorjk/JavaScript-Snake), MIT
- Pacman, [github.com/daleharvey/pacman](https://github.com/daleharvey/pacman), WTFPL
- Javascript Racer, [github.com/jakesgordon/javascript-racer](https://github.com/jakesgordon/javascript-racer), MIT
- Space Invaders, [github.com/StrykerKKD/SpaceInvaders](https://github.com/StrykerKKD/SpaceInvaders), MIT
- Sokoban, [github.com/taniarascia/sokoban](https://github.com/taniarascia/sokoban), MIT
- Infinite Mario, [github.com/robertkleffner/mariohtml5](https://github.com/robertkleffner/mariohtml5), Unlicense
- Floppy Bird, [github.com/nebez/floppybird](https://github.com/nebez/floppybird), Apache-2.0
- Breakout, [github.com/jakesgordon/javascript-breakout](https://github.com/jakesgordon/javascript-breakout), MIT
- 2048, [github.com/gabrielecirulli/2048](https://github.com/gabrielecirulli/2048), MIT

Three of those READMEs say the art is not the author's to license: Mario's sprites are Nintendo's, Floppy
Bird's come from the original Android game and belong to Dong Nguyen and .GEARS, the Racer's are placeholder
art from the Mega Drive OutRun. The licence above covers the code each author wrote. Sokoban's Microban levels
are by David Skinner.

The roster ships silent. Every sound and music file was deleted, which costs nothing: the driver already
aborted every audio request (`playjev/env.py`), the shim forces media elements muted, and Chromium runs with
`--mute-audio`. Every frame in this repository, training or demo, was produced in silence.

[mario]: https://omnijev.github.io/PlayJev/?game=mario
[snake]: https://omnijev.github.io/PlayJev/?game=snake
[tetris]: https://omnijev.github.io/PlayJev/?game=tetris
[2048]: https://omnijev.github.io/PlayJev/?game=2048
[flappy]: https://omnijev.github.io/PlayJev/?game=flappy
[pacman]: https://omnijev.github.io/PlayJev/?game=pacman
[breakout]: https://omnijev.github.io/PlayJev/?game=breakout
[invaders]: https://omnijev.github.io/PlayJev/?game=invaders
[racer]: https://omnijev.github.io/PlayJev/?game=racer
[sokoban]: https://omnijev.github.io/PlayJev/?game=sokoban
