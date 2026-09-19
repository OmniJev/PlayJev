<p align="center">
  <img src="docs/assets/board.webp" alt="Ten browser games, each one being played by PlayJev, with the score it had reached">
</p>

<h1 align="center">PlayJev: A Multimodal JEV-Like Model for Small Games</h1>

<p align="center">
  <a href="https://omnijev.github.io/PlayJev/"><img alt="live demo" src="https://img.shields.io/badge/live%20demo-show%20it-eda100?style=flat-square&labelColor=16181c"></a>
  <a href="https://huggingface.co/OmniJev/PlayJev-0.8B"><img alt="weights on Hugging Face" src="https://img.shields.io/badge/weights-PlayJev--0.8B-ffd21e?style=flat-square&logo=huggingface&logoColor=ffd21e&labelColor=16181c"></a>
  <a href="#-how-a-decision-is-made"><img alt="43 ms per move" src="https://img.shields.io/badge/per%20move-43%20ms-eb6834?style=flat-square&labelColor=16181c"></a>
  <a href="#-results"><img alt="0.53 vs teacher" src="https://img.shields.io/badge/vs%20teacher-0.53-2a78d6?style=flat-square&labelColor=16181c"></a>
</p>

<p align="center">
  <img alt="pixels only" src="https://img.shields.io/badge/input-pixels_only-1c5cab?style=flat-square">
  <img alt="863k frames" src="https://img.shields.io/badge/training-863k_frames-3987e5?style=flat-square">
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
`actions`), and the same page is both training environment and demo tile. Every tile below opens that game on
the demo, with the model playing it.

<table align="center">
  <tr>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=tetris"><img src="docs/assets/thumbs/tetris.png" width="150"><br><b>Tetris</b></a><br>5 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=snake"><img src="docs/assets/thumbs/snake.png" width="150"><br><b>Snake</b></a><br>4 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=pacman"><img src="docs/assets/thumbs/pacman.png" width="150"><br><b>Pacman</b></a><br>4 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=racer"><img src="docs/assets/thumbs/racer.png" width="150"><br><b>Javascript Racer</b></a><br>6 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=invaders"><img src="docs/assets/thumbs/invaders.png" width="150"><br><b>Space Invaders</b></a><br>3 moves</td>
  </tr>
  <tr>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=sokoban"><img src="docs/assets/thumbs/sokoban.png" width="150"><br><b>Sokoban</b></a><br>4 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=mario"><img src="docs/assets/thumbs/mario.png" width="150"><br><b>Infinite Mario</b></a><br>7 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=flappy"><img src="docs/assets/thumbs/flappy.png" width="150"><br><b>Floppy Bird</b></a><br>2 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=breakout"><img src="docs/assets/thumbs/breakout.png" width="150"><br><b>Breakout</b></a><br>3 moves</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=2048"><img src="docs/assets/thumbs/2048.png" width="150"><br><b>2048</b></a><br>4 moves</td>
  </tr>
</table>

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

One model for all ten games, 16 held-out episodes each, argmax move. Random play and the teacher run the same
seeds through the same harness; a teacher is the per-game search program that plays on the game's internal
state, which the model never sees. **vs teacher** is (model - random) / (teacher - random), so 0 is random play
and 1.00 is the teacher. `cloning` is one epoch over 863k teacher-labelled frames, `version 1` and `version 2`
two DAgger rounds on top: the model plays 40k frames per game, the teachers label every frame it visited, one
more epoch. Bold is the released model.

| Game | Random | Cloning | Version 1 | Version 2 | Teacher | vs teacher |
|---|---:|---:|---:|---:|---:|---:|
| Space Invaders | 215 | 400 | 400 | **400** | 400 | 1.00 |
| Racer | 238 | 6211 | 6704 | **6707** | 6712 | 1.00 |
| Sokoban | 6.6 | 57.9 | 102.3 | **102.1** | 102.2 | 1.00 |
| Snake | 1.0 | 77.8 | 107.9 | **89.5** | 114 | 0.79 |
| Pacman | 113 | 1036 | 3209 | **3702** | 7026 | 0.52 |
| Infinite Mario | 613 | 1156 | 1170 | **1764** | 4229 | 0.32 |
| Tetris | 162 | 1034 | 1561 | **4718** | 15288 | 0.30 |
| Floppy Bird | 0.0 | 8.9 | 9.3 | **13.8** | 84.0 | 0.16 |
| Breakout | 496 | 611 | 1552 | **2712** | 16547 | 0.14 |
| 2048 | 1021 | 3174 | 2170 | **3386** | 19593 | 0.13 |
| **mean vs teacher** | | 0.37 | 0.49 | **0.53** | | |

Zero-shot the base model puts 0.7 on option A whatever the frame. Three games end up at their teacher and the
mean at 0.53. What separates the other seven from their teachers is one of four things, and the two rounds
close or halve three of them.

| Problem | Games | How we know | What the rounds did |
|---|---|---|---|
| Covariate shift | Sokoban, Racer, Snake, Pacman | agreement on the teacher's frames 0.95 / 0.84 / 0.99 / 0.87, on their own 0.91 / 0.56 / 0.93 / 0.92 | round 1 brings three of them to the teacher and triples Pacman |
| One frame shows no motion | Breakout, Mario | 16 percent agreement on its own play: cloning learned to read the paddle, which sits under the ball on every teacher frame | 47 percent after round 1, and round 2 doubles the score again |
| Single-step precision | Floppy Bird, Tetris | Flappy matches the teacher on 99.8 percent of frames and dies at 9 pipes on the one it misses | a few hundred such moments in 40k frames, so it takes both rounds |
| Reading tile digits at 448 px | 2048 | 0.48 agreement either way, and the model knows it: confidence 0.25 | relabelling cannot help where the digits are unreadable, resolution can |

The rest of what we measured, one line each.

- **Agreement can move against the score.** Snake scores 89.5 in round 2 against 107.9 in round 1, lower on 12 of
  the 14 held-out seeds the two rounds share (sign-flip permutation p = 0.0073), while its on-policy agreement
  rises more than any other game's. Agreement averages over the frames the model visits and those frames changed,
  455 steps per episode down to 348, so the mean shifts to the easy early game. The closed loop is the instrument
  that sees it.
- **The second frame has to be its own image.** Merged into the vision tower's temporal patch it does nothing; as
  a separate image Breakout gains .096 validation agreement, the largest effect in the ablation, while Mario and
  Racer, the two games whose camera translates, lose at every point.
- **Snake has to be in the training mix.** Hold it and Racer out and two seeds of eight never use the image at
  all: five games stay on the letter prior, three take the most common move for their option list. Snake's label
  is a BFS arrow, uniform over the four moves and decided by the board alone, so no option list predicts it. Put
  it back and every game reads the frame by step 1500.
- **Both halves of an option are read** (`scripts/probe_options.py`, 400 validation frames per game). Neutral
  move names cost 0 to 13 points; rotating the descriptions while the names stay moves the decision in Flappy
  (84 percent), Sokoban (67) and Racer (47). The model that never learned to look follows the name every time.
- **One step late is what real time costs.** Deciding step k+1 from frame k takes the reflex games apart: Snake
  77.8 to 11.4, Tetris 1034 to 248, Flappy 8.9 to 0.2, while Invaders and Racer barely move. Training on labels
  shifted one step (`--label-delay 1`) buys part of it back where the next decision follows from the current
  frame and costs where it depends on what the current move does, so the shift has to be per game.
- **The confidence is worth something.** Below a threshold the decision goes to the teacher
  (`playjev.play --handover`): a third of Breakout's steps handed over by confidence reaches the teacher's score,
  the same share picked at random gets a third of the way.
- **One execution rule.** A move that leaves the observation unchanged (a blocked direction in 2048 or Sokoban is
  a legal no-op) is not repeated on that observation, and the next most probable move goes instead. Without it a
  deterministic policy loops to the step cap and 2048 scores 54. It never fires where the frame changes every step.

## 🏋️ Training Recipe

| Stage | What it is |
|---|---|
| **Teachers** | One search program per game on the game's internal state: BFS (Snake), expectimax (2048), Dellacherie (Tetris), A* with deadlock pruning (Sokoban), exact physics (Floppy Bird), ghost occupancy (Pacman), ball flight (Breakout), dodge-and-aim DP (Invaders), lookahead steering (Racer), physics rollouts (Mario). Soft target: 0.9 on the best move, 0.1 over acceptable ones, 0 on losing ones. |
| **Collection** | 100k frames per game, 448 px JPEGs, teachers playing with 2 to 30 percent random moves so the data covers recoveries. |
| **Fine-tuning** | Full fine-tuning, one epoch over the ten games mixed, batch 64, lr 2e-5, bf16 autocast on fp32 master weights, 3 to 4 h per round on one H200. |
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

