<p align="center">
  <img src="docs/assets/board.png" alt="Ten browser games, each one being played by PlayJev, with the score it had reached">
</p>

<h1 align="center">PlayJev: An Open-Source JEV-Like Model That Plays Small Games</h1>

<p align="center">
  <a href="https://omnijev.github.io/PlayJev/"><img alt="live demo" src="https://img.shields.io/badge/live_demo-play_it-eda100?style=for-the-badge&labelColor=16181c"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.5-0.8B-Base"><img alt="model" src="https://img.shields.io/badge/model-Qwen3.5--0.8B-2a78d6?style=for-the-badge&logo=huggingface&logoColor=white&labelColor=16181c"></a>
  <a href="#-the-ten-games"><img alt="ten games" src="https://img.shields.io/badge/games-10-1baf7a?style=for-the-badge&labelColor=16181c"></a>
  <a href="#-how-a-decision-is-made"><img alt="43 ms per move" src="https://img.shields.io/badge/per_move-43_ms-eb6834?style=for-the-badge&labelColor=16181c"></a>
</p>

<p align="center">
  <img alt="pixels only" src="https://img.shields.io/badge/input-pixels_only-1c5cab?style=flat-square">
  <img alt="863k frames" src="https://img.shields.io/badge/training-863k_frames-3987e5?style=flat-square">
  <a href="#-results"><img alt="0.53 vs teacher" src="https://img.shields.io/badge/vs_teacher-0.53-eda100?style=flat-square"></a>
  <a href="https://github.com/OmniJev/openJev"><img alt="Jev System One" src="https://img.shields.io/badge/contract-Jev_System_One-1baf7a?style=flat-square"></a>
  <a href="LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/licence-Apache_2.0-6d747e?style=flat-square"></a>
</p>

<h3 align="center">🎮 &nbsp;<a href="https://omnijev.github.io/PlayJev/">Play the live demo</a>&nbsp; 🎮</h3>

PlayJev is Qwen3.5-0.8B-Base fine-tuned to play ten classic browser games from raw pixels. One frame goes in,
one forward pass runs, one move comes out, 43 ms on an H200. Every picture above is the trained model playing,
each a frame from a recorded held-out episode with the score it had reached by then.

## 🎮 The Ten Games

Plain HTML5/JS, vendored under `games/<id>/`. Each game exposes the same small hook (`window.pj`:
`start(seed)`, `step(action)`, `frame()`, `score()`, `done()`, `actions`), and the same page serves as both
training environment and demo tile.

|   | Game | Upstream | Licence | Moves |
|---|---|---|---|---:|
| <img src="docs/assets/thumbs/mario.png" width="140"> | Infinite Mario | [robertkleffner/mariohtml5](https://github.com/robertkleffner/mariohtml5) | Unlicense | 7 |
| <img src="docs/assets/thumbs/snake.png" width="140"> | Snake | [patorjk/JavaScript-Snake](https://github.com/patorjk/JavaScript-Snake) | MIT | 4 |
| <img src="docs/assets/thumbs/tetris.png" width="140"> | Tetris | [jakesgordon/javascript-tetris](https://github.com/jakesgordon/javascript-tetris) | MIT | 5 |
| <img src="docs/assets/thumbs/2048.png" width="140"> | 2048 | [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048) | MIT | 4 |
| <img src="docs/assets/thumbs/flappy.png" width="140"> | Floppy Bird | [nebez/floppybird](https://github.com/nebez/floppybird) | Apache-2.0 | 2 |
| <img src="docs/assets/thumbs/pacman.png" width="140"> | Pacman | [daleharvey/pacman](https://github.com/daleharvey/pacman) | WTFPL | 4 |
| <img src="docs/assets/thumbs/breakout.png" width="140"> | Breakout | [jakesgordon/javascript-breakout](https://github.com/jakesgordon/javascript-breakout) | MIT | 3 |
| <img src="docs/assets/thumbs/invaders.png" width="140"> | Space Invaders | [StrykerKKD/SpaceInvaders](https://github.com/StrykerKKD/SpaceInvaders) | MIT | 3 |
| <img src="docs/assets/thumbs/racer.png" width="140"> | Javascript Racer | [jakesgordon/javascript-racer](https://github.com/jakesgordon/javascript-racer) | MIT | 6 |
| <img src="docs/assets/thumbs/sokoban.png" width="140"> | Sokoban | [taniarascia/sokoban](https://github.com/taniarascia/sokoban) | MIT | 4 |

<details>
<summary>Art, licences and the silence</summary>

Three of the upstream projects say in their own READMEs that their art is not theirs to license. Infinite
Mario's sprites are Nintendo's, Floppy Bird's are extracted from the original Android game and belong to Dong
Nguyen and .GEARS, and the Racer's are placeholder art borrowed from the Mega Drive version of OutRun. The
licence in the table covers the code each author wrote. Sokoban's Microban levels are by David Skinner. Every
file we changed is recorded in that game's `vendor.patch`, next to its `NOTES.md` and `TEACHER.md`.

The roster ships silent. Every sound and music file was deleted, which costs nothing, because the driver
already aborted every audio request (`playjev/env.py`), the shim forces media elements muted, and Chromium
runs with `--mute-audio`. Every frame in this repository, training or demo, was produced in silence.
</details>

## 🔍 Why Pixels

Every Jev-shaped game demo before this one feeds the model text. TypeSafe's Mario demo parses the emulator's
RAM into JSON and states that the model does not receive screenshots. The open-jev Doom demo describes each
frame in one line of text. The zero-shot pixel attempts play one game without training, and VideoGameBench
and lmgame-Bench run frontier models with reasoning loops at seconds per move. PlayJev trains a small
vision-language model under the Jev contract and plays ten games from the frame alone, in real time, with a
probability the game loop can act on.

## 🧠 How a Decision Is Made

The model never sees a game's name. It sees the current frame and the list of moves, and it answers with one
of them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/decision-flow-dark.svg">
  <img alt="A game frame and an option list go into one forward pass, which returns a probability for every move" src="docs/assets/decision-flow.svg" width="100%">
</picture>

![The demo showing one Mario frame, the probability the model puts on each of the seven moves, and its confidence through the episode](docs/assets/decision.png)

That is the demo's single game view and the whole model in one picture: the frame on the left is the only
input, the bars on the right are the numbers that come out of the forward pass, and the line below them is
how sure the model was at every step of the episode so far. Moves are shuffled for every training sample, so
position carries no information. One frame per decision by default; for games where velocity matters the
vision tower also takes the previous frame, at no extra token cost. The exact prompt the model is given is in
[docs/MODEL_NOTES.md](docs/MODEL_NOTES.md).

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

- Zero-shot, the base model puts 0.7 on option A whatever the frame. One epoch of cloning puts all ten games
  above random.
- Version 1 closes the games whose only problem was covariate shift: Sokoban, Racer and Snake reach the
  teacher, Pacman triples.
- Version 2 takes the harder ones: Tetris 3x, Mario and Breakout 2x each.
- What is still open: motion from a single frame (Breakout, Mario), single-step precision (Floppy Bird,
  Tetris), and reading tile digits at 448 px (2048).
- Snake drops in version 2, 107.9 to 89.5, the one column that goes backwards, and both agreement metrics
  miss it. Only the closed loop sees it.

<details>
<summary>📋 &nbsp;Every version, every game</summary>

`cloning` is one epoch of behaviour cloning on 863k teacher-labelled frames. `version 1` and `version 2` are
two DAgger rounds on top of it: the current model plays 40k frames per game, the teachers label every frame
it visited, and the model trains one more epoch on those plus the earlier shards.

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

Training cost: 4 h 11 min, 3 h 10 min and 4 h 20 min on one H200.
</details>

<details>
<summary>🔬 &nbsp;What each round fixed, and the Snake regression</summary>

**Covariate shift.** Sokoban, Racer, Snake and Pacman reproduced the teacher on the teacher's own frames
(0.95, 0.84, 0.99 and 0.87 agreement) and lost it on their own (0.91, 0.56, 0.93, 0.92), with the errors
concentrated where they cost most. One round of labels on the model's own states brings three of them to the
teacher's level and triples Pacman; the three saturated ones hold through round 2.

**Snake goes the other way in round 2**, 107.9 to 89.5. It is a real regression and not 16-episode noise: on
the 14 held-out seeds the two rounds share, round 2 is lower on 12, the paired difference is -17.4 with sd
22.5, and a sign-flip permutation test over 200,000 permutations puts it at p = 0.0073. Mean episode length
falls with the score, 455 steps to 348, which is what a snake regression looks like: the model dies earlier,
so it eats less. Both agreement instruments miss it. Validation agreement is flat (.780 to .775) and
on-policy agreement, measured by replaying the recorded episodes through the real game with the teacher
watching, *rises* more for Snake than for any other game (.722 to .824). The reason is that agreement is
averaged over the frames the model visits and those frames changed: total steps over the same 16 episodes
fell 7275 to 5568, so the average shifts toward the easy early game, where a three-segment snake on an empty
board has almost no way to be wrong. In a fatal-on-mistake game, agreement can move against the score.

**Breakout and Mario need motion.** On the teacher's frames the paddle is already under the ball's landing
point, so the cloned policy learned to read the paddle instead of the ball (16 percent agreement on its own
play). DAgger raises that to 47 percent and the score 2.5x, and round 2 doubles both again. A single frame
still does not show the ball's direction, nor Mario's velocity and jump phase, and how the second frame is
delivered decides who benefits. Merged into the vision tower's temporal patch it adds nothing outside
Breakout. Passed as two separate images the model can compare by attention, Breakout gains validation
agreement at every eval point (+.096 on average, the largest effect in the ablation) while Mario and Racer
lose at every point. Those two are the only games in the roster whose camera translates: the frame difference
is dominated by the global shift of the scene, which the model has to discount before any local motion means
anything. For a fixed camera the frame difference *is* the object that moved. Stacking through the patch
convolution mixes the translation into every patch embedding, which is why the merged version looked flat.

**Floppy Bird and Tetris are precision.** Flappy reproduces the teacher on 99.8 percent of frames and dies at
9 pipes on the one missed correction (a second consecutive flap); Tetris drops pieces one move early. Forty
thousand on-policy frames contain a few hundred of those moments, so round 1 barely moves them. Round 2
triples Tetris and lifts Flappy to 0.16.

**2048 is a reading problem** (tile digits at 448 px): 0.48 agreement either way, and the model knows it, its
confidence there is 0.25.

**Anchor game.** An eight-game mix with Snake and Racer held out reproduced a failure mode with two seeds:
five games never left the letter prior and the other three converged to "the most common move for this option
list", with the image pathway unused. With Snake back in the mix (Racer and Pacman held out instead) every
game reads the frame by step 1500, as in the ten-game run. Snake's labels are the ones with no text shortcut
(a BFS arrow, uniform over the four moves, decided by the board alone), and one such game in the mix is what
forces the frame to be read.
</details>

<details>
<summary>👁️ &nbsp;Does it read the move descriptions?</summary>

On 400 validation frames per game, prompt otherwise unchanged (`scripts/probe_options.py`):

- shuffling the moves changes nothing (training permutes them every sample);
- replacing the names by neutral words (alpha, bravo, ...) and keeping the descriptions loses 0 to 2 points
  in six games, 7 in Tetris, 8 in Racer, 13 in Invaders;
- keeping only the names loses nothing, even in Snake, 2048, Pacman and Sokoban, whose names are the same
  four words: which "up" it is comes from the frame;
- rotating the descriptions one move along while the names stay: the decision follows the description in
  Flappy (84 percent), Sokoban (67) and Racer (47), the name in Breakout (95), Mario (82) and Pacman (80),
  and splits in Snake and Tetris. Both halves are read, and a fixed classification head cannot be moved by
  editing a sentence;
- an extra fake move ("hold: keep the current move and do nothing new") gets 2 to 9 percent of the mass in
  seven games and 13 to 18 in the three the model is least sure about (2048, Invaders, Breakout).

The same probes on a model that never learned to look (the eight-game run above) come out the opposite way:
when names and descriptions conflict it follows the name 100 percent of the time in every game, neutral names
collapse it to chance in Flappy and Breakout, and its agreement is at chance wherever a name alone does not
give the move away.
</details>

<details>
<summary>⚙️ &nbsp;The execution rule and the real-time setting</summary>

**Execution rule.** A move that leaves the observation unchanged (a blocked direction in 2048 or Sokoban is a
legal no-op) is not repeated on that observation; the next most probable move is taken. Without it, a
deterministic policy that picks a blocked direction loops to the step cap, and 2048 scores 54. The rule never
fires in the games whose frames change every step; their numbers are identical with and without it.

**Latency.** Applying the same model one step late (the decision from frame k acts at step k+1, which is the
real-time setting at 83 to 100 ms per step) collapses the reflex games: Snake 11, Tetris 248, Breakout 394,
Flappy 0.2, Pacman 502, while Invaders 400 and Racer 5885 barely move. Training on shifted labels
(`--label-delay 1`) helps exactly the games whose next decision follows from the current frame: Snake 11 to
20, Tetris 248 to 306, Breakout 394 to 636, Flappy 0.2 to 2.3, 2048 2191 to 3195. It hurts the ones whose
next decision depends on what the current move does to the board: Mario 820 to 242 (standing still to the cap
in half the episodes), Pacman 502 to 236, Racer 5885 to 5116, Sokoban at random level either way. So the
shift has to be per game, and even where it helps the real-time score stays a fraction of the undelayed one.
Latency has to be attacked in the model too, which is what the second frame is for.
</details>

<details>
<summary>🎚️ &nbsp;Is the confidence worth anything?</summary>

Low-confidence steps are where the errors are. On the model's own play, Tetris agrees with the teacher on 36
percent of the steps below confidence 0.5 and 61 percent overall; Invaders 41 percent against 97 above 0.9;
Racer 34 against 97; Sokoban 3 against 94. The operational test is a System Two behind the model: whenever
confidence falls below a threshold the decision is handed to the teacher (`playjev.play --handover`), and the
control hands the same share of steps over at random (`--handover-random`). Cloning model, 16 episodes per
setting:

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

Handing over a third of Breakout's steps by confidence reaches the teacher's level, and the same share at
random gets a third of the way. The confidence picks the steps that matter, which is what a System One is
for: it knows which decisions to keep and which to pass on. Sokoban is the exception (34 percent handed over,
58 to 71): its failures are boards the model has already deadlocked, and no later decision repairs them.
</details>

## 🏋️ Training Recipe

1. **Teachers.** One program per game plays on the game's internal state: BFS for Snake, expectimax for 2048,
   Dellacherie placement search for Tetris, A* with deadlock pruning for Sokoban, exact physics search for
   Floppy Bird, ghost-occupancy propagation for Pacman, ball-flight simulation for Breakout, a dodge-and-aim
   DP for Space Invaders, lookahead steering for the Racer, physics rollouts for Mario. Each returns a soft
   target: 0.9 on the best move (split on ties), 0.1 over acceptable moves, 0 on moves that lose.
2. **Collection.** Teachers play with 2 to 30 percent random moves so the data covers recoveries, and the
   label is always the teacher's own judgement of the frame. 100k frames per game, 448 px JPEGs, three shards
   per game collected in parallel on the CPU cores next to the GPU.
3. **Fine-tuning.** Full fine-tuning of the 0.8B model against the teacher distribution, one epoch over the
   ten games mixed, batch 64, learning rate 2e-5, fp32 master weights with bf16 autocast.
4. **Closed loop.** The trained model plays 16 held-out episodes per game through the same harness, against
   random play and the teacher on the same seeds. Validation also reports agreement, calibration error, and
   per-position bias.

Curves and every number: [docs/TRAIN_NOTES.md](docs/TRAIN_NOTES.md), [docs/MODEL_NOTES.md](docs/MODEL_NOTES.md),
[docs/BASELINES.md](docs/BASELINES.md).

<details>
<summary>💻 &nbsp;Run it yourself</summary>

Python 3.12, Playwright's Chromium for the games, and a CUDA GPU (about 3 GB for inference, 17 GB for
training at batch 64).

```bash
git clone https://github.com/OmniJev/PlayJev && cd PlayJev
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && playwright install chromium

python -m playjev.bench snake --pages 8 --steps 200        # the harness: random play, env-steps per second
python -m playjev.teacher_eval snake --episodes 16         # the teacher's score on held-out seeds
python -m playjev.collect snake --steps 4000 --shard s0    # (frame, teacher target) pairs under data/snake/s0
python -m playjev.train_sft --games snake --model Qwen/Qwen3.5-0.8B-Base --out ckpt/snake1
python -m playjev.play snake --policy local --ckpt ckpt/snake1/final --record runs/replays
```

`--policy teacher` and `--policy random` give the two reference rows, `--delay 1` applies each decision one
step late. The PBS job scripts we used are under `hpc/`.

Serving: `python -m playjev.serve --ckpt ckpt/snake1/final --port 18732` exposes the checkpoint at
`/v1/systemone` in the OpenJev request shape with frames as the state.
`playjev.play --policy server --url http://127.0.0.1:18732/v1/systemone` plays through it, and the demo page
switches every tile to that server with `?server=http://127.0.0.1:18732`.

The demo itself is in `demo/`, built by `scripts/build_demo.py` from the games and the recorded runs. Serve
that directory and open `index.html` to run it locally.
</details>

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
docs/              HARNESS.md (the hook contract), DESIGN.md, BASELINES.md, MODEL_NOTES.md, TRAIN_NOTES.md
hpc/               PBS job scripts for collection, training and closed-loop play
```

## 🔗 Related

[OpenJev](https://github.com/OmniJev/openJev), the text-state System One server this model plugs into, and
[Awesome-JEV](https://github.com/OmniJev/awesome-jev), the reading list behind System One models.

## ⚖️ Licence

Code and trained weights: Apache-2.0. The games keep their own licences, one file per game under
`games/<id>/`, and the note under the roster says which of them cover the code only.
