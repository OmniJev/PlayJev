# PlayJev

An open System One model that plays ten classic browser games from raw pixels. PlayJev is
Qwen3.5-0.8B-Base fine-tuned so that one forward pass turns a game frame into a calibrated
probability over the game's moves. Nothing is generated and nothing is parsed: the answer is read
from the option-letter logits, the same contract as Jev's Choice primitive and the same confidence
formula. A decision takes 43 ms on one H200 at batch 1, about twice as fast as the games' own step, and
the same GPU serves about 140 decisions per second when several games run at once.

```
frame (JPEG, 448 px) ─┐
                      ├─► one forward pass ─► letter logits ─► softmax ─► {up: 0.05, down: 0.02, left: 0.90, right: 0.03}, confidence 0.87
option list ──────────┘
```

Demo: <https://omnijev.github.io/PlayJev/>. Every tile is the real game replaying a recorded episode of the
trained model step by step, with the model's probabilities drawn beside the picture.

## Why pixels

Every Jev-shaped game demo before this one feeds the model text. TypeSafe's Mario demo parses the
emulator's RAM into JSON and states that the model does not receive screenshots; the open-jev Doom
demo describes each frame in one line of text; the zero-shot pixel attempts play one game without
training. VideoGameBench and lmgame-Bench run frontier models with reasoning loops at seconds per
move. PlayJev trains a small vision-language model under the Jev contract and plays ten games from
the frame alone, in real time, with a probability the game loop can trust.

## Results

<!-- RESULTS: filled from runs/play/*.json when the stage-2 closed loop lands -->

## The ten games

All plain HTML5/JS, vendored under `games/<id>/` with their licences and a `vendor.patch` where we changed a file.
Each game exposes the same tiny hook (`window.pj`: `start(seed)`, `step(action)`, `frame()`, `score()`, `done()`,
`actions`), and the same page serves as environment and demo.

| game | upstream | licence | moves |
|---|---|---|---|
| Infinite Mario | robertkleffner/mariohtml5 | Unlicense (code) | noop, left, right, jump, right jump, right run, right run jump |
| Snake | patorjk/JavaScript-Snake | MIT | up, down, left, right |
| Tetris | jakesgordon/javascript-tetris | MIT | left, right, rotate, drop, none |
| 2048 | gabrielecirulli/2048 | MIT | up, down, left, right |
| Floppy Bird | nebez/floppybird | Apache-2.0 | flap, wait |
| Pacman | daleharvey/pacman | WTFPL | up, down, left, right |
| Breakout | jakesgordon/javascript-breakout | MIT | left, right, stay |
| Space Invaders | StrykerKKD/SpaceInvaders | MIT | left, right, noop |
| Javascript Racer | jakesgordon/javascript-racer | MIT | left, right, faster, slower, left faster, right faster |
| Sokoban | taniarascia/sokoban, Microban levels by David Skinner | MIT | up, down, left, right |

## How a decision is made

The model never sees a game's name. It sees the current frame and the option list, rendered with the frozen
OpenJev prompt, and its answer is the next token after `Answer:`.

```
Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.

<state>
<|vision_start|><|image_pad|><|vision_end|>
</state>

Question: Which move should the player make next?

Options:
A. up: turn the snake to move up
B. down: turn the snake to move down
C. left: turn the snake to move left
D. right: turn the snake to move right

Answer with one letter: A, B, C, D.
Answer:
```

The frame goes in as visual tokens (182 for a 448x416 snake frame). The probability over moves is the float32
softmax over the K letter logits at the last position; confidence is `(p_max - 1/K) / (1 - 1/K)`, Jev's formula,
so 0 is a uniform guess and 1 is certainty. Options are shuffled for every training sample, so the model cannot
learn a letter prior. One frame per decision by default; the vision tower's two-frame temporal patch takes the
previous frame too, without extra tokens, for games where velocity matters.

## Training recipe

1. **Teachers.** One program per game plays on the game's internal state (grids, coordinates, velocities, which
   the model never sees): BFS for Snake, expectimax for 2048, Dellacherie placement search for Tetris, A* with
   deadlock pruning for Sokoban, exact physics search for Floppy Bird, ghost-occupancy propagation for Pacman,
   ball-flight simulation for Breakout, a dodge-and-aim DP for Space Invaders, lookahead steering for the Racer,
   and physics rollouts for Mario. Each returns a soft target: 0.9 on the best move (split on ties), 0.1 spread
   over acceptable moves, 0 on moves that lose. Teacher scores are in [docs/BASELINES.md](docs/BASELINES.md).
2. **Collection.** Teachers play with 2 to 30 percent random moves so the data covers recoveries, and the label is
   always the teacher's own judgement of the frame (DAgger-style). 100k frames per game, 448 px JPEGs, three
   shards per game collected in parallel on the CPU cores next to the GPU.
3. **Supervised fine-tuning.** Full fine-tuning of the 0.8B model, cross-entropy between the teacher distribution
   and the model's softmax over the letter slots, one epoch over the ten games mixed, batch 64, learning rate 2e-5,
   fp32 master weights with bf16 autocast. About 4 hours on one H200 for the ten-game model.
4. **Closed loop.** The trained model plays 16 held-out episodes per game through the same harness, argmax move,
   against a random policy and the teacher on the same seeds. Validation also reports agreement with the teacher,
   ECE and Brier of the chosen move's probability, and the per-letter position bias.

Details, curves and every number: [docs/TRAIN_NOTES.md](docs/TRAIN_NOTES.md) and
[docs/MODEL_NOTES.md](docs/MODEL_NOTES.md).

## Run it

Python 3.12, Playwright's Chromium for the games, and a CUDA GPU for training and play (the 0.8B model fits in
about 3 GB for inference and 17 GB for training with batch 64).

```bash
git clone https://github.com/OmniJev/PlayJev && cd PlayJev
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && playwright install chromium

python -m playjev.bench snake --pages 8 --steps 200         # the harness: random play, env-steps per second
python -m playjev.teacher_eval snake --episodes 16         # the teacher's score on held-out seeds
python -m playjev.collect snake --steps 4000 --shard s0    # (frame, teacher target) pairs under data/snake/s0
python -m playjev.train_sft --games snake --model Qwen/Qwen3.5-0.8B-Base --out ckpt/snake1
python -m playjev.play snake --policy local --ckpt ckpt/snake1/final --record runs/replays
```

`playjev.play --policy teacher` and `--policy random` give the two reference rows; `--delay 1` applies each
decision one step late, the real-time latency model. The HPC job scripts we used are under `hpc/`.

Serving: `python -m playjev.serve --ckpt ckpt/snake1/final --port 18732` exposes the checkpoint at
`/v1/systemone` in the OpenJev request shape with frames as the state
(`{"state": {"frames": ["data:image/jpeg;base64,..."]}, "questions": {"q": {"type": "choice", "criteria": {...}}}}`).
`playjev.play --policy server --url http://127.0.0.1:18732/v1/systemone` plays through it, and the demo page switches
every tile to that server with `?server=http://127.0.0.1:18732`.

## Repository

```
games/<id>/        vendored game, pj.json manifest, pj_hook.js, NOTES.md, TEACHER.md
games/_shared/     pj_shim.js: virtual clock, seeded Math.random, synthetic keys, frame grab
playjev/           env.py (Playwright driver), collect.py, teachers/, model.py, data.py, train_sft.py, play.py, serve.py
demo/              the GitHub Pages site; scripts/build_demo.py assembles it from games/ and runs/replays/
docs/              HARNESS.md (the hook contract), DESIGN.md, BASELINES.md, MODEL_NOTES.md, TRAIN_NOTES.md
hpc/               PBS job scripts for collection, training and closed-loop play
```

## Related

[OpenJev](https://github.com/OmniJev/openJev), the text-state System One server this model plugs into, and
[Awesome-JEV](https://github.com/OmniJev/awesome-jev), the reading list behind System One models.

## Licence

Code and trained weights: Apache-2.0. The games keep their own licences (see each `games/<id>/LICENSE`).
