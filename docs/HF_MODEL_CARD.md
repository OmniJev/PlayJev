---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - game-ai
  - multimodal
  - imitation-learning
  - dagger
  - browser-games
---

# PlayJev 0.8B

Qwen3.5-0.8B-Base fine-tuned to play ten classic browser games from raw pixels. One frame goes in, one forward
pass runs, one move comes out, 43 ms on an H200. The model never sees the game's state and is never told which
game it is playing.

- Code, harness and the ten games: [github.com/OmniJev/PlayJev](https://github.com/OmniJev/PlayJev)
- Live demo, every game playable in the browser: [omnijev.github.io/PlayJev](https://omnijev.github.io/PlayJev/)

## Scores

16 held-out episodes per game, argmax move, episodes capped at 1500 steps. Random play and the teacher run the
same seeds through the same harness. **vs teacher** is (model - random) / (teacher - random), so 0 is random
play and 1.00 is the teacher.

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

## How it decides

The input is the current frame and the list of moves, rendered with the frozen OpenJev prompt. The answer is
the single token after `Answer:`, and the probability over moves is the softmax over the option letters at
that position: one forward pass, no sampling, nothing generated. Moves are shuffled in every training sample,
so position carries no information. Where velocity matters the vision tower also takes the previous frame, at
no extra token cost.

```python
from playjev.model import PlayJevModel      # github.com/OmniJev/PlayJev

model = PlayJevModel("OmniJev/PlayJev-0.8B").load()
d = model.decide([frame], options)[0]       # one forward pass, a probability per move
d.choice, d.probs, d.confidence             # the move it takes, the distribution, how sure it is
```

The prompt, the readout and the option format are in
[docs/MODEL_NOTES.md](https://github.com/OmniJev/PlayJev/blob/main/docs/MODEL_NOTES.md).

## Training

One program per game (BFS, expectimax, placement search, A* with deadlock pruning, physics search) plays on
the game's internal state and labels frames with a soft target. The model trains on 863k such frames, then on
two DAgger rounds where it plays 40k frames per game and the teachers relabel what it visited. Full
fine-tuning, one epoch per round, batch 64, learning rate 2e-5, fp32 master weights with bf16 autocast.
Details in [docs/TRAIN_NOTES.md](https://github.com/OmniJev/PlayJev/blob/main/docs/TRAIN_NOTES.md).

## Limits

Motion from a single frame (Breakout, Mario), single-step precision (Floppy Bird, Tetris), and reading tile
digits at 448 px (2048) are open. One step of latency, which is what real time costs at 83 to 100 ms per step,
takes the reflex games apart.

## Licence

Apache-2.0, same as the base model. The ten games are other people's work and stay in the code repository
under their own licences.
