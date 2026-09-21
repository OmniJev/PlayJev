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

<p align="center">
  <img src="assets/board.webp" alt="Ten browser games, each one being played by PlayJev, with the score it had reached" width="100%">
</p>

<h1 align="center">PlayJev: A Multimodal JEV-Like Model for Small Games</h1>

<p align="center">
  <a href="https://omnijev.github.io/PlayJev/#gallery"><img alt="gallery" src="https://img.shields.io/badge/gallery-play_it-eda100?style=flat-square&labelColor=16181c"></a>
  <a href="https://github.com/OmniJev/PlayJev"><img alt="code" src="https://img.shields.io/badge/code-OmniJev%2FPlayJev-2a78d6?style=flat-square&logo=github&logoColor=white&labelColor=16181c"></a>
  <img alt="0.8B params" src="https://img.shields.io/badge/params-0.8B-8a63d2?style=flat-square&labelColor=16181c">
  <img alt="43 ms per move" src="https://img.shields.io/badge/per_move-43_ms-eb6834?style=flat-square&labelColor=16181c">
  <img alt="0.57 vs teacher" src="https://img.shields.io/badge/vs_teacher-0.57-1baf7a?style=flat-square&labelColor=16181c">
  <img alt="Apache 2.0" src="https://img.shields.io/badge/licence-Apache_2.0-6d747e?style=flat-square&labelColor=16181c">
</p>

## 🎮 Gallery

One model, one prompt, ten games. Every tile opens that game on the demo with the model playing; the number
under each is its score against the teacher (1.00 = matches the teacher, 0 = random play).

<table align="center">
  <tr>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=invaders"><img src="assets/thumbs/invaders.png" width="130"><br><b>Space Invaders</b></a><br>👾 &nbsp;1.00</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=racer"><img src="assets/thumbs/racer.png" width="130"><br><b>Racer</b></a><br>🏎️ &nbsp;1.00</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=sokoban"><img src="assets/thumbs/sokoban.png" width="130"><br><b>Sokoban</b></a><br>📦 &nbsp;1.00</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=snake"><img src="assets/thumbs/snake.png" width="130"><br><b>Snake</b></a><br>🐍 &nbsp;0.90</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=pacman"><img src="assets/thumbs/pacman.png" width="130"><br><b>Pacman</b></a><br>👻 &nbsp;0.56</td>
  </tr>
  <tr>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=tetris"><img src="assets/thumbs/tetris.png" width="130"><br><b>Tetris</b></a><br>🧱 &nbsp;0.37</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=mario"><img src="assets/thumbs/mario.png" width="130"><br><b>Infinite Mario</b></a><br>🍄 &nbsp;0.32</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=2048"><img src="assets/thumbs/2048.png" width="130"><br><b>2048</b></a><br>🔢 &nbsp;0.21</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=flappy"><img src="assets/thumbs/flappy.png" width="130"><br><b>Floppy Bird</b></a><br>🐦 &nbsp;0.18</td>
    <td align="center"><a href="https://omnijev.github.io/PlayJev/?game=breakout"><img src="assets/thumbs/breakout.png" width="130"><br><b>Breakout</b></a><br>🧊 &nbsp;0.14</td>
  </tr>
</table>

## 🏆 Scores

16 held-out episodes per game, argmax move, episodes capped at 1500 steps. Random play and the teacher run the
same seeds through the same harness. **vs teacher** is (model - random) / (teacher - random), so 0 is random
play and 1.00 is the teacher. In the chart the light bars are behaviour cloning and the first two DAgger rounds,
the darkest bar is this release, version 3.

<p align="center">
  <img src="assets/chart.png" alt="Score against the teacher for each game after cloning and after each of the three DAgger rounds" width="100%">
</p>

| Game | Random | PlayJev | Teacher | vs teacher |
|---|---:|---:|---:|---:|
| 👾 Space Invaders | 215 | **400** | 400 | 1.00 |
| 🏎️ Racer | 238 | **6709** | 6712 | 1.00 |
| 📦 Sokoban | 6.6 | **102.1** | 102.2 | 1.00 |
| 🐍 Snake | 1.0 | **102.9** | 114 | 0.90 |
| 👻 Pacman | 113 | **3965** | 7026 | 0.56 |
| 🧱 Tetris | 162 | **5792** | 15288 | 0.37 |
| 🍄 Infinite Mario | 613 | **1757** | 4229 | 0.32 |
| 🔢 2048 | 1021 | **4868** | 19593 | 0.21 |
| 🐦 Floppy Bird | 0.0 | **15.5** | 84.0 | 0.18 |
| 🧊 Breakout | 496 | **2785** | 16547 | 0.14 |
| **mean** | | | | **0.57** |

## 🧭 General Ability

Two held-out sets the training never touched, 200 questions each, asked under the game contract: the picture
or the passage is the state, the question is the instruction, the answers are the options. Each question is
asked twice, with the options in both orders. Version 2, the previous release, trained on games alone and
answered both sets near chance. This release keeps a fifth of every epoch on general image questions and a
fifth on text, and comes out above the base model on MMBench and level with it on MMLU.

<p align="center">
  <img src="assets/general.png" alt="Accuracy on MMBench dev and the MMLU test slice for version 2, version 3 and the base model" width="100%">
</p>

| Model | MMBench dev | MMLU test |
|---|---:|---:|
| Qwen3.5-0.8B-Base | 0.66 | 0.33 |
| Version 2 | 0.37 | 0.22 |
| **Version 3 (this release)** | **0.74** | **0.32** |
| chance | 0.40 | 0.25 |

## 🧠 How It Decides

The model never sees a game's name. It sees the current frame and the list of moves, and it answers with one
of them.

<p align="center">
  <img src="assets/decision-flow.png" alt="A game frame and an option list go into one forward pass, which returns a probability for every move" width="100%">
</p>

The input is the current frame and the list of moves, rendered with the frozen OpenJev prompt. The answer is
the single token after `Answer:`, and the probability over moves is the softmax over the option letters at
that position: one forward pass, no sampling, nothing generated. Moves are shuffled in every training sample,
so position carries no information. One frame per decision: the vision tower's temporal patch of 2 can carry a
previous frame at no extra token cost, and these weights were trained and are served with a single frame.

<p align="center">
  <img src="assets/decision.png" alt="The demo showing one Mario frame, the probability the model puts on each of the seven moves, and its confidence through the episode" width="100%">
</p>

The demo's single game view is the whole model in one picture: the frame on the left is the only input, the
bars are what the forward pass returns, the line below them is how sure it was at every step so far.

```python
from playjev.model import PlayJevModel      # github.com/OmniJev/PlayJev

model = PlayJevModel("OmniJev/PlayJev-0.8B").load()
d = model.decide([frame], options)[0]       # one forward pass, a probability per move
d.choice, d.probs, d.confidence             # the move it takes, the distribution, how sure it is
```

The prompt, the readout and the option format are built in
[playjev/model.py](https://github.com/OmniJev/PlayJev/blob/main/playjev/model.py), and the demo prints the exact
prompt for every game.

## 🏋️ Training

<table align="center">
  <tr>
    <td align="center" width="33%">🤖<br><b>Ten teachers</b><br>one program per game plays on the internal state<br>and labels every frame with a soft target</td>
    <td align="center" width="33%">📸<br><b>2.2M frames</b><br>full fine-tuning of the 0.8B base,<br>a fifth of each epoch on general image and text questions</td>
    <td align="center" width="33%">🔁<br><b>Three DAgger rounds</b><br>the model plays 40k frames per game,<br>the teachers relabel what it visited</td>
  </tr>
</table>

One program per game (BFS, expectimax, placement search, A* with deadlock pruning, physics search) plays on
the game's internal state and labels frames with a soft target. The model clones 863k such frames, then plays
two DAgger rounds of 40k frames per game with the teachers relabelling what it visited. Version 3 starts over
from the base model on all 1.8M frames those rounds produced, with 60 percent of the batches on game frames,
20 percent on general image questions (A-OKVQA, ScienceQA) and 20 percent on text questions (MMLU auxiliary
train, SciQ, ARC), then plays a third round with the same mix. Full fine-tuning, batch 64, learning rate 2e-5
from the base and 1e-5 for a round, fp32 master weights with bf16 autocast. The trainer is
[playjev/train_sft.py](https://github.com/OmniJev/PlayJev/blob/main/playjev/train_sft.py) and the whole recipe
is [scripts/reproduce.sh](https://github.com/OmniJev/PlayJev/blob/main/scripts/reproduce.sh).

## 🔭 Open Problems

- 🏃 **Motion** (Breakout, Mario): one still frame carries no velocity, and these weights get only that one frame.
- 🎯 **Single-step precision** (Floppy Bird, Tetris): the move that decides the episode is a few hundred frames out of 40k.
- 🔍 **Tile digits at 448 px** (2048): relabelling cannot help where the digits are unreadable, resolution can.
- ⏱️ **Latency**: at 83 to 100 ms per step the decision lands one step late, and training on labels shifted one step buys part of it back.

## 📄 Licence

Apache-2.0, same as the base model. The ten games are other people's work and stay in the code repository
under their own licences.
