# PlayJev design (v0, 2026-09-18)

PlayJev is an open System One model that plays ten classic browser games from raw pixels. Each
decision is one forward pass: a game frame goes in, a calibrated probability over the game's
actions comes out, no text is generated. Same output contract as TypeSafe's Jev Choice primitive,
same confidence formula, same `/v1/systemone` server as OpenJev, so the model is a drop-in.
Repo: git@github.com:OmniJev/PlayJev.git (exists, empty). Training on hopper `smallx` (H200).

## 1. Why this is new

Every Jev-shaped game demo so far dodges pixels:

- TypeSafe's own Mario demo (`fhshaik/typesafe-mario`): "NES emulator -> telemetry/RAM parser ->
  structured JSON -> Jev Choice -> controller input", and verbatim "The model does not receive
  screenshots." No scores reported.
- `daseinlabs/open-jev` Doom demo: "describes each frame in a line of text, and lets the server
  rank the action menu" (Gemma 3 4B zero-shot, about 90 ms per request).
- The Gemma open-jev clone "played Doom zero-shot but stood still in Mario, and it dropped the Mario
  demo rather than task-fine-tune for it".
- `AlexWortega/openjev` plays Doom "straight from the pixels through the Qwen3.5 vision tower", but
  zero-shot, one game, no training.
- VideoGameBench (arXiv 2505.18134) and lmgame-Bench (arXiv 2505.15146) run frontier VLMs with
  reasoning loops at seconds per move and report widespread failure.

Nobody has trained a small VLM under the Jev contract to play from pixels. That is the whole
project: one 0.8B model, ten games, real-time, calibrated, and a browser demo where the game and
the probability bars run side by side.

## 2. Model

Primary: `Qwen/Qwen3.5-0.8B-Base`. Verified from HF `config.json`:
`Qwen3_5ForConditionalGeneration`, `image-text-to-text`, 0.87B parameters in total. Text stack
24 layers, hidden 1024, hybrid 3 linear-attention + 1 full-attention, vocab 248,320. Vision tower
12 layers, hidden 768, patch 16, spatial merge 2, temporal patch 2, projecting to 1024.
A 448x448 frame is 784 patches and 196 visual tokens after the merge; 224x224 gives 49.
Temporal patch 2 means a two-frame stack (motion) is native, no hack needed.

Ablations: `Qwen3.5-2B-Base`, `Qwen3.5-4B` (OpenJev's primary, engine already exists),
`SmolVLM-256M-Instruct` as the "how small can it go" point.

## 3. Input and output contract

- State: the current frame (JPEG from the canvas), optionally the previous frame as a second image.
- Question: Choice over the game's actions, rendered with the frozen OpenJev prompt
  (`openjev-letters-v1`): lettered `name: description` options, answer is one uppercase letter.
- Readout: float32 softmax over the option-letter logits at the answer position. Confidence
  `(p_max - 1/K) / (1 - 1/K)` exactly as Jev.
- The model never sees the game's name in the prompt, only the frame and the option list
  (no leaking of which game or which rule to apply).

### 3.1 Image state on the wire

The TypeSafe request shape is kept; the state carries frames instead of text:
`{"state": {"frames": ["data:image/jpeg;base64,..."]}, "questions": {"q": {"type": "choice",
"instructions": "Which move should the player make next?", "criteria": {"up": "turn the snake to move up", ...}}}}`.
One frame or two (previous, current). The response is a normal Choice answer with probabilities per option name
and the Jev confidence. `playjev/play.py` ServerPolicy is the reference client, the demo page uses the same call.

## 4. Environment harness (the same code trains and demos)

Each game gets a small hook file `pj_hook.js` injected after its own scripts:

```
window.pj = {
  reset(seed)         // deterministic restart, seeded RNG (Math.random patched)
  step(action, k)     // apply action for k fixed ticks (game loop driven manually, no rAF)
  frame()             // canvas.toDataURL('image/jpeg', 0.8) or offscreen canvas of the DOM board
  score(), done()     // closed, numeric
  actions()           // ordered list of {name, description}
}
```

Driver: Python + Playwright (chromium headless shell), N pages per process, one process per
CPU core. Measured on the hopper login node: 33 ms per `page.screenshot`; canvas `toDataURL`
is the faster path and is what `frame()` uses. Target: at least 1,000 aggregate env steps per
second on one node (12 cores per GPU slot), enough for on-policy RL with a 0.8B model.
The public demo loads the same page in a real browser and talks to the OpenJev server.

## 5. Training recipe (our reading of RLCD)

Stage A, teachers. One cheap teacher per game, running on the game's internal state (never seen
by the model): Snake BFS-to-food with tail safety; Tetris Dellacherie heuristic; 2048 expectimax
depth 2; Sokoban BFS solver on small levels; Pac-Man BFS to nearest pellet with ghost avoidance;
Breakout paddle tracks predicted ball x; Flappy gap-centre rule; Invaders nearest-column rule;
Racer lane-centre plus brake on curves; Mario heuristic (run right, jump when a gap or enemy is
within reach) with a PPO agent as fallback if the heuristic clears fewer than half the levels.
Teachers give a best action and, where available, a preference over all actions.

Stage B, supervised. Cross-entropy on the letter token against the teacher action on frames from
teacher rollouts, then DAgger rounds (student plays, teacher relabels). One model for all ten
games, random option order per sample.

Stage C, RL for calibrated decisions. Policy = the softmax over letters. GRPO-style policy
gradient on episode return, plus a Brier term against the realised outcome of the chosen action
(teacher agreement, or score gain over a short horizon) so probabilities stay calibrated instead
of collapsing to argmax. Per-game temperature fitted on held-out seeds.

Metrics (all closed): game score and episode length per game; teacher-agreement accuracy;
ECE and Brier of P(chosen action) against agreement; decisions per second. Baselines: random,
zero-shot Qwen3.5-0.8B/2B/4B with the same prompt, teacher ceiling.

## 6. Compute plan

hopper `smallx` (1 to 2 H200, walltime up to 144 h, three jobs per user). Envs on the 12 CPU
cores of the slot, model on the GPU. Weights and data under the project scratch, HF
downloads on the login node with `HF_HUB_DISABLE_XET=1`. Chromium headless shell already
installed at `$PLAYWRIGHT_BROWSERS_PATH` (Playwright 1.63, works on the login node;
verify once inside a compute job).

## 7. Demo

GitHub Pages on the PlayJev repo. Ten game tiles, each with the live canvas and a bar per
action showing the model's probabilities and the Jev confidence. Two serving modes:
live (OpenJev server on a GPU) and replay (recorded rollouts with per-step probabilities
baked in, so the page works with no backend). In-browser WebGPU inference for the 0.8B model
is a stretch goal.

## 8. Open risks

- Mario sprites are Nintendo's; FullScreenMario was DMCA'd in 2016. Decide before publishing
  whether to swap in free art.
- Chromium inside PBS compute nodes: same image as the login node, but unverified.
- Snake result from `RINNECODER/jev-behavior-study` says prompt framing swings Jev from 1/16 to
  14/16 on text state. Our option descriptions are part of the frozen prompt and must not change
  between training and demo.
