# PlayJev model notes (2026-09-18, smoke test on hopper)

What was measured when Qwen3.5-0.8B-Base first read game frames and emitted letter probabilities on a hopper
H200. Code: `playjev/model.py` (PlayJevModel), `playjev/zeroshot.py` (the probe), `hpc/env_setup.sh` (the
environment), `hpc/smoke.pbs` (the job). Raw JSON for every table below is in `runs/smoke/<jobid>/*.json` on
hopper (`$WORK/repo/runs/smoke/`).

## 1. Environment on hopper

`bash hpc/env_setup.sh` on the login node builds `$WORK/.venv` (Python 3.12, uv). The UV cache
had to move to `$SCRATCH/.cache/uv`: `~/.bashrc` points it at `/tmp`, which has 1.4 GB free, and the cu128
stack downloads about 4 GB of wheels.

| package | version |
|---|---|
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 (the Qwen image and video processors refuse to load without it) |
| transformers | 5.17.0 |
| accelerate / peft | 1.15.0 / 0.21.0 |
| flash-linear-attention | 0.5.2, with triton 3.7.1 (torch 2.10 pins 3.6.0, but fla refuses its gated chunk backward on Hopper under triton 3.4 to 3.7.0, wrong results per fla issue #640; the forward path, i.e. every number in this file, was measured under 3.6.0 and is unaffected) |
| tokenizers / safetensors / numpy | 0.23.2 / 0.8.0 / 2.5.3 |
| pillow / playwright / einops | latest / 1.63.0 (matches chromium build 1243 on scratch) / latest |
| causal-conv1d | not installed: no prebuilt wheel for torch 2.10 and no nvcc on the login node; the conv runs on the torch fallback |

Weights (HF cache `$PROJ/hf_home/hub`, all complete):
`Qwen3.5-0.8B-Base` snapshot `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` (1.7 GB), `Qwen3.5-2B-Base` (4.3 GB),
`Qwen3.5-0.8B` instruct snapshot `2fc06364715b967f1860aea9cf38778875588b17`.

Compute node: see section 3 (driver, GPU, versions as printed inside the job).

## 2. Prompt and readout

The Base repo has no `chat_template.jinja` file but its `tokenizer_config.json` embeds the Qwen3.5 chat template,
so both a plain-text prompt and a chat-template prompt were tested. The game's name appears nowhere; the model sees
the frame and the option list.

**Plain template** (`template="plain"`, the default for Base checkpoints). Exact string handed to the processor
for the 4 snake options; the processor expands `<|image_pad|>` into one token per merged 32x32 pixel block:

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

The answer slots are the space-prefixed letter tokens ` A`, ` B`, ` C`, ` D` (each a single token, verified to
round-trip), the standard base-model multiple-choice readout after `Answer:`.

**Chat template** (`template="chat"`): system = the same first sentence, user = the `<state>` block plus the
question, `add_generation_prompt=True, enable_thinking=False`, so the prompt ends with
`<|im_start|>assistant\n<think>\n\n</think>\n\n` and the answer slots are the bare tokens `A`, `B`, `C`, `D`.
`build_prompt` raises if the rendered prompt does not end with the closed think block.

```
<|im_start|>system
Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.<|im_end|>
<|im_start|>user
<state>
<|vision_start|><|image_pad|><|vision_end|>
</state>

Question: Which move should the player make next?

Options:
A. up: turn the snake to move up
...
Answer with one letter: A, B, C, D.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

**Readout.** The inner `Qwen3_5Model` runs once (no cache, left padding so the last position is the answer slot
for every row); the last hidden state is cast to float32 and multiplied by a float32 copy of the tied embedding
matrix; the probability vector is the softmax over the K slot logits; confidence is `(p_max - 1/K) / (1 - 1/K)`.
Two diagnostics come with every decision: `allowed_mass`, the share of the full-vocabulary softmax on the K slots
(is the model about to emit a letter at all?), and `top_token`, the argmax token over the whole vocabulary.

**Tokens.** The snake frames are 448x429 JPEGs; the processor snaps them to 448x416 (multiples of 32), which is
182 visual tokens, 296 input tokens in total with the plain prompt (309 with the chat prompt). A 448x448 frame is
196 visual tokens. A 224x224 frame would be 49 by patch count, but the processor's 65,536-pixel minimum upsamples it
to 256x256 = 64 tokens (the 448x429 snake frame at long side 224 becomes 288x256 = 72; see 3.2).

**Two frames per state.** `frames_per_state=2, stack="temporal"` puts the previous frame in temporal slot 0 and the
current frame in slot 1 of the vision tower's two-frame patch (the layout the video path uses: each patch vector is
`(channel, temporal, 16, 16)`, and the image processor fills both temporal slots with the same frame). Same token
count as one frame, one placeholder in the prompt, no timestamp text. The transformers video path was avoided on
purpose: it pads to at least 4 frames, upsamples anything under 131k pixels (a 224 px frame would be enlarged) and
injects `<t seconds>` strings. `stack="separate"` sends two images and two placeholders (2x visual tokens).

## 3. Measured on the compute node (job 621552, hopper-15, 05:49 to 05:56)

Node: NVIDIA H200 (143,771 MiB), driver 575.57.08, CUDA 12.9 reported by nvidia-smi, compute capability 9.0; the
cu128 wheels run as is. Inside the job: torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.17.0,
fla 0.5.2 with triton 3.6.0 (`fla` importable, so the gated-delta-rule chunk kernels are the triton ones),
causal_conv1d absent (the conv runs the reference torch path; transformers prints one warning per process).

Load times (`from_pretrained` to device, processor included, warm page cache): 0.8B-Base 6.0 s (later processes
3.7 to 9.7 s), 2B-Base 10.0 s, 0.8B instruct 4.3 s. The very first forward on the node took about two minutes
(triton compiling the fla kernels; the cache in `~/.triton` made every later process start within seconds).

### 3.1 Zero-shot behaviour on the 9 snake frames (`runs/bench/snake/*.jpg`)

Options in the frozen order up / down / left / right, prompt as in section 2. "Mass" is the share of the whole
vocabulary softmax on the four letter slots. "Frame L1" is the mean L1 distance between a frame's probability
vector and the mean over frames (0 = the frame changes nothing). "Grey L1" is the L1 between the probabilities on a
flat grey frame and each real frame. The shift test re-runs the frames with the option order rotated by one
(down, left, right, up) and compares the mean probability profile per letter position and per option name across
the two orders: a small letter L1 means the mass stays on the same letter (position bias), a small name L1 means
the mass follows the option (the model reads the options).

| model | prompt | mean p(up, down, left, right) | argmax | conf | mass | frame L1 | grey L1 | shift L1 letter / name |
|---|---|---|---|---|---|---|---|---|
| 0.8B-Base | plain, 1 frame | 0.68, 0.15, 0.05, 0.11 | up 9/9 | 0.57 | 0.964 | 0.024 | 0.078 | 0.12 / 1.27 |
| 0.8B-Base | chat, 1 frame | 0.71, 0.13, 0.03, 0.13 | up 9/9 | 0.62 | 0.982 | 0.033 | 0.148 | 0.20 / 1.43 |
| 0.8B-Base | plain, 2 frames temporal | 0.68, 0.15, 0.06, 0.11 | up 8/8 | 0.58 | 0.981 | 0.032 | 0.077 | 0.08 / 1.23 |
| 0.8B-Base | plain, 2 frames separate | 0.65, 0.16, 0.07, 0.12 | up 8/8 | 0.53 | 0.973 | 0.033 | 0.156 | 0.14 / 1.16 |
| 2B-Base | plain, 1 frame | 0.29, 0.22, 0.27, 0.23 | up 9/9 | 0.05 | 0.997 | 0.043 | 0.062 | 0.07 / 0.29 |
| 2B-Base | chat, 1 frame | 0.60, 0.14, 0.13, 0.13 | up 9/9 | 0.47 | 0.905 | 0.106 | 0.202 | 0.20 / 1.07 |
| 0.8B instruct | chat, thinking off | 0.28, 0.19, 0.21, 0.33 | right 9/9 | 0.10 | 0.919 | 0.065 | 0.127 | 0.14 / 0.42 |

What this says:

- Every checkpoint answers with a letter at the slot: 0.90 to 0.997 of the vocabulary mass sits on the four letters,
  and the top-1 token over the whole vocabulary is a letter on every frame. The readout is measuring the right thing.
- The zero-shot answer is a letter prior, not a decision. 0.8B-Base puts about 0.7 on A whatever the frame; when the
  options are rotated, the mass stays on A (letter L1 0.08 to 0.20) and the per-option profile turns over (name L1
  1.2 to 1.4). 2B-Base plain is near uniform (confidence 0.05). The instruct 0.8B prefers D instead of A.
- The frame is seen but barely matters: real frames differ from each other by 0.02 to 0.04 L1, and a flat grey frame
  moves the vector by 0.06 to 0.20. The 2B chat variant reacts most (frame L1 0.106, grey L1 up to 0.475), still with
  the mass on A.
- Two frames per state (either stacking) change nothing zero-shot; the temporal stack costs the same tokens as one frame.

So the zero-shot baseline for the paper is "the letter prior": accuracy at chance, confidence 0.05 to 0.6 driven by
position bias. Everything above that has to come from training. Per-frame tables are in the JSON files.

### 3.2 Latency and memory (H200, bf16, eager PyTorch, batch = frames per forward)

Columns: visual and total input tokens per sample; `prep` = JPEG decode + processor on the CPU (one thread);
`fwd` = GPU forward including the float32 readout, synchronised; ms per decision and decisions/s count both;
peak = `torch.cuda.max_memory_allocated`. Median of 5 runs after 2 warm-ups.

Qwen3.5-0.8B-Base, plain prompt, one frame:

| long side | batch | visual tok | input tok | prep ms | fwd ms | ms / decision | decisions / s | peak GB |
|---|---|---|---|---|---|---|---|---|
| 448 | 1 | 182 | 296 | 4.8 | 38.1 | 42.9 | 23 | 2.60 |
| 448 | 8 | 182 | 296 | 26.3 | 51.2 | 9.7 | 103 | 2.96 |
| 448 | 32 | 182 | 296 | 108.9 | 136.1 | 7.7 | 131 | 3.43 |
| 224 | 1 | 72 | 186 | 3.8 | 38.0 | 41.8 | 24 | 2.58 |
| 224 | 8 | 72 | 186 | 20.1 | 45.6 | 8.2 | 122 | 2.69 |
| 224 | 32 | 72 | 186 | 95.4 | 89.3 | 5.8 | 173 | 3.10 |

Qwen3.5-0.8B-Base, plain prompt, two frames per state: temporal stack 448 px, batch 1 / 8 / 32: fwd 37.5 / 49.9 /
135.6 ms, 44.0 / 10.2 / 8.5 ms per decision, peak 2.60 / 2.79 / 3.43 GB (same tokens as one frame, the extra prep is
the second JPEG). Separate images, batch 8: 364 visual tokens, 481 input tokens, prep 48.9 ms, fwd 64.6 ms,
14.2 ms per decision, 3.00 GB.

Qwen3.5-2B-Base, plain prompt, one frame:

| long side | batch | visual tok | input tok | prep ms | fwd ms | ms / decision | decisions / s | peak GB |
|---|---|---|---|---|---|---|---|---|
| 448 | 1 | 182 | 296 | 4.7 | 42.2 | 46.9 | 21 | 6.08 |
| 448 | 8 | 182 | 296 | 24.1 | 62.6 | 10.8 | 92 | 6.32 |
| 448 | 32 | 182 | 296 | 103.3 | 209.6 | 9.8 | 102 | 7.15 |
| 224 | 1 | 72 | 186 | 3.8 | 42.3 | 46.1 | 22 | 6.07 |
| 224 | 8 | 72 | 186 | 20.2 | 52.7 | 9.1 | 110 | 6.18 |
| 224 | 32 | 72 | 186 | 91.8 | 133.1 | 7.0 | 142 | 6.56 |

Qwen3.5-0.8B instruct, chat prompt, 448 px, batch 1 / 8 / 32: fwd 37.7 / 50.4 / 138.3 ms, 42.5 / 9.4 / 7.4 ms per
decision, 2.60 / 2.78 / 3.43 GB (same architecture as Base, same numbers).

Reading the table:

- Batch 1 is 38 ms of forward for 0.8B and 42 ms for 2B, the same for 72 and 182 visual tokens: this is kernel
  launch and Python overhead across 36 layers (24 text, 12 vision) plus the transformers M-RoPE index code, not
  compute. A single live decision therefore runs at about 23 per second in eager mode; CUDA graphs or
  `torch.compile` on the fixed per-game shapes are the lever if the demo needs more.
- Batching amortises it: 131 decisions/s at batch 32 for 448 px, 173 at 224 px. At batch 32 the CPU preprocessing
  (3.4 ms per frame, single thread) costs as much as the GPU forward, so an RL loop needs the processor in the env
  workers, not on the learner's thread. Job 621558 profiles the batch-32 forward and extends the sweep to batch 64
  and 128 (section 3.4).
- Memory is small: 2.6 to 3.4 GB for 0.8B, 6.1 to 7.2 GB for 2B at batch 32 inference. Training will be bounded by
  activations, not weights.
- "224 px" is 72 visual tokens rather than 49: the checkpoint's `preprocessor_config.json` sets
  `size.shortest_edge = 65536` (a minimum of 256x256 pixels), so a 224x214 frame is upsampled to 288x256 before
  patching. Passing `min_pixels`/`max_pixels` to the processor call can change that; the smoke test kept the
  checkpoint's defaults because they are what the model was trained with.

### 3.3 Chromium on the compute node

`python -m playjev.bench snake --pages 8 --steps 100` ran inside the job with the playjev venv (playwright 1.63.0)
and `PLAYWRIGHT_BROWSERS_PATH=$SCRATCH/ms-playwright`: 8 pages reset in 1.10 s, 345 env-steps/s over 8 pages,
77 random episodes ended in 100 steps (random snake dies within a few moves, so page reloads dominate this number),
no page errors, no missing shared libraries. The headless shell lives at
`chromium_headless_shell-1243/chrome-headless-shell-linux64/chrome-headless-shell` (the `ldd` line in smoke.pbs
pointed at the old `chrome-linux/headless_shell` path and printed nothing useful; the bench itself is the proof).
The `probe_games_venv` fallback was never needed.

### 3.4 Profile of the batch-32 forward (job 621558, hopper-15, 0.8B-Base, plain, 448 px)

Batch sweep first: batch 32 / 64 / 128 give fwd 135.0 / 261.4 / 513.7 ms, i.e. 7.45 / 7.11 / 7.40 ms per decision
(prep included) and 134 / 141 / 135 decisions/s at 3.43 / 4.30 / 6.03 GB peak. Throughput is flat from batch 32
on: the GPU time grows linearly with the batch, so this eager pipeline tops out at about 140 decisions/s per H200
for 448 px frames (and about 175 at 224 px).

`torch.profiler` on one batch-32 `decide()` (self CUDA total 125 ms, self CPU total 191 ms), grouped:

| where the device time goes | ms | share | note |
|---|---|---|---|
| `aten::copy_` (dtype/device casts, clones) | 38.6 | 31% | 14.2 ms of it is 8 pageable host-to-device copies: `pixel_values` arrives as float32, 784 x 1536 x 4 B = 4.8 MB per frame (the temporal patch duplicates the frame), 154 MB per batch of 32 |
| elementwise kernels + `aten::mul` + `silu` | 41 | 33% | gating, norms and activations of the 18 GatedDeltaNet layers, memory-bound |
| convolutions | 21.8 | 17% | 18 depthwise conv1d on the torch fallback (11.9 ms) plus the vision patch embed; the causal-conv1d kernel would remove most of it |
| matmuls (`mm`, `addmm`, `linear`) | 20.6 | 16% | the actual projections |
| fla `ChunkGatedDeltaRuleFunction` | 12.9 | 10% | 18 calls, 0.7 ms each, triton |
| flash attention | 8.6 | 7% | 384 calls of 22 us: the vision tower runs SDPA once per image per layer (12 x 32), launch-bound |

CPU side: 481 `cudaMemcpyAsync` calls (31 ms) and 3,832 kernel launches per forward; the M-RoPE index code in
transformers builds many small tensors per sample. `aten::_upsample_bicubic2d_aa` (12.8 ms) is the processor's
resize on the CPU, part of `prep`.

What that means for later (not done now): send `pixel_values` as bf16 from pinned memory (or do decode, resize and
normalise on the GPU and upload uint8 frames, 0.6 MB each), build causal-conv1d on a node with nvcc, and try
`torch.compile` for the elementwise chains and CUDA graphs for the launch floor. Each of those is worth tens of
percent; together they should roughly triple the decisions/s. For teacher-labelled SFT none of it matters, since
the teacher runs on `info()` and the model only trains.

## 4. What got in the way

- `AutoProcessor` for Qwen3.5 needs torchvision (both the fast image processor and `Qwen3VLVideoProcessor` refuse to
  import without it); torchvision 0.25.0+cu128 was added to the stack and to `env_setup.sh`.
- `~/.bashrc` puts the uv cache in `/tmp` on the login node, which had 1.4 GB free; the build script overrides
  `UV_CACHE_DIR` and `TMPDIR` to `$SCRATCH/...`.
- causal-conv1d has no prebuilt wheel for torch 2.10 and the login node has no nvcc; skipped, conv on the torch path.
- transformers 5.17 requires `mm_token_type_ids` alongside any multimodal input (it raises otherwise); the processor
  returns it by default and `PlayJevModel` passes it through.
- The sample frames are not all the same size (`cdp.jpg` is 460x440, the rest 448x429), so a batch mixes 196- and
  182-token samples; left padding handles it, and the temporal stack resizes the previous frame to the current one.
- Starting a background job over ssh with `nohup ... &` hangs the ssh call unless stdin is redirected from
  `/dev/null` (the process keeps the session's stdin open); harmless, but it looked like a stuck build.
