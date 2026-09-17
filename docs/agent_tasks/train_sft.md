# Task: SFT trainer for PlayJev (stage B of docs/DESIGN.md section 5)

Prerequisite: `playjev/model.py` (PlayJevModel with build_prompt and the letter readout) exists and the smoke test
in docs/MODEL_NOTES.md passed. Reuse its prompt builder; the training prompt and the inference prompt must be the
same string.

## Data collection (first stage of the job, on the compute node's 12 CPU cores)

Rollouts run on the cluster, not on local-workstation. Inside the training job, before training, run `python -m playjev.collect
<game> --steps N --pages 8 --epsilon 0.1 --shard <job>_<k>` for every game with a registered teacher, three collector
processes at a time (each drives 8 Chromium pages; about 300 to 1000 env-steps/s per process on local-workstation, expect similar
on the node). Start with N = 100,000 frames per game (about 800 MB per game as JPEG). Frames whose seed satisfies
`seed % 10 == 0` are validation by construction (the collector's seeds start at --seed0 and increase; pass a seed0
that is a multiple of 10 so both splits exist). Mario and racer episodes are long, snake and flappy random-heavy
states are short; epsilon 0.1 is the default, use 0.3 for one of the three shards per game so off-teacher states
(recovery situations) are covered.

## Data

`data/<game>/<shard>/records.jsonl` plus `frames/*.jpg` (448 px long side), written by `python -m playjev.collect`.
Fields per line: game, shard, seed, episode, step, frame, prev_frame (null at episode start), actions (ordered names,
same order as the hook's action list), teacher_probs (soft target over actions, sums to 1), teacher_action (argmax),
taken_action (what was actually played; may differ under epsilon), score, reward, done, t. Descriptions for the
options come from the game's `pj.json`/hook action list (load via `playjev.env.load_spec` and the hook's registered
actions, or from `runs/bench/<game>/actions.json` if present).

Train on all games mixed, sampling uniformly per game (not per record, so big-shard games do not dominate).
Held out: seeds are the split key. Records whose `seed % 10 == 0` are validation, never trained on.

## Objective

For each record: prompt = frame image (+ prev_frame when `--two-frame`) + the option list in a **random order per
sample** (permute options, permute teacher_probs the same way). Loss = cross-entropy between the teacher's soft
distribution and the model's softmax over the K option-letter logits at the answer position (KL to soft targets).
Nothing else is trained (no text tokens, no next-token loss). Optionally add `--brier 0.5` for a Brier term on the
same distributions.

Full fine-tuning of the 0.8B model in bf16 with AdamW (lr 2e-5, cosine, warmup 3 percent, weight decay 0.0),
vision tower trainable too (flag `--freeze-vision` to compare). Batch about 64 frames (gradient accumulation as needed),
1 to 3 epochs. Gradient checkpointing on. Save a checkpoint every N steps to the PROJECT project scratch, keep last 2.

## Evaluation (every eval interval and at the end)

1. Validation loss and teacher-agreement accuracy per game (argmax model vs teacher_action).
2. Calibration per game on validation: ECE (15 bins) and Brier of P(model argmax) against "argmax equals
   teacher_action"; mean Jev confidence.
3. Closed loop: play each game with `python -m playjev.play <game> --policy server` against a running server
   (or an in-process policy that wraps the model; add `--policy local --ckpt` to play.py if simpler), 16 episodes,
   held-out seeds, report score mean vs random and vs teacher. Chromium runs on the compute node (verified in the
   smoke test), so this runs inside the training job.

## Order of work (numbers first)

1. Snake only, end to end: collect 100k snake frames in the job (teacher exists, `playjev/teachers/snake.py`), train
   0.8B-Base for one epoch, report validation agreement, ECE, and the closed-loop snake score of the trained model
   versus random (about 1) and the teacher (about 130). This is the first real number of the project; send it as soon
   as it exists, before polishing anything.
2. Then add every game whose teacher is registered in `playjev/teachers/__init__.py` at that time (re-read it), one
   mixed run, per-game eval.
3. Ablations only after 1 and 2 have numbers: 2B-Base, frozen vision tower, two frames (temporal stack), 224 px.

## Facts from the smoke test to reuse (docs/MODEL_NOTES.md has the details)

- `playjev/model.py` has the prompt builder and the letter readout; the plain template puts " A".." D" as
  space-prefixed single tokens after a final "Answer:" line. Training must use build_prompt from the same module.
- Batch-32 forward of 0.8B is 136 ms on H200; single-thread preprocessing (JPEG decode + processor) is 109 ms for the
  same batch, so use DataLoader workers (8 or more) for decoding and the processor, pinned memory, and prefetch. A 448
  px frame is about 182 to 196 visual tokens; left padding across the batch.
- torchvision is required by the processor; the venv at $WORK/.venv has everything. First forward
  on a fresh node takes about two minutes (triton compiles the fla kernels), so warm up before timing.
- Zero-shot is a letter prior (0.7 on A regardless of frame), so option-order permutation per sample during training
  is essential, and the eval must report position bias (mean probability per letter position) alongside accuracy.

## Deliverables

`playjev/train_sft.py`, `hpc/train_sft.pbs` (1 H200, smallx via `-q autox`, walltime 12 h, `-P PROJECT`),
`docs/TRAIN_NOTES.md` with the numbers (loss curves as text, per-game agreement, ECE, closed-loop scores),
and the checkpoint path.
