# PlayJev training notes (stage B, supervised on teacher labels)

Code: `playjev/data.py` (records, permutation, collate), `playjev/train_sft.py` (trainer + eval), `playjev/play.py`
`LocalPolicy` (closed loop with the checkpoint in-process), `hpc/train_sft.pbs` (collect + train + play in one
smallx job). Checkpoints: `$WORK/ckpt/<run>/{step-N,final}` (bf16,
HF format, `PlayJevModel(path)` loads them). Logs: `$WORK/logs/train_<run>_<job>_live.log` and
`<ckpt>/log.jsonl` (one JSON line per logged step and per eval).

## Setup shared by every run

- Data: `python -m playjev.collect <game>` inside the job, three shards per game in parallel on the node's 12 cores
  (epsilon 0.1, 0.1, 0.3; seed0 100000 / 300000 / 500000). Records whose episode seed satisfies `seed % 10 == 0` are
  validation and never trained on. A run trains on its own shards only (`--shards <run>_a <run>_b <run>_c`).
- Sample: the frame (448 px long side JPEG from the hook), the game's options in a fresh random order per sample
  (validation: a permutation fixed by the sample index), the teacher's soft target permuted the same way. The prompt
  is `build_plain_prompt` from `playjev/model.py`, so training and inference strings are identical (see
  MODEL_NOTES section 2). Descriptions come from the game's `pj_hook.js`.
- Objective: cross-entropy between the teacher distribution and the model's softmax over the K letter logits
  (`" A".." D"` rows of the tied embedding, fp32) at the last position. No other token is trained. `--brier w` adds
  a Brier term on the same two distributions.
- Optimisation: full fine-tuning, fp32 master weights, bf16 autocast, AdamW (betas 0.9/0.95, wd 0), lr 2e-5 with 3
  percent warmup then cosine to 10 percent, grad clip 1.0, gradient checkpointing, batch 64 (one micro batch of 64
  on the H200), 10 DataLoader workers doing JPEG decode + processor.
- Eval (every 400 steps and at the end, 2000 fixed validation samples): loss, teacher agreement (argmax vs
  teacher argmax), ECE (15 bins) and Brier of p_max against agreement, mean Jev confidence, mean probability and
  argmax share per letter position (position bias). Closed loop after training: `playjev.play <game> --policy local`
  16 episodes on seeds 5000+ against `--policy random` and `--policy teacher` on the same seeds.
- Note on the diagnostics: the loss constrains only the softmax among the K slots, so the share of the whole
  vocabulary on the letters (`allowed_mass` in PlayJevModel) is not preserved by training. The decision does not
  depend on it.

## Run 1: snake only, 0.8B-Base, one epoch (job 621561, `sft_snake1`, hopper-18)

Collection: 3 x 33,336 records in 154 s (about 220 records/s per collector with the BFS teacher), 100,008 records,
90,062 train / 9,946 val. Training: 1,407 steps of 64, warmup 42.

Step 0 (zero-shot, permuted options): loss 2.004, agreement 0.2375 (chance 0.25), p_max 0.733, ECE 0.50, mean
probability per letter position 0.73 / 0.13 / 0.04 / 0.10, argmax on A for 100 percent of samples.

(training in progress; table filled in below as evals land)
