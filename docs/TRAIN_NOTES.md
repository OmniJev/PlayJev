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

The first job (621561) died at the first backward: fla 0.5.2 refuses its gated chunk backward kernel on Hopper with
triton 3.4 to 3.7.0 (wrong results, fla issue #640). Fix: triton 3.7.1 in the venv (now in `env_setup.sh`), and the
trainer compares the fla forward and backward with the torch reference on random inputs before training (relative
difference 0.3 to 0.8 percent in output and every gradient on this run, within bf16 tolerance; the run aborts if
they disagree, and `PLAYJEV_NO_FLA=1` forces the torch reference). Resubmitted as job 621566 on the same shards.

Training: 60 samples/s (batch 64, about 1.05 s per step), 16.5 GB peak, 25.1 min for the epoch. Train loss 1.78 at
step 1, 0.89 at 100, 0.71 at 400, 0.63 to 0.65 from step 1000 on. Validation (2000 fixed samples):

| step | val loss | agreement | ECE | Brier | mean conf | p per letter (A, B, C, D) |
|---|---|---|---|---|---|---|
| 0 | 2.004 | 0.238 | 0.496 | 0.425 | 0.645 | 0.73 / 0.13 / 0.04 / 0.10 |
| 400 | 0.686 | 0.756 | 0.060 | 0.143 | 0.620 | 0.23 / 0.24 / 0.28 / 0.25 |
| 800 | 0.661 | 0.760 | 0.099 | 0.144 | 0.563 | 0.24 / 0.25 / 0.26 / 0.26 |
| 1200 | 0.640 | 0.694 | 0.105 | 0.124 | 0.596 | 0.24 / 0.24 / 0.26 / 0.26 |
| 1407 (final) | 0.639 | 0.751 | 0.057 | 0.128 | 0.594 | 0.24 / 0.24 / 0.26 / 0.26 |

The letter prior is gone from step 400 on. Agreement moves by several points between evals while the loss keeps
falling: many snake targets are exact ties (0.45 / 0.45 on two equally short paths to the food), where argmax
agreement is a coin flip. From run 2 on the eval also reports agreement against the teacher's whole top set.

Closed loop, 16 episodes each on seeds 5000+ (argmax actions, 8 pages, cap 2000 steps):

| policy | score mean | median | max | episode length | mean conf |
|---|---|---|---|---|---|
| random | 1.0 | 1 | 1 | 3.8 | 1.0 (one-hot) |
| trained 0.8B, one epoch | 78.2 | 71 | 126 | 295 | 0.58 |
| BFS teacher (on `info()`) | 113.7 | 108.5 | 161 | 457 | 0.62 |

The pixel model reaches 69 percent of the teacher's score from frames alone after 25 minutes of training;
zero-shot it was at chance. Play throughput with the model in the loop: 84 env-steps/s over 8 pages (the model
forward per step at batch 8 is about 50 ms, see MODEL_NOTES 3.2). Checkpoint:
`$WORK/ckpt/sft_snake1/final` (step-800 and step-1200 kept as well).

## Run 2: all ten games, 0.8B-Base, one epoch (`sft_all1`, jobs 621575 then 621580)

Collection: 100k frames per game, shards a and b at the game's pj.json `collect.epsilon` (0.02 for 2048, tetris,
flappy; 0.03 sokoban; 0.1 otherwise, snake has no block so 0.1), shard c at 0.3 except sokoban. Collection rates per
game (3 collectors in parallel on the node's 12 cores, seconds for 3 x 33,336 records): snake 157, 2048 208,
tetris 122, breakout 130, invaders 215, flappy 246.

A driver finding on the way, fixed in `playjev/env.py` (the driver owner was unreachable at the time; change is the
one route rule): on the compute nodes every `page.goto(..., wait_until="load")` of flappy hit the 30 s timeout
(job 621575, 0 records; the other games loaded fine there, and flappy loads in 0.6 s on the login node). Cause:
the page creates five `Audio` objects for its `.ogg` sounds at load; media elements delay the document's `load`
event until they have decoded data, and on the nodes that never completed. The driver now aborts requests for
audio files (`.mp3 .ogg .wav .m4a .mid .midi .oga .flac`) alongside off-origin requests; audio is muted anyway, an
aborted source just errors the media element. Verified locally: bench clean and `check_det.py` PASS for flappy,
mario and racer (the games that load sounds); on the node flappy then collected at the same rate as the others.

Flappy shard b died at its first page load with `TargetClosedError` (a Chromium process closed while 24 pages were
launching; `--disable-dev-shm-usage` is the usual remedy, driver owner's call), so flappy has 66,672 records. Data:
966,744 records, 863,381 train / 103,363 val (per game 87k to 91k train, flappy 60k). 13,490 steps of 64, warmup 404,
eval every 1500 steps on 4000 game-balanced validation samples, 51 samples/s cumulative (about 4 h for the epoch).

Zero-shot (step 0), agreement / tie-aware agreement / mean probability on letter A: racer 0.20 / 0.20 / 0.40,
mario 0.24 / 0.24 / 0.35, snake 0.28 / 0.40 / 0.74, tetris 0.17 / 0.17 / 0.62, invaders 0.37 / 0.37 / 0.65,
pacman 0.27 / 0.33 / 0.78, flappy 0.51 / 0.51 / 0.76, 2048 0.26 / 0.26 / 0.78, sokoban 0.24 / 0.24 / 0.68,
breakout 0.31 / 0.32 / 0.69; all 0.286 / 0.306 (chance).

Validation during training (agreement / tie-aware / ECE / mean confidence):

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .642 / .657 | .946 / .946 / .08 / .75 | .856 / .950 / .18 / .59 | .779 / .779 / .12 / .60 | .707 / .710 / .11 / .41 | .677 / .733 / .08 / .48 | .574 / .576 / .14 / .30 | .569 / .569 / .05 / .39 | .564 / .564 / .12 / .36 | .419 / .419 / .02 / .10 | .360 / .362 / .01 / .14 |
| 3000 | .666 / .701 | .993 / .993 / .06 / .87 | .696 / .966 / .07 / .60 | .776 / .776 / .18 / .52 | .670 / .673 / .06 / .48 | .710 / .793 / .10 / .58 | .602 / .607 / .07 / .46 | .701 / .701 / .05 / .56 | .569 / .569 / .10 / .39 | .536 / .536 / .03 / .32 | .414 / .414 / .03 / .22 |
| 4500 | .686 / .720 | .990 / .990 / .05 / .89 | .712 / .969 / .09 / .60 | .792 / .792 / .10 / .64 | .697 / .699 / .10 / .41 | .765 / .848 / .08 / .60 | .647 / .650 / .06 / .51 | .733 / .733 / .07 / .67 | .554 / .554 / .07 / .40 | .564 / .564 / .05 / .27 | .409 / .409 / .04 / .22 |
| 6000 | .709 / .737 | .993 / .993 / .08 / .83 | .772 / .976 / .10 / .59 | .817 / .817 / .12 / .64 | .710 / .713 / .10 / .44 | .758 / .829 / .05 / .65 | .635 / .637 / .05 / .56 | .795 / .795 / .04 / .75 | .597 / .597 / .11 / .40 | .621 / .621 / .06 / .34 | .397 / .397 / .02 / .22 |
| 7500 | .722 / .752 | .985 / .985 / .07 / .82 | .749 / .979 / .08 / .63 | .803 / .803 / .11 / .66 | .705 / .705 / .05 / .50 | .802 / .871 / .06 / .69 | .726 / .728 / .05 / .60 | .814 / .814 / .05 / .81 | .567 / .567 / .08 / .42 | .633 / .633 / .04 / .41 | .436 / .441 / .05 / .27 |
| 9000 | .743 / .774 | .993 / .993 / .08 / .83 | .751 / .976 / .08 / .61 | .827 / .827 / .17 / .59 | .691 / .694 / .05 / .53 | .770 / .853 / .06 / .68 | .792 / .797 / .10 / .62 | .884 / .884 / .04 / .85 | .607 / .607 / .12 / .41 | .657 / .657 / .07 / .41 | .458 / .461 / .03 / .25 |
| 10500 | .763 / .787 | .998 / .998 / .08 / .84 | .830 / .987 / .14 / .59 | .830 / .830 / .13 / .66 | .721 / .721 / .07 / .48 | .795 / .866 / .07 / .65 | .792 / .797 / .09 / .63 | .908 / .908 / .02 / .88 | .620 / .620 / .10 / .44 | .688 / .688 / .08 / .42 | .453 / .456 / .03 / .24 |
| 12000 | .766 / .796 | .998 / .998 / .08 / .85 | .775 / .990 / .08 / .59 | .830 / .830 / .16 / .64 | .697 / .697 / .07 / .44 | .779 / .862 / .07 / .67 | .802 / .805 / .09 / .65 | .930 / .930 / .02 / .90 | .635 / .635 / .12 / .44 | .755 / .755 / .12 / .45 | .453 / .456 / .05 / .25 |
| 13490 (final) | .772 / .804 | .998 / .998 / .08 / .84 | .764 / .990 / .08 / .60 | .844 / .844 / .15 / .64 | .715 / .715 / .07 / .48 | .776 / .873 / .07 / .68 | .810 / .812 / .09 / .67 | .949 / .949 / .02 / .94 | .627 / .627 / .13 / .43 | .755 / .755 / .11 / .48 | .475 / .478 / .05 / .25 |

Training took 4 h 11 min on hopper-15 (56 samples/s, 17 GB peak); train loss 2.13 at step 1, 1.02 at 1000, 0.74 at
7500, 0.70 to 0.73 over the last thousand steps; validation loss 0.708 at the end. Three games are still far from
their teacher at one epoch: 2048 (.475, flat since step 9000), mario (.627) and breakout (.715); sokoban (.949),
flappy (.998) and snake (tie-aware .990) are at the ceiling of what argmax agreement can show. Checkpoint
`$WORK/ckpt/sft_all1/final` (step-10500 and step-12000 kept).

Closed loop on held-out seeds 5000+, 16 episodes per game and policy, 8 pages, cap 1500 steps, all on
`sft_all1/final` (job 621580: argmax, sampled actions, random, teacher; job 621686: argmax with `--delay 1`, the
decision from frame k applied at step k+1). Cells are mean score (median / max); "capped" counts episodes that
reached 1500 steps. Argmax replays and random replays are recorded for the demo (`runs/replays/<game>/`).

<!-- run2-closed-loop -->
| game | trained, argmax | trained, sampled | random | teacher | trained, argmax, delay 1 | argmax: length, conf |
|---|---|---|---|---|---|---|
| snake | 77.8 (78.5 / 121) | 15.2 (15 / 51) | 1.0 (1 / 1) | 113.7 (108.5 / 161) | 11.4 (9.5 / 31, 4 capped) | 296, 0.61 |
| 2048 | 54.5 (32 / 284, 16 capped) | 1304.2 (1230 / 2484) | 1086.8 (1008 / 2424) | 19593.2 (20378 / 22068) | 72.8 (48 / 260, 16 capped) | 1500, 0.16 |
| tetris | 1034.4 (1065 / 1970) | 665.6 (625 / 1670) | - | - | 248.1 (230 / 430) | 252, 0.62 |
| breakout | - | - | - | - | 397.8 (247.5 / 1925) | - |
| flappy | - | - | - | - | 0.2 (0 / 1) | - |
| invaders | - | - | - | - | 400.0 (400 / 400) | - |
| mario | - | - | - | - | 819.9 (645 / 2165) | - |
<!-- /run2-closed-loop -->

Observations while the loop runs (12:15):

- 2048 under argmax is a degenerate loop: all 16 episodes hit the 1500-step cap with a mean score of 54 (random
  1087). A blocked direction is a legal no-op that leaves the board bit-identical, so a deterministic policy that
  picks one keeps picking it forever. The teacher targets give blocked moves zero mass (softmax over expectimax
  values, T = 1200, mean p_max of the targets 0.79 on the smoke shard), so this is the model's reading of the board,
  not the labels: 2048 validation agreement is .475 with p_max about 0.44, calibrated but unsure, and sampling from
  that distribution scores 1304, barely above random. The four-way ranking of near-identical boards full of digits
  is the hardest read in the roster for a 0.8B model after one epoch. Two separate things to fix: the loop (a
  decoding rule: when the frame did not change after an action, drop that action and take the next best; this never
  fires in the real-time games because their frames always change) and the policy itself (more 2048 epochs or
  DAgger on the model's own states).
- Sampled actions lose to argmax everywhere else (snake 15 vs 78, tetris 666 vs 1034), so argmax stays the protocol.
- Delay 1 (job 621686, the same checkpoint acting one step late): snake 77.8 to 11.4, tetris 1034 to 248, flappy 0.25
  pipes; invaders still clears every wave at 400 (left / right / hold with auto-fire barely cares about one step of
  lag). The unmodified teachers collapse harder under the same delay (snake 108 to 1), so the model already carries
  some tolerance, and `sft_all1_d1` is the run that trains for it.

## Run 3: all ten games, delayed labels (`sft_all1_d1`, job 621687, hopper-14)

Same shards, model, optimiser and eval schedule as `sft_all1`, with `--label-delay 1`: frame k is labelled with the
teacher's distribution at record k+1 of the same episode (each episode's last record dropped, 13,426 steps instead of
13,490). Validation targets are shifted the same way, so the rows below are agreement with the teacher's *next*
decision and are not directly comparable to the `sft_all1` table (its targets are the current decision). Closed loop
after training at delay 1 (native) and delay 0. Train loss 1.80 at step 1, 1.12 at 600, 1.06 at 1500 (`sft_all1`:
1.02 at 1000), 57 samples/s.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .569 / .583 | .894 / .902 / .12 / .56 | .711 / .820 / .26 / .27 | .692 / .692 / .16 / .45 | .691 / .691 / .08 / .46 | .634 / .656 / .12 / .36 | .262 / .262 / .08 / .17 | .463 / .463 / .04 / .25 | .536 / .536 / .14 / .36 | .429 / .429 / .07 / .17 | .398 / .400 / .06 / .18 |
| 3000 | .624 / .649 | .899 / .907 / .08 / .69 | .744 / .922 / .12 / .51 | .781 / .781 / .20 / .50 | .691 / .691 / .04 / .60 | .702 / .765 / .08 / .54 | .417 / .417 / .05 / .28 | .557 / .557 / .06 / .40 | .551 / .551 / .10 / .36 | .506 / .506 / .06 / .18 | .410 / .415 / .06 / .18 |

(training in progress)

## Real-time latency (decided 2026-09-18, relayed by session)

The deployed model sees frame k and its answer is applied at step k+1 (one step, 83 to 100 ms in the real-time
games, which covers the 43 ms batch-1 inference). `playjev.play --delay 1` reproduces that in the closed loop; the
unmodified teachers collapse under it (snake 108 to 1, flappy 30 to 0 pipes, measured by session). Training for it
needs no new data: `--label-delay 1` in `playjev/data.py` labels frame k with the teacher's decision at record k+1
of the same episode (consecutive steps only, each episode's last record dropped), and the validation metrics of a
delayed run are computed against those shifted targets.

Runs queued after `sft_all1`: `play_eval.pbs` on `sft_all1/final` at delay 1 (how much an undelayed model loses in
real time); `sft_all1_d1` (same shards, `--label-delay 1`, closed loop at delay 1 and 0); then the two-frame
temporal-stack variant with the delayed label. Turn-based games (2048, sokoban, tetris as stepped here) are trained
with the same shifted labels for a single model unless the per-game numbers say it hurts.
