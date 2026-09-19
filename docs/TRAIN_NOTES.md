# PlayJev training notes (stage B, supervised on teacher labels)

Code: `playjev/data.py` (records, permutation, collate), `playjev/train_sft.py` (trainer + eval), `playjev/play.py`
`LocalPolicy` (closed loop with the checkpoint in-process), `hpc/train_sft.pbs` (collect + train + play in one
smallx job). Checkpoints: `$CKPT_ROOT/<run>/{step-N,final}` (bf16,
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
`$CKPT_ROOT/sft_snake1/final` (step-800 and step-1200 kept as well).

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
`$CKPT_ROOT/sft_all1/final` (step-10500 and step-12000 kept).

Closed loop on held-out seeds 5000+, 16 episodes per game and policy, 8 pages, cap 1500 steps, all on
`sft_all1/final` (job 621580: argmax, sampled actions, random, teacher; job 621686: argmax with `--delay 1`, the
decision from frame k applied at step k+1). Cells are mean score (median / max); "capped" counts episodes that
reached 1500 steps. Argmax replays and random replays are recorded for the demo (`runs/replays/<game>/`).

<!-- run2-closed-loop -->
| game | trained, argmax | trained, sampled | random | teacher | trained, argmax, delay 1 | argmax: length, conf |
|---|---|---|---|---|---|---|
| snake | 77.8 (78.5 / 121) | 15.2 (15 / 51) | 1.0 (1 / 1) | 113.7 (108.5 / 161) | 11.4 (9.5 / 31, 4 capped) | 296, 0.61 |
| 2048 | 54.5 (32 / 284, 16 capped) | 1304.2 (1230 / 2484) | 1086.8 (1008 / 2424) | 19593.2 (20378 / 22068) | 72.8 (48 / 260, 16 capped) | 1500, 0.16 |
| tetris | 1034.4 (1065 / 1970) | 665.6 (625 / 1670) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) | 248.1 (230 / 430) | 252, 0.62 |
| breakout | 798.4 (415 / 2825) | 660.3 (595 / 2030) | 578.1 (272.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) | 397.8 (247.5 / 1925) | 135, 0.42 |
| flappy | 8.9 (7 / 23) | 6.0 (4.5 / 20) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) | 0.2 (0 / 1) | 241, 0.88 |
| invaders | 400.0 (400 / 400) | 400.0 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) | 400.0 (400 / 400) | 203, 0.48 |
| mario | 1156.4 (837 / 2877) | 1001.5 (637.5 / 2885) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) | 819.9 (645 / 2165) | 43, 0.41 |
| pacman | 1209.4 (1040 / 4100) | 601.2 (590 / 1610) | 113.1 (60 / 360) | 7190.0 (7510 / 8080, 15 capped) | 453.8 (440 / 1180) | 197, 0.71 |
| racer | 6211.2 (6225.4 / 6712.4, 15 capped) | 5962.9 (5991.05 / 6406.7, 16 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.65 / 6715.1) | 5885.1 (5911.15 / 6411.9, 16 capped) | 1500, 0.58 |
| sokoban | 57.9 (101 / 104) | 26.2 (1.5 / 103) | 6.6 (0 / 101) | 102.2 (102 / 104) | 6.6 (0 / 101) | 245, 0.82 |
<!-- /run2-closed-loop -->

Job 621580 finished at 12:45 (5 h 24 min in all). Read as the model's share of the teacher's gain over random,
(model - random) / (teacher - random), on the same seeds: invaders 1.00 (every policy but random clears the
waves), racer 0.92, snake 0.68, sokoban 0.54, then a cliff: mario 0.15, pacman 0.15, flappy 0.11, tetris 0.06,
breakout 0.01, 2048 below random (the argmax loop). Validation agreement does not predict this order: flappy
(.998) and tetris (.810) sit at the bottom in play while their agreement is near the top, and the step by
step relabel of a flappy death shows why: 89 of 90 decisions match the teacher and the one miss is a recovery move
(a second consecutive flap) that the teacher's own trajectories almost never contain. Covariate shift, so the next
round is DAgger (job 621845, `dagger1`: the model plays 40k frames per game, the teacher labels, one more epoch
from the `sft_all1` checkpoint on those plus run 1's shard a).

### The same checkpoint under the executor rule (job 621844, 12:49 to 13:38)

`play.py` default since commit 26f9cfa: a move whose step left the observation unchanged (same frame
bytes, same score, not done) is banned on that observation and the best other move is taken; the ban clears when the
observation changes; the recorded probabilities are untouched, only the executed index. Same seeds, cap and episode
count as above; random and teacher rerun under the same rule (both recorded for the demo). Games whose frames always
change reproduce the plain numbers exactly (snake, tetris, flappy, invaders, mario, racer, sokoban), which doubles
as a determinism check across nodes; 2048, breakout and pacman do have unchanged observations (a blocked slide, the
ball waiting on the paddle, the start-of-life pause) and change.

<!-- run2-guarded -->
| game | trained, argmax | random | teacher | trained, argmax, delay 1 | no-op steps seen / redirected (delay 0) |
|---|---|---|---|---|---|
| snake | 77.8 (78.5 / 121) | 1.0 (1 / 1) | 113.7 (108.5 / 161) | 11.4 (9.5 / 31, 4 capped) | 0 / 0 |
| 2048 | 3174.2 (3164 / 5808) | 1021.0 (860 / 2424) | 19593.2 (20378 / 22068) | 2191.2 (2218 / 4680) | 854 / 853 |
| tetris | 1034.4 (1065 / 1970) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) | 248.1 (230 / 430) | 0 / 0 |
| breakout | 611.2 (305 / 2825) | 495.9 (232.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) | 394.1 (265 / 1925) | 56 / 56 |
| flappy | 8.9 (7 / 23) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) | 0.2 (0 / 1) | 0 / 0 |
| invaders | 400.0 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) | 400.0 (400 / 400) | 0 / 0 |
| mario | 1156.4 (837 / 2877) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) | 819.9 (645 / 2165) | 0 / 0 |
| pacman | 1035.6 (1040 / 2520) | 113.1 (60 / 360) | 7025.6 (7175 / 8080, 15 capped) | 501.9 (440 / 1380) | 15 / 15 |
| racer | 6211.2 (6225.4 / 6712.4, 15 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.65 / 6715.1) | 5885.1 (5911.15 / 6411.9, 16 capped) | 0 / 0 |
| sokoban | 57.9 (101 / 104) | 6.6 (0 / 101) | 102.2 (102 / 104) | 6.6 (0 / 101) | 155 / 155 |
<!-- /run2-guarded -->

Observations while the loop ran (12:15):

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
| 4500 | .628 / .661 | .919 / .927 / .07 / .71 | .676 / .919 / .07 / .49 | .779 / .779 / .20 / .50 | .694 / .694 / .07 / .48 | .680 / .760 / .03 / .55 | .501 / .501 / .08 / .31 | .571 / .571 / .06 / .49 | .514 / .514 / .12 / .30 | .562 / .562 / .09 / .22 | .402 / .405 / .02 / .18 |
| 6000 | .663 / .686 | .944 / .952 / .07 / .76 | .770 / .922 / .13 / .52 | .781 / .781 / .15 / .55 | .707 / .707 / .08 / .46 | .729 / .804 / .05 / .61 | .509 / .509 / .07 / .30 | .640 / .640 / .03 / .51 | .548 / .548 / .09 / .37 | .581 / .581 / .07 / .29 | .432 / .434 / .04 / .19 |
| 7500 | .679 / .698 | .952 / .960 / .07 / .76 | .820 / .937 / .16 / .56 | .774 / .774 / .09 / .63 | .715 / .715 / .05 / .61 | .758 / .818 / .06 / .64 | .545 / .545 / .06 / .40 | .677 / .677 / .03 / .59 | .548 / .548 / .05 / .43 | .615 / .615 / .07 / .33 | .398 / .400 / .06 / .19 |
| 9000 | .696 / .719 | .967 / .975 / .08 / .79 | .770 / .944 / .13 / .53 | .794 / .794 / .15 / .58 | .702 / .702 / .06 / .54 | .785 / .835 / .07 / .63 | .613 / .613 / .04 / .51 | .736 / .736 / .05 / .65 | .548 / .548 / .08 / .39 | .632 / .632 / .11 / .28 | .420 / .424 / .04 / .17 |
| 10500 | .701 / .724 | .965 / .972 / .07 / .79 | .790 / .939 / .14 / .54 | .805 / .805 / .15 / .58 | .704 / .704 / .06 / .52 | .751 / .823 / .08 / .64 | .672 / .672 / .10 / .50 | .734 / .734 / .05 / .70 | .551 / .551 / .08 / .38 | .613 / .613 / .08 / .30 | .439 / .444 / .05 / .19 |
| 12000 | .711 / .733 | .962 / .970 / .07 / .80 | .790 / .944 / .14 / .54 | .792 / .792 / .11 / .63 | .709 / .709 / .06 / .54 | .782 / .835 / .06 / .66 | .677 / .677 / .08 / .56 | .791 / .791 / .04 / .73 | .561 / .561 / .07 / .41 | .634 / .634 / .07 / .34 | .427 / .432 / .05 / .18 |
| 13426 (final) | .724 / .741 | .970 / .977 / .07 / .80 | .858 / .949 / .21 / .54 | .799 / .799 / .14 / .59 | .712 / .712 / .05 / .52 | .782 / .845 / .09 / .66 | .684 / .684 / .08 / .55 | .820 / .820 / .05 / .78 | .553 / .553 / .06 / .41 | .649 / .649 / .08 / .35 | .427 / .432 / .04 / .20 |

Training took 3 h 55 min on hopper-14 (57 samples/s); validation loss 0.843 at the end against 0.708 for the unshifted
run, so predicting the teacher's next decision from the current frame is harder everywhere, and hardest where the
next decision depends on what the current one does to the board: sokoban .820 (unshifted .949), tetris .684 (.810),
2048 .427 (.475), invaders .649 (.755). The real-time games that move on their own lose little: flappy .970 (.998),
snake .858 (.764, higher: the shifted snake target has fewer ties), racer .799 (.844), pacman .782 (.776), mario .553
(.627), breakout .712 (.715). Closed loop (delay 1, delay 0, sampled, random, teacher; under the executor rule since
the mirror's play.py changed at 12:49) below.

<!-- run3-closed-loop -->
| game | this run, delay 1 | `sft_all1`, delay 1 | this run, delay 0 | `sft_all1`, delay 0 | this run, sampled | random | teacher |
|---|---|---|---|---|---|---|---|
| snake | 19.6 (17.5 / 36) | 11.4 (9.5 / 31, 4 capped) | 1.0 (1 / 1) | 77.8 (78.5 / 121) | 5.6 (3.5 / 20) | 1.0 (1 / 1) | 113.7 (108.5 / 161) |
| 2048 | 3195.0 (2804 / 6680) | 2191.2 (2218 / 4680) | 2680.2 (2318 / 5436) | 3174.2 (3164 / 5808) | 1004.8 (1124 / 1452) | 1021.0 (860 / 2424) | 19593.2 (20378 / 22068) |
| tetris | 306.2 (195 / 880) | 248.1 (230 / 430) | 203.1 (210 / 260) | 1034.4 (1065 / 1970) | 191.9 (195 / 260) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) |
| breakout | 635.9 (270 / 2530) | 394.1 (265 / 1925) | 979.7 (430 / 3350) | 611.2 (305 / 2825) | 623.4 (552.5 / 1695) | 495.9 (232.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) |
| flappy | 2.3 (2 / 8) | 0.2 (0 / 1) | 0.0 (0 / 0) | 8.9 (7 / 23) | 0.0 (0 / 0) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) |
| invaders | 400.0 (400 / 400) | 400.0 (400 / 400) | 400.0 (400 / 400) | 400.0 (400 / 400) | 391.9 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) |
| mario | 242.0 (211 / 667, 8 capped) | 819.9 (645 / 2165) | 247.0 (211 / 667, 7 capped) | 1156.4 (837 / 2877) | 689.4 (591.5 / 2170) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) |
| pacman | 236.2 (80 / 1370) | 501.9 (440 / 1380) | 418.1 (345 / 1420) | 1035.6 (1040 / 2520) | 561.9 (295 / 1440) | 113.1 (60 / 360) | 7025.6 (7175 / 8080, 15 capped) |
| racer | 5116.0 (5064.35 / 6110.5, 16 capped) | 5885.1 (5911.15 / 6411.9, 16 capped) | 4873.0 (4923.5 / 5371.9, 16 capped) | 6211.2 (6225.4 / 6712.4, 15 capped) | 5356.8 (5340.5 / 5869.6, 16 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.65 / 6715.1) |
| sokoban | 6.4 (0 / 101) | 6.6 (0 / 101) | 6.3 (0 / 101) | 57.9 (101 / 104) | 6.4 (0 / 101) | 6.6 (0 / 101) | 102.2 (102 / 104) |
<!-- /run3-closed-loop -->

Job 621687 finished at 16:05. Under one step of latency the shifted labels help the games where the next decision
follows from the current frame (snake 11.4 to 19.6, tetris 248 to 306, breakout 394 to 636, flappy 0.2 to 2.3,
2048 2191 to 3195) and hurt the ones where the next decision depends on what the current move does (mario 820 to
242, standing still until the cap in half the episodes; pacman 502 to 236; racer 5885 to 5116; sokoban 6.6 either
way, and 6.3 at delay 0 against 57.9 unshifted). At delay 0 the model is out of phase by design (snake 1.0). Two
things follow: the label shift has to be per game (delay 1 for the real-time games only, 0 for 2048 and sokoban,
and mario needs the two-frame read of its jump phase before a shifted label can work), and even where it helps the
real-time score stays a fraction of the undelayed one, so latency has to be attacked in the model too (two frames
for velocity, run 4). Curious side result: at delay 0 the shifted model plays breakout better than the unshifted
one (980 vs 611, medians 430 vs 305), a label that points one step ahead of the ball is a better paddle target;
16 episodes, so treat it as a hint.


## Run 4: two frames per state, delayed labels (`sft_all1_d1_2f`, job 621820, started 12:26)

`sft_all1_d1` with `--two-frame` (previous and current frame temporally stacked into the same visual tokens, see
MODEL_NOTES section 2; at an episode start the current frame is repeated) and `--label-delay 1`, same shards, same
schedule, 56 samples/s (the stack adds no tokens, so no cost). Closed loop after training at delay 1 and 0 with
`--two-frame`. The target is what a single frame cannot show: velocity and jump phase (mario, breakout, flappy).
Rows are against the shifted targets, comparable with the `sft_all1_d1` table above and not with `sft_all1`.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .597 / .614 | .896 / .904 / .11 / .58 | .716 / .866 / .21 / .35 | .686 / .686 / .16 / .46 | .717 / .717 / .07 / .51 | .622 / .642 / .05 / .43 | .486 / .486 / .08 / .25 | .493 / .493 / .09 / .38 | .543 / .543 / .09 / .37 | .458 / .458 / .03 / .23 | .368 / .371 / .04 / .22 |
| 3000 | .661 / .679 | .894 / .902 / .10 / .61 | .785 / .909 / .16 / .54 | .761 / .761 / .17 / .52 | .759 / .759 / .05 / .61 | .736 / .780 / .09 / .60 | .529 / .529 / .07 / .37 | .645 / .645 / .08 / .55 | .548 / .548 / .12 / .34 | .559 / .559 / .10 / .20 | .415 / .420 / .07 / .17 |
| 4500 | .670 / .696 | .934 / .942 / .09 / .69 | .706 / .914 / .10 / .53 | .779 / .779 / .17 / .54 | .757 / .757 / .09 / .54 | .758 / .799 / .09 / .55 | .585 / .585 / .08 / .39 | .685 / .685 / .03 / .58 | .509 / .509 / .11 / .30 | .571 / .571 / .11 / .23 | .434 / .437 / .05 / .18 |
| 6000 | .678 / .698 | .934 / .942 / .05 / .78 | .785 / .929 / .12 / .57 | .787 / .787 / .13 / .60 | .749 / .749 / .09 / .53 | .743 / .787 / .04 / .62 | .552 / .552 / .11 / .35 | .682 / .682 / .10 / .64 | .553 / .553 / .10 / .38 | .593 / .593 / .06 / .32 | .415 / .420 / .03 / .21 |
| 7500 | .694 / .717 | .929 / .937 / .07 / .74 | .780 / .942 / .14 / .55 | .784 / .784 / .07 / .67 | .757 / .757 / .06 / .61 | .770 / .828 / .07 / .64 | .616 / .616 / .06 / .48 | .754 / .754 / .05 / .71 | .548 / .548 / .05 / .44 | .627 / .627 / .07 / .37 | .385 / .390 / .04 / .20 |
| 9000 | .707 / .733 | .952 / .960 / .07 / .80 | .752 / .937 / .12 / .55 | .787 / .787 / .14 / .59 | .759 / .759 / .05 / .62 | .768 / .833 / .07 / .64 | .618 / .618 / .08 / .47 | .810 / .810 / .05 / .73 | .548 / .548 / .07 / .41 | .666 / .666 / .11 / .33 | .417 / .422 / .03 / .18 |
| 10500 | .722 / .741 | .960 / .967 / .07 / .79 | .803 / .942 / .16 / .54 | .805 / .805 / .15 / .59 | .762 / .762 / .08 / .55 | .797 / .840 / .10 / .64 | .654 / .654 / .09 / .45 | .791 / .791 / .05 / .72 | .571 / .571 / .10 / .39 | .676 / .676 / .12 / .33 | .412 / .417 / .02 / .18 |
| 12000 | .727 / .748 | .960 / .967 / .07 / .79 | .805 / .947 / .14 / .55 | .807 / .807 / .13 / .62 | .775 / .775 / .08 / .59 | .789 / .845 / .08 / .64 | .649 / .649 / .06 / .50 | .828 / .828 / .05 / .79 | .586 / .586 / .11 / .40 | .692 / .692 / .11 / .37 | .395 / .400 / .04 / .18 |
| 13426 (final) | .726 / .744 | .960 / .967 / .06 / .81 | .843 / .942 / .18 / .55 | .781 / .781 / .14 / .59 | .785 / .785 / .07 / .61 | .768 / .838 / .09 / .66 | .644 / .644 / .05 / .51 | .847 / .847 / .03 / .80 | .561 / .561 / .07 / .41 | .678 / .678 / .10 / .39 | .405 / .410 / .04 / .19 |

Against the single-frame delayed run at the same step the second frame is worth +.028 overall at step 1500 (.597 vs
.569) and +.037 at 3000 (.661 vs .624): tetris .529 vs .417, sokoban .645 vs .557, breakout .759 vs .691, snake .785
vs .744, invaders .559 vs .506; mario (.548 vs .551), flappy and racer are unchanged so far.

Final (4 h 13 min, hopper-15): .726 / .744 against .724 / .741 for the single-frame delayed run, so the mid-training
lead did not survive the epoch; validation loss 0.826 vs 0.843. Per game the second frame ends ahead on breakout
(.785 vs .712), sokoban (.847 vs .820) and invaders (.678 vs .649), behind on tetris (.644 vs .684), racer and
snake, and mario moves from .553 to .561 only: the temporally stacked patch does not give this model a velocity
read after one epoch. The stack puts both frames through the patch embedding that Qwen3.5 pretrained on
duplicated frames (it averages the two temporal slots), so motion has to be learned from scratch inside the
convolution; `--stack separate` (two images, twice the visual tokens) or a frame-difference channel are the
alternatives if the closed loop below does not show a velocity effect either.

<!-- run4-closed-loop -->
| game | this run, delay 1 | `sft_all1`, delay 1 | this run, delay 0 | `sft_all1`, delay 0 | this run, sampled | random | teacher |
|---|---|---|---|---|---|---|---|
| snake | 19.8 (16 / 51) | 11.4 (9.5 / 31, 4 capped) | 1.0 (1 / 1) | 77.8 (78.5 / 121) | 6.9 (6.5 / 19) | 1.0 (1 / 1) | 113.7 (108.5 / 161) |
| 2048 | 3285.2 (3234 / 5912) | 2191.2 (2218 / 4680) | 2780.2 (2740 / 5436) | 3174.2 (3164 / 5808) | 1254.5 (1162 / 2468) | 1021.0 (860 / 2424) | 19593.2 (20378 / 22068) |
| tetris | 253.8 (200 / 660) | 248.1 (230 / 430) | 195.6 (200 / 260) | 1034.4 (1065 / 1970) | 200.6 (205 / 240) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) |
| breakout | 976.2 (730 / 4280) | 394.1 (265 / 1925) | 628.1 (355 / 3025) | 611.2 (305 / 2825) | 845.0 (530 / 3960) | 495.9 (232.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) |
| flappy | 2.8 (2.5 / 9) | 0.2 (0 / 1) | 0.0 (0 / 0) | 8.9 (7 / 23) | 0.0 (0 / 0) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) |
| invaders | 400.0 (400 / 400) | 400.0 (400 / 400) | 400.0 (400 / 400) | 400.0 (400 / 400) | 400.0 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) |
| mario | 235.0 (211 / 507, 8 capped) | 819.9 (645 / 2165) | 256.0 (211 / 667, 8 capped) | 1156.4 (837 / 2877) | 646.9 (636 / 1642) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) |
| pacman | 405.0 (120 / 1560) | 501.9 (440 / 1380) | 600.0 (575 / 1350) | 1035.6 (1040 / 2520) | 256.9 (235 / 520) | 113.1 (60 / 360) | 7025.6 (7175 / 8080, 15 capped) |
| racer | 4904.5 (5260 / 6263.6, 16 capped) | 5885.1 (5911.15 / 6411.9, 16 capped) | 4712.2 (4796.7 / 5991, 16 capped) | 6211.2 (6225.4 / 6712.4, 15 capped) | 5188.3 (5197.55 / 5651.7, 16 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.65 / 6715.1) |
| sokoban | 6.6 (0 / 101) | 6.6 (0 / 101) | 6.4 (0 / 101) | 57.9 (101 / 104) | 6.6 (0 / 101) | 6.6 (0 / 101) | 102.2 (102 / 104) |
<!-- /run4-closed-loop -->

Job 621820 finished at 17:39. Against the single-frame shifted run at delay 1: breakout 976 vs 636 (the ball's
velocity is what a paddle needs, and the second frame carries it), pacman 405 vs 236, mario 647 vs 242 only in the
sampled column (argmax 235 vs 242, still standing still to the cap in half the episodes), snake 19.8 vs 19.6, 2048
3285 vs 3195, flappy 2.75 vs 2.3, tetris 254 vs 306, racer 4905 vs 5116, sokoban unchanged at random level. So the
temporal stack is worth having for breakout and nothing else at this scale; mario's jump phase is not read from
it. The separate-stack ablation (`sft_all1_2fs`, job 622271, delay 0, unshifted) tests whether two images the
model can compare by attention do better.


## Run 5: DAgger round 1 (`dagger1`, job 621962, hopper-13, collection 14:18 to 15:05, training from 15:06)

Why: the closed-loop deaths of `sft_all1` are missed corrections in states the teacher never visits (the
step-by-step relabel of a flappy death: 89 of 90 decisions agree, the miss is a second consecutive flap). So the
model plays and the teacher labels: `playjev.collect --actor local --ckpt sft_all1/final --epsilon 0.05` (argmax
with the executor rule, 5 percent random moves), 3 x 13,336 frames per game, seeds 700000 / 800000 / 900000, 4 to
5 minutes per game with three model copies on the H200 (sokoban 8 min; teacher give-ups end deadlocked sokoban
episodes, 60 to 75 per shard, and flappy episodes the teacher marks doomed, about 50 per shard). Three collectors
with the default 12 torch threads each thrash the 12-core node (9 records per second, job 621845); with
`OMP_NUM_THREADS=3` each does 37 per second. Training: one epoch from the `sft_all1` checkpoint at lr 1e-5 on
run 1's shard a plus the three on-policy shards (733k records, 655k train, 65k to 66k per game), 10,236 steps, eval
every 1500 on 4000 samples drawn from the validation records of all four shards (teacher-state and model-state
samples mixed), closed loop argmax only (random and teacher come from job 621844 on the same seeds).

Step 0 is the `sft_all1` checkpoint on this mixed validation set, so the drop against its own table (.772) is the
off-distribution gap on the model's own states: breakout .487 (teacher-state .715), racer .663 (.844), mario .512
(.627), sokoban .740 (.949), tetris .713 (.810), invaders .675 (.755); pacman .862 and flappy .977 are higher on
their own states.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | .682 / .711 | .977 / .979 / .07 / .84 | .786 / .987 / .10 / .62 | .663 / .663 / .06 / .62 | .487 / .490 / .18 / .48 | .862 / .931 / .08 / .72 | .713 / .723 / .05 / .67 | .740 / .740 / .11 / .79 | .512 / .512 / .07 / .40 | .675 / .675 / .10 / .47 | .411 / .411 / .06 / .24 |
| 1500 | .736 / .766 | .977 / .979 / .06 / .85 | .781 / .985 / .09 / .60 | .701 / .701 / .06 / .58 | .738 / .741 / .10 / .45 | .840 / .917 / .12 / .69 | .680 / .688 / .07 / .60 | .911 / .911 / .04 / .89 | .603 / .603 / .11 / .42 | .706 / .706 / .10 / .42 | .434 / .437 / .06 / .22 |
| 3000 | .728 / .764 | .984 / .987 / .08 / .82 | .710 / .980 / .11 / .60 | .718 / .718 / .11 / .55 | .638 / .641 / .04 / .45 | .840 / .924 / .12 / .67 | .690 / .701 / .06 / .55 | .924 / .924 / .03 / .89 | .600 / .600 / .09 / .43 | .704 / .704 / .11 / .44 | .473 / .473 / .08 / .20 |
| 4500 | .734 / .764 | .982 / .984 / .07 / .84 | .781 / .992 / .09 / .61 | .713 / .713 / .09 / .58 | .610 / .613 / .08 / .50 | .845 / .917 / .09 / .71 | .708 / .716 / .07 / .61 | .938 / .938 / .04 / .91 | .591 / .591 / .09 / .43 | .740 / .740 / .11 / .47 | .439 / .442 / .05 / .23 |
| 6000 | .741 / .776 | .987 / .990 / .06 / .85 | .733 / .990 / .09 / .60 | .730 / .730 / .08 / .59 | .626 / .628 / .08 / .52 | .843 / .924 / .10 / .72 | .723 / .736 / .07 / .63 | .927 / .927 / .04 / .93 | .629 / .629 / .10 / .46 | .755 / .755 / .08 / .55 | .457 / .460 / .04 / .27 |
| 7500 | .753 / .784 | .987 / .990 / .07 / .84 | .753 / .992 / .08 / .62 | .740 / .740 / .11 / .57 | .628 / .631 / .05 / .52 | .874 / .921 / .09 / .72 | .716 / .728 / .07 / .64 | .966 / .966 / .02 / .94 | .600 / .600 / .10 / .43 | .777 / .777 / .10 / .52 | .501 / .504 / .06 / .27 |
| 9000 | .754 / .783 | .982 / .984 / .07 / .85 | .771 / .995 / .08 / .61 | .742 / .742 / .11 / .60 | .590 / .592 / .11 / .55 | .876 / .921 / .09 / .72 | .754 / .761 / .06 / .66 | .956 / .956 / .03 / .95 | .610 / .610 / .08 / .45 | .772 / .772 / .12 / .53 | .496 / .501 / .04 / .28 |
| 10236 (final) | .763 / .791 | .990 / .992 / .07 / .85 | .804 / .995 / .12 / .61 | .747 / .747 / .10 / .60 | .595 / .597 / .13 / .58 | .867 / .926 / .10 / .73 | .751 / .764 / .05 / .68 | .966 / .966 / .02 / .96 | .620 / .620 / .07 / .47 | .786 / .786 / .11 / .55 | .517 / .522 / .06 / .31 |

Training took 3 h 8 min (57.6 samples/s). On the mixed validation set the round moves the model from .682 to .763
(.711 to .791 tie-aware): sokoban .740 to .966, invaders .675 to .786, mario .512 to .620, 2048 .411 to .517, racer
.663 to .747, tetris .713 to .751, breakout .487 to .595 (the noisiest column, 390 samples). Closed loop below.

<!-- run5-closed-loop -->
| game | `dagger1`, argmax | `sft_all1`, argmax | random | teacher | share of teacher gain: dagger1 / sft_all1 |
|---|---|---|---|---|---|
| snake | 107.9 (106 / 156) | 77.8 (78.5 / 121) | 1.0 (1 / 1) | 113.7 (108.5 / 161) | 0.95 / 0.68 |
| 2048 | 2170.0 (1962 / 6228) | 3174.2 (3164 / 5808) | 1021.0 (860 / 2424) | 19593.2 (20378 / 22068) | 0.06 / 0.12 |
| tetris | 1561.2 (1615 / 4710) | 1034.4 (1065 / 1970) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) | 0.09 / 0.06 |
| breakout | 1551.6 (1115 / 6140) | 611.2 (305 / 2825) | 495.9 (232.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) | 0.07 / 0.01 |
| flappy | 9.3 (7 / 31) | 8.9 (7 / 23) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) | 0.11 / 0.11 |
| invaders | 400.0 (400 / 400) | 400.0 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) | 1.00 / 1.00 |
| mario | 1169.6 (781 / 2891) | 1156.4 (837 / 2877) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) | 0.15 / 0.15 |
| pacman | 3209.4 (2275 / 8320, 1 capped) | 1035.6 (1040 / 2520) | 113.1 (60 / 360) | 7025.6 (7175 / 8080, 15 capped) | 0.45 / 0.13 |
| racer | 6704.1 (6711.3 / 6715, 1 capped) | 6211.2 (6225.4 / 6712.4, 15 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.65 / 6715.1) | 1.00 / 0.92 |
| sokoban | 102.3 (102 / 104) | 57.9 (101 / 104) | 6.6 (0 / 101) | 102.2 (102 / 104) | 1.00 / 0.54 |
<!-- /run5-closed-loop -->

Job 621962 finished at 18:34 (collection 47 min, training 3 h 8 min, closed loop 20 min). One DAgger round from
the model's own states: sokoban and racer reach the teacher (102.3 of 102.2, every level solved; 6704 of 6712, the
model laps in 15 of 16 episodes), snake 95 percent of the teacher (107.9 of 113.7, from 68 percent), pacman
tripled (1036 to 3209, one episode capped), breakout 2.5x (611 to 1552), tetris 1.5x (1034 to 1561). Unchanged:
invaders (already solved), mario (1170 vs 1156: a perception limit, the jump phase is not in a single frame, and
on-policy labels do not add what the frame does not show), flappy (9.3 vs 8.9: the correction step is a rare event,
a few hundred labelled instances in 40k frames against 60k hover frames at .99 agreement; a round that oversamples
the last steps before each death is the next move), and 2048 lower (2170 vs 3174; 16 episodes with max 6228, the
board reading has not improved and the number is within its noise). Read as the share of the teacher's gain over
random: seven games at or above 0.45 were four before the round.


## Run 6: eight games, snake and racer held out (`sft_hold8`, job 622160, hopper, from 16:08)

The base for the transfer experiment (the plan: fine-tune `sft_hold8/final` on 1k / 3k / 10k frames of
snake and racer against the base model on the same frames). Exactly the `sft_all1` recipe on the `sft_all1`
shards of the other eight games: 686k train records, 10,715 steps, same schedule. Columns for snake and racer are
empty by construction.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | .302 / .307 | .496 / .496 / .54 / .52 | - | - | .331 / .331 / .38 / .54 | .303 / .339 / .48 / .71 | .208 / .208 / .41 / .53 | .262 / .262 / .42 / .57 | .224 / .224 / .17 / .30 | .372 / .372 / .28 / .48 | .218 / .220 / .56 / .71 |
| 1500 | .433 / .438 | .876 / .880 / .08 / .60 | - | - | .655 / .655 / .05 / .46 | .205 / .245 / .09 / .05 | .275 / .275 / .02 / .11 | .237 / .237 / .06 / .06 | .552 / .552 / .13 / .33 | .361 / .361 / .02 / .07 | .268 / .268 / .02 / .06 |

Early observation, to be checked against the final row: at step 1500 this run has seen 12k samples per game
(`sft_all1` had 9.6k at the same step) and is far behind on the games it shares with `sft_all1`: pacman .205
against .677, tetris .275 against .574, sokoban .237 against .569, invaders .361 against .419, with near-uniform
outputs (mean confidence .05 to .11) where `sft_all1` was already committed; flappy (.876 vs .946), breakout (.655
vs .707) and mario (.552 vs .564) are close. Same code on the mirror (trainer at aa3656f), same shards, same
schedule up to the epoch length. At step 3000 nothing had moved (.434 / .441, loss flat at 1.19 since step 1000):
pacman, sokoban, 2048, invaders and tetris output the letter prior for every frame (pacman prob per letter .22 /
.29 / .24 / .25, argmax always the second letter, mean confidence .05), while flappy (.876), breakout (.655) and
mario (.556) had learned by step 1500 and stayed there. Job 622160 (hopper-30) was stopped at 17:12 and the run
resubmitted as 622258 with `--seed 1`, everything else identical (the stuck checkpoints are kept under
`ckpt/sft_hold8_seed0_stuck/`). If the second seed learns, the first was plateau luck; if it sticks too, the two
removed games are what pulls the hard games off the letter prior early in training (snake has the most decisive
targets in the roster), which is itself a transfer result and changes how the held-out base should be built.

Seed 1 (job 622258, from 17:41) reproduced seed 0 exactly in kind: at step 1500 .441 / .448, pacman .281, tetris
.276, sokoban .246, invaders .337, 2048 .262 with confidence .03 to .10 and the argmax sitting on the second and
third letters whatever the frame (2048: the second letter in 100 percent of samples), while flappy (.900), breakout
(.691) and mario (.545) score exactly what the most common action of each game scores when chosen by name (their
argmax per letter position is flat, so the choice follows the option text). So the eight-game run converges to a
text-only policy: it identifies the game from the option names and plays that game's marginal action, and never
opens the image pathway; `sft_all1` at the same step reads the frame for every game (pacman .677 with confidence
.48). Two seeds rule out luck. The difference to `sft_all1` is the two removed games: snake's labels have no
text-only shortcut at all (the BFS direction is uniform over the four arrows and depends only on the frame), so
snake supplies the largest image-dependent gradient early in training and opens the pathway for everyone; racer is
similar. The warmup length also differs (321 steps against 404, a side effect of the shorter epoch), which a control
with `--warmup 0.06` can rule out; hypothesis first, control when a slot is free. Stopped 622258 at 18:14
(checkpoints under `ckpt/sft_hold8_seed1_stuck/`).

Consequence for the transfer experiment: the held-out base keeps snake. Job 622333 (`sft_hold8b`) holds out racer
and pacman instead (pacman shares snake's four arrows, which makes the transfer question sharper), same recipe.

`sft_hold8b` (snake in, racer and pacman out; 10,709 steps, warmup 321 like the stuck runs) at step 1500:

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .620 / .637 | .967 / .975 / .07 / .79 | .826 / .965 / .15 / .57 | - | .705 / .705 / .10 / .41 | - | .497 / .497 / .07 / .33 | .545 / .545 / .07 / .40 | .542 / .542 / .10 / .35 | .494 / .494 / .06 / .19 | .380 / .380 / .04 / .23 |
| 3000 | .661 / .681 | .943 / .951 / .07 / .81 | .814 / .965 / .11 / .63 | - | .701 / .701 / .11 / .45 | - | .552 / .554 / .07 / .42 | .713 / .713 / .03 / .61 | .557 / .557 / .09 / .39 | .627 / .627 / .08 / .33 | .380 / .382 / .02 / .18 |
| 4500 | .656 / .688 | .975 / .982 / .08 / .79 | .714 / .969 / .06 / .59 | - | .703 / .703 / .10 / .41 | - | .560 / .560 / .06 / .42 | .739 / .739 / .05 / .63 | .550 / .550 / .09 / .37 | .659 / .659 / .09 / .36 | .345 / .347 / .05 / .20 |
| 6000 | .702 / .725 | .976 / .984 / .08 / .83 | .797 / .973 / .10 / .60 | - | .719 / .719 / .06 / .52 | - | .636 / .640 / .06 / .50 | .810 / .810 / .02 / .76 | .559 / .559 / .08 / .40 | .702 / .702 / .08 / .44 | .414 / .416 / .06 / .22 |
| 7500 | .700 / .730 | .978 / .984 / .09 / .81 | .743 / .979 / .08 / .62 | - | .715 / .715 / .08 / .48 | - | .659 / .663 / .07 / .53 | .856 / .856 / .03 / .85 | .546 / .546 / .07 / .40 | .685 / .685 / .07 / .43 | .418 / .420 / .04 / .23 |

Every game is off the prior (tetris .497, sokoban .545, invaders .494, 2048 .380, all near `sft_all1` at the same
step) with snake back in the mix, and the warmup is the same 321 steps as in the two stuck runs, so the warmup
confound is gone: **the mix needs a game whose labels cannot be predicted from the option text; snake is that game
here.** The rest of this section keeps the eight-game table for the record.



## Run 7: per-game label shift (`sft_all1_dpg`, job 622256, from 17:13)

`--label-delay 1 --no-delay-games 2048 sokoban`: the eight real-time games get the shifted label of run 3, the two
turn-based games keep the current decision (nothing moves there without input, so the deployed model has no
latency to absorb). Same shards, schedule and eval as runs 3 and 4 (13,442 steps); closed loop argmax at delay 1
and 0, no refs. Rows are against the per-game targets (shifted except 2048 and sokoban).

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .574 / .601 | .860 / .868 / .08 / .66 | .706 / .918 / .12 / .51 | .743 / .743 / .09 / .59 | .688 / .688 / .05 / .58 | .625 / .672 / .07 / .44 | .291 / .291 / .09 / .22 | .513 / .513 / .08 / .42 | .540 / .540 / .10 / .38 | .436 / .436 / .03 / .12 | .352 / .359 / .04 / .18 |
| 3000 | .627 / .648 | .917 / .925 / .10 / .64 | .772 / .908 / .14 / .52 | .766 / .766 / .22 / .46 | .691 / .691 / .10 / .43 | .698 / .759 / .08 / .50 | .403 / .403 / .02 / .27 | .650 / .650 / .06 / .49 | .547 / .547 / .13 / .32 | .482 / .482 / .03 / .24 | .359 / .361 / .09 / .21 |
| 4500 | .662 / .687 | .945 / .953 / .07 / .76 | .757 / .931 / .12 / .51 | .772 / .772 / .17 / .53 | .709 / .709 / .11 / .41 | .724 / .787 / .07 / .56 | .491 / .491 / .06 / .36 | .724 / .724 / .06 / .57 | .569 / .569 / .10 / .38 | .547 / .547 / .06 / .23 | .398 / .400 / .03 / .19 |
| 6000 | .673 / .703 | .940 / .948 / .07 / .77 | .691 / .921 / .07 / .55 | .790 / .790 / .16 / .57 | .717 / .717 / .03 / .56 | .749 / .808 / .06 / .63 | .538 / .538 / .06 / .35 | .795 / .795 / .02 / .72 | .550 / .550 / .08 / .38 | .550 / .550 / .04 / .29 | .427 / .429 / .04 / .23 |
| 7500 | .689 / .716 | .960 / .968 / .08 / .77 | .734 / .939 / .08 / .55 | .777 / .777 / .10 / .64 | .688 / .688 / .05 / .55 | .763 / .829 / .09 / .64 | .592 / .592 / .05 / .45 | .809 / .809 / .03 / .76 | .550 / .550 / .06 / .41 | .576 / .576 / .07 / .32 | .443 / .443 / .04 / .21 |
| 9000 | .710 / .729 | .965 / .973 / .09 / .76 | .826 / .954 / .19 / .52 | .782 / .782 / .18 / .52 | .712 / .712 / .09 / .44 | .770 / .824 / .10 / .58 | .621 / .621 / .10 / .40 | .853 / .853 / .04 / .78 | .557 / .557 / .09 / .38 | .574 / .574 / .10 / .21 | .448 / .451 / .05 / .19 |
| 10500 | .713 / .734 | .963 / .970 / .08 / .80 | .813 / .954 / .16 / .54 | .787 / .787 / .13 / .60 | .693 / .693 / .06 / .56 | .768 / .834 / .09 / .64 | .636 / .636 / .09 / .52 | .861 / .861 / .06 / .87 | .564 / .564 / .05 / .43 | .610 / .610 / .09 / .29 | .441 / .441 / .04 / .25 |
| 12000 | .727 / .747 | .965 / .973 / .10 / .78 | .806 / .949 / .15 / .54 | .801 / .801 / .13 / .61 | .696 / .696 / .07 / .52 | .792 / .843 / .08 / .65 | .665 / .665 / .06 / .55 | .914 / .914 / .02 / .91 | .567 / .567 / .09 / .39 | .615 / .615 / .08 / .32 | .453 / .455 / .05 / .24 |

(training in progress)

## Run 8: two frames as two images (`sft_all1_2fs`, job 622271, from 18:14)

`--two-frame --stack separate` with unshifted labels at delay 0: previous and current frame as two images with two
placeholders (twice the visual tokens, 34 samples/s instead of 57, 21 GB), so the model can compare them by
attention instead of through the patch convolution of run 4. Same shards and schedule as `sft_all1` (13,490 steps);
closed loop argmax at delay 0 with `--two-frame --stack separate`, no refs. Read against the `sft_all1` table;
mario and breakout are the readout.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1500 | .623 / .649 | .970 / .970 / .14 / .67 | .728 / .942 / .06 / .58 | .725 / .725 / .11 / .59 | .707 / .710 / .12 / .39 | .671 / .719 / .07 / .48 | .495 / .500 / .09 / .28 | .600 / .600 / .06 / .47 | .564 / .564 / .09 / .39 | .405 / .405 / .03 / .15 | .384 / .387 / .03 / .14 |
| 3000 | .675 / .716 | .988 / .988 / .08 / .82 | .649 / .961 / .11 / .59 | .792 / .792 / .17 / .56 | .721 / .723 / .06 / .53 | .735 / .832 / .11 / .56 | .629 / .635 / .09 / .43 | .757 / .757 / .05 / .65 | .564 / .564 / .10 / .39 | .531 / .531 / .09 / .30 | .394 / .394 / .04 / .20 |
| 4500 | .701 / .730 | .983 / .983 / .05 / .87 | .743 / .974 / .08 / .61 | .782 / .782 / .10 / .65 | .739 / .739 / .10 / .46 | .783 / .841 / .09 / .59 | .706 / .711 / .06 / .62 | .807 / .807 / .04 / .74 | .529 / .529 / .11 / .33 | .574 / .574 / .05 / .33 | .369 / .369 / .05 / .17 |

(training in progress; `sft_all1` at 1500 was .642 / .657, mario .564)

### What the second frame buys, per game (paired against `sft_all1` at the same step)

Both runs share shards, schedule, seed and eval set, so the two logs can be read step for step. Agreement of
`sft_all1_2fs` minus `sft_all1`, at the four eval points after the curves settle:

| step | breakout | tetris | pacman | invaders | 2048 | snake | sokoban | flappy | mario | racer |
|---|---|---|---|---|---|---|---|---|---|---|
| 6000 | +.082 | +.091 | +.016 | -.029 | -.007 | +.005 | +.063 | +.000 | -.033 | -.016 |
| 7500 | +.082 | +.063 | +.000 | -.017 | -.037 | +.024 | +.051 | +.000 | -.003 | -.005 |
| 9000 | +.122 | -.003 | +.030 | +.014 | -.007 | +.013 | +.000 | +.002 | -.025 | +.000 |
| 10500 | +.096 | +.010 | +.014 | +.019 | +.030 | -.021 | -.007 | -.002 | -.033 | -.019 |

Breakout gains at every point (+.096 mean) and is the largest effect in the table. Mario and racer lose at every
point (-.024 and -.010 mean). The other eight games change sign between points, so their differences are eval noise
at this sample size (371 to 2000 frames per game).

The split is not about how fast a game moves, it is about what moves. Breakout, tetris and pacman have a fixed
camera: the frame difference is the object that moved, so the second image hands the model a velocity it cannot
read from one frame (where the ball is going, how fast a piece is falling). Mario (`games/mario/pj_hook.js`, a
side-scrolling Infinite Mario) and racer (`games/racer/pj_hook.js`, a pseudo-3D track where the camera advances
along z) are the only two games whose camera translates, and they are exactly the two that lose. There the frame
difference is dominated by the global shift of the whole scene, and the model has to discount that shift before
any local motion means anything, so the second image costs more attention than it returns. Flappy is saturated at
.995 and cannot show either sign.

This also explains run 4: stacking the two frames through the patch convolution (`--stack temporal`) mixes the
translation into every patch embedding, which is the worst case for the two camera-moving games and gave a run
that looked flat overall. Separate images let the fixed-camera games win without forcing the moving-camera games
to pay in the same tensor.

## Handover: when the model asks the teacher for help

The System One contract says the confidence is supposed to mean something operationally, not just be calibrated on
a validation split. The test is to put the scripted teacher behind the model and hand it every step whose Jev
confidence falls below a threshold (`playjev.play --handover TAU`, tau 0 the model alone, tau 1.01 the teacher
alone). Sweeping tau traces a curve from the model's score to the teacher's, with the share of steps the teacher
actually took on the horizontal axis.

The curve on its own proves nothing: giving a third of the steps to a much stronger player raises the score whether
or not the trigger is any good. The control is `--handover-random RATE`, which hands over the same share of steps
chosen uniformly at random, so the two runs differ only in *which* steps the teacher got. A third control,
`--handover-invert`, uses the same threshold the other way round and hands over the steps whose confidence is
*above* it; if the trigger carries signal this has to land below the random control, not merely below the real one.

### `sft_all1`, matched pairs (jobs 1372654 and 1372977)

| game | rate | by confidence | at random, same rate |
|---|---:|---:|---:|
| Breakout | 0.16 | **3423** | 1257 |
| Breakout | 0.34 | **15139** | 6192 |
| Tetris | 0.02 | **1133** | 1119 |
| Tetris | 0.14 | **2976** | 1610 |
| Tetris | 0.33 | **9109** | 3177 |
| Tetris | 0.63 | **15156** | 8473 |
| Pacman | 0.16 | **1739** | 1594 |
| Pacman | 0.22 | **3573** | 2947 |
| Snake | 0.004 | **82.1** | 82.4 |
| Snake | 0.37 | **90.1** | 81.9 |
| Snake | 0.45 | **97.8** | 90.4 |
| Snake | 0.52 | **112.4** | 92.1 |

Confidence wins 11 of 12 pairs; the one it does not win (snake at 0.4 percent handed over) is 82.1 against 82.4,
which is four steps out of a thousand and inside the episode noise. The gap grows with the rate, which is what a
useful trigger looks like: the first steps it picks are the ones that matter most.

### `dagger1`, matched pairs (jobs 1373506 and 1373508 and the sweep they control)

The same experiment on the DAgger checkpoint, which is the one the demo ships. Rates are matched to within a
percentage point because the control is given the measured rate of the confidence run, not the threshold.

| game | rate | alone | by confidence | at random, same rate | teacher |
|---|---:|---:|---:|---:|---:|
| Tetris | 0.14 | 1459 | **7341** | 3167 | 15288 |
| Tetris | 0.26 | 1459 | **13581** | 3881 | 15288 |
| Breakout | 0.33 | 1486 | **6025** | 2810 | 16547 |
| Breakout | 0.49 | 1486 | **11696** | 4761 | 16547 |
| 2048 | 0.17 | 2071 | **4209** | 3104 | 19593 |
| 2048 | 0.80 | 2071 | **19534** | 14393 | 19593 |
| Mario | 0.35 | 1170 | **2920** | 1353 | 4229 |
| Mario | 0.87 | 1170 | **4369** | 3619 | 4229 |
| Pacman | 0.16 | 3318 | **2785** | 2093 | 7026 |
| Pacman | 0.22 | 3318 | **5725** | 3932 | 7026 |
| Flappy | 0.03 | 9.3 | 11.7 | 12.1 | 84 |
| Flappy | 0.15 | 9.3 | **56.8** | 20.2 | 84 |
| Snake | 0.40 | 101 | **105** | 94 | 114 |
| Snake | 0.53 | 101 | **119** | 98 | 114 |

Thirteen pairs out of fourteen. The sharpest one is Tetris at a quarter of the steps: picked by confidence those
steps are worth 13581, which is 89 percent of what the teacher scores playing every step itself, and the same share
of steps picked at random is worth 3881, which is 25 percent. The model keeps three quarters of the decisions and
gives up eleven percent of the score. Snake at half the steps scores 119 against the teacher's own 114, which is
the one place where the pair beats both of its members.

The one loss is Flappy at three percent of the steps, 11.7 against 12.1, on a game whose scores are single-digit
pipe counts; by fifteen percent it is 56.8 against 20.2. Pacman at sixteen percent is worth a note of its own:
both legs come in under the model playing alone (2785 and 2093 against 3318), so a handover that is too thin makes
Pacman worse whichever steps it takes. Two policies chasing different routes through the same maze interrupt each
other, and only at twenty-two percent does the confidence leg clear the model alone.

Games where the model already plays at the teacher's level carry no signal here. Invaders is a ceiling (random 215,
teacher 400 with every single episode at exactly 400, model also 400), so nothing can move. Sokoban after DAgger is
at the teacher's score already. Both are left out of the handover reading rather than counted as flat.

### Handing over the confident steps instead (job 1373555)

`play.py --handover-invert` keeps the threshold and reverses the test, so the teacher is given the steps the model
is *most* sure about and the model keeps the ones it is unsure about. If the gain came from the teacher simply
playing some of the steps, this would pay the same. If it came from picking which steps, this should pay nothing.

| game | rate | alone | by confidence | at random | inverted |
|---|---:|---:|---:|---:|---:|
| Tetris | 0.10 to 0.14 | 1459 | **7341** | 3167 | 1411 |
| Breakout | 0.31 to 0.33 | 1486 | **6025** | 2810 | 2131 |

Inverted Tetris scores 1411 where the model alone scores 1459: a tenth of the steps handed to a teacher that plays
Tetris eleven times better than the model buys nothing at all. Breakout inverted is 2131, below the 2810 that the
same share of randomly chosen steps is worth. The ordering is the one the claim predicts, confidence above random
above inverted, and the inverted leg sits at or under playing alone.

A sanity leg at threshold 0.9 inverted hands over 0.05 percent of Tetris steps and returns 1459.375, the model's
own score to the last digit, which says the harness is not leaking anything through the handover path itself.

## Transfer: what the other eight games are worth on a new one

`sft_hold8b` is trained on eight games with racer and pacman held out. The question is what that checkpoint is
worth as a starting point when a new game arrives: fine-tune the held-out game from `sft_hold8b` and, on the same
frames with the same recipe, from raw `Qwen3.5-0.8B-Base`. Agreement is with the teacher on 2000 validation frames
of that game, never trained on; score is 16 held-out episodes through the closed loop.

| game | starts from | frames | agreement | score | vs teacher |
|---|---|---:|---:|---:|---:|
| Racer | base | 0 | n/a | 1 | -0.04 |
| Racer | hold8 | 0 | n/a | 48 | -0.03 |
| Racer | base | 1000 | 0.392 | 48 | -0.03 |
| Racer | base | 3000 | 0.473 | 801 | 0.09 |
| Racer | base | 10000 | 0.529 | 616 | 0.06 |
| Racer | hold8 | 1000 | 0.669 | **4867** | **0.72** |
| Racer | hold8 | 3000 | 0.726 | 3608 | 0.52 |
| Racer | hold8 | 10000 | 0.759 | **6313** | **0.94** |
| Pacman | base | 0 | n/a | 53 | -0.01 |
| Pacman | hold8 | 0 | n/a | 84 | 0.00 |
| Pacman | base | 1000 | 0.242 | 53 | -0.01 |
| Pacman | base | 3000 | 0.263 | 50 | -0.01 |
| Pacman | base | 10000 | 0.241 | 53 | -0.01 |
| Pacman | hold8 | 1000 | 0.314 | 30 | -0.01 |
| Pacman | hold8 | 3000 | 0.344 | 55 | -0.01 |
| Pacman | hold8 | 10000 | **0.771** | 654 | 0.08 |

Racer is the clean one. A thousand frames from the eight-game checkpoint reaches 0.72 of the teacher; ten thousand
frames from the base model reaches 0.06. That is a tenfold difference in frames and the smaller number still wins
by twelve times the score. At ten thousand frames the eight-game start is at 0.94 of the teacher.

### The base arm answers with a constant, and that is the finding, not a bug

The base arm's agreement looked broken until the per-letter argmax was read next to it. Every base leg emits the
same answer on every validation frame. Which constant it picks is all that separates a leg that scores at chance
from one that looks like it learned something.

| run | agreement | confidence | argmax over the letter slots (2000 val frames) |
|---|---:|---:|---|
| `tr_pacman_1000_base` | 0.2415 | 0.042 | 1.00 / 0 / 0 / 0 |
| `tr_pacman_3000_base` | 0.2625 | 0.022 | 0 / 1.00 / 0 / 0 |
| `tr_pacman_10000_base` | 0.2410 | 0.009 | 0.996 / 0.004 / 0 / 0 |
| `tr_pacman_10000_base` lr 5e-5 | 0.2405 | 0.033 | 0 / 0 / 0 / 1.00 |
| `tr_pacman_10000_base` lr 1e-4 | 0.2555 | 0.019 | 0 / 0 / 1.00 / 0 |
| `tr_pacman_10000_hold8` | 0.7710 | 0.647 | 0.243 / 0.263 / 0.264 / 0.231 |
| `tr_racer_3000_base` | 0.4730 | 0.340 | 0.174 / 0.162 / 0.164 / 0.173 / 0.168 / 0.161 |
| `tr_racer_10000_base` lr 5e-5 | 0.4730 | 0.329 | 0.174 / 0.162 / 0.164 / 0.173 / 0.168 / 0.161 |
| `tr_racer_10000_base` lr 1e-4 | 0.4730 | 0.318 | 0.174 / 0.162 / 0.164 / 0.173 / 0.168 / 0.161 |
| `tr_racer_10000_base` lr 2e-5 | 0.5285 | 0.377 | 0.172 / 0.198 / 0.169 / 0.098 / 0.189 / 0.175 |
| `tr_racer_10000_hold8` | 0.7590 | 0.637 | 0.172 / 0.180 / 0.156 / 0.175 / 0.155 / 0.164 |

Confidence is `(pmax - 1/K) / (1 - 1/K)`, the margin of the winning letter over uniform.

Pacman's base legs collapse onto a **letter**. The probabilities stay flat (0.24 to 0.26 on each of the four slots,
margin 0.9 to 4 percent above uniform) and an infinitesimal constant bias decides the argmax, so the same slot wins
on all 2000 frames. Options are permuted per sample, so a constant letter is a uniformly random action: 0.24
agreement is chance, and the closed loop scores 53 against a random policy's 113.

Racer's base legs collapse onto an **action**. Three independently trained checkpoints (3000 frames at 2e-5, 10000
at 5e-5, 10000 at 1e-4) return argmax histograms identical to four decimals and agreement identical to four
decimals. That can only happen if all three emit the same action on every frame: the permutation then scatters it
across the six letters with its own fixed histogram, which is what the identical rows are. The action is `faster`,
whose share of the teacher's frames is 0.477, and the agreement is 0.473. The closed loop confirms it from the other
side: `tr_racer_3000_base` and `tr_racer_10000_base` at lr 1e-4 score 801.19375 with all 16 episodes capped and 81
skipped no-ops apiece, the same numbers to the last digit, because holding one key is a deterministic trajectory.

So the 801 in the transfer table is not small positive transfer. It is what this racer pays for holding the
accelerator, and the honest reading of the base column is that nothing in it reads the frame.

Two things say this is the measurement and not a broken pipeline. The same data, recipe, learning rate and step
count take `sft_hold8b` to 0.77 on pacman and 0.76 on racer, with flat argmax histograms and confidence above 0.6,
so nothing about the arm is unable to learn. And the learning rate does not decide it: 2e-5, 5e-5 and 1e-4 all land
on the same constant, and the one leg that moves off it (racer 10000 at 2e-5, argmax dipping to 0.098 on one slot,
agreement 0.529) still scores 616.

Ten thousand frames at three epochs is 468 optimizer steps. In 468 steps a raw base model learns which letters are
legal answers and, when the option text carries a marginal worth taking, which single answer is most often right.
It does not get as far as reading the frame. This is the same wall as the anchor-game finding: a mix without snake
never opens the image pathway either.

This is what the `sft_hold8_shuf` control is for. Shuffling the targets within each game keeps every game's action
marginal and destroys only the frame-to-action mapping, so a checkpoint trained on it carries the answer format and
the marginal and no game skill. If the shuffled start lands on the base arm's plateau, then `hold8 - shuf` is the
part that is actual transferred game knowledge and `shuf - base` is the part that is just knowing how to answer.

## Real-time latency (decided 2026-09-18)

The deployed model sees frame k and its answer is applied at step k+1 (one step, 83 to 100 ms in the real-time
games, which covers the 43 ms batch-1 inference). `playjev.play --delay 1` reproduces that in the closed loop; the
unmodified teachers collapse under it (snake 108 to 1, flappy 30 to 0 pipes, measured on the same harness). Training for it
needs no new data: `--label-delay 1` in `playjev/data.py` labels frame k with the teacher's decision at record k+1
of the same episode (consecutive steps only, each episode's last record dropped), and the validation metrics of a
delayed run are computed against those shifted targets.

Runs queued after `sft_all1`: `play_eval.pbs` on `sft_all1/final` at delay 1 (how much an undelayed model loses in
real time); `sft_all1_d1` (same shards, `--label-delay 1`, closed loop at delay 1 and 0); then the two-frame
temporal-stack variant with the delayed label. Turn-based games (2048, sokoban, tetris as stepped here) are trained
with the same shifted labels for a single model unless the per-game numbers say it hurts.

## Run 9: DAgger round 2 (`dagger2`, job 622353, hopper-30, collection from 21:57, training 4 h 20 min)

Why: round 1 moved eight of ten games and left breakout, tetris, 2048 and flappy far from the teacher, so the same
loop runs again with `dagger1` as the actor. Collection is identical to run 5 except for the checkpoint:
`playjev.collect --actor local --ckpt dagger1/final --epsilon 0.05`, 3 x 13,336 frames per game, seeds 700000 /
800000 / 900000, 4 to 5 minutes per game (snake 276 s, tetris 243 s, mario 348 s, sokoban 497 s). Teacher give-ups
appear in flappy only (51 to 56 per shard); every other game ends 0 episodes early, including sokoban, which is the
first round where the actor stops deadlocking it.

Training: one epoch from `dagger1/final` at lr 1e-5 on `sft_all1_a dagger1_a dagger2_a dagger2_b dagger2_c`,
13,796 steps at 56.6 samples/s, eval every 1500 on 4000 samples mixed across the five shards, closed loop argmax on
the same 16 held-out seeds. One knob changed besides the shards: `--boost-last 10 4`, which repeats the last ten
records of every episode shorter than 500 steps four times. That takes 733k records to 973,942 (883,000 train,
90,942 val). Read the step-0 row against run 5's final row: it is the same checkpoint on a different mixed
validation set, so .763 becomes .737.

| step | all | flappy | snake | racer | breakout | pacman | tetris | sokoban | mario | invaders | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | .737 / .767 | .997 / .997 / .07 / .85 | .780 / .982 / .12 / .60 | .764 / .764 / .13 / .62 | .581 / .581 / .15 / .59 | .813 / .911 / .08 / .69 | .740 / .742 / .03 / .66 | .824 / .824 / .11 / .91 | .638 / .638 / .08 / .49 | .804 / .804 / .12 / .57 | .467 / .469 / .06 / .32 |
| 1500 | .733 / .766 | .987 / .990 / .07 / .86 | .762 / .990 / .08 / .61 | .737 / .737 / .08 / .62 | .633 / .633 / .05 / .48 | .800 / .906 / .08 / .68 | .723 / .723 / .05 / .61 | .911 / .911 / .04 / .91 | .588 / .588 / .07 / .45 | .751 / .751 / .09 / .49 | .460 / .465 / .03 / .27 |
| 3000 | .740 / .774 | .995 / .995 / .08 / .83 | .770 / .987 / .09 / .59 | .759 / .759 / .14 / .54 | .629 / .629 / .07 / .54 | .782 / .909 / .11 / .68 | .723 / .723 / .08 / .60 | .916 / .916 / .03 / .92 | .628 / .628 / .10 / .45 | .788 / .788 / .11 / .52 | .444 / .448 / .05 / .31 |
| 4500 | .762 / .791 | .995 / .997 / .07 / .86 | .760 / .987 / .09 / .60 | .759 / .759 / .10 / .61 | .645 / .645 / .06 / .53 | .851 / .914 / .07 / .71 | .749 / .752 / .06 / .66 | .913 / .913 / .04 / .87 | .672 / .672 / .14 / .46 | .812 / .812 / .08 / .62 | .493 / .498 / .06 / .37 |
| 6000 | .761 / .797 | .992 / .995 / .07 / .86 | .734 / .990 / .05 / .60 | .772 / .772 / .11 / .62 | .700 / .700 / .07 / .54 | .820 / .922 / .08 / .69 | .752 / .752 / .05 / .65 | .866 / .866 / .06 / .88 | .697 / .697 / .14 / .48 | .798 / .798 / .13 / .55 | .505 / .509 / .06 / .38 |
| 7500 | .771 / .800 | .997 / .997 / .07 / .85 | .762 / .985 / .08 / .61 | .775 / .775 / .11 / .62 | .590 / .590 / .11 / .54 | .851 / .919 / .08 / .70 | .769 / .771 / .09 / .65 | .945 / .945 / .02 / .93 | .700 / .700 / .16 / .47 | .809 / .809 / .09 / .59 | .538 / .542 / .05 / .38 |
| 9000 | .768 / .805 | .992 / .992 / .07 / .86 | .714 / .990 / .08 / .60 | .764 / .764 / .13 / .59 | .657 / .657 / .04 / .52 | .820 / .916 / .10 / .69 | .759 / .762 / .06 / .65 | .928 / .928 / .04 / .95 | .720 / .720 / .17 / .49 | .804 / .804 / .13 / .57 | .547 / .552 / .06 / .37 |
| 10500 | .777 / .813 | .997 / .997 / .08 / .85 | .744 / .992 / .07 / .59 | .791 / .791 / .13 / .60 | .660 / .660 / .06 / .53 | .805 / .916 / .09 / .70 | .752 / .754 / .07 / .68 | .960 / .960 / .02 / .94 | .707 / .707 / .16 / .48 | .819 / .819 / .12 / .61 | .559 / .566 / .06 / .39 |
| 12000 | .789 / .821 | .995 / .995 / .07 / .85 | .772 / .990 / .09 / .59 | .777 / .777 / .12 / .60 | .681 / .681 / .06 / .56 | .815 / .916 / .09 / .70 | .783 / .786 / .05 / .69 | .980 / .980 / .02 / .98 | .702 / .702 / .14 / .50 | .830 / .830 / .12 / .60 | .573 / .577 / .04 / .41 |
| 13500 | .788 / .821 | .992 / .992 / .07 / .85 | .772 / .990 / .09 / .59 | .794 / .794 / .11 / .63 | .674 / .674 / .05 / .57 | .813 / .919 / .09 / .70 | .803 / .803 / .05 / .72 | .980 / .980 / .02 / .97 | .700 / .700 / .15 / .49 | .822 / .822 / .12 / .58 | .556 / .561 / .06 / .43 |
| 13796 (final) | .788 / .820 | .997 / .997 / .07 / .85 | .775 / .990 / .08 / .60 | .791 / .791 / .12 / .63 | .640 / .640 / .07 / .57 | .818 / .922 / .08 / .70 | .800 / .803 / .06 / .73 | .980 / .980 / .01 / .97 | .705 / .705 / .14 / .49 | .822 / .822 / .12 / .59 | .577 / .580 / .05 / .43 |

Validation agreement goes .737 to .788 on the mixed set, every column flat or up, snake flat at .775 with its
tie-aware number at .990. Closed loop:

<!-- run9-closed-loop -->
| game | `dagger2`, argmax | `dagger1`, argmax | random | teacher | share of teacher gain: dagger2 / dagger1 | mean length: dagger2 / dagger1 |
|---|---|---|---|---|---|---|
| snake | 89.5 (88.5 / 131) | 107.9 (106 / 156) | 1.0 (1 / 1) | 113.7 (108.5 / 161) | 0.79 / 0.95 | 348 / 455 |
| 2048 | 3385.8 (2948 / 7492) | 2170.0 (1962 / 6228) | 1021.0 (860 / 2424) | 19593.2 (20378 / 22068) | 0.13 / 0.06 | 268 / 198 |
| tetris | 4717.5 (4645 / 9920) | 1561.2 (1615 / 4710) | 161.9 (165 / 220) | 15287.5 (15340 / 16160, 16 capped) | 0.30 / 0.09 | 656 / 334 |
| breakout | 2711.9 (1810 / 8700) | 1551.6 (1115 / 6140) | 495.9 (232.5 / 2110) | 16546.6 (17850 / 22045, 16 capped) | 0.14 / 0.07 | 336 / 246 |
| pacman | 3701.9 (2370 / 8260, 3 capped) | 3209.4 (2275 / 8320, 1 capped) | 113.1 (60 / 360) | 7025.6 (7175 / 8080, 15 capped) | 0.52 / 0.45 | 713 / 613 |
| flappy | 13.8 (13 / 30) | 9.3 (7 / 31) | 0.0 (0 / 0) | 84.0 (84 / 84, 16 capped) | 0.16 / 0.11 | 323 / 247 |
| mario | 1763.9 (1526 / 5216) | 1169.6 (781 / 2891) | 613.2 (627 / 1009) | 4228.6 (5224.5 / 5485) | 0.32 / 0.15 | 66 / 46 |
| invaders | 400.0 (400 / 400) | 400.0 (400 / 400) | 215.0 (215 / 290) | 400.0 (400 / 400) | 1.00 / 1.00 | 180 / 190 |
| racer | 6706.7 (6712.2 / 6714, 1 capped) | 6704.1 (6711.3 / 6715, 1 capped) | 238.3 (236.1 / 343.7, 16 capped) | 6711.7 (6711.7 / 6715.1) | 1.00 / 1.00 | 1448 / 1453 |
| sokoban | 102.1 (102 / 104) | 102.3 (102 / 104) | 6.6 (0 / 101) | 102.2 (102 / 104) | 1.00 / 1.00 | 50 / 54 |

Mean share of the teacher gain 0.488 to 0.535. Tetris triples (0.09 to 0.30), mario doubles (0.15 to 0.32), 2048
doubles, breakout doubles, flappy and pacman move; the three saturated games hold. Snake falls, 107.9 to 89.5, and
that is the only column that moves the wrong way.

### The snake fall is a real regression, not 16-episode noise

`dagger1` and `dagger2` share 14 of their 16 held-out seeds. The two sets are not identical because `playjev.play`
runs 8 pages and hands the next unused seed to whichever page finishes first, so the 16 recorded episodes are the
first 16 to end and a faster run reaches higher seeds (`dagger1` covers 5000 to 5016, `dagger2` 5000 to 5018, both
skipping 5008). On the 14 in common, `dagger2` is lower on 12, higher on 1, level on 1; the paired
difference is -17.4 on average with sd 22.5. A two-sided sign-flip permutation test over 200,000 permutations puts
that at p = 0.0073. The mean episode length falls with the score, 455 steps to 348, which is what a snake
regression looks like: the model dies earlier, so it eats less.

### What `--boost-last 10 4` actually touches

The flag's docstring says it reweights "a few hundred frames among tens of thousands". That premise is false in
eight of ten games. Measured on the `dagger2_a/b/c` shards, per game: share of episodes shorter than the 500-step
threshold, median episode length, and the extra training weight the boost adds.

| game | boosted share | median episode | extra weight |
|---|---|---|---|
| mario | 0.99 | 53 | +34.5% |
| flappy | 1.00 | 110 | +21.8% |
| sokoban | 0.93 | 75 | +19.6% |
| 2048 | 1.00 | 170 | +14.9% |
| invaders | 1.00 | 187 | +13.9% |
| snake | 0.98 | 174 | +13.5% |
| tetris | 0.96 | 284 | +8.9% |
| breakout | 0.87 | 277 | +7.8% |
| pacman | 0.69 | 265 | +4.9% |
| racer | 0.50 | 834 | +1.6% |

The 500-step threshold is not a filter in eight games, it is a blanket. Raising it to each game's own `max_steps`
cap was considered and dropped: the audit says that pushes the boosted share to 1.00 everywhere, which removes the
only two games (racer 0.50, pacman 0.69) where the flag still selects anything.

### Three instruments, one disagreement

Relabelling both runs' recordings through the real game with the teacher watching
(`scripts/relabel_replays.py`) gives on-policy agreement on the states the model actually visits. It rises for all
ten games from `dagger1` to `dagger2`, and snake rises the most.

| game | on-policy agreement d1 to d2 | ECE d1 to d2 | closed loop |
|---|---|---|---|
| snake | .722 to .824 (+.102) | .291 to .285 | 108 to 90 |
| invaders | .840 to .896 | .138 to .157 | flat at the cap |
| 2048 | .475 to .556 | .024 to .034 | 2170 to 3386 |
| tetris | .680 to .759 | .028 to .049 | 1561 to 4718 |
| racer | .719 to .751 | .051 to .068 | flat at the cap |
| mario | .593 to .612 | .070 to .082 | 1170 to 1764 |
| breakout | .468 to .483 | .207 to .164 | 1552 to 2712 |
| pacman | .846 to .849 | .130 to .127 | 3209 to 3702 |
| flappy | .949 to .951 | .044 to .041 | 9 to 14 |
| sokoban | .990 to 1.000 | .005 to .004 | flat at the cap |

Snake gains 10 agreement points on its own states while losing 17 percent of its score. The mechanism is that
agreement is averaged over visited frames and the visited frames changed: total steps over the same 16 episodes
fell 7275 to 5568, so the average shifts toward the easy early game, where a three-segment snake on an empty board
has almost no way to be wrong. Validation agreement is blind in the other direction, .780 to .775, flat. Only the
closed loop sees it. For a fatal-on-mistake game the two agreement instruments can move against the score, and
on-policy agreement can move the wrong way with confidence.

### The single-variable ablation (jobs 623073, 623074)

The shards and the boost both changed between the two rounds, so the boost is separated from the data with
`SKIP_COLLECT=1 DATA_RUN=dagger2`, which retrains on the exact shards `dagger2` used and costs no collection time.

    dagger2nb   the same run with --boost-last removed
    dagger2ns   --boost-last 10 4 --boost-games <the nine games that are not snake>

`--boost-games` is new in `playjev/data.py` and `playjev/train_sft.py`: it limits `--boost-last` to a subset and
leaves the rest of the roster at weight one. Both jobs start from `dagger1/final` at lr 1e-5 on the five `dagger2`
shards, 4.5 h each. If `dagger2nb` reproduces the tetris and mario gains, the boost buys nothing and goes; if it
loses them, the gains are the boost's and `dagger2ns` is the shipping configuration.

### Two limbs that are dead, so they do not get tried again

**Collection depth does not sort the outcome.** The ratio of the collection's median episode length to the deployed
mean length is flappy 0.34, pacman 0.37, tetris 0.43, snake 0.50, racer 0.58, 2048 0.63, mario 0.80, breakout 0.82,
invaders 1.04, sokoban 1.51. Snake sits mid-pack and the shallowest game in the roster improved.

**The collection epsilon is not the discriminator either.** Snake was the only game in the roster with no `collect`
block in its `pj.json`, so a bare `python -m playjev.collect snake` fell through to the generic 0.1 while the other
fatal-on-mistake games declare 0.02. It now declares `{"collect": {"epsilon": 0.02}}` for consistency, but that
changes none of the numbers above: `hpc/train_sft.pbs` passes `--epsilon` explicitly on every path except sokoban's
teacher round, and both DAgger rounds ran at `COLLECT_EPS=0.05`.

## Motion in a single frame (jobs 622710, 622711, 622712, breakout)

The Jev contract is one forward pass per frame, and a still frame carries position but not velocity: the same
picture of a breakout ball appears rising and falling. `playjev/motion.py` folds the k = 3 most recent frames into
one image, so the composite still costs one image and 182 visual tokens and the contract is untouched. Two
composers, against a single-frame control:

    plain  the control, raw current frame
    ghost  the pixels that moved are pasted back at rising alpha (0.30 for t-2, 0.60 for t-1, current solid),
           masked by a 25/255 per-pixel difference, so only movement ghosts and direction reads off the brightness
           gradient
    rgbt   the three frames become the R, G and B channels of one image; static pixels stay grey, moving ones
           leave a coloured fringe whose colour order gives the direction

The composites are pre-rendered into shards, so the trainer needs no flag of its own and `scripts/play_motion.py`
composes the same way at play time. All three arms: breakout only, the same three `sft_all1` shards, one epoch from
the 0.8B base, 1357 steps, the same 16 held-out seeds, cap 1500.

| arm | closed loop | agreement | ECE | mean length |
|---|---|---|---|---|
| plain | 642.8 (400 / 2410) | .692 | .067 | 135 |
| ghost | 878.4 (542.5 / 3285) | .692 | .037 | 134 |
| rgbt | 532.8 (352.5 / 2145) | .749 | .070 | 118 |

Ghost is worth 37 percent of the score over the control at identical agreement, with ECE cut from .067 to .037.
Rgbt is the opposite shape: it wins the offline metric by 6 points and loses 17 percent of the score. The two
results together say that on breakout, agreement against the teacher is not what the score is made of, and a
composer can buy agreement by making the frames easier to imitate while making them harder to act on.
