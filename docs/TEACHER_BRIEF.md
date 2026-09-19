# Brief for teacher-policy agents (PlayJev)

A teacher plays one game well using the hook's `info()` (internal state the model never sees) and returns a
probability vector over the game's actions. Its rollouts become the supervised targets for the vision model.
Repo root: this checkout, venv `source .venv/bin/activate`.

Read first: `docs/HARNESS.md` (contract), `games/<id>/NOTES.md` and `games/<id>/pj_hook.js` (what `info()`
contains for your game and what one step means), `playjev/teachers/__init__.py` and `playjev/teachers/snake.py`
(the reference teacher), `playjev/teacher_eval.py` and `playjev/collect.py` (how teachers are run).

## Contract

- File `playjev/teachers/<id>.py` with a class `<Name>Teacher(Teacher)`; register it in the `_REGISTRY` dict of
  `playjev/teachers/__init__.py` (one line; that dict is the only shared file you touch).
- `reset()` at episode start, `act(obs) -> list[float]` each step, where `obs["info"]` is the hook's info() and
  `obs["score"]`, `obs["t"]`, `obs["steps"]` are available. Return a distribution over `self.actions` (order given at
  construction; use `self.idx[name]`). Sum to 1, no NaNs.
- Soft targets are wanted, not just argmax: put most of the mass on the best action(s), a small share on other
  acceptable actions, zero on actions that lose (die, waste the move) when you can tell. Ties split evenly. Snake
  uses 0.9 / 0.1; use the same convention unless the game calls for something else and say why.
- Pure Python, no per-step cost above a few milliseconds on average (the env runs at 300 to 1000 steps/s across
  8 pages; a teacher that takes 50 ms per step halves throughput). Search-based teachers (2048 expectimax, tetris
  placement search) must fit this budget; cache per-episode plans where the game is turn-based.
- Stateful per episode is fine (previous positions, a plan of key presses toward a chosen placement).
- Optional `giveup() -> bool`: return True once the episode cannot be won any more (deadlocked board, unrecoverable
  state); the collector ends the episode early so hundreds of fallback labels are not recorded. Keep it conservative:
  a wrong True throws away a live episode.
- Per-game collection defaults go in `games/<id>/pj.json` as `"collect": {"epsilon": 0.03}` (random-action rate used
  by `playjev.collect` when no `--epsilon` is passed). Choose it so off-teacher states are covered but the share of
  fallback labels stays small (say what share you measured in TEACHER.md).

## Deliverables

1. The teacher file plus the registry line.
2. `python -m playjev.teacher_eval <id> --pages 8 --episodes 16` output: mean, median and max score, mean episode
   length, capped count. Compare against the random policy (`python -m playjev.play <id> --policy random`) and
   against whatever human or published numbers you know for the game; the teacher should be far above random and
   at least a competent human.
3. `python -m playjev.collect <id> --steps 2000 --shard smoke --epsilon 0.1` runs clean; inspect 3 frames with their
   teacher_probs and confirm the labels make sense against the picture.
4. `games/<id>/TEACHER.md`: the algorithm, its hyperparameters, the numbers above, known failure modes, cost per step.

Rules: only `playjev/teachers/<id>.py`, the registry line, `games/<id>/TEACHER.md`. Do not edit hooks or the driver;
if `info()` lacks something you need, write exactly what in TEACHER.md and in your final report (the hook owner adds
it). Never kill processes by name or pattern on this shared machine. Budget 60 minutes per game. Final report: the
algorithm in three lines, the eval numbers, cost per step, and any info() request.
