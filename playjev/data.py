"""Teacher-labelled SFT data: records written by `python -m playjev.collect`, one sample per decision.

A sample is the frame (plus the previous frame when two frames per state are used), the game's option list in a
random order (a fresh permutation per sample, the teacher's soft target permuted the same way), and the letter
position of the teacher's argmax. The prompt is rendered with the same `build_plain_prompt` the inference model
uses, so train and test strings are identical by construction. Descriptions come from the game's hook (the
records carry the action names only).

    python -m playjev.data snake            # count records, split sizes, one rendered sample
"""
from __future__ import annotations

import functools
import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .model import DEFAULT_INSTRUCTIONS, build_plain_prompt, stack_temporal_patches

ROOT = Path(__file__).resolve().parents[1]  # same as env.ROOT, without importing playwright
GAMES_DIR = ROOT / "games"
DATA_ROOT = ROOT / "data"
VAL_MOD = 10  # records whose seed % VAL_MOD == 0 are validation, never trained on


@dataclass
class Record:
    game: str
    shard_dir: Path
    frame: str  # relative to shard_dir
    prev_frame: str | None
    names: tuple[str, ...]  # action names in the hook's order
    probs: list[float]  # teacher's soft target over names
    teacher_action: int
    seed: int
    history: tuple[str, ...] = ()  # the moves already made in this episode, oldest first (see episode_history)


def all_game_options(game: str) -> list[dict]:
    """Every option the game declares, in the hook's order. Source: runs/bench/<game>/actions.json when present,
    else the `name: '...', description: '...'` pairs in the game's pj_hook.js."""
    path = ROOT / "runs" / "bench" / game / "actions.json"
    if path.exists():
        return json.loads(path.read_text())
    src = (GAMES_DIR / game / "pj_hook.js").read_text()
    pairs = re.findall(r"""name:\s*(['"])(.*?)\1\s*,\s*description:\s*(['"])(.*?)\3""", src)
    return [{"name": n, "description": d} for _, n, _, d in pairs]


def game_options(game: str, names: Sequence[str]) -> list[dict]:
    """Option dicts (name, description) in the given order."""
    opts = all_game_options(game)
    by = {o["name"]: o for o in opts}
    missing = [n for n in names if n not in by]
    if missing:
        raise ValueError(f"{game}: no description for actions {missing}; known {sorted(by)}")
    return [{"name": n, "description": by[n]["description"]} for n in names]


# --------------------------------------------------------------------------------- option augmentation
# The option list of a game is the same k names in the same wording for all of its 86k frames, so a model can learn
# the letter mapping instead of reading the list. These four rewrites keep the question answerable from the frame
# and make the list worth reading: an option from a mechanic this game does not have, one fewer option, a synonym
# for a name, another wording of the ask.
DIRECTIONS = ("up", "down", "left", "right")
ACTION_CLASS = {"stay": "idle", "wait": "idle", "noop": "idle", "none": "idle", "flap": "rise", "jump": "rise",
                "faster": "speed", "slower": "speed", "rotate": "piece", "drop": "piece"}
SYNONYMS = {"up": ("move up", "go up"), "down": ("move down", "go down"), "left": ("move left", "go left"),
            "right": ("move right", "go right"), "stay": ("hold still", "stay put"), "wait": ("hold", "sit still"),
            "noop": ("no move", "hold position"), "none": ("no input", "leave it"), "flap": ("flap once", "beat the wings"),
            "jump": ("hop", "leap"), "faster": ("accelerate", "speed up"), "slower": ("brake", "slow down"),
            "rotate": ("turn the piece", "spin the piece"), "drop": ("hard drop", "send it down")}
PARAPHRASES = ("Which move should the player make next?", "What should the player do next?",
               "Choose the player's next action.", "Which of these moves comes next?",
               "Pick the move to play from this state.")


def action_class(name: str) -> str:
    """A direction, or a compound holding one, is `move`: every game has those, so they are never a distractor."""
    if any(w in name.split() for w in DIRECTIONS):
        return "move"
    return ACTION_CLASS.get(name, "other")


@functools.lru_cache(maxsize=1)
def _roster_actions() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[tuple[str, str], ...]]]:
    """(game -> its action names, class -> the (name, description) pairs the roster uses for that class)."""
    names, pool = {}, {}
    for gdir in sorted(p for p in GAMES_DIR.iterdir() if (p / "pj_hook.js").exists()):
        opts = all_game_options(gdir.name)
        names[gdir.name] = tuple(o["name"] for o in opts)
        for o in opts:
            pool.setdefault(action_class(o["name"]), {}).setdefault(o["name"], o["description"])
    return names, {c: tuple(sorted(d.items())) for c, d in pool.items()}


@dataclass
class GameAug:
    """Per-sample probabilities. 0 everywhere (the default) renders exactly what the released model trained on."""
    distractor: float = 0.0
    prune: float = 0.0
    paraphrase: float = 0.0
    rename: float = 0.0
    max_distractors: int = 2

    @property
    def on(self) -> bool:
        return max(self.distractor, self.prune, self.paraphrase, self.rename) > 0

    @property
    def extra_slots(self) -> int:
        return self.max_distractors if self.distractor > 0 else 0


def apply_game_aug(rng: random.Random, game: str, opts: list[dict], target: list[float], teacher: int,
                   aug: GameAug) -> tuple[list[dict], list[float], int, str | None]:
    """Rewrite one sample's option list. Returns (options, target, teacher index, instruction override)."""
    items = [[dict(o), t, i == teacher] for i, (o, t) in enumerate(zip(opts, target))]
    if len(items) >= 3 and rng.random() < aug.prune:
        # drop one clearly-worse option and renormalise below: with a softmax teacher the reduced question has the
        # same answer and the same order among the survivors. Near-ties are never dropped, so the answer cannot move.
        best = max(t for _, t, _ in items)
        cand = [i for i, (_, t, is_t) in enumerate(items) if not is_t and t < 0.5 * best]
        if cand:
            items.pop(rng.choice(cand))
    if rng.random() < aug.distractor:
        roster, pool = _roster_actions()
        blocked = {action_class(n) for n in roster.get(game, ())} | {"move"}
        present = {it[0]["name"] for it in items}
        cand = [(n, d) for c, pairs in pool.items() if c not in blocked for n, d in pairs if n not in present]
        if cand:
            rng.shuffle(cand)
            for n, d in cand[: 1 if rng.random() < 0.7 else aug.max_distractors]:
                items.append([{"name": n, "description": d}, 0.0, False])
    if rng.random() < aug.rename:
        used = {it[0]["name"] for it in items}
        for it in items:
            alts = [x for x in SYNONYMS.get(it[0]["name"], ()) if x not in used]
            if alts:
                new = rng.choice(alts)
                used.discard(it[0]["name"]); used.add(new); it[0]["name"] = new
    total = sum(t for _, t, _ in items)
    out_t = [t / total for _, t, _ in items] if total > 0 else [1 / len(items)] * len(items)
    t_idx = next(i for i, (_, _, is_t) in enumerate(items) if is_t)
    instr = rng.choice(PARAPHRASES) if rng.random() < aug.paraphrase else None
    return [it[0] for it in items], out_t, t_idx, instr


def shift_labels(records: list[Record], episode_keys: list[tuple], steps: list[int], delay: int) -> list[Record]:
    """Real-time labels: the deployed model sees frame k and its answer is applied `delay` steps later, so the target
    for frame k becomes the teacher's decision at record k + delay of the same episode (consecutive steps only; the
    last `delay` records of every episode, and any record without a consecutive successor, are dropped)."""
    by_ep: dict[tuple, dict[int, int]] = {}
    for i, (key, step) in enumerate(zip(episode_keys, steps)):
        by_ep.setdefault(key, {})[step] = i
    out = []
    for i, (key, step) in enumerate(zip(episode_keys, steps)):
        j = by_ep[key].get(step + delay)
        if j is None:
            continue
        r = records[i]
        out.append(Record(game=r.game, shard_dir=r.shard_dir, frame=r.frame, prev_frame=r.prev_frame, names=r.names,
                          probs=records[j].probs, teacher_action=records[j].teacher_action, seed=r.seed,
                          history=r.history))
    return out


def boost_last_steps(keys: list[tuple], steps: list[int], seeds: list[int], last_k: int, times: int,
                     short_than: int = 500) -> list[int]:
    """Indices to append so that the last `last_k` records of every short episode appear `times` times in training.
    Short (fewer than `short_than` steps; every cap in the roster is at least 500) means the episode ended by a death,
    a solved level or a teacher give-up, so its last steps hold the corrections a DAgger round is collected for and
    which are otherwise a few hundred frames among tens of thousands. Validation seeds are never boosted."""
    by_ep: dict[tuple, list[tuple[int, int]]] = {}
    for i, (key, step) in enumerate(zip(keys, steps)):
        by_ep.setdefault(key, []).append((step, i))
    extra: list[int] = []
    for key, items in by_ep.items():
        length = max(st for st, _ in items) + 1
        if length >= short_than or seeds[items[0][1]] % VAL_MOD == 0:
            continue
        extra.extend(i for st, i in items if st >= length - last_k for _ in range(times - 1))
    return extra


# ----------------------------------------------------------------------------------- moves already made
# A still frame carries position but not velocity, and part of the velocity is the player's own: the paddle's
# drift, whether the bird was flapped, how long Mario has been running. That part is recoverable from the moves
# already made, which the records carry as `taken_action` and which cost no visual tokens to pass along.
#
# Two rules the text has to keep. It names moves, never letters: the option order is reshuffled per sample, so a
# letter would mean something different in the history than in the list. And it carries `taken_action`, what was
# actually played, never `teacher_action`: the teacher's choice at the previous steps is the teacher's policy,
# and a model given that reads the answer off the history instead of the frame.
HISTORY_PREFIX = "Moves already made, oldest first:"
HISTORY_PAD = "-"  # this step is before the episode had that many moves; keeps the line a fixed length


def render_history(moves: Sequence[str]) -> str:
    """The history line as the model sees it. A fixed number of slots whatever the step, so the line's length
    never tells the model how far into the episode it is."""
    return f"{HISTORY_PREFIX} {', '.join(moves)}"


def episode_history(keys: Sequence[tuple], steps: Sequence[int], taken: Sequence[int],
                    names: Sequence[Sequence[str]], k: int) -> list[tuple[str, ...]]:
    """For every record, the names of the k moves played before it in its own episode, oldest first, padded at
    the front when the episode is younger than k steps. A record whose predecessor is missing from the shard
    gets padding there rather than a move from further back, so the line never claims a move that did not
    immediately precede this frame."""
    at: dict[tuple, dict[int, int]] = {}
    for i, (key, step) in enumerate(zip(keys, steps)):
        at.setdefault(key, {})[step] = i
    out = []
    for i, (key, step) in enumerate(zip(keys, steps)):
        row = []
        for back in range(k, 0, -1):
            j = at[key].get(step - back)
            a = taken[j] if j is not None else None
            row.append(names[j][a] if (j is not None and a is not None and 0 <= a < len(names[j])) else HISTORY_PAD)
        out.append(tuple(row))
    return out


def load_records(games: Sequence[str], data_root: Path = DATA_ROOT, shards: Sequence[str] | None = None,
                 label_delay: int = 0, no_delay_games: Sequence[str] = (), boost_last: tuple[int, int] | None = None,
                 boost_games: Sequence[str] = (), history: int = 0) -> list[Record]:
    """`no_delay_games` keep their labels unshifted even when `label_delay` is set: the turn-based games (2048,
    sokoban) do nothing until the player acts, so the deployed model has no latency to absorb there. `boost_last`
    = (k, times) repeats the last k records of every short episode `times` times (see boost_last_steps);
    `boost_games`, when non-empty, limits that to those games and leaves the rest of the roster untouched."""
    out: list[Record] = []
    for game in games:
        delay = 0 if game in no_delay_games else label_delay
        gdir = data_root / game
        if not gdir.exists():
            raise FileNotFoundError(f"no data for {game} under {data_root}")
        for shard_dir in sorted(p for p in gdir.iterdir() if (p / "records.jsonl").exists()):
            if shards and shard_dir.name not in shards:
                continue
            recs, keys, steps, taken = [], [], [], []
            with open(shard_dir / "records.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    recs.append(Record(game=r["game"], shard_dir=shard_dir, frame=r["frame"], prev_frame=r.get("prev_frame"),
                                       names=tuple(r["actions"]), probs=r["teacher_probs"], teacher_action=r["teacher_action"],
                                       seed=r["seed"]))
                    keys.append((r["seed"], r["episode"])); steps.append(r["step"]); taken.append(r.get("taken_action"))
            if history:  # before boosting and shifting, which duplicate and drop records
                for rec, h in zip(recs, episode_history(keys, steps, taken, [rec.names for rec in recs], history)):
                    object.__setattr__(rec, "history", h)
            if boost_last and (not boost_games or game in boost_games):
                for i in boost_last_steps(keys, steps, [r.seed for r in recs], *boost_last):
                    recs.append(recs[i]); keys.append(keys[i]); steps.append(steps[i])
            out.extend(shift_labels(recs, keys, steps, delay) if delay else recs)
    return out


def split_records(records: Sequence[Record]) -> tuple[list[Record], list[Record]]:
    train = [r for r in records if r.seed % VAL_MOD != 0]
    val = [r for r in records if r.seed % VAL_MOD == 0]
    return train, val


class SFTDataset(Dataset):
    """Decodes the frame(s) and draws the option permutation. Training draws a fresh permutation every time;
    validation uses a permutation fixed by the sample index so repeated evals compare like with like."""

    def __init__(self, records: Sequence[Record], two_frame: bool = False, long_side: int | None = None,
                 train: bool = True, seed: int = 0, aug: GameAug | None = None,
                 instructions: str = DEFAULT_INSTRUCTIONS):
        self.records = list(records)
        self.two_frame, self.long_side, self.train, self.seed = two_frame, long_side, train, seed
        self.aug = aug if (aug is not None and aug.on and train) else None  # validation is never augmented
        self.instructions = instructions
        self.options = {}  # game -> options in the record order (all records of a game share the order)
        for r in self.records:
            if r.game not in self.options:
                self.options[r.game] = game_options(r.game, r.names)
            elif tuple(o["name"] for o in self.options[r.game]) != r.names:
                raise ValueError(f"{r.game}: action order differs between shards")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, shard_dir: Path, rel: str) -> Image.Image:
        im = Image.open(shard_dir / rel).convert("RGB")
        if self.long_side and max(im.size) != self.long_side:
            s = self.long_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        return im

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        k = len(r.names)
        # train: torch.initial_seed() differs per DataLoader worker and per epoch, so forked workers do not repeat
        # each other's permutations; val: fixed by the index so repeated evals are comparable
        rng = random.Random(torch.initial_seed() + i) if self.train else random.Random(self.seed * 1_000_003 + i)
        opts, target, teacher, instr = self.options[r.game], r.probs, r.teacher_action, None
        if self.aug is not None:  # draws nothing when off, so an unaugmented run reproduces byte for byte
            opts, target, teacher, instr = apply_game_aug(rng, r.game, list(opts), list(target), teacher, self.aug)
            k = len(opts)
        perm = list(range(k))
        rng.shuffle(perm)  # perm[pos] = original option index shown at letter position pos
        item = {"image": self._image(r.shard_dir, r.frame), "options": [opts[j] for j in perm],
                "target": [target[j] for j in perm], "teacher_pos": perm.index(teacher), "game": r.game,
                "n_opts": k, "kind": "game", "state_text": render_history(r.history) if r.history else None,
                "instructions": instr or self.instructions}
        if self.two_frame:
            # at an episode start there is no previous frame: repeat the current one (a still, like a single image)
            item["prev"] = self._image(r.shard_dir, r.prev_frame) if r.prev_frame else item["image"]
        return item


class Collate:
    """Renders prompts and runs the processor. Lives in the DataLoader workers (the processor is the slow part)."""

    def __init__(self, processor, two_frame: bool = False, stack: str = "temporal", instructions: str = DEFAULT_INSTRUCTIONS):
        self.processor, self.two_frame, self.stack, self.instructions = processor, two_frame, stack, instructions
        ip = processor.image_processor
        self.patch_size, self.temporal_patch_size = ip.patch_size, ip.temporal_patch_size

    def __call__(self, items: list[dict]) -> dict:
        two = self.two_frame and all(it.get("prev") is not None for it in items)
        has_img = [it.get("image") is not None for it in items]
        if any(has_img) != all(has_img):
            raise ValueError("a batch mixes frame states with text states; draw batches with TypeBatchSampler")
        n_ph = (2 if (two and self.stack == "separate") else 1) if all(has_img) else 0
        texts = [build_plain_prompt(it["options"], it.get("instructions") or self.instructions, n_ph,
                                    it.get("state_text")) for it in items]
        if not all(has_img):
            enc = self.processor(text=texts, return_tensors="pt", padding=True)
        elif not two:
            enc = self.processor(text=texts, images=[it["image"] for it in items], return_tensors="pt", padding=True)
        elif self.stack == "separate":
            enc = self.processor(text=texts, images=[[it["prev"], it["image"]] for it in items], return_tensors="pt", padding=True)
        else:
            cur = [it["image"] for it in items]
            prev = [it["prev"] if it["prev"].size == it["image"].size else it["prev"].resize(it["image"].size, Image.BILINEAR) for it in items]
            enc = self.processor(text=texts, images=cur, return_tensors="pt", padding=True)
            prev_pv = self.processor.image_processor(prev, return_tensors="pt")["pixel_values"]
            enc["pixel_values"] = stack_temporal_patches(prev_pv, enc["pixel_values"], self.temporal_patch_size, self.patch_size)
        kmax = max(it["n_opts"] for it in items)
        targets = torch.zeros(len(items), kmax)
        for row, it in enumerate(items):
            targets[row, : it["n_opts"]] = torch.tensor(it["target"], dtype=torch.float32)
        batch = {k: v for k, v in enc.items() if hasattr(v, "shape")}
        batch["targets"] = targets
        batch["n_opts"] = torch.tensor([it["n_opts"] for it in items])
        batch["teacher_pos"] = torch.tensor([it["teacher_pos"] for it in items])
        batch["games"] = [it["game"] for it in items]
        batch["kinds"] = [it.get("kind", "game") for it in items]
        return batch


class GameBalancedSampler(Sampler[int]):
    """Uniform over games, then without replacement within a game (each game's indices are shuffled and cycled,
    reshuffled when exhausted). One epoch = len(records) draws. With one game this is a plain shuffle."""

    def __init__(self, games: Sequence[str], seed: int = 0, num_samples: int | None = None):
        self.by_game: dict[str, list[int]] = {}
        for i, g in enumerate(games):
            self.by_game.setdefault(g, []).append(i)
        self.num_samples = num_samples or len(games)
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed * 7919 + self.epoch)
        names = sorted(self.by_game)
        cycles = {g: [] for g in names}
        for _ in range(self.num_samples):
            g = rng.choice(names) if len(names) > 1 else names[0]
            if not cycles[g]:
                cycles[g] = self.by_game[g][:]
                rng.shuffle(cycles[g])
            yield cycles[g].pop()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("games", nargs="+")
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--label-delay", type=int, default=0); p.add_argument("--no-delay-games", nargs="*", default=[])
    p.add_argument("--boost-last", type=int, nargs=2, default=None, metavar=("K", "TIMES"))
    p.add_argument("--boost-games", nargs="*", default=[], help="limit --boost-last to these games (default: all)")
    a = p.parse_args()
    recs = load_records(a.games, Path(a.data_root), label_delay=a.label_delay, no_delay_games=a.no_delay_games,
                        boost_last=tuple(a.boost_last) if a.boost_last else None, boost_games=a.boost_games)
    train, val = split_records(recs)
    print(f"{len(recs)} records, {len(train)} train, {len(val)} val (seed % {VAL_MOD} == 0)")
    for g in a.games:
        n = sum(r.game == g for r in recs)
        print(f"  {g}: {n} records, {sum(r.game == g for r in val)} val; options {game_options(g, recs[[r.game for r in recs].index(g)].names)}")
    ds = SFTDataset(train[:1], train=False)
    it = ds[0]
    print("one rendered prompt:")
    print(build_plain_prompt(it["options"]))
    print("target", it["target"], "teacher at position", it["teacher_pos"], "image", it["image"].size)


if __name__ == "__main__":
    main()
