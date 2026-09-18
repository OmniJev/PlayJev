"""Teacher-labelled SFT data: records written by `python -m playjev.collect`, one sample per decision.

A sample is the frame (plus the previous frame when two frames per state are used), the game's option list in a
random order (a fresh permutation per sample, the teacher's soft target permuted the same way), and the letter
position of the teacher's argmax. The prompt is rendered with the same `build_plain_prompt` the inference model
uses, so train and test strings are identical by construction. Descriptions come from the game's hook (the
records carry the action names only).

    python -m playjev.data snake            # count records, split sizes, one rendered sample
"""
from __future__ import annotations

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


def game_options(game: str, names: Sequence[str]) -> list[dict]:
    """Option dicts (name, description) in the given order. Source: runs/bench/<game>/actions.json when present,
    else the `name: '...', description: '...'` pairs in the game's pj_hook.js."""
    path = ROOT / "runs" / "bench" / game / "actions.json"
    if path.exists():
        opts = json.loads(path.read_text())
    else:
        src = (GAMES_DIR / game / "pj_hook.js").read_text()
        pairs = re.findall(r"""name:\s*(['"])(.*?)\1\s*,\s*description:\s*(['"])(.*?)\3""", src)
        opts = [{"name": n, "description": d} for _, n, _, d in pairs]
    by = {o["name"]: o for o in opts}
    missing = [n for n in names if n not in by]
    if missing:
        raise ValueError(f"{game}: no description for actions {missing}; known {sorted(by)}")
    return [{"name": n, "description": by[n]["description"]} for n in names]


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
                          probs=records[j].probs, teacher_action=records[j].teacher_action, seed=r.seed))
    return out


def load_records(games: Sequence[str], data_root: Path = DATA_ROOT, shards: Sequence[str] | None = None,
                 label_delay: int = 0) -> list[Record]:
    out: list[Record] = []
    for game in games:
        gdir = data_root / game
        if not gdir.exists():
            raise FileNotFoundError(f"no data for {game} under {data_root}")
        for shard_dir in sorted(p for p in gdir.iterdir() if (p / "records.jsonl").exists()):
            if shards and shard_dir.name not in shards:
                continue
            recs, keys, steps = [], [], []
            with open(shard_dir / "records.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    recs.append(Record(game=r["game"], shard_dir=shard_dir, frame=r["frame"], prev_frame=r.get("prev_frame"),
                                       names=tuple(r["actions"]), probs=r["teacher_probs"], teacher_action=r["teacher_action"],
                                       seed=r["seed"]))
                    keys.append((r["seed"], r["episode"])); steps.append(r["step"])
            out.extend(shift_labels(recs, keys, steps, label_delay) if label_delay else recs)
    return out


def split_records(records: Sequence[Record]) -> tuple[list[Record], list[Record]]:
    train = [r for r in records if r.seed % VAL_MOD != 0]
    val = [r for r in records if r.seed % VAL_MOD == 0]
    return train, val


class SFTDataset(Dataset):
    """Decodes the frame(s) and draws the option permutation. Training draws a fresh permutation every time;
    validation uses a permutation fixed by the sample index so repeated evals compare like with like."""

    def __init__(self, records: Sequence[Record], two_frame: bool = False, long_side: int | None = None,
                 train: bool = True, seed: int = 0):
        self.records = list(records)
        self.two_frame, self.long_side, self.train, self.seed = two_frame, long_side, train, seed
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
        perm = list(range(k))
        rng.shuffle(perm)  # perm[pos] = original option index shown at letter position pos
        opts = [self.options[r.game][j] for j in perm]
        target = [r.probs[j] for j in perm]
        item = {"image": self._image(r.shard_dir, r.frame), "options": opts, "target": target,
                "teacher_pos": perm.index(r.teacher_action), "game": r.game, "n_opts": k}
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
        n_ph = 2 if (self.two_frame and self.stack == "separate") else 1
        texts = [build_plain_prompt(it["options"], self.instructions, n_ph) for it in items]
        if not self.two_frame:
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
    p.add_argument("--label-delay", type=int, default=0)
    a = p.parse_args()
    recs = load_records(a.games, Path(a.data_root), label_delay=a.label_delay)
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
