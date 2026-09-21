"""Non-game samples and the sampler that mixes them into a game run.

A game frame teaches the model one thing: read this picture, pick one of these ten games' moves. After one epoch
of nothing else the model answers every other typed question at chance and puts no mass at all on the letter
slots (runs/general). The fix is to keep the other two kinds of state in the stream while the games train:

  text   the state is a passage or a question stem, no frame at all
  image  the state is a picture that is not a game frame
  game   a frame, as before, with the option rewrites in data.GameAug

Every kind renders through the same `build_plain_prompt`, so the model sees one contract with three kinds of
state. Kinds cannot share a batch (the processor takes one image list per text list, and a text state has no
image), so `TypeBatchSampler` draws each batch from a single kind and shuffles the order of the batches.

    python -m playjev.mixdata runs/aux/text.jsonl runs/aux/image.jsonl    # counts, one rendered sample of each
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .data import VAL_MOD
from .model import build_plain_prompt

KINDS = ("game", "image", "text")


@dataclass
class AuxRecord:
    kind: str  # "text" or "image"
    source: str  # the dataset it came from, for the mixture report and for holding a source out
    uid: str
    question: str  # the instruction slot
    options: list[dict]  # name, optional description
    answer: int  # index into options
    state: str | None = None  # text body of the state
    image: str | None = None  # path relative to the manifest

    def __post_init__(self) -> None:
        if self.kind not in ("text", "image"):
            raise ValueError(f"{self.uid}: kind {self.kind!r} is not text or image")
        if self.kind == "text" and not (self.state or "").strip():
            raise ValueError(f"{self.uid}: a text state needs a state body")
        if self.kind == "image" and not self.image:
            raise ValueError(f"{self.uid}: an image state needs an image")
        if not 0 <= self.answer < len(self.options):
            raise ValueError(f"{self.uid}: answer {self.answer} outside {len(self.options)} options")


def load_aux(paths: Sequence[Path | str]) -> list[AuxRecord]:
    """Read jsonl manifests. Image paths are resolved against the manifest's own directory."""
    out: list[AuxRecord] = []
    for path in paths:
        path = Path(path)
        # split on "\n" only: str.splitlines also breaks on U+2028 / U+0085 / the C1 separators, which occur
        # inside real passages and would cut a record in half
        for line in path.read_text().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            r = AuxRecord(kind=d["kind"], source=d.get("source", path.stem), uid=d["uid"], question=d["question"],
                          options=[o if isinstance(o, dict) else {"name": o} for o in d["options"]],
                          answer=d["answer"], state=d.get("state"),
                          image=str(path.parent / d["image"]) if d.get("image") else None)
            out.append(r)
    return out


def split_aux(records: Sequence[AuxRecord]) -> tuple[list[AuxRecord], list[AuxRecord]]:
    """Same 1-in-VAL_MOD holdout the game records use, keyed by uid so it survives reshuffles and reruns."""
    def bucket(r: AuxRecord) -> int:
        return int(hashlib.sha1(f"{r.source}/{r.uid}".encode()).hexdigest()[:8], 16) % VAL_MOD
    return [r for r in records if bucket(r)], [r for r in records if not bucket(r)]


class AuxDataset(Dataset):
    """One item per record, in the shape SFTDataset emits: option order drawn per sample, soft target on the
    answer. `soft` is the mass the answer keeps; the rest is spread over the other options, the shape the game
    teacher already produces, so the loss sees one kind of target."""

    def __init__(self, records: Sequence[AuxRecord], long_side: int | None = None, train: bool = True,
                 seed: int = 0, soft: float = 0.9):
        self.records, self.long_side, self.train, self.seed, self.soft = list(records), long_side, train, seed, soft

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, path: str) -> Image.Image:
        im = Image.open(path).convert("RGB")
        if self.long_side and max(im.size) != self.long_side:
            s = self.long_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        return im

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        k = len(r.options)
        rng = random.Random(torch.initial_seed() + i) if self.train else random.Random(self.seed * 1_000_003 + i)
        perm = list(range(k))
        rng.shuffle(perm)
        spread = (1.0 - self.soft) / (k - 1) if k > 1 else 0.0
        target = [self.soft if j == r.answer else spread for j in perm]
        return {"image": self._image(r.image) if r.image else None, "options": [r.options[j] for j in perm],
                "target": target, "teacher_pos": perm.index(r.answer), "game": r.source, "n_opts": k,
                "kind": r.kind, "state_text": r.state, "instructions": r.question}


class MixedDataset(Dataset):
    """Concatenates the game dataset and the aux datasets and remembers what each global index is, so the
    sampler can draw a batch of one kind."""

    def __init__(self, parts: Sequence[Dataset]):
        self.parts = list(parts)
        self.index: list[tuple[int, int]] = []
        self.kinds: list[str] = []
        self.groups: list[str] = []  # game name, or aux source: the axis each kind is balanced over
        for p, part in enumerate(self.parts):
            aux = isinstance(part, AuxDataset)
            for i in range(len(part)):
                self.index.append((p, i))
                self.kinds.append(part.records[i].kind if aux else "game")
                self.groups.append(part.records[i].source if aux else part.records[i].game)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        p, j = self.index[i]
        return self.parts[p][j]


class TypeBatchSampler(Sampler[list[int]]):
    """Batches of one kind at a time, in a shuffled order, at the requested mixture.

    `ratios` is the share of *batches* each kind gets (so also the share of samples, the batch size being fixed).
    Within a kind the draw is uniform over its groups (the ten games, or the aux sources), then without
    replacement inside a group, the same cycle GameBalancedSampler uses: a small source is revisited rather than
    exhausted, and a large one is not allowed to take over."""

    def __init__(self, kinds: Sequence[str], groups: Sequence[str], batch_size: int, ratios: dict[str, float],
                 seed: int = 0, num_batches: int | None = None):
        if abs(sum(ratios.values()) - 1) > 1e-6:
            raise ValueError(f"mixture {ratios} does not sum to 1")
        self.by_kind: dict[str, dict[str, list[int]]] = {}
        for i, (k, g) in enumerate(zip(kinds, groups)):
            self.by_kind.setdefault(k, {}).setdefault(g, []).append(i)
        missing = [k for k, w in ratios.items() if w > 0 and k not in self.by_kind]
        if missing:
            raise ValueError(f"mixture asks for {missing} but no sample has that kind")
        self.ratios = {k: w for k, w in ratios.items() if w > 0}
        self.batch_size, self.seed, self.epoch = batch_size, seed, 0
        # One epoch is one pass over the game frames, whatever the mixture: the other kinds are added on top of
        # the run that already works, they do not take its place. Without a game part, one pass over everything.
        if num_batches:
            self.num_batches = num_batches
        elif "game" in self.ratios:
            n_game = sum(len(v) for v in self.by_kind["game"].values())
            self.num_batches = int(n_game / (batch_size * self.ratios["game"]))
        else:
            self.num_batches = len(kinds) // batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def counts(self) -> dict[str, int]:
        return {k: sum(len(v) for v in g.values()) for k, g in sorted(self.by_kind.items())}

    def report(self) -> str:
        """Pool size, draws per epoch and how often a sample comes back, per kind: a small source revisited
        three times an epoch is a different run from one seen once, and the difference belongs in the log."""
        pool = self.counts()
        rows = []
        for k in sorted(self.ratios):
            drawn = int(self.num_batches * self.ratios[k]) * self.batch_size
            rows.append(f"{k} {pool[k]} pool / {drawn} drawn ({drawn / max(pool[k], 1):.2f}x, {self.ratios[k]:.0%})")
        return f"{self.num_batches} batches of {self.batch_size}: " + "; ".join(rows)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed * 7919 + self.epoch)
        kinds = sorted(self.ratios)
        weights = [self.ratios[k] for k in kinds]
        cycles: dict[str, list[int]] = {}
        for _ in range(self.num_batches):
            kind = rng.choices(kinds, weights)[0]
            names = sorted(self.by_kind[kind])
            batch = []
            while len(batch) < self.batch_size:
                g = rng.choice(names) if len(names) > 1 else names[0]
                key = f"{kind}/{g}"
                if not cycles.get(key):
                    cycles[key] = self.by_kind[kind][g][:]
                    rng.shuffle(cycles[key])
                batch.append(cycles[key].pop())
            yield batch


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("manifests", nargs="+")
    p.add_argument("--soft", type=float, default=0.9)
    a = p.parse_args()
    recs = load_aux(a.manifests)
    train, val = split_aux(recs)
    by_source: dict[str, int] = {}
    for r in recs:
        by_source[f"{r.kind}/{r.source}"] = by_source.get(f"{r.kind}/{r.source}", 0) + 1
    print(f"{len(recs)} aux records, {len(train)} train, {len(val)} val")
    for k, n in sorted(by_source.items()):
        print(f"  {k}: {n}")
    ds = AuxDataset(train, train=False, soft=a.soft)
    seen = set()
    for i in range(len(ds)):
        if train[i].kind in seen:
            continue
        seen.add(train[i].kind)
        it = ds[i]
        print(f"\n---- {train[i].kind} / {train[i].source} / {train[i].uid}"
              f"{' / image ' + str(it['image'].size) if it['image'] else ''}")
        print(build_plain_prompt(it["options"], it["instructions"], 1 if it["image"] else 0, it["state_text"]))
        print("target", [round(t, 3) for t in it["target"]], "answer at", it["teacher_pos"])
        if len(seen) == 2:
            break


if __name__ == "__main__":
    main()
