"""Convert public multiple-choice sets into the Choice rendering playjev.mixdata reads.

Two manifests come out, one line per sample: text.jsonl (no image, the state is a passage or a question stem)
and image.jsonl (a picture that is not a game frame). Images are re-encoded to the same 448 px long side the
game frames use. Held out and never converted: MMBench dev and the MMLU test slice, the two instruments in
runs/general, and every validation / test split of the sources below.

    python scripts/build_aux.py --out runs/aux            # everything
    python scripts/build_aux.py --out runs/aux --limit 200 --only sciq aokvqa

  text   mmlu_aux  cais/mmlu all/auxiliary_train    ~100k   broad, no passage
         arc       allenai/ai2_arc  Easy + Challenge  3.4k  science, no passage
         sciq      allenai/sciq train               11.7k   most items carry a support passage
  image  aokvqa    HuggingFaceM4/A-OKVQA train      17.1k   natural photographs, 4 choices
         sciqa     derek-thomas/ScienceQA train     12.7k   diagrams and photographs, some text only
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

from PIL import Image

SPLITTERS = str.maketrans({c: " " for c in "\v\f\x1c\x1d\x1e\x85\u2028\u2029"})  # str.splitlines breaks on these


def scrub(text: str) -> str:
    """json.dumps(ensure_ascii=False) leaves U+2028 and friends as themselves, and a reader that uses
    splitlines then cuts the record in two. They carry nothing a passage needs, so they become spaces.
    A real newline is escaped by json.dumps and breaks nothing, so a passage keeps its paragraphs."""
    return str(text).translate(SPLITTERS).strip() if text else text


ASKS = ("Which option is correct?", "Which of these is right?", "Pick the correct option.", "Which answer is correct?")
LONG_SIDE = 448


def ask_for(uid: str) -> str:
    """A question stem in the state still needs something in the instruction slot. Four wordings, picked by the
    uid so a rerun writes the same file, one of them the wording runs/general/eval_text.py uses."""
    return ASKS[int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % len(ASKS)]


def save_image(im: Image.Image, path: Path) -> None:
    im = im.convert("RGB")
    if max(im.size) != LONG_SIDE:
        s = LONG_SIDE / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, "JPEG", quality=88)


class Progress:
    def __init__(self, name: str, total: int):
        self.name, self.total, self.n, self.t0, self.last = name, total, 0, time.time(), 0.0

    def tick(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        if now - self.last < 1 and self.n < self.total:
            return
        self.last = now
        frac = self.n / max(self.total, 1)
        bar = "#" * int(30 * frac) + "." * (30 - int(30 * frac))
        rate = self.n / max(now - self.t0, 1e-9)
        eta = (self.total - self.n) / max(rate, 1e-9)
        sys.stderr.write(f"\r  {self.name:9s} [{bar}] {self.n}/{self.total} {rate:.0f}/s eta {eta / 60:.1f} min")
        sys.stderr.flush()
        if self.n >= self.total:
            sys.stderr.write("\n")


def text_item(source: str, uid: str, options: list[str], answer: int, question: str, state: str | None) -> dict:
    """A text state. With a passage the passage is the state and the question is the instruction; without one
    the stem is the state and the instruction is a fixed ask, which is how the held-out text set is rendered."""
    if state and state.strip():
        return {"kind": "text", "source": source, "uid": uid, "state": scrub(state), "question": scrub(question),
                "options": [{"name": o} for o in options], "answer": answer}
    return {"kind": "text", "source": source, "uid": uid, "state": scrub(question), "question": ask_for(uid),
            "options": [{"name": o} for o in options], "answer": answer}


def clean(options: list[str]) -> list[str] | None:
    opts = [" ".join(scrub(o).split()) for o in options]  # an option name is one line by construction
    if len(opts) < 2 or len(set(opts)) != len(opts) or any(not o for o in opts):
        return None  # a duplicated or empty option makes the answer ambiguous
    return opts


# --------------------------------------------------------------------------------------- sources
def src_mmlu_aux(limit: int):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train", streaming=False)
    n = min(len(ds), limit) if limit else len(ds)
    pr = Progress("mmlu_aux", n)
    for i in range(n):
        r = ds[i]
        opts = clean(r["choices"])
        pr.tick()
        if opts and 0 <= r["answer"] < len(opts):
            yield text_item("mmlu_aux", f"mmlu_aux-{i}", opts, r["answer"], r["question"], None), None


def src_arc(limit: int):
    from datasets import load_dataset
    for cfg in ("ARC-Easy", "ARC-Challenge"):
        ds = load_dataset("allenai/ai2_arc", cfg, split="train")
        n = min(len(ds), limit) if limit else len(ds)
        pr = Progress(cfg[4:].lower(), n)
        for i in range(n):
            r = ds[i]
            opts, labels = clean(r["choices"]["text"]), list(r["choices"]["label"])
            pr.tick()
            if opts and r["answerKey"] in labels:
                yield text_item("arc", f"{cfg}-{r['id']}", opts, labels.index(r["answerKey"]), r["question"], None), None


def src_sciq(limit: int):
    from datasets import load_dataset
    ds = load_dataset("allenai/sciq", split="train")
    n = min(len(ds), limit) if limit else len(ds)
    pr = Progress("sciq", n)
    for i in range(n):
        r = ds[i]
        rng = random.Random(f"sciq-{i}")
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        opts = clean([opts[j] for j in order])
        pr.tick()
        if opts:
            yield text_item("sciq", f"sciq-{i}", opts, order.index(0), r["question"], r.get("support")), None


def src_aokvqa(limit: int):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="train")
    n = min(len(ds), limit) if limit else len(ds)
    pr = Progress("aokvqa", n)
    for i in range(n):
        r = ds[i]
        opts = clean(r["choices"])
        pr.tick()
        if not opts or not 0 <= r["correct_choice_idx"] < len(opts):
            continue
        uid = f"aokvqa-{r.get('question_id', i)}"
        yield ({"kind": "image", "source": "aokvqa", "uid": uid, "state": None, "question": scrub(r["question"]),
                "options": [{"name": o} for o in opts], "answer": r["correct_choice_idx"],
                "image": f"img/aokvqa/{uid}.jpg"}, r["image"])


def src_sciqa(limit: int):
    from datasets import load_dataset
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    n = min(len(ds), limit) if limit else len(ds)
    pr = Progress("sciqa", n)
    for i in range(n):
        r = ds[i]
        opts = clean(r["choices"])
        pr.tick()
        if not opts or not 0 <= r["answer"] < len(opts):
            continue
        uid = f"sciqa-{i}"
        hint, q = scrub(r.get("hint") or ""), scrub(r["question"])
        if r.get("image") is None:
            yield text_item("sciqa", uid, opts, r["answer"], q, hint), None
        else:
            yield ({"kind": "image", "source": "sciqa", "uid": uid, "state": hint or None, "question": q,
                    "options": [{"name": o} for o in opts], "answer": r["answer"],
                    "image": f"img/sciqa/{uid}.jpg"}, r["image"])


SOURCES = {"mmlu_aux": src_mmlu_aux, "arc": src_arc, "sciq": src_sciq, "aokvqa": src_aokvqa, "sciqa": src_sciqa}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/aux")
    p.add_argument("--only", nargs="*", default=sorted(SOURCES))
    p.add_argument("--limit", type=int, default=0, help="first N rows of each source (smoke runs)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(out / "text.jsonl", "w") as ft, open(out / "image.jsonl", "w") as fi:
        for name in a.only:
            if name not in SOURCES:
                raise SystemExit(f"unknown source {name}; have {sorted(SOURCES)}")
            for item, image in SOURCES[name](a.limit):
                if image is not None:
                    save_image(image, out / item["image"])
                    fi.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    ft.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts[f"{item['kind']}/{item['source']}"] = counts.get(f"{item['kind']}/{item['source']}", 0) + 1
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")
    print(f"total {sum(counts.values())} -> {out}/text.jsonl {out}/image.jsonl")


if __name__ == "__main__":
    main()
