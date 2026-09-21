"""Can the released model answer a typed question that is not a game move?

Same contract as the games: one image plus a typed option list in, a probability per option out,
one forward pass, nothing generated. Only the content is outside the training distribution.

    python scripts/general/eval_general.py <ckpt> <tag>
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.model import PlayJevModel

ckpt, tag = sys.argv[1], sys.argv[2]
root = Path(__file__).resolve().parents[2] / "runs" / "general" / "mmbench_en"
items = [json.loads(l) for l in (root / "items.jsonl").read_text().splitlines()]
m = PlayJevModel(ckpt).load()

def run(reverse: bool):
    hits = picks = 0
    letters, confs, correct_flags = [], [], []
    for it in items:
        opts = list(it["options"]); gold = it["answer_idx"]
        if reverse:
            opts = opts[::-1]; gold = len(opts) - 1 - gold
        q = (it["hint"] + "\n\n" + it["question"]).strip() if it["hint"] else it["question"]
        d = m.decide([(root / "img" / it["img"]).read_bytes()], [{"name": o} for o in opts], instructions=q)[0]
        pick = list(d.probs).index(max(d.probs)) if not isinstance(d.probs, dict) else None
        if pick is None:
            names = list(d.probs); pick = names.index(d.choice)
        hits += int(pick == gold); picks += 1
        letters.append(pick); confs.append(d.confidence); correct_flags.append(pick == gold)
    chance = sum(1 / len(i["options"]) for i in items) / len(items)
    dist = {k: letters.count(k) for k in sorted(set(letters))}
    print(f"[{tag}{' reversed' if reverse else ''}] acc {hits/picks:.3f}  chance {chance:.3f}  "
          f"n {picks}  answer-slot distribution {dist}  mean confidence {sum(confs)/len(confs):.3f}")
    return {"acc": hits / picks, "chance": chance, "dist": dist, "conf": sum(confs) / len(confs)}

t0 = time.time()
out = {"ckpt": ckpt, "normal": run(False), "reversed": run(True), "seconds": round(time.time() - t0)}
(Path(__file__).resolve().parents[2] / "runs" / "general" / f"result_{tag}.json").write_text(json.dumps(out, indent=1))
