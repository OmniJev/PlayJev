"""Text-only typed decisions under today's contract: a blank frame, the question in the
instruction slot, the choices as options. One forward pass, letter-logit readout.

    python scripts/general/eval_text.py <ckpt> <tag> [blank|text]

`blank` (default) is the shipped contract: a blank 448 px frame as the state, the question in the
instruction slot. `text` is the contract of playjev/model.py render_state with a text body: the
question stem is the state, the instruction is the same fixed sentence for every item. The README reports the
`text` form, mean of the given option order and the reversed one.
"""
import io, json, sys, time
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.model import PlayJevModel

ckpt, tag = sys.argv[1], sys.argv[2]
form = sys.argv[3] if len(sys.argv) > 3 else "blank"
ASK = "Which option is correct?"
items = [json.loads(l) for l in (Path(__file__).resolve().parents[2] / "runs" / "general" / "mmlu_test" / "items.jsonl").read_text().splitlines()]
buf = io.BytesIO(); Image.new("RGB", (448, 448), "white").save(buf, "JPEG", quality=92)
BLANK = buf.getvalue()
m = PlayJevModel(ckpt).load()

def run(reverse):
    hits = 0; picks = []; confs = []; mass = []; tops = {}
    for it in items:
        opts = list(it["options"]); gold = it["answer_idx"]
        if reverse:
            opts = opts[::-1]; gold = len(opts) - 1 - gold
        if form == "text":
            d = m.decide([None], [{"name": o} for o in opts], instructions=ASK,
                         frames_per_state=0, state_text=it["question"])[0]
        else:
            d = m.decide([BLANK], [{"name": o} for o in opts], instructions=it["question"])[0]
        pick = d.choice
        hits += int(pick == gold); picks.append(pick); confs.append(d.confidence); mass.append(d.allowed_mass)
        tops[d.top_token] = tops.get(d.top_token, 0) + 1
    dist = {k: picks.count(k) for k in sorted(set(picks))}
    print(f"[{tag}/{form}{' reversed' if reverse else ''}] acc {hits/len(items):.3f}  chance 0.250  n {len(items)}  "
          f"answer-slot {dist}  confidence {sum(confs)/len(confs):.3f}  letter-mass {sum(mass)/len(mass):.3f}  "
          f"top tokens {sorted(tops.items(), key=lambda x: -x[1])[:4]}")
    return {"acc": hits / len(items), "dist": dist, "conf": sum(confs) / len(confs)}

out = {"ckpt": ckpt, "form": form, "normal": run(False), "reversed": run(True)}
(Path(__file__).resolve().parents[2] / "runs" / "general" / f"text_{tag}_{form}.json").write_text(json.dumps(out, indent=1))
