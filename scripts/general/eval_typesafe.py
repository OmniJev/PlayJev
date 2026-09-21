"""The TypeSafe public evaluation rows (evals.typesafe.ai, 365 rows over 20 workflows) under the text contract of
playjev/model.py: the state serialised as JSON is the state text, the question is the instruction, the options are
the choices. One forward pass per row, letter-logit readout, no frame.

    python scripts/general/eval_typesafe.py <ckpt> <tag> [rows.jsonl]

Prints accuracy against the reference label overall and per primitive, next to Jev's own answers on the same rows.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from playjev.model import PlayJevModel

ckpt, tag = sys.argv[1], sys.argv[2]
src = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).resolve().parents[2] / "runs" / "general" / "typesafe365.jsonl"
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
m = PlayJevModel(ckpt).load()


def jev_pick(row):
    """Jev's own argmax on the row, from the answer the public evaluation page records for it."""
    j = row.get("jev")
    if not j:
        return None
    if row["primitive"] == "noul":
        probs = [j["noul"], 1 - j["noul"]]
    else:
        probs = [j["probabilities"].get(o["id"], 0.0) for o in row["options"]]
    return max(range(len(probs)), key=lambda i: probs[i])


t0 = time.time()
hits, jev_hits = {}, {}
confs, mass, picks = [], [], []
per_row, preds = [], []
for i, r in enumerate(rows):
    opts = r["options"]  # the label is an index into this order
    names = [f"{o['id']}: {o['description']}" if o["description"] and o["description"] != o["id"] else o["id"] for o in opts]
    state = json.dumps(r["state"], ensure_ascii=False, indent=1)
    d = m.decide([None], [{"name": n} for n in names], instructions=r["question"], frames_per_state=0, state_text=state)[0]
    ok = d.choice == r["label"]
    p = r["primitive"]
    hits.setdefault(p, []).append(ok)
    jp = jev_pick(r)
    if jp is not None:
        jev_hits.setdefault(p, []).append(jp == r["label"])
    confs.append(d.confidence); mass.append(d.allowed_mass); picks.append(d.choice)
    per_row.append({"id": r["id"], "primitive": p, "pick": d.choice, "label": r["label"], "confidence": d.confidence})
    preds.append({"id": r["id"], "option_ids": [o["id"] for o in opts], "probabilities": [float(x) for x in d.probs]})
    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)

allh = [h for v in hits.values() for h in v]
allj = [h for v in jev_hits.values() for h in v]
line = f"[{tag}/typesafe] acc {sum(allh) / len(allh):.3f}  n {len(allh)}  " + \
       "  ".join(f"{p} {sum(v) / len(v):.3f} (n {len(v)})" for p, v in sorted(hits.items())) + \
       f"  |  jev {sum(allj) / len(allj):.3f} (n {len(allj)})  " + \
       "  ".join(f"{p} {sum(v) / len(v):.3f}" for p, v in sorted(jev_hits.items())) + \
       f"  |  confidence {sum(confs) / len(confs):.3f}  letter-mass {sum(mass) / len(mass):.3f}  answer-slot " + \
       str({k: picks.count(k) for k in sorted(set(picks))})
print(line)
out = {"ckpt": ckpt, "rows": str(src), "acc": sum(allh) / len(allh), "n": len(allh),
       "per_primitive": {p: sum(v) / len(v) for p, v in hits.items()},
       "jev": {"acc": sum(allj) / len(allj), "n": len(allj), "per_primitive": {p: sum(v) / len(v) for p, v in jev_hits.items()}},
       "confidence": sum(confs) / len(confs), "letter_mass": sum(mass) / len(mass), "seconds": round(time.time() - t0),
       "per_row": per_row}
gen = Path(__file__).resolve().parents[2] / "runs" / "general"
(gen / f"typesafe_{tag}.json").write_text(json.dumps(out, indent=1))
# the same predictions in the openjev prediction shape, for bench/score_direct.py of the OpenJev repository
(gen / f"typesafe_{tag}_pred.jsonl").write_text("\n".join(json.dumps(x) for x in preds) + "\n")
