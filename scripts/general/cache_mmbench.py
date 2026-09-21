"""Cache a slice of MMBench dev locally: image + question + options + answer."""
import io, json, sys
from datasets import load_dataset
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
LANG = sys.argv[2] if len(sys.argv) > 2 else "EN"
out = Path(__file__).resolve().parents[2] / "runs" / "general" / f"mmbench_{LANG.lower()}"
(out / "img").mkdir(parents=True, exist_ok=True)
ds = load_dataset(f"lmms-lab/MMBench_{LANG}", split="dev", streaming=True)
rows = []
for i, r in enumerate(ds):
    if len(rows) >= N:
        break
    opts = [(L, r[L]) for L in "ABCD" if isinstance(r.get(L), str) and r[L] not in ("nan", "", None)]
    if len(opts) < 2 or r["answer"] not in [L for L, _ in opts]:
        continue
    p = out / "img" / f"{len(rows):04d}.jpg"
    r["image"].convert("RGB").save(p, "JPEG", quality=92)
    rows.append({"i": len(rows), "img": p.name, "question": r["question"],
                 "hint": r.get("hint") if isinstance(r.get("hint"), str) and r["hint"] != "nan" else "",
                 "options": [t for _, t in opts], "answer_letter": r["answer"],
                 "answer_idx": [L for L, _ in opts].index(r["answer"]),
                 "category": r.get("l2-category", "")})
(out / "items.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
print(f"cached {len(rows)} items to {out}")
