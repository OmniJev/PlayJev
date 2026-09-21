"""Cache a held-out text multiple-choice set (MMLU test, mixed subjects)."""
import json, sys
from pathlib import Path
from datasets import load_dataset

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = Path(__file__).resolve().parents[2] / "runs" / "general" / "mmlu_test"
out.mkdir(parents=True, exist_ok=True)
ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
rows, seen = [], {}
for r in ds:
    s = r["subject"]
    if seen.get(s, 0) >= 4:          # spread over subjects instead of taking one block
        continue
    if len(r["choices"]) != 4:
        continue
    seen[s] = seen.get(s, 0) + 1
    rows.append({"i": len(rows), "question": r["question"], "options": list(r["choices"]),
                 "answer_idx": int(r["answer"]), "subject": s})
    if len(rows) >= N:
        break
(out / "items.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
print(f"cached {len(rows)} items over {len(seen)} subjects to {out}")
