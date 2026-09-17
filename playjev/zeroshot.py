"""Zero-shot probe: run PlayJevModel.decide() over a directory of frames and report what comes out.

    python -m playjev.zeroshot --model Qwen/Qwen3.5-0.8B-Base --game snake --frames runs/bench/snake \
        --latency 1,8,32 --latency-sizes 448,224 --out runs/smoke/q08b_plain.json

Prints the exact prompt, a probability table per frame, position-bias and frame-sensitivity checks,
and a latency / peak-memory grid. Options come from a game hook (--game, parsed out of pj_hook.js),
from `name:description` pairs (--options), or from a JSON list (--options-json).
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path

from PIL import Image

from .model import LETTERS, DEFAULT_INSTRUCTIONS, PlayJevModel

ROOT = Path(__file__).resolve().parents[1]


def options_from_hook(game: str) -> list[dict]:
    src = (ROOT / "games" / game / "pj_hook.js").read_text()
    pairs = re.findall(r"name:\s*'([^']*)'\s*,\s*description:\s*'([^']*)'", src)
    if not pairs:
        raise SystemExit(f"no `name: '...', description: '...'` pairs found in games/{game}/pj_hook.js")
    return [{"name": n, "description": d} for n, d in pairs]


def parse_options(a) -> list[dict]:
    if a.options_json:
        return json.loads(Path(a.options_json).read_text())
    if a.options:
        out = []
        for s in a.options:
            name, _, desc = s.partition(":")
            out.append({"name": name.strip(), "description": desc.strip()})
        return out
    if a.game:
        return options_from_hook(a.game)
    raise SystemExit("give --game, --options or --options-json")


def load_frames(d: Path) -> list[tuple[str, bytes]]:
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"no frames in {d}")
    return [(p.name, p.read_bytes()) for p in files]


def grey_frame(like: bytes) -> Image.Image:
    w, h = Image.open(__import__("io").BytesIO(like)).size
    return Image.new("RGB", (w, h), (128, 128, 128))


def make_items(frames: list[bytes], frames_per_state: int) -> list:
    if frames_per_state == 1:
        return list(frames)
    return [(frames[i - 1], frames[i]) for i in range(1, len(frames))]


def l1(p, q) -> float:
    return sum(abs(x - y) for x, y in zip(p, q))


def fmt_probs(p) -> str:
    return " ".join(f"{x:5.3f}" for x in p)


def main(a) -> None:
    import torch

    frames_dir = Path(a.frames)
    options = parse_options(a)
    k = len(options)
    names = [o["name"] for o in options]
    named = load_frames(frames_dir)
    fnames, frames = [n for n, _ in named], [b for _, b in named]
    report: dict = {"model": a.model, "template": a.template, "frames_dir": str(frames_dir), "n_frames": len(frames),
                    "options": options, "frames_per_state": a.frames_per_state, "stack": a.stack}

    model = PlayJevModel(a.model, device=a.device, template=a.template).load()
    import transformers
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
           "python": platform.python_version(), "kernels": model.kernels,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    report["env"] = env
    report["load_seconds"] = model.load_seconds
    print(f"[zeroshot] {a.model} ({a.template}) loaded in {model.load_seconds:.1f}s on {env['gpu']}; kernels {model.kernels}")
    print(f"[zeroshot] torch {env['torch']} cuda {env['cuda']} transformers {env['transformers']} python {env['python']}")

    prompt = model.build_prompt(options, a.instructions, a.frames_per_state, a.stack)
    report["prompt"] = prompt
    report["slot_tokens"] = [model.slot_text(l) for l in LETTERS[:k]]
    print("[zeroshot] exact prompt (before <|image_pad|> expansion):")
    print(json.dumps(prompt, ensure_ascii=False))
    print(f"[zeroshot] answer slot tokens: {report['slot_tokens']}")

    # ---- probabilities per frame
    items = make_items(frames, a.frames_per_state)
    item_names = fnames if a.frames_per_state == 1 else [f"{fnames[i-1]}+{fnames[i]}" for i in range(1, len(fnames))]
    dec = model.decide(items, options, a.instructions, a.frames_per_state, a.stack, batch_size=a.batch, long_side=a.long_side)
    t = model.last_timing
    report["tokens"] = {"input": t.input_tokens, "visual": t.visual_tokens}
    print(f"[zeroshot] tokens per sample: {t.input_tokens} total, {t.visual_tokens} visual")
    head = f"{'frame':40s} " + " ".join(f"{LETTERS[i]}:{names[i][:5]:5s}"[:7].ljust(5) for i in range(k)) + "  choice      conf   mass  top"
    print(head)
    rows = []
    for nm, d in zip(item_names, dec):
        print(f"{nm[:40]:40s} {fmt_probs(d.probs)}  {names[d.choice]:10s} {d.confidence:5.3f}  {d.allowed_mass:5.3f}  {d.top_token!r}")
        rows.append({"frame": nm, "probs": d.probs, "choice": names[d.choice], "confidence": d.confidence,
                     "allowed_mass": d.allowed_mass, "top_token": d.top_token})
    report["decisions"] = rows

    mean_p = [statistics.fmean(d.probs[i] for d in dec) for i in range(k)]
    hist = {n: sum(1 for d in dec if names[d.choice] == n) for n in names}
    sens = [l1(d.probs, mean_p) for d in dec]
    summary = {"mean_probs": dict(zip(names, mean_p)), "argmax_hist": hist,
               "mean_confidence": statistics.fmean(d.confidence for d in dec),
               "mean_allowed_mass": statistics.fmean(d.allowed_mass for d in dec),
               "frame_sensitivity_l1_mean": statistics.fmean(sens), "frame_sensitivity_l1_max": max(sens),
               "top_tokens": sorted({d.top_token for d in dec})}
    print(f"[summary] mean probs {dict((n, round(p, 3)) for n, p in zip(names, mean_p))}")
    print(f"[summary] argmax histogram {hist}; mean confidence {summary['mean_confidence']:.3f}; "
          f"mean allowed mass {summary['mean_allowed_mass']:.3f}; top tokens {summary['top_tokens']}")
    print(f"[summary] frame sensitivity: L1 distance of each frame's probs to the mean, mean {summary['frame_sensitivity_l1_mean']:.3f}, "
          f"max {summary['frame_sensitivity_l1_max']:.3f} (0 = the frame changes nothing)")

    # ---- control: a flat grey frame with the same prompt
    if not a.no_control:
        g = grey_frame(frames[0])
        gitems = [g] if a.frames_per_state == 1 else [(g, g)]
        gd = model.decide(gitems, options, a.instructions, a.frames_per_state, a.stack, long_side=a.long_side)[0]
        dist = [l1(d.probs, gd.probs) for d in dec]
        summary["grey_control"] = {"probs": gd.probs, "l1_to_real_mean": statistics.fmean(dist), "l1_to_real_max": max(dist)}
        print(f"[control] grey frame probs {fmt_probs(gd.probs)} top {gd.top_token!r}; L1 to real frames mean {statistics.fmean(dist):.3f} max {max(dist):.3f}")

    # ---- position bias: rotate the option order by one and see whether the mass follows the letter or the name
    if not a.no_shift_check:
        shifted = options[1:] + options[:1]
        sd = model.decide(items, options=shifted, instructions=a.instructions, frames_per_state=a.frames_per_state,
                          stack=a.stack, batch_size=a.batch, long_side=a.long_side)
        by_letter = [statistics.fmean(d.probs[i] for d in sd) for i in range(k)]
        by_name = {o["name"]: by_letter[i] for i, o in enumerate(shifted)}
        summary["shift1"] = {"order": [o["name"] for o in shifted], "mean_probs_by_letter": by_letter, "mean_probs_by_name": by_name}
        print(f"[shift] order {[o['name'] for o in shifted]}")
        print(f"[shift] mean prob per letter position: original {fmt_probs(mean_p)} | shifted {fmt_probs(by_letter)}")
        print(f"[shift] mean prob per option name under the shifted order: {dict((n, round(p, 3)) for n, p in by_name.items())}")
        # mass follows the letter when the per-letter profile is stable across orders
        letter_l1 = l1(mean_p, by_letter)
        name_l1 = l1(mean_p, [by_name[n] for n in names])
        summary["shift1"]["l1_letter_profile"] = letter_l1
        summary["shift1"]["l1_name_profile"] = name_l1
        print(f"[shift] L1 between orders: per-letter profile {letter_l1:.3f}, per-name profile {name_l1:.3f} "
              f"(small letter L1 = position bias, small name L1 = the model reads the options)")
    report["summary"] = summary

    # ---- latency and memory grid
    if a.latency:
        batches = [int(x) for x in a.latency.split(",")]
        sizes = [int(x) for x in a.latency_sizes.split(",")] if a.latency_sizes else [None]
        grid = []
        print(f"{'size':>5s} {'batch':>5s} {'vis_tok':>7s} {'in_tok':>6s} {'prep_ms':>8s} {'fwd_ms':>8s} {'ms/dec':>7s} {'dec/s':>7s} {'peak_GB':>7s}")
        for size in sizes:
            for b in batches:
                batch_items = [items[i % len(items)] for i in range(b)]
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(a.warmup):
                    model.decide(batch_items, options, a.instructions, a.frames_per_state, a.stack, batch_size=b, long_side=size)
                preps, fwds = [], []
                for _ in range(a.repeats):
                    model.decide(batch_items, options, a.instructions, a.frames_per_state, a.stack, batch_size=b, long_side=size)
                    preps.append(model.last_timing.prep_s)
                    fwds.append(model.last_timing.forward_s)
                prep_ms, fwd_ms = 1000 * statistics.median(preps), 1000 * statistics.median(fwds)
                peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
                row = {"long_side": size, "batch": b, "visual_tokens": model.last_timing.visual_tokens,
                       "input_tokens": model.last_timing.input_tokens, "prep_ms": prep_ms, "forward_ms": fwd_ms,
                       "ms_per_decision": (prep_ms + fwd_ms) / b, "decisions_per_s": 1000 * b / (prep_ms + fwd_ms),
                       "forward_ms_per_decision": fwd_ms / b, "peak_gb": peak}
                grid.append(row)
                print(f"{str(size):>5s} {b:5d} {row['visual_tokens']:7d} {row['input_tokens']:6d} {prep_ms:8.1f} {fwd_ms:8.1f} "
                      f"{row['ms_per_decision']:7.2f} {row['decisions_per_s']:7.1f} {peak:7.2f}")
        report["latency"] = grid

    # ---- where does a forward spend its time? one batch under the torch profiler
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        b = a.profile
        batch_items = [items[i % len(items)] for i in range(b)]
        model.decide(batch_items, options, a.instructions, a.frames_per_state, a.stack, batch_size=b, long_side=a.long_side)
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
        with profile(activities=acts, record_shapes=False) as prof:
            model.decide(batch_items, options, a.instructions, a.frames_per_state, a.stack, batch_size=b, long_side=a.long_side)
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        print(f"[profile] batch {b}: top ops by device time")
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
        print(f"[profile] batch {b}: top ops by CPU time")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"[zeroshot] wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--template", default="plain", choices=["plain", "chat"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--frames", required=True, help="directory of JPEG/PNG frames")
    p.add_argument("--game", help="read the options out of games/<game>/pj_hook.js")
    p.add_argument("--options", nargs="*", help="name:description pairs")
    p.add_argument("--options-json")
    p.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    p.add_argument("--frames-per-state", type=int, default=1, choices=[1, 2])
    p.add_argument("--stack", default="temporal", choices=["temporal", "separate"])
    p.add_argument("--long-side", type=int, default=None, help="rescale frames so the long side is this many px")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--latency", default="", help="comma list of batch sizes to time, e.g. 1,8,32")
    p.add_argument("--latency-sizes", default="", help="comma list of long sides for the timing grid, e.g. 448,224")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no-control", action="store_true")
    p.add_argument("--no-shift-check", action="store_true")
    p.add_argument("--profile", type=int, default=0, help="run one batch of this size under torch.profiler and print the op table")
    p.add_argument("--out")
    main(p.parse_args())
