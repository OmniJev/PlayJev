"""Supervised fine-tuning of a Qwen3.5 VLM on teacher-labelled frames.

Loss: cross-entropy between the teacher's soft distribution over the options and the model's softmax over the
option-letter logits at the answer position (a KL to soft targets), optionally plus a Brier term on the same two
distributions. Nothing else is trained: no next-token loss, no text targets. Option order is a fresh random
permutation per sample (playjev/data.py), which is what breaks the zero-shot letter prior.

Precision: fp32 master weights and optimizer states, bf16 autocast for the forward and backward, the letter
readout in fp32. Checkpoints are saved in bf16 in HF format, so `PlayJevModel(ckpt_dir)` loads them directly.

    python -m playjev.train_sft --games snake --model <hf path> --out <ckpt dir> --epochs 1
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import Collate, GameAug, GameBalancedSampler, SFTDataset, load_records, render_history, split_records
from .mixdata import AuxDataset, MixedDataset, TypeBatchSampler, load_aux, split_aux
from .model import LETTERS, DEFAULT_INSTRUCTIONS, choice_confidence

ROOT = Path(__file__).resolve().parents[1]


def log_line(path: Path, rec: dict) -> None:
    rec["time"] = time.time()
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def slot_ids_for(tokenizer, k: int) -> list[int]:
    """Token ids of the space-prefixed letters (the plain-template answer slots), each verified single-token."""
    ids = []
    for letter in LETTERS[:k]:
        enc = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"slot {letter!r} is not a single token")
        ids.append(enc[0])
    return ids


def slot_logits(model, batch: dict, slot_ids: torch.Tensor, device) -> torch.Tensor:
    """Forward the inner model under bf16 autocast, read the last position, and project it onto the letter rows of
    the (tied) output embedding in fp32. Returns (B, Kmax) fp32 logits; positions >= n_opts are masked to -inf."""
    keys = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    inputs = {k: batch[k].to(device, non_blocking=True) for k in keys if k in batch}  # a text state has no image keys
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model.model(**inputs, use_cache=False, return_dict=True)
    h = out.last_hidden_state[:, -1, :].float()  # left padding: the last position is the answer slot for every row
    w = model.get_output_embeddings().weight[slot_ids].float()  # (Kmax, H)
    logits = h @ w.T
    n_opts = batch["n_opts"].to(device)
    mask = torch.arange(logits.shape[1], device=device)[None, :] < n_opts[:, None]
    return logits.masked_fill(~mask, float("-inf"))


def soft_ce(logits: torch.Tensor, targets: torch.Tensor, brier: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    if targets.shape[1] < logits.shape[1]:  # a batch of small-K games against the global slot count
        targets = F.pad(targets, (0, logits.shape[1] - targets.shape[1]))
    logp = F.log_softmax(logits, -1)
    ce = -(targets * logp.masked_fill(targets == 0, 0.0)).sum(-1)  # 0 * -inf is nan; zero-target slots contribute 0
    loss = ce.mean()
    if brier:
        p = logp.exp()
        loss = loss + brier * ((p - targets) ** 2).sum(-1).mean()
    return loss, ce


def ece(conf: list[float], correct: list[bool], bins: int = 15) -> float:
    n = len(conf)
    if n == 0:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if idx:
            acc = statistics.fmean(correct[i] for i in idx)
            avg = statistics.fmean(conf[i] for i in idx)
            total += len(idx) / n * abs(acc - avg)
    return total


def overall_agreement(ev: dict) -> float:
    """Sample-weighted agreement over every group an evaluate() call reports."""
    rows = [v for k, v in ev.items() if k != "all"]
    n = sum(r["n"] for r in rows)
    return sum(r["agreement"] * r["n"] for r in rows) / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, slot_ids, device, brier: float, max_batches: int | None = None) -> dict:
    """Validation loss, teacher agreement, calibration (ECE 15 bins and Brier of p_max against agreement),
    mean Jev confidence and the mean probability per letter position, all per game and overall."""
    model.eval()
    per: dict[str, dict] = defaultdict(lambda: {"ce": [], "correct": [], "top": [], "pmax": [], "conf": [], "pos_prob": None, "pos_argmax": None, "n": 0})
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        logits = slot_logits(model, batch, slot_ids, device)
        targets = batch["targets"].to(device)
        _, ce = soft_ce(logits, targets, 0.0)
        probs = F.softmax(logits, -1)
        pred = probs.argmax(-1)
        pmax = probs.max(-1).values
        teacher_pos = batch["teacher_pos"].to(device)
        tmax = targets.max(-1).values
        for row, game in enumerate(batch["games"]):
            k = int(batch["n_opts"][row])
            s = per[game]
            p = probs[row, :k].tolist()
            s["ce"].append(float(ce[row])); s["correct"].append(bool(pred[row] == teacher_pos[row]))
            s["top"].append(bool(targets[row, pred[row]] >= tmax[row] - 1e-6))  # argmax inside the teacher's top set (ties)
            s["pmax"].append(float(pmax[row])); s["conf"].append(choice_confidence(p))
            if s["pos_prob"] is None or len(s["pos_prob"]) < k:  # option counts differ within a source
                s["pos_prob"] = (s["pos_prob"] or []) + [0.0] * (k - len(s["pos_prob"] or []))
                s["pos_argmax"] = (s["pos_argmax"] or []) + [0] * (k - len(s["pos_argmax"] or []))
            for j in range(k):
                s["pos_prob"][j] += p[j]
            s["pos_argmax"][int(pred[row])] += 1
            s["n"] += 1
    model.train()
    out = {}
    for game, s in per.items():
        n = s["n"]
        out[game] = {"n": n, "loss": statistics.fmean(s["ce"]), "agreement": statistics.fmean(s["correct"]), "agreement_top": statistics.fmean(s["top"]),
                     "ece": ece(s["pmax"], s["correct"]), "brier": statistics.fmean((c - float(ok)) ** 2 for c, ok in zip(s["pmax"], s["correct"])),
                     "confidence": statistics.fmean(s["conf"]), "pmax": statistics.fmean(s["pmax"]),
                     "prob_per_letter": [x / n for x in s["pos_prob"]], "argmax_per_letter": [x / n for x in s["pos_argmax"]]}
    tot = sum(v["n"] for v in out.values())
    if len(out) > 1:
        out["all"] = {"n": tot, "loss": sum(v["loss"] * v["n"] for v in out.values()) / tot,
                      "agreement": sum(v["agreement"] * v["n"] for v in out.values()) / tot,
                      "agreement_top": sum(v["agreement_top"] * v["n"] for v in out.values()) / tot}
    return out


def save_ckpt(model, processor, out_dir: Path, keep: int, tag: str, meta: dict) -> Path:
    path = out_dir / tag
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in model.state_dict().items()}
    model.save_pretrained(path, state_dict=state, safe_serialization=True)
    processor.save_pretrained(path)
    (path / "playjev_train.json").write_text(json.dumps(meta, indent=1))
    steps = sorted((p for p in out_dir.glob("step-*") if p.is_dir()), key=lambda p: int(p.name.split("-")[1]))
    for old in steps[:-keep] if keep > 0 else []:
        shutil.rmtree(old, ignore_errors=True)
    return path


def check_linear_attention_kernels(device, tol: float = 5e-2) -> dict:
    """Compare the kernel transformers picked for the gated delta rule (fla triton, or tilelang) with the torch
    reference, forward and backward, on random inputs shaped like a Qwen3.5-0.8B layer. fla 0.5.2 refuses its
    Hopper backward under triton 3.4 to 3.7.0 (wrong results, fla issue #640); with triton 3.7.1 it runs, and this
    check is what says the gradients are right before any training step spends GPU time on them. Set
    PLAYJEV_NO_FLA=1 to force the torch reference for the whole run instead."""
    import importlib
    import torch.nn.functional as F  # noqa: F811

    m = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    fast = m.torch_chunk_gated_delta_rule  # module attribute: the kernel wrapper (fla when installed)
    ref = getattr(fast, "__wrapped__", None)
    while ref is not None and hasattr(ref, "__wrapped__"):
        ref = ref.__wrapped__
    if ref is None:
        return {"checked": False, "reason": "no wrapped reference found"}
    torch.manual_seed(0)
    b, t, hk, hv, dk, dv = 2, 200, 16, 16, 128, 128
    mk = lambda *shape, dtype=torch.bfloat16: torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    inputs = {"query": mk(b, t, hk, dk), "key": mk(b, t, hk, dk), "value": mk(b, t, hv, dv),
              "g": (-torch.rand(b, t, hv, device=device) * 0.5).requires_grad_(), "beta": torch.rand(b, t, hv, device=device, dtype=torch.bfloat16).requires_grad_()}
    outs, grads = [], []
    for fn in (fast, ref):
        args = {k: v.detach().clone().requires_grad_() for k, v in inputs.items()}
        o, _ = fn(args["query"], args["key"], args["value"], g=args["g"], beta=args["beta"], initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=True)
        o = o.float()
        (o * torch.linspace(-1, 1, o.numel(), device=device).view_as(o)).sum().backward()
        outs.append(o.detach()); grads.append({k: v.grad.float() for k, v in args.items()})
    rel = lambda a, b: float((a - b).abs().max() / (b.abs().max() + 1e-6))
    res = {"checked": True, "kernel": getattr(fast, "__module__", "?"), "out": rel(outs[0], outs[1]),
           **{f"grad_{k}": rel(grads[0][k], grads[1][k]) for k in inputs}}
    res["ok"] = all(v <= tol for k, v in res.items() if k.startswith("grad_") or k == "out")
    return res


def force_torch_linear_attention() -> None:
    """Swap the gated-delta-rule kernels for the torch reference implementations for this process."""
    import importlib

    m = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        fn = getattr(m, name)
        ref = fn
        while hasattr(ref, "__wrapped__"):
            ref = ref.__wrapped__
        setattr(m, name, ref)


def main(a: argparse.Namespace) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.jsonl"
    print(f"[train] games {a.games} model {a.model} out {out_dir}")

    # ---- data
    records = load_records(a.games, Path(a.data_root), shards=a.shards or None, label_delay=a.label_delay,
                           no_delay_games=a.no_delay_games, boost_last=tuple(a.boost_last) if a.boost_last else None,
                           boost_games=a.boost_games, history=a.history)
    if a.history:
        shown = [r for r in records[:2000] if any(m != "-" for m in r.history)]
        print(f"[train] history {a.history}: the state carries the moves already made, e.g. "
              f"{render_history(shown[0].history) if shown else '(none in the first 2000 records)'}")
    if a.boost_last:
        print(f"[train] boost: the last {a.boost_last[0]} records of every episode shorter than 500 steps appear {a.boost_last[1]} times"
              + (f" (only for {' '.join(a.boost_games)})" if a.boost_games else ""))
    if a.label_delay:
        print(f"[train] label delay {a.label_delay}: frame k is labelled with the teacher's decision at step k+{a.label_delay}"
              + (f" (not for {' '.join(a.no_delay_games)})" if a.no_delay_games else ""))
    train_recs, val_recs = split_records(records)
    if a.limit:  # smoke tests: the first `limit` training records of every game, a few validation records each
        train_recs = [r for g in a.games for r in [x for x in train_recs if x.game == g][: a.limit]]
        val_recs = [r for g in a.games for r in [x for x in val_recs if x.game == g][: max(4, a.limit // 4)]]
    if a.subsample:  # data-efficiency runs: a seeded random subset of `subsample` training records per game, full validation
        rng = random.Random(a.seed + 7); sub = []
        for g in a.games:
            pool = [x for x in train_recs if x.game == g]; rng.shuffle(pool); sub.extend(pool[: a.subsample])
        train_recs = sub
    if a.shuffle_labels:  # control: the targets of one game are dealt out to its own frames at random, so the run
        # learns the answer format and the option text of every game while the frame says nothing about the answer
        rng = random.Random(a.seed + 11); shuffled = list(train_recs)
        for g in a.games:
            idx = [i for i, r in enumerate(shuffled) if r.game == g]
            labels = [(shuffled[i].probs, shuffled[i].teacher_action) for i in idx]
            rng.shuffle(labels)
            for i, (probs, act) in zip(idx, labels):
                shuffled[i] = dataclasses.replace(shuffled[i], probs=probs, teacher_action=act)
        kept = statistics.fmean(shuffled[i].teacher_action == train_recs[i].teacher_action for i in range(len(train_recs)))
        train_recs = shuffled
        print(f"[train] shuffle-labels: targets dealt out within each game, {kept:.3f} land on their own frame by chance"
              f" (validation is untouched, so its agreement reads what the run actually learned)")
    print(f"[train] {len(records)} records: {len(train_recs)} train, {len(val_recs)} val; per game "
          f"{ {g: sum(r.game == g for r in train_recs) for g in a.games} }")
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=Path(a.model).exists())
    processor.tokenizer.padding_side = "left"
    collate = Collate(processor, two_frame=a.two_frame, stack=a.stack)
    aug = GameAug(*a.aug) if any(a.aug) else None
    train_ds = SFTDataset(train_recs, two_frame=a.two_frame, long_side=a.long_side, train=True, seed=a.seed, aug=aug)
    val_ds = SFTDataset(val_recs, two_frame=a.two_frame, long_side=a.long_side, train=False, seed=a.seed)
    sampler = GameBalancedSampler([r.game for r in train_recs], seed=a.seed)
    loader_kw = dict(num_workers=a.workers, collate_fn=collate, pin_memory=device.type == "cuda",
                     persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None)
    aux_recs, aux_val_loaders, k_aux = [], {}, 0
    if a.aux:
        aux_recs = load_aux(a.aux)
        aux_train, aux_val = split_aux(aux_recs)
        k_aux = max(len(r.options) for r in aux_recs)
        mix = dict(zip(("game", "image", "text"), a.mix)) if a.mix else None
        if mix is None:
            raise SystemExit("--aux needs --mix GAME IMAGE TEXT")
        mixed = MixedDataset([train_ds, AuxDataset(aux_train, long_side=a.long_side, train=True, seed=a.seed,
                                                   soft=a.aux_soft)])
        batch_sampler = TypeBatchSampler(mixed.kinds, mixed.groups, a.micro_batch, mix, seed=a.seed)
        train_loader = DataLoader(mixed, batch_sampler=batch_sampler, **loader_kw)
        sampler = batch_sampler
        print(f"[train] mixture: {batch_sampler.report()}")
        # one validation loader per non-game kind: mixing kinds inside a batch is what the type sampler exists
        # to prevent, and the two kinds are read separately anyway
        for kind in ("image", "text"):
            part = [r for r in aux_val if r.kind == kind][: a.eval_samples]
            if part:
                ds = AuxDataset(part, long_side=a.long_side, train=False, seed=a.seed, soft=a.aux_soft)
                aux_val_loaders[kind] = DataLoader(ds, batch_size=a.micro_batch, shuffle=False, **loader_kw)
        print(f"[train] aux: {len(aux_train)} train, {len(aux_val)} val "
              f"{ {k: sum(r.kind == k for r in aux_recs) for k in ('image', 'text')} }")
    else:
        train_loader = DataLoader(train_ds, batch_size=a.micro_batch, sampler=sampler, drop_last=True, **loader_kw)
    # validation: a fixed, game-balanced subset so every eval sees the same samples
    val_idx = list(GameBalancedSampler([r.game for r in val_recs], seed=a.seed, num_samples=min(len(val_recs), a.eval_samples)))
    val_loader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=a.micro_batch, shuffle=False, **loader_kw)

    # ---- linear-attention kernels: verify the fast path against the torch reference, or switch it off
    if os.environ.get("PLAYJEV_NO_FLA"):
        force_torch_linear_attention()
        print("[train] PLAYJEV_NO_FLA set: gated delta rule runs on the torch reference implementation")
    elif device.type == "cuda":
        chk = check_linear_attention_kernels(device)
        print(f"[train] linear-attention kernel check: {json.dumps(chk)}")
        if chk.get("checked") and not chk["ok"]:
            raise SystemExit("fast gated-delta-rule kernel disagrees with the torch reference; set PLAYJEV_NO_FLA=1 or fix triton")

    # ---- model
    t0 = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.float32, device_map={"": str(device)},
                                                        local_files_only=Path(a.model).exists())
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.freeze_vision:
        for p in model.model.visual.parameters():
            p.requires_grad_(False)
    model.train()
    kmax = max([len(r.names) for r in records] + [k_aux]) + (aug.extra_slots if aug else 0)
    slot_ids = torch.tensor(slot_ids_for(processor.tokenizer, kmax), device=device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] loaded in {time.perf_counter() - t0:.1f}s; trainable params {n_train / 1e6:.1f}M; slots {slot_ids.tolist()}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=a.wd, fused=device.type == "cuda")
    accum = max(1, a.batch // a.micro_batch)
    steps_per_epoch = len(train_loader) // accum
    total_steps = a.max_steps or steps_per_epoch * a.epochs
    warmup = max(1, int(a.warmup * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return a.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine to 10 percent

    print(f"[train] batch {a.batch} = {accum} x {a.micro_batch}; {steps_per_epoch} steps/epoch, {total_steps} total, warmup {warmup}")
    meta = {"args": vars(a), "records": len(records), "train": len(train_recs), "val": len(val_recs), "kmax": kmax}

    # ---- eval before training: the zero-shot letter prior, for the record
    ev = evaluate(model, val_loader, slot_ids, device, a.brier)
    print(f"[eval step 0] {json.dumps(ev)}")
    log_line(log_path, {"step": 0, "eval": ev})

    step, micro, t_start, seen = 0, 0, time.perf_counter(), 0
    running: list[float] = []
    done = False
    for epoch in range(a.epochs if not a.max_steps else 10**6):
        sampler.set_epoch(epoch)
        for batch in train_loader:
            logits = slot_logits(model, batch, slot_ids, device)
            loss, _ = soft_ce(logits, batch["targets"].to(device), a.brier)
            (loss / accum).backward()
            running.append(loss.detach().item()); seen += batch["targets"].shape[0]; micro += 1
            if micro % accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % a.log_every == 0 or step == 1:
                el = time.perf_counter() - t_start
                rec = {"step": step, "epoch": epoch, "loss": statistics.fmean(running[-accum * a.log_every:]), "lr": lr_at(step - 1),
                       "grad_norm": float(gn), "samples_per_s": seen / el, "elapsed_s": el,
                       "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
                print(f"[step {step}/{total_steps}] loss {rec['loss']:.4f} lr {rec['lr']:.2e} gnorm {rec['grad_norm']:.2f} "
                      f"{rec['samples_per_s']:.1f} samples/s mem {rec['mem_gb']:.1f} GB elapsed {el / 60:.1f} min")
                log_line(log_path, rec)
            if a.eval_every and step % a.eval_every == 0 and step < total_steps:
                ev = evaluate(model, val_loader, slot_ids, device, a.brier)
                for kind, loader in aux_val_loaders.items():
                    ev[f"aux_{kind}"] = overall_agreement(evaluate(model, loader, slot_ids, device, a.brier))
                print(f"[eval step {step}] {json.dumps(ev)}")
                log_line(log_path, {"step": step, "eval": ev})
            if a.save_every and step % a.save_every == 0 and step < total_steps:
                p = save_ckpt(model, processor, out_dir, a.keep, f"step-{step}", {**meta, "step": step})
                print(f"[train] saved {p}")
            if step >= total_steps:
                done = True
                break
        if done:
            break

    ev = evaluate(model, val_loader, slot_ids, device, a.brier)
    for kind, loader in aux_val_loaders.items():
        ev[f"aux_{kind}"] = overall_agreement(evaluate(model, loader, slot_ids, device, a.brier))
    print(f"[eval final] {json.dumps(ev)}")
    log_line(log_path, {"step": step, "eval": ev, "final": True})
    p = save_ckpt(model, processor, out_dir, a.keep, "final", {**meta, "step": step, "eval": ev})
    print(f"[train] saved final checkpoint {p}; {step} steps, {seen} samples, {(time.perf_counter() - t_start) / 60:.1f} min")
    (out_dir / "final_eval.json").write_text(json.dumps(ev, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--shards", nargs="*", default=[], help="use only these shard names (default: every shard under data/<game>/)")
    p.add_argument("--label-delay", type=int, default=0, help="label frame k with the teacher's decision at step k+delay (real-time latency)")
    p.add_argument("--no-delay-games", nargs="*", default=[], help="games whose labels stay unshifted under --label-delay (turn-based: 2048 sokoban)")
    p.add_argument("--boost-last", type=int, nargs=2, default=None, metavar=("K", "TIMES"), help="repeat the last K records of every short (< 500 steps) episode TIMES times (DAgger: oversample the steps before a death)")
    p.add_argument("--history", type=int, default=0, metavar="K", help="put the K moves already made in this episode into the state, oldest first (names, from taken_action)")
    p.add_argument("--boost-games", nargs="*", default=[], help="apply --boost-last to these games only (default: every game in --games)")
    p.add_argument("--model", required=True, help="HF id or local snapshot path")
    p.add_argument("--out", required=True, help="checkpoint directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (overrides epochs)")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.03)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--brier", type=float, default=0.0)
    p.add_argument("--freeze-vision", action="store_true")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--two-frame", action="store_true")
    p.add_argument("--stack", default="temporal", choices=["temporal", "separate"])
    p.add_argument("--long-side", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--eval-samples", type=int, default=2000)
    p.add_argument("--save-every", type=int, default=400)
    p.add_argument("--keep", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="use only this many training records (smoke tests)")
    p.add_argument("--subsample", type=int, default=0, help="random subset of this many training records per game (transfer / data-efficiency runs)")
    p.add_argument("--aux", nargs="*", default=[], help="jsonl manifests of non-game samples (playjev.mixdata)")
    p.add_argument("--mix", nargs=3, type=float, default=None, metavar=("GAME", "IMAGE", "TEXT"),
                   help="share of batches per kind, e.g. 0.6 0.2 0.2; one epoch stays one pass over the game frames")
    p.add_argument("--aux-soft", type=float, default=0.9, help="mass the answer keeps in a non-game target")
    p.add_argument("--aug", nargs=4, type=float, default=[0.0, 0.0, 0.0, 0.0],
                   metavar=("DISTRACTOR", "PRUNE", "PARAPHRASE", "RENAME"),
                   help="per-sample probability of each option rewrite on game frames (playjev.data.GameAug)")
    p.add_argument("--shuffle-labels", action="store_true", help="control: permute the training targets within each game so the frame carries no information about the answer")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
