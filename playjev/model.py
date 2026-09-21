"""PlayJevModel: game frames in, a probability over lettered actions out, one forward pass.

The prompt is the frozen OpenJev wording (`openjev-letters-v1`, omnijev/openjev/prompt.py) with the
frame(s) standing in for the text state. Nothing about the game is in the prompt (no name, no rules):
the model sees the pixels and the option list only. The readout is a float32 softmax restricted to the
option-letter tokens at the last position, and the confidence is Jev's (p_max - 1/K) / (1 - 1/K).

Two prompt templates, fixed at construction:
  plain  raw text, for the Base checkpoints. Ends with "Answer:" so the answer slots are the
         space-prefixed letter tokens " A", " B", ... (the standard base-model MCQ readout).
  chat   the checkpoint's chat template with thinking disabled; the prompt ends with the closed
         <think></think> block and the answer slots are the bare letter tokens "A", "B", ...

Two frames per state are supported two ways. `stack="temporal"` puts the previous frame in temporal
slot 0 and the current frame in slot 1 of the vision tower's 2-frame patch (the layout a Qwen video
uses), so a state costs the same visual tokens as one frame and the prompt has one placeholder.
`stack="separate"` sends two images and two placeholders.
"""
from __future__ import annotations

import importlib.util
import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

PROMPT_VERSION = "playjev-letters-v1"  # openjev-letters-v1 wording, frames as the state

# Frozen OpenJev system sentence (beat the "You are a System One decision model" wording by 3 to 6 points).
SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
DEFAULT_INSTRUCTIONS = "Which move should the player make next?"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
FRAME_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"  # the processor expands <|image_pad|> to one token per merged patch
PLAIN_ANSWER_CUE = "Answer:"


def options_block(options: Sequence[dict]) -> str:
    """Lettered `name: description` lines, exactly as OpenJev renders a Choice question."""
    if len(options) > len(LETTERS):
        raise ValueError(f"{len(options)} options exceed the {len(LETTERS)} letter slots")
    lines = []
    for letter, opt in zip(LETTERS, options):
        desc = (opt.get("description") or "").strip()
        lines.append(f"{letter}. {opt['name']}: {desc}" if desc else f"{letter}. {opt['name']}")
    return "\n".join(lines)


def render_suffix(options: Sequence[dict], instructions: str = DEFAULT_INSTRUCTIONS) -> str:
    letters = ", ".join(LETTERS[: len(options)])
    return f"Question: {instructions}\n\nOptions:\n{options_block(options)}\n\nAnswer with one letter: {letters}."


def render_state(n_placeholders: int, text: str | None = None) -> str:
    """The state block: frame placeholders first, then a text body if the state has one. A state may be
    frames (a game), text (a document, a question stem), or both; it may not be empty."""
    parts = [FRAME_PLACEHOLDER] * n_placeholders
    if text and text.strip():
        parts.append(text.strip())
    if not parts:
        raise ValueError("a state needs at least one frame or a text body")
    return "<state>\n" + "\n".join(parts) + "\n</state>"


def build_plain_prompt(options: Sequence[dict], instructions: str = DEFAULT_INSTRUCTIONS, n_placeholders: int = 1,
                       state_text: str | None = None) -> str:
    return (f"{SYSTEM_PROMPT}\n\n{render_state(n_placeholders, state_text)}\n\n"
            f"{render_suffix(options, instructions)}\n{PLAIN_ANSWER_CUE}")


def stack_temporal_patches(prev_pv, cur_pv, temporal_patch_size: int = 2, patch_size: int = 16):
    """Merge two single-frame patch tensors into one two-frame patch tensor.

    The image processor lays each patch out as (channel, temporal, patch, patch) with the frame repeated in both
    temporal slots. A video puts consecutive frames in the two slots, so we do the same: previous frame in slot 0,
    current frame in slot 1. Token count and grid are unchanged. Both tensors must come from same-size frames."""
    if prev_pv.shape != cur_pv.shape:
        raise ValueError("previous and current frames must have the same size for temporal stacking")
    c, t, p = 3, temporal_patch_size, patch_size
    prev = prev_pv.view(-1, c, t, p, p).clone()
    prev[:, :, 1:] = cur_pv.view(-1, c, t, p, p)[:, :, 1:]
    return prev.view(-1, c * t * p * p)


def choice_confidence(probs: Sequence[float]) -> float:
    k = len(probs)
    if k == 1:
        return 1.0
    return (max(probs) - 1.0 / k) / (1.0 - 1.0 / k)


@dataclass
class Decision:
    probs: list[float]  # over the options, in the order given
    choice: int  # argmax index into options
    confidence: float  # Jev Choice confidence
    allowed_mass: float  # share of the full-vocabulary softmax that lands on the K answer slots
    top_token: str  # most likely next token over the whole vocabulary (diagnostic: is the model answering at all?)


@dataclass
class Timing:
    prep_s: float = 0.0  # JPEG decode + processor
    forward_s: float = 0.0  # GPU forward including the readout, synchronised
    input_tokens: int = 0  # per sample
    visual_tokens: int = 0  # per sample
    extra: dict[str, Any] = field(default_factory=dict)


class PlayJevModel:
    def __init__(self, model_id: str, device: str = "cuda:0", dtype: str = "bfloat16", template: str = "plain") -> None:
        if template not in ("plain", "chat"):
            raise ValueError("template must be 'plain' or 'chat'")
        self.model_id, self.device_str, self.dtype_str, self.template = model_id, device, dtype, template
        self.model = self.processor = None
        self.slot_ids: list[int] = []
        self.load_seconds = 0.0
        self.last_timing = Timing()

    # ------------------------------------------------------------------ setup
    def load(self) -> "PlayJevModel":
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        t0 = time.perf_counter()
        self.device = torch.device(self.device_str)
        self.torch_dtype = getattr(torch, self.dtype_str)
        local = Path(self.model_id).exists()
        self.processor = AutoProcessor.from_pretrained(self.model_id, local_files_only=local)
        self.processor.tokenizer.padding_side = "left"  # the last position is the answer slot for every row
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=self.torch_dtype, device_map={"": str(self.device)}, local_files_only=local)
        self.model.eval()
        # Readout in float32. bf16 logits at magnitude 25 to 50 are quantised to 0.125, which lets options tie;
        # the head weight is tied to the embedding, so keep a float32 copy for the readout instead of retyping it.
        self.head_w32 = self.model.get_output_embeddings().weight.detach().float()
        self.slot_ids = self._verify_slots()
        ip = self.processor.image_processor
        self.patch_size, self.merge_size, self.temporal_patch_size = ip.patch_size, ip.merge_size, ip.temporal_patch_size
        self.image_token_id = self.model.config.image_token_id
        self.kernels = {name: importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")}
        self.load_seconds = time.perf_counter() - t0
        return self

    def refresh_head(self) -> None:
        """Call after the weights change (training) so the float32 readout copy follows."""
        self.head_w32 = self.model.get_output_embeddings().weight.detach().float()

    def slot_text(self, letter: str) -> str:
        return f" {letter}" if self.template == "plain" else letter

    def _verify_slots(self) -> list[int]:
        tok = self.processor.tokenizer
        ids = []
        for letter in LETTERS:
            text = self.slot_text(letter)
            enc = tok.encode(text, add_special_tokens=False)
            if len(enc) != 1 or tok.decode(enc) != text:
                raise ValueError(f"answer slot {text!r} is not a single round-trip token in {self.model_id}")
            ids.append(enc[0])
        if len(set(ids)) != len(ids):
            raise ValueError("answer slot tokens collide")
        return ids

    # ----------------------------------------------------------------- prompt
    def build_prompt(self, options: Sequence[dict], instructions: str = DEFAULT_INSTRUCTIONS,
                     frames_per_state: int = 1, stack: str = "temporal", state_text: str | None = None) -> str:
        """The exact text handed to the processor (before <|image_pad|> is expanded to one token per merged patch)."""
        n = 0 if frames_per_state == 0 else (1 if frames_per_state == 1 or stack == "temporal" else frames_per_state)
        if self.template == "plain":
            return build_plain_prompt(options, instructions, n, state_text)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_state(n, state_text) + "\n\n" + render_suffix(options, instructions)}]
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                            enable_thinking=False)
        if not text.endswith("</think>\n\n"):
            raise ValueError("chat prompt does not end with a closed think block; the readout would hit a reasoning token")
        return text

    # ----------------------------------------------------------------- images
    @staticmethod
    def _to_image(frame: Any, long_side: int | None) -> Image.Image:
        im = frame if isinstance(frame, Image.Image) else Image.open(io.BytesIO(frame))
        im = im.convert("RGB")
        if long_side and max(im.size) != long_side:
            s = long_side / max(im.size)
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        return im

    def _stack_temporal(self, prev_pv, cur_pv):
        return stack_temporal_patches(prev_pv, cur_pv, self.temporal_patch_size, self.patch_size)

    def _prepare(self, items: Sequence[Any], prompts: Sequence[str], frames_per_state: int, stack: str,
                 long_side: int | None):
        b = len(items)
        if len(prompts) != b:
            raise ValueError("one prompt per state")
        if frames_per_state == 0:
            return self.processor(text=list(prompts), return_tensors="pt", padding=True)
        if frames_per_state == 1:
            imgs = [self._to_image(x, long_side) for x in items]
            enc = self.processor(text=list(prompts), images=imgs, return_tensors="pt", padding=True)
        elif stack == "separate":
            imgs = [[self._to_image(f, long_side) for f in item] for item in items]
            enc = self.processor(text=list(prompts), images=imgs, return_tensors="pt", padding=True)
        elif stack == "temporal":
            if frames_per_state != 2:
                raise ValueError("temporal stacking takes exactly 2 frames per state")
            cur = [self._to_image(item[-1], long_side) for item in items]
            prev = [self._to_image(item[0], long_side) for item in items]
            prev = [p if p.size == c.size else p.resize(c.size, Image.BILINEAR) for p, c in zip(prev, cur)]
            enc = self.processor(text=list(prompts), images=cur, return_tensors="pt", padding=True)
            prev_enc = self.processor.image_processor(prev, return_tensors="pt")
            if not (prev_enc["image_grid_thw"] == enc["image_grid_thw"]).all():
                raise ValueError("previous and current frames resize to different grids")
            enc["pixel_values"] = self._stack_temporal(prev_enc["pixel_values"], enc["pixel_values"])
        else:
            raise ValueError("stack must be 'temporal' or 'separate'")
        return enc

    # ---------------------------------------------------------------- forward
    def _forward(self, enc):
        import torch

        enc = {k: v.to(self.device) for k, v in enc.items() if hasattr(v, "to")}
        kw = {k: enc[k] for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids") if k in enc}
        with torch.inference_mode():
            out = self.model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                   use_cache=False, return_dict=True, **kw)
            h = out.last_hidden_state[:, -1, :].float()  # left padding: the last position is the answer slot
            logits = h @ self.head_w32.T  # (B, V) float32
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return logits

    def _readout(self, logits, k: int) -> list[Decision]:
        import torch

        slots = torch.tensor(self.slot_ids[:k], device=logits.device)
        sel = logits[:, slots]
        probs = torch.softmax(sel, -1)
        mass = (torch.logsumexp(sel, -1) - torch.logsumexp(logits, -1)).exp()
        top = logits.argmax(-1)
        out = []
        for row in range(logits.shape[0]):
            p = probs[row].tolist()
            out.append(Decision(probs=p, choice=max(range(k), key=p.__getitem__), confidence=choice_confidence(p),
                                allowed_mass=float(mass[row]), top_token=self.processor.tokenizer.decode([int(top[row])])))
        return out

    # ----------------------------------------------------------------- public
    def decide(self, frames: Sequence[Any], options: Sequence[dict], instructions: str = DEFAULT_INSTRUCTIONS,
               frames_per_state: int = 1, stack: str = "temporal", batch_size: int = 32,
               long_side: int | None = None,
               state_text: str | Sequence[str | None] | None = None) -> list[Decision]:
        """One decision per state. `frames` holds one item per state: JPEG bytes (or a PIL image) for
        frames_per_state=1, else a (previous, current) pair. With `frames_per_state=0` the state is
        `state_text` alone and `frames` only sets the batch length (pass a list of None). `state_text` is
        one string for every state, or a sequence with one entry per state when the states carry different
        text. All states share the option list. `long_side` optionally rescales frames."""
        if self.model is None:
            self.load()
        if frames_per_state not in (0, 1, 2):
            raise ValueError("frames_per_state must be 0, 1 or 2")
        texts = ([state_text] * len(frames) if state_text is None or isinstance(state_text, str)
                 else list(state_text))
        if len(texts) != len(frames):
            raise ValueError("state_text must be one string or one per state")
        prompts = [self.build_prompt(options, instructions, frames_per_state, stack, t) for t in texts]
        k = len(options)
        decisions: list[Decision] = []
        prep = fwd = 0.0
        timing = Timing()
        for start in range(0, len(frames), batch_size):
            items = frames[start:start + batch_size]
            t0 = time.perf_counter()
            enc = self._prepare(items, prompts[start:start + batch_size], frames_per_state, stack, long_side)
            t1 = time.perf_counter()
            logits = self._forward(enc)
            t2 = time.perf_counter()
            prep += t1 - t0
            fwd += t2 - t1
            decisions.extend(self._readout(logits, k))
            timing.input_tokens = int(enc["attention_mask"][0].sum())
            timing.visual_tokens = int((enc["input_ids"][0] == self.image_token_id).sum())  # 0 for a text state
        timing.prep_s, timing.forward_s = prep, fwd
        self.last_timing = timing
        return decisions
