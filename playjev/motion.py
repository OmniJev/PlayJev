"""Put motion into a single still frame.

The Jev contract gives the policy one forward pass per frame. A still frame
carries position but not velocity, so a ball, a falling piece or a jumping
sprite is ambiguous: the same picture appears going up and going down. These
composers fold the k most recent frames into one image that still costs one
image (182 visual tokens at 448 px), so the token budget and the contract are
untouched.

Two modes are offered:

  ghost  the pixels that moved are pasted back over the current frame at a
         rising alpha, so the trail reads oldest-faintest to newest-solid.
         Direction is recoverable from the brightness gradient alone.
  rgbt   the k frames become the R, G and B channels of one image. Anything
         static stays grey; anything moving leaves a coloured fringe whose
         colour order gives the direction.

Equal weights across k frames are deliberately not offered: k identical copies
of a symmetric average carry no ordering, so the composite says that something
moved and refuses to say which way.

k = 3 is the default because two frames give velocity and three give
acceleration, which is what a bouncing ball and a falling piece need.
"""

from PIL import Image, ImageChops

K = 3
GHOST_ALPHAS = (0.30, 0.60)  # t-2, t-1; the current frame is drawn solid
GHOST_THRESH = 25  # 0..255 per-pixel difference that counts as movement


def _rgb(f):
    return f if f.mode == "RGB" else f.convert("RGB")


def ghost(frames, alphas=GHOST_ALPHAS, thresh=GHOST_THRESH):
    """Threshold-masked trail: only moved pixels ghost, background and the current object stay clean.

    A pixel gets a ghost when an older frame differs from the current one there (something was there and
    left). The current frame is then pasted back solid wherever the older frames agreed with each other
    and disagree with the current one (something just arrived there), so the newest position is never
    washed out by the background the older frames carry. Trail brightness rises with time, which is what
    makes the direction readable."""
    fs = [_rgb(f) for f in frames][-(len(alphas) + 1):]
    while len(fs) < len(alphas) + 1:
        fs.insert(0, fs[0])
    cur = fs[-1]
    fs = [f if f.size == cur.size else f.resize(cur.size) for f in fs]
    olds, out = fs[:-1], cur.copy()
    hot = lambda a, b: ImageChops.difference(a, b).convert("L").point(lambda v, t=thresh: 255 if v > t else 0)
    for f, a in zip(olds, alphas):
        out = Image.composite(Image.blend(out, f, a), out, hot(f, cur))
    agree = ImageChops.invert(hot(olds[0], olds[-1])) if len(olds) > 1 else Image.new("L", cur.size, 255)
    arrived = ImageChops.multiply(agree, hot(olds[-1], cur))
    return Image.composite(cur, out, arrived)


def rgbtime(frames):
    """Oldest -> R, middle -> G, newest -> B. Static pixels stay grey."""
    fs = [_rgb(f) for f in frames][-3:]
    while len(fs) < 3:
        fs.insert(0, fs[0])
    size = fs[-1].size
    ch = [f.convert("L") if f.size == size else f.resize(size).convert("L") for f in fs]
    return Image.merge("RGB", tuple(ch))


def compose(frames, mode):
    """frames: oldest .. newest, at least one. Returns a PIL RGB image."""
    if not frames:
        raise ValueError("compose needs at least one frame")
    if mode == "none":
        return _rgb(frames[-1])
    if mode == "ghost":
        return ghost(frames)
    if mode == "rgbt":
        return rgbtime(frames)
    raise ValueError(f"unknown motion mode {mode!r}")


MODES = ("none", "ghost", "rgbt")
