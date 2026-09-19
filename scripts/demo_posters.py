"""Still frames the demo puts behind a tile before its game has painted.

    python scripts/demo_posters.py

Without one a tile is a black square until the iframe loads, and again for a moment between
episodes. The still is the game's own screenshot from docs/assets/games/, resized; the tile
positions it with the same vertical alignment it crops the live view with, so the picture does
not jump when the game takes over.
"""
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "docs/assets/games", ROOT / "demo/assets/poster"
LONG_SIDE = 560

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("*.png")):
        im = Image.open(f).convert("RGB")
        k = LONG_SIDE / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        dst = OUT / (f.stem + ".webp")
        im.save(dst, "WEBP", quality=78, method=6)
        print(f"{dst.relative_to(ROOT)}  {im.width}x{im.height}  {dst.stat().st_size / 1024:.0f} KB")

if __name__ == "__main__":
    main()
