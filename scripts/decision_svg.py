"""The decision diagram for the README: frame + option list, one forward pass, a probability per move.

Light and dark SVGs, drawn at fixed coordinates so GitHub renders them identically everywhere.
Arrowheads are explicit paths and the font is a plain stack, since the viewer's browser draws this.

    python scripts/decision_svg.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

W, H = 960, 252
MOVES = [("up", 0.05), ("down", 0.02), ("left", 0.90), ("right", 0.03)]

LIGHT = dict(bg="#ffffff", ink="#16181c", muted="#6d747e", fill="#f6f7f9", stroke="#e3e6ea",
             line="#c9ced5", accent="#2a78d6", dim="#bcd6f5", track="#eef0f2")
DARK = dict(bg="#0c0d0f", ink="#e9eaec", muted="#8f97a2", fill="#15171a", stroke="#262a30",
            line="#3a3f47", accent="#3987e5", dim="#1f4878", track="#1b1e22")


def box(c, x, y, w, h, title, sub):
    return f"""  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c['fill']}" stroke="{c['stroke']}" stroke-width="1.5"/>
  <text x="{x + w / 2}" y="{y + h / 2 - 6}" text-anchor="middle" font-size="19" font-weight="600" fill="{c['ink']}">{title}</text>
  <text x="{x + w / 2}" y="{y + h / 2 + 18}" text-anchor="middle" font-size="14.5" fill="{c['muted']}">{sub}</text>"""


def arrow(c, x0, y0, x1, y1):
    """Straight run with a solid head at (x1, y1)."""
    return f"""  <path d="M {x0} {y0} L {x1 - 9} {y1}" stroke="{c['line']}" stroke-width="2.4" fill="none"/>
  <path d="M {x1} {y1} L {x1 - 11} {y1 - 6.5} L {x1 - 11} {y1 + 6.5} Z" fill="{c['line']}"/>"""


def svg(dark=False):
    c = DARK if dark else LIGHT
    cy = 128
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'role="img" aria-label="A game frame and an option list go into one forward pass, '
             f'which gives a probability for every move">',
             f'  <rect width="{W}" height="{H}" fill="{c["bg"]}"/>',
             f'  <g font-family="{FONT}">']

    parts.append(box(c, 30, cy - 90, 200, 76, "game frame", "the only input"))
    parts.append(box(c, 30, cy + 14, 200, 76, "option list", "up &#183; down &#183; left &#183; right"))

    # the two inputs join, then one line enters the pass
    parts.append(f'  <path d="M 230 {cy - 52} H 262 V {cy + 52} H 230" stroke="{c["line"]}" '
                 f'stroke-width="2.4" fill="none" stroke-linejoin="round"/>')
    parts.append(arrow(c, 262, cy, 300, cy))

    parts.append(f'  <rect x="300" y="{cy - 56}" width="240" height="112" rx="14" fill="{c["fill"]}" '
                 f'stroke="{c["accent"]}" stroke-width="2"/>')
    parts.append(f'  <text x="420" y="{cy - 6}" text-anchor="middle" font-size="21" font-weight="700" '
                 f'fill="{c["ink"]}">one forward pass</text>')
    parts.append(f'  <text x="420" y="{cy + 20}" text-anchor="middle" font-size="14.5" '
                 f'fill="{c["muted"]}">PlayJev 0.8B</text>')

    parts.append(arrow(c, 540, cy, 596, cy))

    parts.append(f'  <text x="656" y="{cy - 76}" font-size="14.5" fill="{c["muted"]}">'
                 f'a probability for every move</text>')
    x0, full = 668, 200
    for i, (name, p) in enumerate(MOVES):
        y = cy - 51 + i * 34
        hot = p == max(v for _, v in MOVES)
        parts.append(f'  <text x="{x0 - 14}" y="{y + 14.5}" text-anchor="end" font-size="16" '
                     f'fill="{c["ink"]}">{name}</text>')
        parts.append(f'  <rect x="{x0}" y="{y}" width="{full}" height="20" rx="4" fill="{c["track"]}"/>')
        parts.append(f'  <rect x="{x0}" y="{y}" width="{max(4, round(full * p))}" height="20" rx="4" '
                     f'fill="{c["accent"] if hot else c["dim"]}"/>')
        parts.append(f'  <text x="{x0 + full + 14}" y="{y + 14.5}" font-size="16" '
                     f'font-weight="{700 if hot else 400}" fill="{c["ink"] if hot else c["muted"]}">'
                     f'{p:.2f}</text>')

    parts += ["  </g>", "</svg>", ""]
    p = OUT / ("decision-flow-dark.svg" if dark else "decision-flow.svg")
    p.write_text("\n".join(parts), encoding="utf-8")
    return p


if __name__ == "__main__":
    for dark in (False, True):
        print("  ", svg(dark))
