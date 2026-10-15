"""Transparent title-card PNGs (intro + outro) with a soft glow and shadow.

Font sizes scale with min(width, height), so landscape and vertical videos both work.
Cinzel + Cormorant Garamond are used when present (see scripts/get_fonts.sh);
otherwise it falls back to DejaVu / Pillow's default font.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT_DIRS = [Path(__file__).parent / "fonts", Path.cwd() / "fonts"]
FALLBACKS = ["DejaVuSerif.ttf", "DejaVuSans.ttf"]

GOLD, PALE, VIOLET, LILAC = (246, 226, 170), (236, 232, 245), (196, 170, 255), (220, 204, 255)


def _font(name, size, weight=None):
    for d in FONT_DIRS:
        p = d / name
        if p.exists():
            f = ImageFont.truetype(str(p), size)
            if weight:
                try: f.set_variation_by_axes([weight])
                except Exception: pass
            return f
    for fb in FALLBACKS:
        try: return ImageFont.truetype(fb, size)
        except OSError: pass
    return ImageFont.load_default(size)


def card(W, H, lines, out):
    """lines: (text, font file, size as fraction of min(W,H), weight, rgb, letter-spacing frac, y frac)"""
    S = min(W, H)
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    for text, ff, sz, wg, rgb, track, yf in lines:
        if not text: continue
        px = int(sz * S)
        while True:  # shrink until the line fits in 90% of the width
            f = _font(ff, px, wg); sp = track * px
            widths = [d.textlength(c, font=f) for c in text]
            total = sum(widths) + sp * (len(text) - 1)
            if total <= 0.9 * W or px < 8: break
            px = int(px * 0.92)
        x = (W - total) / 2; y = yf * H
        for c, w in zip(text, widths):
            d.text((x, y), c, font=f, fill=rgb + (255,), anchor="lm")
            x += w + sp
    glow = base.filter(ImageFilter.GaussianBlur(S * 0.012))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    shadow.putalpha(base.split()[3].filter(ImageFilter.GaussianBlur(S * 0.035))
                    .point(lambda v: int(min(255, v * 0.95))))
    img = Image.alpha_composite(shadow, glow)
    img = Image.alpha_composite(img, glow)
    Image.alpha_composite(img, base).save(out)


def make(W, H, prefix, title, artist="", year="", tagline="", url=""):
    """Writes <prefix>_intro.png and <prefix>_outro.png."""
    card(W, H, [
        (artist.upper(), "Cinzel.ttf", 0.030, 500, PALE, 0.45, 0.39),
        (title.upper(), "Cinzel.ttf", 0.072, 700, GOLD, 0.12, 0.47),
        (year, "Cormorant.ttf", 0.046, 600, LILAC, 0.6, 0.56),
    ], f"{prefix}_intro.png")
    credit = "  ·  ".join(x for x in (title, year, artist) if x)
    card(W, H, [
        (credit, "Cormorant.ttf", 0.040, 500, PALE, 0.02, 0.44),
        (tagline, "Cormorant.ttf", 0.034, 400, VIOLET, 0.02, 0.51),
        (url.upper(), "Cinzel.ttf", 0.022, 600, GOLD, 0.4, 0.585),
    ], f"{prefix}_outro.png")
