#!/usr/bin/env python3
"""generate_f_4_fixed.py
Figure 4: eight annotated longitudinal sections, two per sample.

Follows exactly the same conventions as Figure 3:
  - double-column width, 169 mm at 600 dpi = 3992 px
  - panel designators lower case, in parentheses, 11 pt, NOT bold,
    INSIDE the panel at top left, white on a semi-transparent black box
  - sample name above the panel, 9 pt, not bold
  - NO bbox_inches='tight' (it silently overrides figsize)

Run from the folder holding 01.tif ... 08.tif:
    cd ~/Desktop/figure_4
    python3 generate_f_4_fixed.py
"""

from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- settings
INPUT_DIR = Path(".")
OUTPUT = "Figure4"
CROP_TOP = 0.09          # removes burnt-in "s1-long" header
WIDTH_MM = 169.0         # TPJ double column
DPI = 600

# explicit order - never rely on sorted(), file names have lied before
FILES = ["01.tif", "02.tif", "03.tif", "04.tif",
         "05.tif", "06.tif", "07.tif", "08.tif"]

LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

# one sample name per ROW: each row holds the two longitudinal views of
# the same sample, so the name is centred once over the pair
ROW_TITLES = [
    "Inowroc\u0142aw, 0 mM",
    "Inowroc\u0142aw, 1000 mM",
    "Ciechocinek, 0 mM",
    "Ciechocinek, 1000 mM",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})


def load_img(p, crop_top=0.0):
    im = Image.open(p)
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if crop_top > 0:
        w, h = im.size
        im = im.crop((0, int(h * crop_top), w, h))
    return im


paths = [INPUT_DIR / f for f in FILES]
missing = [p.name for p in paths if not p.exists()]
if missing:
    raise SystemExit(
        "MISSING: " + ", ".join(missing) +
        f"\nYou are in: {Path.cwd()}\n"
        "Copy eight longitudinal files here as 01.tif ... 08.tif,\n"
        "in the order S1 long, S1 long2, S4 long, S4 long2, "
        "S5 long, S5 long2, S8 long, S8 long2."
    )

imgs = [load_img(p, CROP_TOP) for p in paths]
for p, im in zip(paths, imgs):
    print(f"{im.size}  {p.name}")

# page geometry ------------------------------------------------------------
width_in = WIDTH_MM / 25.4                       # 6.654 in -> 3992 px @600
ar = sum(im.height / im.width for im in imgs) / len(imgs)   # mean aspect
panel_w = width_in / 2
height_in = 4 * panel_w * ar * 1.14              # 1.14 leaves room for titles

fig, axes = plt.subplots(4, 2, figsize=(width_in, height_in), dpi=DPI)
axes = axes.flat

for ax, im, letter in zip(axes, imgs, LETTERS):
    ax.imshow(im)
    ax.axis("off")
    # panel letter inside the panel, top left
    ax.text(0.015, 0.975, letter, transform=ax.transAxes,
            fontsize=11, fontweight="normal", color="white",
            ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.55,
                      edgecolor="none", pad=2.0))

# NO bbox_inches here, and no tight_layout: both can override figsize
fig.subplots_adjust(left=0.004, right=0.996, top=0.972, bottom=0.004,
                    wspace=0.02, hspace=0.11)

# one centred row title per sample, placed above the row from the real
# axes geometry (set_title on a single axes would sit over one panel only)
axlist = fig.axes
for r, title in enumerate(ROW_TITLES):
    top = axlist[2 * r].get_position().y1
    fig.text(0.5, top + 0.004, title, ha="center", va="bottom",
             fontsize=9, fontweight="normal")

fig.savefig(f"{OUTPUT}.tif", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{OUTPUT}.png", dpi=DPI)
plt.close(fig)

w, h = Image.open(f"{OUTPUT}.tif").size
print(f"\nsaved {OUTPUT}.tif  {w} x {h} px")
print(f"printed width {w / DPI * 25.4:.1f} mm   (target {WIDTH_MM} mm)")
if abs(w - 3992) > 40:
    print("!! width off target - check that no 'tight' option crept back in")
