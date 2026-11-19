# generate_f_3.py  -- Figure 3 for TPJ
#
# TPJ specs: composite figure, 600 dpi, width 169 mm, fonts >= 8 pt at print size.
# 169 mm = 6.654 in  ->  6.654 * 600 = 3992 px wide.

from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

INPUT_DIR     = Path(".")
OUTPUT_PREFIX = "Figure3"          # TPJ naming: Figure3.tif
CROP_TOP      = 0.09
DPI           = 600

FIG_WIDTH_IN  = 6.654              # 169 mm -- do NOT change
FIG_HEIGHT_IN = 5.82               # keeps the original 7.087:6.2 proportion

# Explicit order. Rename your source files to match, or edit this list.
# The script will tell you exactly what it matched before it draws anything.
EXPECTED = ["I-0", "I-1000", "C-0", "C-1000"]

TITLES = [
    "Inowroc\u0142aw, 0 mM",
    "Inowroc\u0142aw, 1000 mM",
    "Ciechocinek, 0 mM",
    "Ciechocinek, 1000 mM",
]

# TPJ: lower case letters, no smaller than 8 pt Helvetica. Arial is
# metrically identical and available everywhere. Letters go inside the
# panel, as agreed for photographic figures; the sample name stays above.
LETTERS         = ["(a)", "(b)", "(c)", "(d)"]
PANEL_LETTER_PT = 11
TITLE_PT        = 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "savefig.dpi": DPI,
})


def load_img(p, crop_top=0):
    im = Image.open(p)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if crop_top > 0:
        w, h = im.size
        im = im.crop((0, int(h * crop_top), w, h))
    return im


VALID_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

# Collect inputs, excluding anything this script itself produced.
files = sorted([
    p for p in INPUT_DIR.iterdir()
    if p.suffix.lower() in VALID_EXT
    and not p.name.startswith(OUTPUT_PREFIX)
    and not p.name.startswith("_")
])

if len(files) != 4:
    raise SystemExit(
        f"Figure 3 needs exactly 4 source images, found {len(files)}:\n  "
        + "\n  ".join(f.name for f in files)
    )

print("Panel order -- CHECK THIS before using the output:")
for tag, f, t, l in zip(EXPECTED, files, TITLES, LETTERS):
    print(f"  {l} {t:<28s} <- {f.name}   (should be {tag})")
print()

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

for ax, fpath, title, letter in zip(axes.flat, files, TITLES, LETTERS):
    ax.imshow(load_img(fpath, CROP_TOP))
    ax.axis("off")
    # sample name above the panel, no letter in it
    ax.text(0, 1.03, title, transform=ax.transAxes,
            fontsize=TITLE_PT, fontweight="normal", ha="left", va="bottom")
    # panel letter inside the panel, white on the dark background
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_PT, fontweight="normal", color="white",
            ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.55, pad=2.5, edgecolor="none"))

GAP = 0.10
fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                    wspace=GAP, hspace=GAP)

# No bbox_inches='tight' -- it silently overrides figsize and shrinks the figure.
fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=DPI)
fig.savefig(f"{OUTPUT_PREFIX}.tif", dpi=DPI,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{OUTPUT_PREFIX}.pdf")

print(f"saved {OUTPUT_PREFIX}.png / .tif / .pdf")
print(f"declared size : {FIG_WIDTH_IN*25.4:.1f} x {FIG_HEIGHT_IN*25.4:.1f} mm")
print(f"expected width: {round(FIG_WIDTH_IN*DPI)} px")
