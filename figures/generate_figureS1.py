"""
Figure S1 - manual measurement overlays for the four annotated cross-sections.
Supporting Information figure backing Table S6.

Usage
-----
1. Put the four manual-measurement images in one folder.
2. Edit the PANELS list below so each filename maps to the right sample.
3. Run from that folder:   python3 generate_figureS1_fixed.py

Nothing is cropped: panels are letterboxed with black to a common aspect ratio,
so all four render at identical size without cutting any overlay or scale bar.

TPJ specs applied here:
  - width 169 mm (6.654 in); at 600 dpi that is 3992 px
  - panel letters (a)-(d), lower case, in parentheses, Helvetica/Arial,
    not bold, drawn inside the panel; the sample name sits above it
  - saved as FigureS1.tif (LZW) + .png + .pdf
"""
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- settings
INPUT_DIR     = Path(".")
OUTPUT_PREFIX = "FigureS1"
DPI           = 600
FIG_WIDTH_IN  = 6.654        # 169 mm, TPJ full page width -- do NOT change
GAP           = 0.04
BLACK_CUTOFF  = 12

# TPJ: "Labelling on figures should be no smaller than 8pt Helvetica.
#       Figure sections should be designated with lower case letters."
PANEL_LETTER_PT = 11
TITLE_PT        = 9

# >>> EDIT THESE FOUR FILENAMES <<<
# Order fixes the panel letters; do not rely on alphabetical sorting.
PANELS = [
    ("S1_I0_measured.tif",     "(a)", "Inowroc\u0142aw, 0 mM"),
    ("S1_I1000_measured.tif",  "(b)", "Inowroc\u0142aw, 1000 mM"),
    ("S1_C0_measured.tif",     "(c)", "Ciechocinek, 0 mM"),
    ("S1_C1000_measured.tif",  "(d)", "Ciechocinek, 1000 mM"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "savefig.dpi": DPI,
})

# ---------------------------------------------------------------- helpers
def trim_border(im, cutoff=BLACK_CUTOFF):
    """Remove the uniform black padding each export carries."""
    a = np.asarray(im.convert("L"))
    rows = np.where(a.max(axis=1) > cutoff)[0]
    cols = np.where(a.max(axis=0) > cutoff)[0]
    if rows.size == 0 or cols.size == 0:
        return im
    return im.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))

def pad_to_aspect(im, target_aspect, fill=(0, 0, 0)):
    """Letterbox onto a canvas of the target aspect, content centred."""
    w, h = im.size
    if w / h < target_aspect:
        new_w, new_h = int(round(h * target_aspect)), h
    else:
        new_w, new_h = w, int(round(w / target_aspect))
    canvas = Image.new("RGB", (new_w, new_h), fill)
    canvas.paste(im, ((new_w - w) // 2, (new_h - h) // 2))
    return canvas

# ---------------------------------------------------------------- load
missing = [f for f, _, _ in PANELS if not (INPUT_DIR / f).exists()]
if missing:
    raise FileNotFoundError(
        "Missing source image(s): " + ", ".join(missing) +
        "\nEdit the PANELS list at the top of this script."
    )

imgs = []
for fname, _, _ in PANELS:
    im = Image.open(INPUT_DIR / fname)
    if im.mode != "RGB":
        im = im.convert("RGB")
    imgs.append(trim_border(im))

print("Panel order -- CHECK THIS before using the output:")
for (fname, letter, title), im in zip(PANELS, imgs):
    print(f"  {letter} {title:<24s} <- {fname}   trimmed to {im.size[0]}x{im.size[1]} px")
print()

target_aspect = max(w / h for w, h in (i.size for i in imgs))
imgs = [pad_to_aspect(i, target_aspect) for i in imgs]
print("panel aspect:", round(target_aspect, 3),
      "| sizes:", [i.size for i in imgs])

# ---------------------------------------------------------------- draw
panel_w = (FIG_WIDTH_IN - 0.04 * FIG_WIDTH_IN) / 2
panel_h = panel_w / target_aspect
fig_h   = 2 * panel_h * (1 + GAP) + 0.45

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN, fig_h))
for ax, img, (_, letter, title) in zip(axes.flat, imgs, PANELS):
    ax.imshow(img, interpolation="lanczos")
    ax.set_anchor("C")
    ax.axis("off")
    # sample name above the panel, no letter in it
    ax.text(0.0, 1.015, title, transform=ax.transAxes,
            fontsize=TITLE_PT, fontweight="normal",
            ha="left", va="bottom", color="black")
    # panel letter inside the panel, white on the black letterbox
    ax.text(0.015, 0.985, letter, transform=ax.transAxes,
            fontsize=PANEL_LETTER_PT, fontweight="normal", color="white",
            ha="left", va="top",
            bbox=dict(facecolor="black", alpha=0.55, pad=2.5, edgecolor="none"))

fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01,
                    wspace=GAP, hspace=GAP * 2.2)

# No bbox_inches: it crops after drawing and silently discards figsize.
fig.savefig(f"{OUTPUT_PREFIX}.png", dpi=DPI)
fig.savefig(f"{OUTPUT_PREFIX}.pdf")
fig.savefig(f"{OUTPUT_PREFIX}.tif", dpi=DPI, pil_kwargs={"compression": "tiff_lzw"})

print(f"\nsaved {OUTPUT_PREFIX}.png / .pdf / .tif  at {DPI} dpi")
print(f"declared size : {FIG_WIDTH_IN*25.4:.1f} x {fig_h*25.4:.1f} mm")
print(f"expected width: {round(FIG_WIDTH_IN*DPI)} px")
