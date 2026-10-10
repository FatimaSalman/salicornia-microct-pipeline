# generate_figure4.py
# Figure 4: annotated longitudinal sections, 8 panels, new titles without S1/S4/S5/S8
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# ---------- settings ----------
INPUT_DIR = Path(".")
OUTPUT_PREFIX = "Figure4_final"
CROP_TOP = 0.09  # نفس الفكرة: قص العنوان القديم s1-long إلخ
DPI = 600

TITLES = [
    "a) Inowrocław 0 mM — long1",
    "b) Inowrocław 0 mM — long2",
    "c) Inowrocław 1000 mM — long1",
    "d) Inowrocław 1000 mM — long2",
    "e) Ciechocinek 0 mM — long1",
    "f) Ciechocinek 0 mM — long2",
    "g) Ciechocinek 1000 mM — long1",
    "h) Ciechocinek 1000 mM — long2",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "savefig.dpi": DPI,
})

def load_img(p, crop_top=0):
    im = Image.open(p)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
    if crop_top > 0:
        w, h = im.size
        y0 = int(h * crop_top)
        im = im.crop((0, y0, w, h))
    return im

files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in [".tif",".tiff",".png",".jpg",".jpeg"]])
print(f"found {len(files)} files:", [f.name for f in files])
assert len(files) == 8, f"Figure 4 needs exactly 8 images, found {len(files)}"

fig, axes = plt.subplots(4, 2, figsize=(7.087, 10.0))
axes = axes.flat

for ax, fpath, title in zip(axes, files, TITLES):
    img = load_img(fpath, CROP_TOP)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=3)

plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
plt.savefig(f"{OUTPUT_PREFIX}.png", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{OUTPUT_PREFIX}.tif", dpi=DPI, bbox_inches="tight")
plt.savefig(f"{OUTPUT_PREFIX}.pdf", bbox_inches="tight")
print(f"saved {OUTPUT_PREFIX}.png/.tif/.pdf")