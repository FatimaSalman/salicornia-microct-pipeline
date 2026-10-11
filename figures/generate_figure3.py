# generated_f_3.py
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

INPUT_DIR = Path(".")
OUTPUT_PREFIX = "Figure3_final"
CROP_TOP = 0.09
DPI = 600

TITLES = [
    "a) Inowrocław, 0 mM",
    "b) Inowrocław, 1000 mM",
    "c) Ciechocinek, 0 mM",
    "d) Ciechocinek, 1000 mM",
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

VALID_EXT = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
files = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in VALID_EXT])
print(f"found {len(files)} files:", [f.name for f in files])
assert len(files) == 4, f"Figure 3 needs exactly 4 images, found {len(files)}"

fig, axes = plt.subplots(2, 2, figsize=(7.087, 6.2))
axes = axes.flat

for ax, fpath, title in zip(axes, files, TITLES):
    img = load_img(fpath, CROP_TOP)
    ax.imshow(img)          # ✅ بدون aspect="auto" — نفس أبعاد الصورة الأصلية
    ax.axis("off")
    # ✅ العنوان كـ text فوق الـ axes مباشرة، مسافة ثابتة كنسبة من ارتفاع الصورة
    ax.text(0, 1.03, title, transform=ax.transAxes,
             fontsize=9, fontweight="bold", ha="left", va="bottom")

# ✅ نفس القيمة بالضبط أفقي وعمودي = نفس المسافة البصرية
GAP = 0.10
fig.subplots_adjust(
    left=0.02, right=0.98,
    top=0.94, bottom=0.02,
    wspace=GAP,
    hspace=GAP,
)

plt.savefig(f"{OUTPUT_PREFIX}.png", dpi=DPI)
plt.savefig(f"{OUTPUT_PREFIX}.tif", dpi=DPI)
plt.savefig(f"{OUTPUT_PREFIX}.pdf")
print(f"saved {OUTPUT_PREFIX}.png/.tif/.pdf")