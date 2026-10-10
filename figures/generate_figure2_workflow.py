import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np
import tifffile
import os

# ============================================================
# 1) تحميل الصور
# ============================================================
def load_img(path):
    """يحمّل الصورة بأمان، ويطبّع الصور 16-bit للعرض بتباين جيد"""
    if path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(path)
        if img.dtype in (np.uint16, np.int16):
            img = img.astype(np.float32)
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) / (p98 - p2 + 1e-9), 0, 1)
        return img
    else:
        return np.array(Image.open(path).convert('RGB'))

FILES = {
    'A1': 'PanelA1_crosssection_FINAL.tif',
    'A2': 'PanelA2_longitudinal_FINAL.tif',
    'B1': 'PanelB1_FINAL.png',
    'B2': 'PanelB2_FINAL.png',
    'C1': 'PanelC1_s5_longitudinal_zaveraged.tif',
    'C2': 'PanelC2_s5_crosssection_zaveraged.tif',
    'D':  'PanelD_boundary_methods.png',
    'E':  'PanelE_labels_vs_prediction.png',
    'F':  'PanelF_s8_3d_slicer.png',
}

for k, v in FILES.items():
    if not os.path.exists(v):
        raise FileNotFoundError(f"❌ الملف غير موجود: {v}")

imgs = {k: load_img(v) for k, v in FILES.items()}

def aspect(img):
    h, w = img.shape[0], img.shape[1]
    return w / h

# ============================================================
# 2) حساب نسب الأبعاد (aspect ratios) لكل صورة/مجموعة
# ============================================================
aA1, aA2 = aspect(imgs['A1']), aspect(imgs['A2'])
aB1, aB2 = aspect(imgs['B1']), aspect(imgs['B2'])
aC1, aC2 = aspect(imgs['C1']), aspect(imgs['C2'])
aD = aspect(imgs['D'])
aE = aspect(imgs['E'])
aF = aspect(imgs['F'])

# مجموعة A: A1 وA2 جنب بعض (نفس الارتفاع)
groupA_aspect = aA1 + aA2
# مجموعة B: B1 فوق B2 (نفس العرض) — 3D Slicer
groupB_aspect = 1 / (1 / aB1 + 1 / aB2)
# مجموعة C: C1 وC2 جنب بعض
groupC_aspect = aC1 + aC2

row1_widths = [groupA_aspect, groupB_aspect, groupC_aspect]
row2_aspect = aD
row3_widths = [aE, aF]

# ============================================================
# 3) حساب أبعاد الشكل النهائي (180mm عرض، حفاظ على النسب)
# ============================================================
W = 1.0
H1 = W / sum(row1_widths)
H2 = W / row2_aspect
H3 = W / sum(row3_widths)

FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
scale = FIG_WIDTH_IN / W
FIG_HEIGHT_IN = (H1 + H2 + H3) * scale

DPI = 600

fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI)

outer = gridspec.GridSpec(
    3, 1, height_ratios=[H1, H2, H3],
    hspace=0.06, figure=fig
)

# ---------------- الصف 1: A | B | C ----------------
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[0],
    width_ratios=row1_widths, wspace=0.04
)

gA = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=row1[0],
    width_ratios=[aA1, aA2], wspace=0.02
)
axA1 = fig.add_subplot(gA[0])
axA2 = fig.add_subplot(gA[1])

gB = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row1[1],
    height_ratios=[1/aB1, 1/aB2], hspace=0.02
)
axB1 = fig.add_subplot(gB[0])
axB2 = fig.add_subplot(gB[1])

gC = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=row1[2],
    width_ratios=[aC1, aC2], wspace=0.02
)
axC1 = fig.add_subplot(gC[0])
axC2 = fig.add_subplot(gC[1])

# ---------------- الصف 2: D (عرض كامل) ----------------
axD = fig.add_subplot(outer[1])

# ---------------- الصف 3: E | F ----------------
row3 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[2],
    width_ratios=row3_widths, wspace=0.04
)
axE = fig.add_subplot(row3[0])
axF = fig.add_subplot(row3[1])

# ============================================================
# 4) عرض الصور + ليبل البانل
# ============================================================
def show(ax, img, label='', label_position=(0.01,0.99)):
    cmap = 'gray' if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if label:
        ax.text(
            label_position[0], label_position[1], label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', color='white',
            ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.55, pad=3, edgecolor='none')
        )

show(axA1, imgs['A1'], 'A')
show(axA2, imgs['A2'])

show(axB1, imgs['B1'], 'B', label_position=(0.97, 0.97))
axB1.texts[-1].set_ha('right')
show(axB2, imgs['B2'])

show(axC1, imgs['C1'], 'C')
show(axC2, imgs['C2'])

show(axD,  imgs['D'],  'D')
show(axE,  imgs['E'],  'E')

show(axF,  imgs['F'],  'F', label_position=(0.97, 0.97))
axF.texts[-1].set_ha('right')
# ============================================================
# 5) الحفظ (PNG + TIFF + PDF)
# ============================================================
plt.savefig('Figure2_final.png', dpi=DPI, bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure2_final.tiff', dpi=DPI, bbox_inches='tight', pad_inches=0.03)
plt.savefig('Figure2_final.pdf', bbox_inches='tight', pad_inches=0.03)

print("✅ تم الحفظ بنجاح:")
print(f"   - Figure2_final.png  ({DPI} dpi)")
print(f"   - Figure2_final.tiff ({DPI} dpi)")
print(f"   - Figure2_final.pdf  (vector)")
print(f"   أبعاد الشكل: {FIG_WIDTH_IN*25.4:.1f}mm × {FIG_HEIGHT_IN*25.4:.1f}mm")