#!/usr/bin/env python3
"""
Figure 2 - composite workflow figure, sized for The Plant Journal.

Changed from the previous version:

  1. FIG_WIDTH_MM is now 169, the full page width TPJ allows (was 180).

  2. bbox_inches='tight' has been removed from the saves. That option crops
     the canvas after drawing, which silently discarded the size you had
     asked for: a figure declared at 180 mm came out as 3324 px, i.e. 141 mm
     at 600 dpi. The margins are now set to zero on the GridSpec instead, so
     the saved file is exactly FIG_WIDTH_MM wide and the point sizes below
     are true printed points.

  3. Panel letters are set in points that are correct at the final size.

After running, the TIFF should be 3992 px wide. Check with:
    python3 figure_audit.py .
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from PIL import Image
import numpy as np
import tifffile
import os

# ---------------------------------------------------------------- settings --
FIG_WIDTH_MM = 169      # TPJ full page width
DPI          = 600      # composite of images and lettering
PANEL_LETTER_PT = 11    # true printed points; TPJ minimum is 8

# TPJ: "Labelling on figures should be no smaller than 8pt Helvetica.
#       Figure sections should be designated with lower case letters."
# Arial is metrically identical to Helvetica and is available everywhere.
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
})

# ============================================================
# 1) Load the images
# ============================================================
def load_img(path):
    """Load an image safely, normalising 16-bit images for good display contrast."""
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
    'C2': 'PanelC2_s8_texture_z492.tif',
    'D':  'PanelD_boundary_methods.png',
    'E':  'PanelE_labels_vs_prediction.png',
    'F':  'PanelF_s8_3d_slicer.png',
}

for k, v in FILES.items():
    if not os.path.exists(v):
        raise FileNotFoundError(f"file not found: {v}")

imgs = {k: load_img(v) for k, v in FILES.items()}

def aspect(img):
    h, w = img.shape[0], img.shape[1]
    return w / h

# ============================================================
# 2) Aspect ratio of each image and of each group
# ============================================================
aA1, aA2 = aspect(imgs['A1']), aspect(imgs['A2'])
aB1, aB2 = aspect(imgs['B1']), aspect(imgs['B2'])
aC1, aC2 = aspect(imgs['C1']), aspect(imgs['C2'])
aD = aspect(imgs['D'])
aE = aspect(imgs['E'])
aF = aspect(imgs['F'])

groupA_aspect = aA1 + aA2            # A1 and A2 side by side
groupB_aspect = 1 / (1 / aB1 + 1 / aB2)   # B1 above B2
groupC_aspect = aC1 + aC2            # C1 and C2 side by side

row1_widths = [groupA_aspect, groupB_aspect, groupC_aspect]
row2_aspect = aD
row3_widths = [aE, aF]

# ============================================================
# 3) Final figure dimensions
# ============================================================
HSPACE, WSPACE = 0.06, 0.04

W  = 1.0
H1 = W / sum(row1_widths)
H2 = W / row2_aspect
H3 = W / sum(row3_widths)

FIG_WIDTH_IN  = FIG_WIDTH_MM / 25.4
scale         = FIG_WIDTH_IN / W
# the gaps between rows add height, so account for them
gap_in        = HSPACE * ((H1 + H2 + H3) / 3) * scale * 2
FIG_HEIGHT_IN = (H1 + H2 + H3) * scale + gap_in

fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=DPI)

# margins at zero: the figure edge is the image edge, so the declared
# width survives the save
outer = gridspec.GridSpec(
    3, 1, height_ratios=[H1, H2, H3],
    hspace=HSPACE, figure=fig,
    left=0.0, right=1.0, top=1.0, bottom=0.0
)

# ---------------- row 1: A | B | C ----------------
row1 = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer[0],
    width_ratios=row1_widths, wspace=WSPACE
)

gA = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=row1[0], width_ratios=[aA1, aA2], wspace=0.02
)
axA1 = fig.add_subplot(gA[0]); axA2 = fig.add_subplot(gA[1])

gB = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row1[1], height_ratios=[1/aB1, 1/aB2], hspace=0.02
)
axB1 = fig.add_subplot(gB[0]); axB2 = fig.add_subplot(gB[1])

gC = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=row1[2], width_ratios=[aC1, aC2], wspace=0.02
)
axC1 = fig.add_subplot(gC[0]); axC2 = fig.add_subplot(gC[1])

# ---------------- row 2: D (full width) ----------------
axD = fig.add_subplot(outer[1])

# ---------------- row 3: E | F ----------------
row3 = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer[2], width_ratios=row3_widths, wspace=WSPACE
)
axE = fig.add_subplot(row3[0]); axF = fig.add_subplot(row3[1])

# ============================================================
# 4) Draw the images and the panel labels
# ============================================================
def show(ax, img, label='', label_position=(0.01, 0.99), ha='left'):
    cmap = 'gray' if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if label:
        ax.text(
            label_position[0], label_position[1], label, transform=ax.transAxes,
            fontsize=PANEL_LETTER_PT, fontweight='normal', color='white',
            ha=ha, va='top',
            bbox=dict(facecolor='black', alpha=0.55, pad=2.5, edgecolor='none')
        )

show(axA1, imgs['A1'], '(a)')
show(axA2, imgs['A2'])

show(axB1, imgs['B1'], '(b)', label_position=(0.97, 0.97), ha='right')
show(axB2, imgs['B2'])

show(axC1, imgs['C1'], '(c)')
show(axC2, imgs['C2'])

show(axD,  imgs['D'],  '(d)')
show(axE,  imgs['E'],  '(e)')
show(axF,  imgs['F'],  '(f)', label_position=(0.97, 0.97), ha='right')

# ============================================================
# 5) Save
# ============================================================
plt.savefig('Figure2.tif', dpi=DPI, pil_kwargs={'compression': 'tiff_lzw'})
plt.savefig('Figure2.png', dpi=DPI)
plt.savefig('Figure2.pdf')

px = round(FIG_WIDTH_IN * DPI)
print("saved:")
print(f"   Figure2.tif  {DPI} dpi, LZW")
print(f"   Figure2.png  {DPI} dpi")
print(f"   Figure2.pdf  vector")
print(f"   declared size: {FIG_WIDTH_MM:.0f} x {FIG_HEIGHT_IN*25.4:.1f} mm")
print(f"   expected width: {px} px  (verify with figure_audit.py)")
