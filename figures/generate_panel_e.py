#!/usr/bin/env python3
"""generate_panel_e.py
Panel E: Reference training labels vs RF prediction,
for a slice actually used in training (z=451, index=2).
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")


import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from skimage.transform import resize

# ==========================================
# Paths
# ==========================================
LABELS_PATH = os.path.join(DATA, "Salicornia_8/image_folder/Salicornia8_/labelled-stack.tif")
PREDICTION_PATH = os.path.join(DATA, "Salicornia_8/image_folder/Salicornia8_/MLresults/Salicornia8_fullstack_prediction.tif")

LABEL_INDEX = 2         # corresponds to full_z = 451
FULL_Z = 451
CC_CLASS_IN_PREDICTION = -1

# ==========================================
# Load data
# ==========================================
labels_stack = tifffile.imread(LABELS_PATH)
prediction_stack = np.squeeze(tifffile.imread(PREDICTION_PATH))

label_slice = labels_stack[LABEL_INDEX]          # (1078, 884), values {0,51,102,153}
pred_slice = prediction_stack[FULL_Z]            # (539, 442), values include -1 for CC

print("Label slice shape:", label_slice.shape)
print("Prediction slice shape:", pred_slice.shape)

# ==========================================
# Resize label slice DOWN to match prediction resolution
# (nearest-neighbor to preserve categorical values)
# ==========================================
label_slice_resized = resize(
    label_slice,
    output_shape=pred_slice.shape,
    order=0,
    preserve_range=True,
    anti_aliasing=False,
).astype(np.uint8)

print("Resized label slice shape:", label_slice_resized.shape)
print("Unique values after resize:", np.unique(label_slice_resized))

# ==========================================
# Crop both around the CC region for a clean, zoomed comparison
# ==========================================
def crop_around_mask(mask, padding=30):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - padding), min(mask.shape[0], ys.max() + padding + 1)
    x0, x1 = max(0, xs.min() - padding), min(mask.shape[1], xs.max() + padding + 1)
    return y0, y1, x0, x1

cc_mask_for_crop = label_slice_resized == 153
y0, y1, x0, x1 = crop_around_mask(cc_mask_for_crop, padding=40)

label_crop = label_slice_resized[y0:y1, x0:x1]
pred_crop_raw = pred_slice[y0:y1, x0:x1]
pred_crop_binary = (pred_crop_raw == CC_CLASS_IN_PREDICTION).astype(np.uint8)

# ==========================================
# Plot: Panel E (2 subplots)
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=300)

# --- E1: Reference labels (categorical, 4 colors) ---
label_colors = {0: "black", 51: "#5b8fd1", 102: "#f0c419", 153: "#e63946"}
cmap = ListedColormap([label_colors[v] for v in sorted(label_colors)])
value_to_index = {v: i for i, v in enumerate(sorted(label_colors))}
label_crop_indexed = np.vectorize(value_to_index.get)(label_crop)

axes[0].imshow(label_crop_indexed, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
axes[0].set_title("Reference training labels\n Training slice z = 451",
                   fontsize=9, fontweight="bold")
axes[0].set_xticks([]); axes[0].set_yticks([])

legend_elements = [
    Patch(facecolor="black", edgecolor="gray", label="0: Background"),
    Patch(facecolor="#5b8fd1", label="51: Far context"),
    Patch(facecolor="#f0c419", label="102: Near context"),
    Patch(facecolor="#e63946", label="153: Central cylinder"),
]
axes[0].legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=6.5, frameon=False)

# --- E2: RF Prediction (binary CC mask) ---
axes[1].imshow(pred_crop_binary, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title("Predicted central-cylinder class\nSame slice, z=451",
                   fontsize=9, fontweight="bold")
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.tight_layout()
plt.savefig("Figure2_PanelE_labels_vs_prediction.png", dpi=300, bbox_inches="tight")
plt.show()
print("\n✅ Saved: Figure2_PanelE_labels_vs_prediction.png")