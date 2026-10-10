
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

pred_path = os.path.join(DATA, "Salicornia 1, Inowroclaw 0 mM/image_folder/Salicornia1_/MLresults/Salicornia1_fullstack_prediction.tif")

pred = tifffile.imread(pred_path)
print("Full stack shape:", pred.shape)

# Take the middle-ish reference slice (500) for visual comparison with
# everything we've inspected so far.
z = 500
pred_slice = pred[z]

# --- Same fixed center/radii used to build the original labels ---
# NOTE: prediction stack is at rescale_factor=2 -> 450x450 instead of 900x900,
# so the center and radii (in voxels) must be halved to match this scale.
com_y_full, com_x_full = 520, 591
voxel_um = 2.5 * 2  # effective voxel size after rescale_factor=2 -> 5um
com_y, com_x = com_y_full / 2, com_x_full / 2

r_cc_pc = 87.5 / voxel_um
r_pc_pt = 350 / voxel_um
r_pt_out = 450 / voxel_um

unique_vals = np.unique(pred_slice)
print("Unique values in this slice:", unique_vals)

# --- Discrete color map, one distinct color per unique value ---
cmap = plt.get_cmap('tab10', len(unique_vals))
val_to_idx = {v: i for i, v in enumerate(unique_vals)}
idx_slice = np.vectorize(val_to_idx.get)(pred_slice)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(idx_slice, cmap=cmap, vmin=-0.5, vmax=len(unique_vals) - 0.5)

# Discrete colorbar labeled with the ACTUAL pixel values, not the color index
cbar = plt.colorbar(im, ax=ax, ticks=range(len(unique_vals)))
cbar.ax.set_yticklabels([str(v) for v in unique_vals])
cbar.set_label("Predicted class value")

# --- Overlay reference circles at the scaled radii ---
for r_um, r_vox, color in [
    (87.5, r_cc_pc, 'white'),
    (350, r_pc_pt, 'black'),
    (450, r_pt_out, 'red'),
]:
    circle = plt.Circle((com_x, com_y), r_vox, fill=False,
                         color=color, linewidth=2, linestyle='--', label=f'{r_um}um (from labels)')
    ax.add_patch(circle)

ax.plot(com_x, com_y, '+', color='yellow', markersize=15)
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f"Predicted classes - slice {z} (rescaled 450x450)\nCompare colors to dashed reference circles")

plt.tight_layout()
plt.savefig("S1_prediction_class_identification.png", dpi=150, bbox_inches='tight')
print("Saved S1_prediction_class_identification.png")