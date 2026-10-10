
import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

pred_path=os.path.join(DATA, "Salicornia 4, Inowroclaw 1000 mM/image_folder/Salicornia4_/MLresults/Salicornia4_fullstack_prediction.tif"),

z = 400  # Z axis is typically NOT affected by rescale_factor (that only resizes X,Y)

com_y_full, com_x_full = 795, 1051
r_cc_um = 279
rescale_factor = 4          # <-- corrected from 2 to 4
voxel_um_original = 2.50001997

pred = tifffile.imread(pred_path)
print("S4 full stack shape:", pred.shape)

pred_slice = pred[z]
unique_vals = np.unique(pred_slice)
print("S4 unique values in slice", z, ":", unique_vals)

voxel_um_eff = voxel_um_original * rescale_factor
com_y = com_y_full / rescale_factor
com_x = com_x_full / rescale_factor
r_cc_vox = r_cc_um / voxel_um_eff

print(f"Scaled center: ({com_y:.1f}, {com_x:.1f}), cc radius in voxels: {r_cc_vox:.1f}")

cmap = plt.get_cmap('tab10', len(unique_vals))
val_to_idx = {v: i for i, v in enumerate(unique_vals)}
idx_slice = np.vectorize(val_to_idx.get)(pred_slice)

fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(idx_slice, cmap=cmap, vmin=-0.5, vmax=len(unique_vals) - 0.5)
cbar = plt.colorbar(im, ax=ax, ticks=range(len(unique_vals)))
cbar.ax.set_yticklabels([str(v) for v in unique_vals])
cbar.set_label("Predicted class value")

circle = plt.Circle((com_x, com_y), r_cc_vox, fill=False,
                     color='white', linewidth=2, linestyle='--',
                     label=f'cc radius = {r_cc_um}um (from log)')
ax.add_patch(circle)
ax.plot(com_x, com_y, '+', color='yellow', markersize=15)
ax.legend(loc='upper right', fontsize=9)
ax.set_title(f"S4 - predicted classes, slice {z} (rescale_factor=4)\nWhich color falls inside the dashed cc circle?")

plt.tight_layout()
plt.savefig("S4_prediction_class_identification_v2.png", dpi=150, bbox_inches='tight')
print("Saved S4_prediction_class_identification_v2.png")