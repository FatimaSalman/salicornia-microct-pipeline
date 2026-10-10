import tifffile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Same fixed parameters used to build the label_stack ---
com_y, com_x = 520, 591
voxel_um = 2.5

r_cc_pc = 87.5 / voxel_um
r_pc_pt = 350 / voxel_um
r_pt_out = 450 / voxel_um

check_slices = [80, 800]  # the two furthest from the reference slice (500)

fig, axes = plt.subplots(1, len(check_slices), figsize=(9 * len(check_slices), 9))
if len(check_slices) == 1:
    axes = [axes]

with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    for ax, z in zip(axes, check_slices):
        s_img = tif.pages[z].asarray().astype(float)

        ax.imshow(s_img, cmap='gray',
                  vmin=np.percentile(s_img, 1), vmax=np.percentile(s_img, 99))

        for r_um, r_vox, color in [
            (87.5, r_cc_pc, 'lime'),
            (350, r_pc_pt, 'orange'),
            (450, r_pt_out, 'red'),
        ]:
            circle = plt.Circle((com_x, com_y), r_vox, fill=False,
                                 color=color, linewidth=2, label=f'{r_um}um')
            ax.add_patch(circle)

        ax.plot(com_x, com_y, '+', color='yellow', markersize=15)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f"Slice {z} - fixed center ({com_y},{com_x})")

plt.tight_layout()
plt.savefig("S1_center_drift_check.png", dpi=150, bbox_inches='tight')
print("Saved S1_center_drift_check.png")