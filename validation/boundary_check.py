import tifffile
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

com_y, com_x = 520, 591

check_slices = [150, 900]
fig, axes = plt.subplots(1, 2, figsize=(16,8))

with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    for ax, z in zip(axes, check_slices):
        s_img = tif.pages[z].asarray().astype(float)
        ax.imshow(s_img, cmap='gray', vmin=np.percentile(s_img,1), vmax=np.percentile(s_img,99))
        for r_um, color in [(90,'lime'), (190,'orange'), (250,'red')]:
            r_vox = r_um / 2.5
            circle = plt.Circle((com_x, com_y), r_vox, fill=False, color=color, linewidth=2)
            ax.add_patch(circle)
        ax.plot(com_x, com_y, '+', color='yellow', markersize=12)
        ax.set_title(f"Z={z}")

plt.savefig("S1_boundary_stability_check.png", dpi=150, bbox_inches='tight')
print("Saved S1_boundary_stability_check.png")