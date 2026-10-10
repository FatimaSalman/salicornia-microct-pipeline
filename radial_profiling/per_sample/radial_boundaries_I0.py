import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    mid_z = 500
    slice_img = tif.pages[mid_z].asarray().astype(float)

print("Slice shape:", slice_img.shape)

# Known center from earlier session (texture-based method)
com_y, com_x = 520, 591

local_std = ndimage.generic_filter(slice_img, np.std, size=9)

yy, xx = np.mgrid[0:slice_img.shape[0], 0:slice_img.shape[1]]
r = np.sqrt((yy-com_y)**2 + (xx-com_x)**2)

max_r_um = 500
radial_profile, radii_um = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile.append(local_std[mask].mean())
        radii_um.append(rad_vox * 2.5)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(radii_um, radial_profile, linewidth=2)
ax.axvline(87.5, color='green', linestyle='--', label='cc boundary (known~87.5um)')
ax.set_xlabel("Distance from center (um)")
ax.set_ylabel("Local texture (std dev)")
ax.set_title("S1 slice 500 - radial texture profile")
ax.legend()
ax.grid(True)
plt.savefig("S1_radial_profile.png", dpi=150, bbox_inches='tight')
print("Saved S1_radial_profile.png")

# Extended profile + visual boundary check, combined
max_r_um = 700
radial_profile2, radii_um2 = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile2.append(local_std[mask].mean())
        radii_um2.append(rad_vox * 2.5)

fig2, ax2 = plt.subplots(figsize=(11,5))
ax2.plot(radii_um2, radial_profile2, linewidth=2)
ax2.axvline(87.5, color='green', linestyle='--', label='cc (87.5um)')
ax2.axvline(350, color='orange', linestyle='--', label='pc/pt candidate (350um)')
ax2.set_xlabel("Distance from center (um)")
ax2.set_ylabel("Local texture (std dev)")
ax2.set_title("S1 - EXTENDED radial profile")
ax2.legend()
ax2.grid(True)
plt.savefig("S1_radial_extended.png", dpi=150, bbox_inches='tight')
print("Saved S1_radial_extended.png")


# Visual check with candidate circles
fig3, ax3 = plt.subplots(figsize=(11,11))
ax3.imshow(slice_img, cmap='gray', vmin=slice_img.min(), vmax=np.percentile(slice_img,99))
for r_um, color in [(87.5, 'lime'), (350, 'orange'), (450, 'red')]:
    r_vox = r_um / 2.5
    circle = plt.Circle((com_x, com_y), r_vox, fill=False, color=color, linewidth=2, label=f'{r_um}um')
    ax3.add_patch(circle)
ax3.plot(com_x, com_y, '+', color='yellow', markersize=15)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_title("S1 - boundary visual check")
plt.savefig("S1_visual_check.png", dpi=150, bbox_inches='tight')
print("Saved S1_visual_check.png")

'''
reference_slices = [80, 200, 350, 500, 650, 800]  # spread across 1000 slices

r_cc_pc = 87.5 / 2.5
r_pc_pt = 350 / 2.5
r_pt_out = 450 / 2.5

label_slices = []
with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    for z in reference_slices:
        s_img = tif.pages[z].asarray().astype(float)
        lbl = np.zeros(s_img.shape, dtype=np.uint8)
        lbl[r <= r_cc_pc] = 153
        lbl[(r > r_cc_pc) & (r <= r_pc_pt)] = 102
        lbl[(r > r_pc_pt) & (r <= r_pt_out)] = 51
        label_slices.append(lbl)
        print(f"Processed slice {z}")

label_stack = np.stack(label_slices, axis=0)
print("Shape:", label_stack.shape, "Unique:", np.unique(label_stack))

tifffile.imwrite("image_folder/Salicornia1_/labelled-stack.tif", label_stack, photometric='minisblack')
print("Saved labelled-stack.tif")
'''