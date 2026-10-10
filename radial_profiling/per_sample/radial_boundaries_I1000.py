import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read just ONE slice from the big stack (memory-safe)
with tifffile.TiffFile("Salicornia_4_stack.tif") as tif:
    mid_z = 400  # middle slice
    slice_img = tif.pages[mid_z].asarray().astype(float)

print("Slice shape:", slice_img.shape)

# We already know the center from earlier session: Y=795, X=1051
com_y, com_x = 795, 1051

local_std = ndimage.generic_filter(slice_img, np.std, size=9)

yy, xx = np.mgrid[0:slice_img.shape[0], 0:slice_img.shape[1]]
r = np.sqrt((yy-com_y)**2 + (xx-com_x)**2)

max_r_um = 700
radial_profile, radii_um = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile.append(local_std[mask].mean())
        radii_um.append(rad_vox * 2.5)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(radii_um, radial_profile, linewidth=2)
ax.axvline(279, color='green', linestyle='--', label='cc boundary (known=279um)')
ax.set_xlabel("Distance from center (um)")
ax.set_ylabel("Local texture (std dev)")
ax.set_title("S4 slice 400 - radial texture profile")
ax.legend()
ax.grid(True)
plt.savefig("S4_radial_profile.png", dpi=150, bbox_inches='tight')
print("Saved S4_radial_profile.png")

# Extend the profile further to find outer pt boundary
max_r_um = 900
radial_profile2, radii_um2 = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile2.append(local_std[mask].mean())
        radii_um2.append(rad_vox * 2.5)

fig2, ax2 = plt.subplots(figsize=(11,5))
ax2.plot(radii_um2, radial_profile2, linewidth=2)
ax2.axvline(279, color='green', linestyle='--', label='cc/pc (279um)')
ax2.axvline(470, color='orange', linestyle='--', label='pc/pt candidate (470um)')
ax2.set_xlabel("Distance from center (um)")
ax2.set_ylabel("Local texture (std dev)")
ax2.set_title("S4 slice 400 - EXTENDED radial profile")
ax2.legend()
ax2.grid(True)
plt.savefig("S4_radial_profile_extended.png", dpi=150, bbox_inches='tight')
print("Saved S4_radial_profile_extended.png")

# Extend even further to find the true outer stem edge
max_r_um = 1400
radial_profile3, radii_um3 = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile3.append(local_std[mask].mean())
        radii_um3.append(rad_vox * 2.5)

fig3, ax3 = plt.subplots(figsize=(13,5))
ax3.plot(radii_um3, radial_profile3, linewidth=2)
ax3.axvline(279, color='green', linestyle='--', label='cc/pc (279um)')
ax3.axvline(470, color='orange', linestyle='--', label='pc/pt (470um)')
ax3.set_xlabel("Distance from center (um)")
ax3.set_ylabel("Local texture (std dev)")
ax3.set_title("S4 slice 400 - FULL extended radial profile")
ax3.legend()
ax3.grid(True)
plt.savefig("S4_radial_profile_full.png", dpi=150, bbox_inches='tight')
print("Saved S4_radial_profile_full.png")

# Visual check of all candidate boundaries
fig4, ax4 = plt.subplots(figsize=(12,12))
ax4.imshow(slice_img, cmap='gray', vmin=slice_img.min(), vmax=np.percentile(slice_img,99))

boundaries = {279: 'lime', 470: 'orange', 1050: 'red'}
for r_um, color in boundaries.items():
    r_vox = r_um / 2.5
    circle = plt.Circle((com_x, com_y), r_vox, fill=False, color=color, linewidth=2, label=f'{r_um}um')
    ax4.add_patch(circle)

ax4.plot(com_x, com_y, '+', color='yellow', markersize=15)
ax4.legend(loc='upper right', fontsize=10)
ax4.set_title("S4 - all candidate boundaries visual check")
plt.savefig("S4_all_boundaries_visual.png", dpi=150, bbox_inches='tight')
print("Saved S4_all_boundaries_visual.png")

# Build 6 reference slices + label stack for S4 (memory-safe, slice-by-slice)
reference_slices = [80, 200, 320, 480, 600, 720]  # spread across 800 total, avoiding edges

r_cc_pc = 279 / 2.5
r_pc_pt = 470 / 2.5
r_pt_out = 1050 / 2.5

raw_slices_8bit = []
label_slices = []

with tifffile.TiffFile("Salicornia_4_stack.tif") as tif:
    for z in reference_slices:
        s_img = tif.pages[z].asarray().astype(float)

        lbl = np.zeros(s_img.shape, dtype=np.uint8)
        lbl[r <= r_cc_pc] = 153
        lbl[(r > r_cc_pc) & (r <= r_pc_pt)] = 102
        lbl[(r > r_pc_pt) & (r <= r_pt_out)] = 51

        clipped = np.clip(s_img, 12000, 50000)
        s_8bit = ((clipped - 12000) / (50000 - 12000) * 255.0).astype(np.uint8)

        raw_slices_8bit.append(s_8bit)
        label_slices.append(lbl)
        print(f"Processed slice {z}")

label_stack = np.stack(label_slices, axis=0)
print("label_stack shape:", label_stack.shape)
print("Unique values:", np.unique(label_stack))

tifffile.imwrite("image_folder/Salicornia4_/labelled-stack.tif", label_stack, photometric='minisblack')
print("Saved labelled-stack.tif directly into image_folder")