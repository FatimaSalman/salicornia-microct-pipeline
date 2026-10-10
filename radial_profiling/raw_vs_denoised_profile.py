import tifffile
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Load the raw slice and crop the same region used before ---
with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    mid_z = 500
    slice_raw_full = tif.pages[mid_z].asarray().astype(float)

row_start, row_end = 300, 700
col_start, col_end = 400, 800
raw_crop = slice_raw_full[row_start:row_end, col_start:col_end]

# Center of mass, in crop coordinates
com_y_crop = 520 - row_start  # = 220
com_x_crop = 591 - col_start  # = 191

# --- IMPORTANT: the crop is only 400x400, centered off-center at (220,191).
# Distance from center to the nearest edge (left side) is only ~191 px
# = ~478um. Past that, circles start leaving the crop, so radii beyond
# roughly 450um are only partially covered (biased average, not a full
# ring) -- treat anything past that line as unreliable / not a full circle.
max_reliable_um = min(com_y_crop, raw_crop.shape[0]-com_y_crop,
                       com_x_crop, raw_crop.shape[1]-com_x_crop) * 2.5
print(f"Max reliable radius (full ring inside crop): ~{max_reliable_um:.0f}um")

# --- Compute local_std and radius map ONCE, on the raw crop ---
local_std = ndimage.generic_filter(raw_crop, np.std, size=9)
yy, xx = np.mgrid[0:raw_crop.shape[0], 0:raw_crop.shape[1]]
r = np.sqrt((yy - com_y_crop) ** 2 + (xx - com_x_crop) ** 2)

# --- Extended radial profile ---
max_r_um = 700  # requested; will exceed crop on some sides, see note above
radial_profile2, radii_um2 = [], []
for rad_vox in range(0, int(max_r_um / 2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox + 4)
    if mask.sum() > 0:
        radial_profile2.append(local_std[mask].mean())
        radii_um2.append(rad_vox * 2.5)

fig2, ax2 = plt.subplots(figsize=(11, 5))
ax2.plot(radii_um2, radial_profile2, linewidth=2)
ax2.axvline(87.5, color='green', linestyle='--', label='cc (87.5um)')
ax2.axvline(350, color='orange', linestyle='--', label='pc/pt candidate (350um)')
ax2.axvline(max_reliable_um, color='red', linestyle=':',
            label=f'crop edge (~{max_reliable_um:.0f}um) - beyond = partial ring only')
ax2.set_xlabel("Distance from center (um)")
ax2.set_ylabel("Local texture (std dev)")
ax2.set_title("S1 - EXTENDED radial profile (raw crop)")
ax2.legend()
ax2.grid(True)
plt.savefig("S1_radial_extended.png", dpi=150, bbox_inches='tight')
print("Saved S1_radial_extended.png")

# --- Visual check with candidate circles, on the same raw crop ---
fig3, ax3 = plt.subplots(figsize=(11, 11))
ax3.imshow(raw_crop, cmap='gray', vmin=raw_crop.min(), vmax=np.percentile(raw_crop, 99))
for r_um, color in [(87.5, 'lime'), (350, 'orange'), (450, 'red')]:
    r_vox = r_um / 2.5
    circle = plt.Circle((com_x_crop, com_y_crop), r_vox, fill=False, color=color, linewidth=2, label=f'{r_um}um')
    ax3.add_patch(circle)
ax3.plot(com_x_crop, com_y_crop, '+', color='yellow', markersize=15)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_title("S1 - boundary visual check (raw crop)")
plt.savefig("S1_visual_check.png", dpi=150, bbox_inches='tight')
print("Saved S1_visual_check.png")

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