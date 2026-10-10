import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

arr = tifffile.imread("Salicornia_8_stack.tif")
print("Shape:", arr.shape)

mid_z = 150
slice_img = arr[mid_z, :, :].astype(float)

local_std = ndimage.generic_filter(slice_img, np.std, size=9)
texture_mask = local_std > np.percentile(local_std, 90)
dilated = ndimage.binary_dilation(texture_mask, iterations=3)
filled = ndimage.binary_fill_holes(dilated)
filled = ndimage.binary_erosion(filled, iterations=3)
labeled, num = ndimage.label(filled)
sizes = ndimage.sum(filled, labeled, range(1, num+1))
largest = np.argmax(sizes) + 1
solid_mask = (labeled == largest)
com_y, com_x = ndimage.center_of_mass(solid_mask)
print(f"Center: Y={com_y:.0f}, X={com_x:.0f}")

yy, xx = np.mgrid[0:slice_img.shape[0], 0:slice_img.shape[1]]
r = np.sqrt((yy-com_y)**2 + (xx-com_x)**2)

max_r_um = 600

# --- Texture profile (already have this) ---
radial_profile, radii_um = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile.append(local_std[mask].mean())
        radii_um.append(rad_vox * 2.5)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(radii_um, radial_profile, linewidth=2)
ax.set_xlabel("Distance from center (um)")
ax.set_ylabel("Local texture (std dev)")
ax.set_title(f"S8 slice {mid_z} - TEXTURE radial profile")
ax.grid(True)
plt.savefig("radial_full_profile.png", dpi=150, bbox_inches='tight')
print("Saved radial_full_profile.png")

# --- NEW: Raw intensity profile ---
raw_profile, radii_um2 = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        raw_profile.append(slice_img[mask].mean())
        radii_um2.append(rad_vox * 2.5)

fig2, ax2 = plt.subplots(figsize=(11,5))
ax2.plot(radii_um2, raw_profile, linewidth=2, color='darkorange')
ax2.set_xlabel("Distance from center (um)")
ax2.set_ylabel("Mean raw intensity")
ax2.set_title(f"S8 slice {mid_z} - RAW INTENSITY radial profile")
ax2.grid(True)
plt.savefig("radial_intensity_profile.png", dpi=150, bbox_inches='tight')
print("Saved radial_intensity_profile.png")

print(f"Center for reuse: Y={com_y:.1f}, X={com_x:.1f}")

# Visual check of all boundaries together
fig3, ax3 = plt.subplots(figsize=(11,11))
ax3.imshow(slice_img, cmap='gray', vmin=slice_img.min(), vmax=np.percentile(slice_img,99))

boundaries_um = {215: 'cyan', 470: 'orange', 590: 'red'}
for r_um, color in boundaries_um.items():
    r_vox = r_um / 2.5
    circle = plt.Circle((com_x, com_y), r_vox, fill=False, color=color, linewidth=2, label=f'{r_um}um')
    ax3.add_patch(circle)

ax3.plot(com_x, com_y, '+', color='yellow', markersize=15)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_title(f"S8 slice {mid_z} - all boundary candidates")
plt.savefig("all_boundaries_check.png", dpi=150, bbox_inches='tight')
print("Saved all_boundaries_check.png")

# Build the automatic label image using the confirmed boundaries
label_img = np.zeros(slice_img.shape, dtype=np.uint8)

# Order matters: draw from outside in, or use masks (safer)
r_pt_out = 590 / 2.5
r_pc_pt = 470 / 2.5
r_cc_pc = 215 / 2.5

label_img[r <= r_cc_pc] = 153        # cc (merged with pi)
label_img[(r > r_cc_pc) & (r <= r_pc_pt)] = 102   # pc
label_img[(r > r_pc_pt) & (r <= r_pt_out)] = 51   # pt
# everything beyond r_pt_out stays 0 (background)

tifffile.imwrite("S8_slice150_labels_AUTO.tif", label_img, photometric='minisblack')
print("Saved S8_slice150_labels_AUTO.tif")
print("Unique values:", np.unique(label_img))

# Visual check: overlay label boundaries on raw image
fig4, ax4 = plt.subplots(figsize=(11,11))
ax4.imshow(slice_img, cmap='gray', vmin=slice_img.min(), vmax=np.percentile(slice_img,99))
ax4.contour(label_img, levels=[50,100,150], colors=['orange','yellow','cyan'], linewidths=1.5)
ax4.set_title("Auto-generated labels overlaid on raw slice")
plt.savefig("labels_overlay_check.png", dpi=150, bbox_inches='tight')
print("Saved labels_overlay_check.png")


# Build a clean labeled overlay WITH a proper color legend
import matplotlib.patches as mpatches

# Assign distinct display colors (different from the pixel values used for training)
display_colors = {
    153: ('cyan', 'cc + pi (merged)'),
    102: ('lime', 'pc (parenchyma cortex)'),
    51:  ('orange', 'pt (palisade tissue)'),
    0:   ('black', 'background'),
}

# Build an RGB visualization image from the label image
label_rgb = np.zeros((*label_img.shape, 3), dtype=np.uint8)
color_map_rgb = {
    153: (0, 255, 255),   # cyan
    102: (0, 255, 0),     # lime
    51:  (255, 165, 0),   # orange
    0:   (0, 0, 0),       # black (background, will show raw image instead)
}
for val, rgb in color_map_rgb.items():
    label_rgb[label_img == val] = rgb

fig5, ax5 = plt.subplots(figsize=(12,11))
# Show raw image as base, then overlay colored labels with transparency
ax5.imshow(slice_img, cmap='gray', vmin=slice_img.min(), vmax=np.percentile(slice_img,99))
overlay = np.ma.masked_where(label_img == 0, label_img)
cmap_overlay = matplotlib.colors.ListedColormap(['orange','lime','cyan'])
bounds = [1, 76, 127, 200]
norm = matplotlib.colors.BoundaryNorm(bounds, cmap_overlay.N)
ax5.imshow(overlay, cmap=cmap_overlay, norm=norm, alpha=0.45)

# Legend
legend_patches = [
    mpatches.Patch(color='orange', label='pt - palisade tissue'),
    mpatches.Patch(color='lime', label='pc - parenchyma cortex'),
    mpatches.Patch(color='cyan', label='cc + pi (merged)'),
]
ax5.legend(handles=legend_patches, loc='upper right', fontsize=11, framealpha=0.9)
ax5.set_title(f"S8 slice {mid_z} - labeled tissue map with legend")
plt.savefig("labels_with_legend.png", dpi=150, bbox_inches='tight')
print("Saved labels_with_legend.png")

# Apply same boundaries to remaining 5 reference slices + build combined stacks
reference_slices = [150, 300, 450, 550, 700, 850]

raw_slices_8bit = []
label_slices = []

for z in reference_slices:
    s_img = arr[z, :, :].astype(float)
    lbl = np.zeros(s_img.shape, dtype=np.uint8)
    lbl[r <= r_cc_pc] = 153
    lbl[(r > r_cc_pc) & (r <= r_pc_pt)] = 102
    lbl[(r > r_pc_pt) & (r <= r_pt_out)] = 51

    # Convert this slice to matching 8-bit (same contrast stretch as GRID file)
    LOW, HIGH = 12000, 50000
    clipped = np.clip(s_img, LOW, HIGH)
    s_8bit = ((clipped - LOW) / (HIGH - LOW) * 255.0).astype(np.uint8)

    raw_slices_8bit.append(s_8bit)
    label_slices.append(lbl)
    print(f"Processed slice {z}")

raw_stack = np.stack(raw_slices_8bit, axis=0)
label_stack = np.stack(label_slices, axis=0)

print("raw_stack shape:", raw_stack.shape)
print("label_stack shape:", label_stack.shape)
print("Label unique values:", np.unique(label_stack))

tifffile.imwrite("Salicornia8__label_stack.tif", label_stack, photometric='minisblack')
print("Saved Salicornia8__label_stack.tif")