import tifffile
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the denoised crop saved earlier
denoised_crop = tifffile.imread("S1_slice500_crop_denoised.tif")
print("Denoised crop shape:", denoised_crop.shape)

# Center within the crop coordinates (crop was rows 300:700, cols 400:800)
# Original full-frame center was approx (520, 591) -> shift into crop coords
com_y_crop = 520 - 300  # = 220
com_x_crop = 591 - 400  # = 191

local_std = ndimage.generic_filter(denoised_crop, np.std, size=9)

yy, xx = np.mgrid[0:denoised_crop.shape[0], 0:denoised_crop.shape[1]]
r = np.sqrt((yy-com_y_crop)**2 + (xx-com_x_crop)**2)

max_r_um = 350
radial_profile, radii_um = [], []
for rad_vox in range(0, int(max_r_um/2.5), 4):
    mask = (r >= rad_vox) & (r < rad_vox+4)
    if mask.sum() > 0:
        radial_profile.append(local_std[mask].mean())
        radii_um.append(rad_vox * 2.5)

fig, ax = plt.subplots(figsize=(11,5))
ax.plot(radii_um, radial_profile, linewidth=2)
ax.axvline(87.5, color='green', linestyle='--', label='cc candidate (87.5um)')
ax.set_xlabel("Distance from center (um)")
ax.set_ylabel("Local texture (std dev)")
ax.set_title("S1 - radial profile on DENOISED crop")
ax.legend()
ax.grid(True)
plt.savefig("S1_denoised_radial_profile.png", dpi=150, bbox_inches='tight')
print("Saved S1_denoised_radial_profile.png")


fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(denoised_crop, cmap='gray', vmin=np.percentile(denoised_crop,1), vmax=np.percentile(denoised_crop,99))

for r_um, color in [(90, 'lime'), (120, 'cyan'), (190, 'orange'), (250, 'red')]:
    r_vox = r_um / 2.5
    circle = plt.Circle((com_x_crop, com_y_crop), r_vox, fill=False, color=color, linewidth=2, label=f'{r_um}um')
    ax.add_patch(circle)

ax.plot(com_x_crop, com_y_crop, '+', color='yellow', markersize=15)
ax.legend(loc='upper right', fontsize=10)
ax.set_title("S1 denoised - boundary candidates")
plt.savefig("S1_denoised_boundaries_check.png", dpi=150, bbox_inches='tight')
print("Saved S1_denoised_boundaries_check.png")