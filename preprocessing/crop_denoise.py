import tifffile
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_nl_means
try:
    from skimage.restoration import estimate_sigma
    _HAS_PYWT = True
except Exception:
    _HAS_PYWT = False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Load one mid-stack slice ---
with tifffile.TiffFile("Salicornia_1_stack.tif") as tif:
    mid_z = 500
    slice_raw = tif.pages[mid_z].asarray().astype(float)

# --- Crop around the central cylinder ---
# From the earlier full-frame images, the central cylinder sits roughly at
# rows 350-650, cols 450-750 (out of ~900x900). Adjust these to center it
# tightly with a bit of margin if it looks off in your data.
row_start, row_end = 300, 700
col_start, col_end = 400, 800
slice_crop = slice_raw[row_start:row_end, col_start:col_end]

# --- Noise estimate on the crop only (local, not affected by radial trend) ---
if _HAS_PYWT:
    sigma_est = np.mean(estimate_sigma(slice_crop))
else:
    laplacian = ndimage.laplace(slice_crop)
    sigma_est = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745

# --- Strong NLM denoising, tuned for the crop ---
# h raised relative to before (1.15 -> 1.6) since we no longer need to
# preserve the faint radial pattern outside the ROI, and can push harder for
# a cleaner boundary inside it. patch_size/patch_distance bumped slightly too.
slice_denoised = denoise_nl_means(
    slice_crop,
    h=1.6 * sigma_est,
    patch_size=7,
    patch_distance=9,
    fast_mode=False,
)

# --- Plot: crop raw vs crop denoised ---
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(slice_crop, cmap='gray',
               vmin=np.percentile(slice_crop, 1), vmax=np.percentile(slice_crop, 99))
axes[0].set_title(f"Raw crop [{row_start}:{row_end}, {col_start}:{col_end}]")

axes[1].imshow(slice_denoised, cmap='gray',
               vmin=np.percentile(slice_denoised, 1), vmax=np.percentile(slice_denoised, 99))
axes[1].set_title("Strong NLM denoised (ROI only)")

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("S1_crop_denoise_comparison.png", dpi=150, bbox_inches='tight')
print("Saved S1_crop_denoise_comparison.png")

# --- Save the denoised crop as a standalone TIFF for manual labeling ---
tifffile.imwrite("S1_slice500_crop_denoised.tif", slice_denoised.astype(np.float32))
print("Saved S1_slice500_crop_denoised.tif")