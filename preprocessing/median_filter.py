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

# --- Step 1: illumination / radial-gradient correction ---
# Remove the slow-varying radial pattern (reconstruction/cupping artifact)
# by subtracting a heavily-blurred version of the image (a crude flat-field).
# NOTE: bumped way up from 50 -> the radial gradient spans hundreds of pixels,
# so sigma needs to be large enough to capture only that slow trend, not
# local structure.
sigma_bg = 150  # tune further if rings are still visible after this
background = ndimage.gaussian_filter(slice_raw, sigma=sigma_bg)
slice_corrected = slice_raw - background

# --- Step 2: Non-Local Means denoising on the corrected image ---
if _HAS_PYWT:
    sigma_est = np.mean(estimate_sigma(slice_corrected))
else:
    # Fallback noise estimate (median absolute deviation of Laplacian),
    # doesn't need PyWavelets. Roughly comparable in scale to estimate_sigma.
    laplacian = ndimage.laplace(slice_corrected)
    sigma_est = np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745
# fast_mode=False is slower but avoids the block-based approximation used by
# fast_mode=True, which can sometimes introduce visible seams between
# processing blocks. We're testing this to check whether the vertical line
# seen previously was a fast_mode artifact or something real in the data.
slice_denoised = denoise_nl_means(
    slice_corrected,
    h=1.15 * sigma_est,
    patch_size=5,
    patch_distance=6,
    fast_mode=False,
)

# --- Also show median-filtered corrected version for comparison ---
slice_median_after = ndimage.median_filter(slice_corrected, size=3)

# --- Plot: raw vs illumination-corrected vs NLM-denoised ---
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

axes[0].imshow(slice_raw, cmap='gray',
               vmin=np.percentile(slice_raw, 1), vmax=np.percentile(slice_raw, 99))
axes[0].set_title("Raw S1")

axes[1].imshow(slice_corrected, cmap='gray',
               vmin=np.percentile(slice_corrected, 1), vmax=np.percentile(slice_corrected, 99))
axes[1].set_title(f"Illumination-corrected (sigma_bg={sigma_bg})")

axes[2].imshow(slice_denoised, cmap='gray',
               vmin=np.percentile(slice_denoised, 1), vmax=np.percentile(slice_denoised, 99))
axes[2].set_title("Corrected + NLM denoised")

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("S1_denoise_comparison_v2.png", dpi=150, bbox_inches='tight')
print("Saved S1_denoise_comparison_v2.png")

# --- Optional: save the corrected slice as a standalone TIFF to inspect at full res ---
tifffile.imwrite("S1_slice500_corrected_denoised.tif", slice_denoised.astype(np.float32))
print("Saved S1_slice500_corrected_denoised.tif")