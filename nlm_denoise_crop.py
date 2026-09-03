#!/usr/bin/env python3
"""
nlm_denoise_crop.py
====================
Applies Non-Local Means (NLM) denoising to a cropped region-of-interest
around the stem center in a single microCT slice, and compares the
raw vs. denoised radial texture profile.

Why crop first: applying illumination/background correction across the
FULL frame does not work well when noise amplitude itself increases
with distance from center (heteroscedastic, cone-beam-related noise),
rather than a simple brightness gradient. Restricting denoising to a
tight crop around the tissue of interest sidesteps this entirely.

Usage
-----
    python3 nlm_denoise_crop.py \
        --stack path/to/stack.tif \
        --slice 500 \
        --center-y 520 --center-x 591 \
        --half-width 400 \
        --voxel-um 2.5 \
        --known-radius-um 87.5
"""
import argparse
import numpy as np
import tifffile
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


def estimate_noise_sigma(img):
    if _HAS_PYWT:
        return np.mean(estimate_sigma(img))
    # Fallback (no PyWavelets dependency): MAD of the Laplacian
    laplacian = ndimage.laplace(img)
    return np.median(np.abs(laplacian - np.median(laplacian))) / 0.6745


def radial_profile(img, com_y, com_x, max_r_um, voxel_um, step_vox=4, window=9):
    local_std = ndimage.generic_filter(img, np.std, size=window)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.sqrt((yy - com_y) ** 2 + (xx - com_x) ** 2)
    radii, profile = [], []
    for rad_vox in range(0, int(max_r_um / voxel_um), step_vox):
        mask = (r >= rad_vox) & (r < rad_vox + step_vox)
        if mask.sum() > 0:
            profile.append(local_std[mask].mean())
            radii.append(rad_vox * voxel_um)
    return np.array(radii), np.array(profile)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True)
    ap.add_argument("--slice", type=int, required=True)
    ap.add_argument("--center-y", type=float, required=True)
    ap.add_argument("--center-x", type=float, required=True)
    ap.add_argument("--half-width", type=int, required=True, help="Crop half-width in voxels (keep well inside frame bounds)")
    ap.add_argument("--voxel-um", type=float, required=True)
    ap.add_argument("--known-radius-um", type=float, default=None, help="Reference cc radius to mark on the profile plot")
    ap.add_argument("--patch-size", type=int, default=7)
    ap.add_argument("--patch-distance", type=int, default=9)
    ap.add_argument("--h-factor", type=float, default=1.6, help="NLM filter strength = h_factor * estimated sigma")
    ap.add_argument("--out-prefix", default=None)
    args = ap.parse_args()

    prefix = args.out_prefix or args.stack.split("/")[-1].rsplit(".", 1)[0]

    with tifffile.TiffFile(args.stack) as tif:
        slice_raw = tif.pages[args.slice].asarray().astype(float)

    row0, row1 = int(args.center_y - args.half_width), int(args.center_y + args.half_width)
    col0, col1 = int(args.center_x - args.half_width), int(args.center_x + args.half_width)
    crop = slice_raw[row0:row1, col0:col1]
    print(f"Crop window: rows[{row0}:{row1}], cols[{col0}:{col1}], shape={crop.shape}")

    sigma_est = estimate_noise_sigma(crop)
    print(f"Estimated noise sigma: {sigma_est:.3f}")

    denoised = denoise_nl_means(
        crop,
        h=args.h_factor * sigma_est,
        patch_size=args.patch_size,
        patch_distance=args.patch_distance,
        fast_mode=False,
    )

    # --- Save raw vs denoised comparison image ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(crop, cmap='gray', vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))
    axes[0].set_title("Raw crop")
    axes[1].imshow(denoised, cmap='gray', vmin=np.percentile(denoised, 1), vmax=np.percentile(denoised, 99))
    axes[1].set_title(f"NLM denoised (h={args.h_factor}*sigma, patch={args.patch_size}/{args.patch_distance})")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f"{prefix}_crop_denoise_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved {prefix}_crop_denoise_comparison.png")

    tifffile.imwrite(f"{prefix}_slice{args.slice}_crop_denoised.tif", denoised.astype(np.float32))
    print(f"Saved {prefix}_slice{args.slice}_crop_denoised.tif")

    # --- Radial profile: raw vs denoised, same crop-local center ---
    com_y_crop, com_x_crop = args.half_width, args.half_width
    max_r_um = (args.known_radius_um * 3) if args.known_radius_um else (args.half_width * args.voxel_um * 0.9)

    radii_raw, profile_raw = radial_profile(crop, com_y_crop, com_x_crop, max_r_um, args.voxel_um)
    radii_dn, profile_dn = radial_profile(denoised, com_y_crop, com_x_crop, max_r_um, args.voxel_um)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.plot(radii_raw, profile_raw, label="Raw", color='gray', linewidth=2)
    ax2.plot(radii_dn, profile_dn, label="Denoised", color='tab:blue', linewidth=2)
    if args.known_radius_um:
        ax2.axvline(args.known_radius_um, color='green', linestyle='--', label=f'known radius ({args.known_radius_um} um)')
    ax2.set_xlabel("Distance from center (um)")
    ax2.set_ylabel("Local texture (std dev)")
    ax2.set_title(f"{prefix} - radial profile, raw vs denoised")
    ax2.legend()
    ax2.grid(True)
    plt.savefig(f"{prefix}_radial_profile_raw_vs_denoised.png", dpi=150, bbox_inches='tight')
    print(f"Saved {prefix}_radial_profile_raw_vs_denoised.png")


if __name__ == "__main__":
    main()
