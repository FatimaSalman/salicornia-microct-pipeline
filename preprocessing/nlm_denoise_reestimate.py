#!/usr/bin/env python3
"""
nlm_denoise_reestimate.py
==========================
For low-contrast volumes (e.g. I-0) where the raw radial-texture profile
is confounded by heteroscedastic noise and irregular (star-shaped) tissue
geometry, this script:

  1. Crops a region around the stem centre.
  2. Estimates the noise level (wavelet-based sigma estimator).
  3. Applies non-local-means denoising (patch_size=7, patch_distance=9,
     h=1.6*sigma, fast_mode=False), exactly as described in the manuscript.
  4. Recomputes the radial texture profile on the denoised crop.
  5. Reports candidate boundary radii and saves a diagnostic plot.

Usage
-----
    python3 nlm_denoise_reestimate.py \
        --raw Salicornia_1_stack.tif \
        --slice 500 \
        --cx 590.74 --cy 519.41 \
        --crop-half-size-um 400 \
        --voxel-size-um 2.5 \
        --save-plot Salicornia1_denoised_radial_profile.png
"""
import argparse
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def windowed_std(img, window=9):
    img = img.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.clip(mean_sq - mean**2, 0, None)
    return np.sqrt(variance)


def radial_profile(texture_map, cx, cy, max_radius_px):
    h, w = texture_map.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_int = np.clip(r.astype(int), 0, max_radius_px)

    profile = np.zeros(max_radius_px + 1)
    counts = np.zeros(max_radius_px + 1)
    np.add.at(profile, r_int.ravel(), texture_map.ravel())
    np.add.at(counts, r_int.ravel(), 1)
    counts[counts == 0] = 1
    return profile / counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--slice", type=int, required=True)
    ap.add_argument("--cx", type=float, required=True)
    ap.add_argument("--cy", type=float, required=True)
    ap.add_argument("--crop-half-size-um", type=float, default=400,
                    help="Half-width of crop box around stem centre, in µm")
    ap.add_argument("--voxel-size-um", type=float, default=2.5)
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--max-radius-um", type=float, default=None,
                    help="Default: same as crop-half-size-um")
    ap.add_argument("--save-plot", default=None)
    ap.add_argument("--save-denoised-tif", default=None)
    args = ap.parse_args()

    voxel = args.voxel_size_um
    half_px = int(round(args.crop_half_size_um / voxel))
    max_radius_um = args.max_radius_um or args.crop_half_size_um
    max_radius_px = int(round(max_radius_um / voxel))

    print(f"Loading slice {args.slice} from {args.raw} ...")
    with tifffile.TiffFile(args.raw) as tif:
        img = tif.pages[args.slice].asarray().astype(np.float32)
    h, w = img.shape

    # --- Crop around stem centre ---
    x0 = int(round(args.cx - half_px)); x1 = int(round(args.cx + half_px))
    y0 = int(round(args.cy - half_px)); y1 = int(round(args.cy + half_px))
    x0c, x1c = max(x0, 0), min(x1, w)
    y0c, y1c = max(y0, 0), min(y1, h)

    crop = img[y0c:y1c, x0c:x1c]
    new_cx = args.cx - x0c
    new_cy = args.cy - y0c
    print(f"Cropped to shape {crop.shape}; new centre in crop = ({new_cx:.2f}, {new_cy:.2f})")

    # --- Estimate noise sigma (wavelet-based) ---
    sigma_est = estimate_sigma(crop, channel_axis=None)
    h_param = 1.6 * sigma_est
    print(f"Estimated noise sigma: {sigma_est:.2f}  ->  NLM h = {h_param:.2f}")

    print("Applying non-local-means denoising (may take 1-2 min)...")
    denoised = denoise_nl_means(
        crop, patch_size=7, patch_distance=9, h=h_param,
        fast_mode=False, preserve_range=True
    )

    if args.save_denoised_tif:
        tifffile.imwrite(args.save_denoised_tif, denoised.astype(np.float32))
        print(f"Saved denoised crop to {args.save_denoised_tif}")

    # --- Recompute radial texture profile on denoised crop ---
    texture = windowed_std(denoised, window=args.window)
    profile = radial_profile(texture, new_cx, new_cy, max_radius_px)
    radii_um = np.arange(len(profile)) * voxel

    peaks, _ = find_peaks(profile, prominence=np.std(profile) * 0.3)
    print(f"\nCandidate boundary radii AFTER denoising:")
    for p in peaks:
        print(f"  r = {radii_um[p]:.1f} µm  (texture value = {profile[p]:.1f})")
    if not peaks.size:
        print("  [!] Still no clear peaks — try adjusting window/crop size.")

    if args.save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(crop, cmap='gray',
                        vmin=np.percentile(crop, 1), vmax=np.percentile(crop, 99))
        axes[0].plot(new_cx, new_cy, 'y+', markersize=15, markeredgewidth=2)
        axes[0].set_title("Raw crop (before denoising)"); axes[0].axis('off')

        axes[1].imshow(denoised, cmap='gray',
                        vmin=np.percentile(denoised, 1), vmax=np.percentile(denoised, 99))
        axes[1].plot(new_cx, new_cy, 'y+', markersize=15, markeredgewidth=2)
        colors = plt.cm.rainbow(np.linspace(0, 1, max(len(peaks), 1)))
        for idx, p in enumerate(peaks):
            circle = plt.Circle((new_cx, new_cy), p, fill=False,
                                 color=colors[idx], linewidth=1.5,
                                 label=f"{radii_um[p]:.0f} µm")
            axes[1].add_patch(circle)
        axes[1].set_title("Denoised crop + boundaries")
        axes[1].legend(loc='upper right', fontsize=8); axes[1].axis('off')

        axes[2].plot(radii_um, profile, 'k-', linewidth=1)
        for idx, p in enumerate(peaks):
            axes[2].axvline(radii_um[p], color=colors[idx], linestyle='--',
                             label=f"{radii_um[p]:.0f} µm")
        axes[2].set_xlabel("Distance from stem centre (µm)")
        axes[2].set_ylabel("Local texture (std dev)")
        axes[2].set_title("Radial texture profile (denoised)")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150)
        print(f"\nSaved diagnostic plot to {args.save_plot}")


if __name__ == "__main__":
    main()