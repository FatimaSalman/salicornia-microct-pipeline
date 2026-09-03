#!/usr/bin/env python3
"""
radial_boundary_estimation.py
==============================
Estimates tissue-boundary radii in a microCT cross-section by computing
a local-texture-vs-radius profile. Local texture (windowed standard
deviation) rises sharply at transitions between anatomically distinct
tissue layers (e.g. central cylinder -> parenchyma cortex -> palisade
tissue), producing peaks/steps that can be read off the profile plot
and confirmed visually by overlaying candidate circles on the slice.

This does NOT assume a fixed set of boundaries — you inspect the output
plot and image, then note down the radii (in micrometers) you find
plausible for use in build_label_stack.py.

Usage
-----
    python3 radial_boundary_estimation.py \
        --stack path/to/stack.tif \
        --slice 500 \
        --voxel-um 2.5 \
        --max-radius-um 700 \
        --candidate-radii 90 190 250

If --center-y/--center-x are not given, the script estimates the stem
center automatically using a "solid-shape" method: threshold on local
texture, dilate, fill holes, erode, then take the centroid of the
largest connected component. This is more robust than a simple
intensity threshold on noisy / low-contrast volumes.

Outputs
-------
    <stack_stem>_radial_profile.png   - texture vs. radius plot
    <stack_stem>_boundary_check.png   - candidate circles overlaid on the slice
"""
import argparse
import numpy as np
import tifffile
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def estimate_center_solid_shape(slice_img, texture_window=9, percentile=90,
                                 dilate_iter=3, erode_iter=3):
    """Robust stem-center estimate for noisy / low-contrast slices.

    Simple intensity thresholds fail when background noise has similar
    or higher amplitude than the tissue signal (common in low-CNR
    volumes). Thresholding on LOCAL TEXTURE instead, then filling and
    taking the centroid of the largest solid blob, is far more robust.
    """
    local_std = ndimage.generic_filter(slice_img, np.std, size=texture_window)
    texture_mask = local_std > np.percentile(local_std, percentile)
    dilated = ndimage.binary_dilation(texture_mask, iterations=dilate_iter)
    filled = ndimage.binary_fill_holes(dilated)
    solid = ndimage.binary_erosion(filled, iterations=erode_iter)

    labeled, num = ndimage.label(solid)
    if num == 0:
        raise RuntimeError("No connected tissue region found; check texture_window/percentile.")
    sizes = ndimage.sum(solid, labeled, range(1, num + 1))
    largest = np.argmax(sizes) + 1
    largest_mask = (labeled == largest)
    com_y, com_x = ndimage.center_of_mass(largest_mask)
    return com_y, com_x, local_std


def radial_texture_profile(local_std, com_y, com_x, max_r_um, voxel_um, step_vox=4):
    yy, xx = np.mgrid[0:local_std.shape[0], 0:local_std.shape[1]]
    r = np.sqrt((yy - com_y) ** 2 + (xx - com_x) ** 2)

    radii_um, profile = [], []
    for rad_vox in range(0, int(max_r_um / voxel_um), step_vox):
        mask = (r >= rad_vox) & (r < rad_vox + step_vox)
        if mask.sum() > 0:
            profile.append(local_std[mask].mean())
            radii_um.append(rad_vox * voxel_um)
    return np.array(radii_um), np.array(profile), r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, help="Path to the raw microCT TIFF stack")
    ap.add_argument("--slice", type=int, required=True, help="Slice index to analyze")
    ap.add_argument("--voxel-um", type=float, required=True, help="Isotropic voxel size in micrometers")
    ap.add_argument("--max-radius-um", type=float, default=700, help="Max radius to profile (um)")
    ap.add_argument("--center-y", type=float, default=None, help="Manual center Y (voxels); auto-estimated if omitted")
    ap.add_argument("--center-x", type=float, default=None, help="Manual center X (voxels); auto-estimated if omitted")
    ap.add_argument("--candidate-radii", type=float, nargs="*", default=[],
                     help="Candidate boundary radii (um) to overlay on the check image, e.g. 90 190 250")
    ap.add_argument("--out-prefix", default=None, help="Output filename prefix (default: derived from --stack)")
    args = ap.parse_args()

    prefix = args.out_prefix or args.stack.split("/")[-1].rsplit(".", 1)[0]

    with tifffile.TiffFile(args.stack) as tif:
        slice_img = tif.pages[args.slice].asarray().astype(float)
    print(f"Loaded slice {args.slice}, shape={slice_img.shape}")

    if (args.center_y is None) != (args.center_x is None):
        ap.error("--center-y and --center-x must be provided together, or both omitted.")
    
    if args.center_y is None or args.center_x is None:
        com_y, com_x, local_std = estimate_center_solid_shape(slice_img)
        print(f"Auto-estimated center: Y={com_y:.1f}, X={com_x:.1f}")
    else:
        com_y, com_x = args.center_y, args.center_x
        local_std = ndimage.generic_filter(slice_img, np.std, size=9)
        print(f"Using manual center: Y={com_y:.1f}, X={com_x:.1f}")

    radii_um, profile, r_grid = radial_texture_profile(
        local_std, com_y, com_x, args.max_radius_um, args.voxel_um
    )

    # --- Plot 1: radial texture profile ---
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(radii_um, profile, linewidth=2)
    for cr in args.candidate_radii:
        ax.axvline(cr, linestyle='--', label=f'{cr} um')
    ax.set_xlabel("Distance from center (um)")
    ax.set_ylabel("Local texture (std dev)")
    ax.set_title(f"{prefix} slice {args.slice} - radial texture profile")
    if args.candidate_radii:
        ax.legend()
    ax.grid(True)
    plt.savefig(f"{prefix}_radial_profile.png", dpi=150, bbox_inches='tight')
    print(f"Saved {prefix}_radial_profile.png")

    # --- Plot 2: visual boundary check ---
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.imshow(slice_img, cmap='gray', vmin=np.percentile(slice_img, 1), vmax=np.percentile(slice_img, 99))
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(args.candidate_radii), 1)))
    for cr, color in zip(args.candidate_radii, colors):
        r_vox = cr / args.voxel_um
        circle = plt.Circle((com_x, com_y), r_vox, fill=False, color=color, linewidth=2, label=f'{cr} um')
        ax2.add_patch(circle)
    ax2.plot(com_x, com_y, '+', color='yellow', markersize=15)
    if args.candidate_radii:
        ax2.legend(loc='upper right', fontsize=10)
    ax2.set_title(f"{prefix} slice {args.slice} - candidate boundaries")
    plt.savefig(f"{prefix}_boundary_check.png", dpi=150, bbox_inches='tight')
    print(f"Saved {prefix}_boundary_check.png")

    print(f"\nCenter for reuse in other scripts: com_y={com_y:.1f}, com_x={com_x:.1f}")


if __name__ == "__main__":
    main()
