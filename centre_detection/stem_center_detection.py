#!/usr/bin/env python3
"""
stem_center_detection.py
=========================
Determines the stem centre (x, y) in full-frame voxel coordinates using
the "solid-shape" approach described in the manuscript:

  1. Compute a local-texture map (windowed std-dev) on a representative
     cross-section from the 8-bit GRID stack.
  2. Threshold at the 90th percentile -> rough binary mask.
  3. Dilate -> fill internal holes -> erode -> solid stem silhouette.
  4. Centroid of the largest connected component = stem centre.

This is more robust than simple intensity thresholding, which gave an
unstable centre in the noisiest volume (I-0) per the manuscript.

Usage
-----
    python3 stem_center_detection.py \
        --grid image_folder/Salicornia1_/Salicornia1_GRID-8bit.tif \
        --slice 500 \
        --save-plot Salicornia1_center_check.png
"""
import argparse
import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage import uniform_filter
from skimage.measure import label, regionprops
from skimage.morphology import dilation, erosion, disk
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNG
import matplotlib.pyplot as plt


def windowed_std(img, window=9):
    """Fast windowed standard deviation via box-filter identity:
    std = sqrt(E[x^2] - E[x]^2)."""
    img = img.astype(np.float64)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img**2, size=window)
    variance = np.clip(mean_sq - mean**2, 0, None)
    return np.sqrt(variance)


def find_stem_center(slice_img, window=9, dilate_radius=5, verbose=True):
    # 1. local texture map
    texture = windowed_std(slice_img, window=window)

    # 2. threshold at 90th percentile
    thresh = np.percentile(texture, 90)
    mask = texture >= thresh
    if verbose:
        print(f"Texture threshold (90th pct): {thresh:.2f}")
        print(f"Initial mask coverage: {mask.sum()/mask.size*100:.2f}%")

    # 3. dilate -> fill holes -> erode
    struct = disk(dilate_radius)
    mask_dilated = dilation(mask, struct)
    mask_filled = ndimage.binary_fill_holes(mask_dilated)
    mask_solid = erosion(mask_filled, struct)

    # 4. largest connected component -> centroid
    labeled = label(mask_solid)
    if labeled.max() == 0:
        raise RuntimeError("No connected component found — adjust --window / --dilate-radius.")

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    cy, cx = largest.centroid  # regionprops gives (row, col) = (y, x)

    if verbose:
        print(f"Found {len(regions)} connected component(s).")
        print(f"Largest component area: {largest.area} px")
        print(f"Centroid (x, y): ({cx:.2f}, {cy:.2f})")

    return cx, cy, mask_solid, texture


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="Path to GRID-8bit.tif stack")
    ap.add_argument("--slice", type=int, required=True, help="Representative slice index")
    ap.add_argument("--window", type=int, default=9, help="Local std-dev window size (default 9)")
    ap.add_argument("--dilate-radius", type=int, default=5, help="Structuring element radius")
    ap.add_argument("--save-plot", default=None, help="Optional path to save diagnostic PNG")
    args = ap.parse_args()

    print(f"Loading slice {args.slice} from {args.grid} ...")
    with tifffile.TiffFile(args.grid) as tif:
        img = tif.pages[args.slice].asarray()
    print(f"Slice shape: {img.shape}, dtype: {img.dtype}")

    cx, cy, mask_solid, texture = find_stem_center(
        img, window=args.window, dilate_radius=args.dilate_radius)

    print(f"\n=== RESULT ===")
    print(f"Stem centre (x, y) in full-frame voxel coords: ({cx:.2f}, {cy:.2f})")

    if args.save_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray'); axes[0].set_title("Raw slice")
        axes[1].imshow(texture, cmap='inferno'); axes[1].set_title("Local texture (std-dev)")
        axes[2].imshow(img, cmap='gray')
        axes[2].contour(mask_solid, colors='cyan', linewidths=1)
        axes[2].plot(cx, cy, 'y+', markersize=20, markeredgewidth=2)
        axes[2].set_title("Solid mask + centroid")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=150)
        print(f"Saved diagnostic plot to {args.save_plot}")


if __name__ == "__main__":
    main()