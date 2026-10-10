"""
check_alignment.py
Decide objectively whether a reference mask is stored in the same axis order
as the raw image stack, instead of judging an overlay by eye.

For a given slice it segments the tissue from the raw image (Otsu), then for
BOTH orientations of the mask it measures:
  * distance between the mask centroid and the tissue centroid
  * the fraction of mask voxels that fall inside the tissue
The correct orientation has a small centroid distance and a containment
fraction close to 1.

Usage
-----
    python3 check_alignment.py --raw stack.tif --mask mask.nrrd --z 500
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

_spec = importlib.util.spec_from_file_location(
    "cmv", str(Path(__file__).with_name("check_mask_volume.py")))
cmv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmv)


def otsu(img):
    img = img.astype(np.float64)
    lo, hi = np.percentile(img, [0.1, 99.9])
    hist, edges = np.histogram(np.clip(img, lo, hi), bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    m = np.cumsum(hist * centers)
    tot_w, tot_m = w[-1], m[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (tot_m * w / tot_w - m) ** 2 / (w * (tot_w - w))
    return centers[np.nanargmax(between)]


def tissue_mask(raw):
    """Segment the specimen robustly: smooth away reconstruction noise,
    threshold, then keep only the largest connected component."""
    sm = ndimage.median_filter(raw.astype(np.float32), size=5)
    t = otsu(sm)
    b = sm > t
    b = ndimage.binary_opening(b, np.ones((5, 5)))
    lab, n = ndimage.label(b)
    if n == 0:
        return b, t, 0
    sizes = ndimage.sum(b, lab, range(1, n + 1))
    b = lab == (int(np.argmax(sizes)) + 1)
    b = ndimage.binary_fill_holes(b)
    return b, t, n


def centroid(binary):
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return None
    return float(ys.mean()), float(xs.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--z", type=int, required=True)
    args = ap.parse_args()

    stack = tifffile.memmap(args.raw, mode="r")
    print(f"raw stack shape : {stack.shape}")
    raw = np.asarray(stack[args.z])
    print(f"raw slice shape : {raw.shape}")

    arr, sizes, spacing, fields = cmv.load_array(args.mask)
    arr = np.asarray(arr)
    # NRRD sizes are fastest-axis-first; numpy wants the reverse
    vol = arr.reshape(sizes[::-1])
    print(f"mask volume shape: {vol.shape}  (from NRRD sizes {sizes})")

    # locate the axis whose length matches the number of slices
    zaxis = next((i for i, n in enumerate(vol.shape) if n == stack.shape[0]), 0)
    vol = np.moveaxis(vol, zaxis, 0)
    msk = vol[args.z] > 0
    print(f"mask slice shape : {msk.shape}  (z axis was {zaxis})")

    tissue, t, ncomp = tissue_mask(raw)
    tc = centroid(tissue)
    frac = tissue.sum() / tissue.size
    print(f"\ntissue threshold : {t:.1f}   ({ncomp} components before "
          f"keeping the largest)")
    print(f"tissue centroid  : ({tc[0]:.1f}, {tc[1]:.1f})  "
          f"[{tissue.sum():,} px, {frac*100:.1f}% of the frame]")
    if frac > 0.45:
        print("  !! the segmentation covers nearly half the frame; it has "
              "probably latched onto background noise. Treat the centroid "
              "test below as unreliable and rely on the intensity test.")

    gmean = float(raw.mean())
    print(f"mean raw value   : {gmean:.1f} over the whole slice")

    best, results = None, []
    for name, m in (("as stored", msk), ("transposed", msk.T)):
        if m.shape != raw.shape:
            print(f"\n{name:>11}: shape {m.shape} != raw {raw.shape}, skipped")
            continue
        mc = centroid(m)
        if mc is None:
            print(f"\n{name:>11}: mask empty on this slice")
            continue
        d = float(np.hypot(mc[0] - tc[0], mc[1] - tc[1]))
        inside = float((m & tissue).sum()) / max(int(m.sum()), 1)
        print(f"\n{name:>11}: centroid ({mc[0]:.1f}, {mc[1]:.1f})  "
              f"[{m.sum():,} px]")
        print(f"             distance to tissue centroid : {d:.1f} px")
        print(f"             fraction inside tissue      : {inside:.3f}")
        mi = float(raw[m].mean())
        print(f"             mean raw value under mask   : {mi:.1f}  "
              f"({mi/gmean:.2f}x the slice mean)")
        results.append((name, d, inside, mi))

    print("\n" + "=" * 54)
    if len(results) == 2 and results[0][1] == results[1][1]:
        print("Both orientations are identical on this slice; the test cannot "
              "discriminate. Use a sample whose slices are not square.")
    elif results:
        best = min(results, key=lambda r: r[1])
        print(f"Closest to the tissue centroid : {best[0]} "
              f"({best[1]:.1f} px, {best[2]*100:.1f}% inside tissue)")
        bright = max(results, key=lambda r: r[3])
        print(f"Highest intensity under mask   : {bright[0]} "
              f"({bright[3]:.1f})")
        if len(results) == 1:
            print("Only one orientation matches the raw slice shape, so the "
                  "axis order is already determined by the shapes alone.")
        elif best[0] == bright[0]:
            print(f"Both tests agree: use '{best[0]}'.")
        else:
            print("The two tests disagree. Inspect the overlay manually "
                  "before depositing.")


if __name__ == "__main__":
    main()
