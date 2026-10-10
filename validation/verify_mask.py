"""
verify_mask.py
Overlay each reference mask on its raw image stack and confirm that the two
are in the same axis order and the same place.

Reads ImageJ-style stacks correctly (a single IFD holding every slice) by
memory-mapping the file instead of walking tif.pages, and finds the z axis of
the mask by matching the slice count rather than assuming a position.

Writes one PNG per sample and prints a numeric summary.
"""

import os

# Root of the data tree. Override with:  export SALICORNIA_DATA=/path
DATA = os.environ.get("SALICORNIA_DATA", ".")

import numpy as np
import nrrd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

BASE = DATA


def tissue_mask(raw):
    """Segment the specimen: median filter away reconstruction noise,
    Otsu threshold, keep the largest connected component, fill holes."""
    sm = ndimage.median_filter(raw.astype(np.float32), size=5)
    lo, hi = np.percentile(sm, [0.1, 99.9])
    hist, edges = np.histogram(np.clip(sm, lo, hi), bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w, m = np.cumsum(hist), np.cumsum(hist * centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        between = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    t = centers[np.nanargmax(between)]

    b = ndimage.binary_opening(sm > t, np.ones((5, 5)))
    lab, n = ndimage.label(b)
    if n:
        sizes = ndimage.sum(b, lab, range(1, n + 1))
        b = ndimage.binary_fill_holes(lab == int(np.argmax(sizes)) + 1)
    return b, float(t)


def centroid(binary):
    ys, xs = np.nonzero(binary)
    return (None if ys.size == 0 else (float(ys.mean()), float(xs.mean())))


def verify(tiff_path, mask_path, z_test, sample_name, is_npy=False):
    print(f"\n=== {sample_name} ===")

    stack = tifffile.memmap(tiff_path, mode="r")
    if stack.ndim != 3:
        print(f"  !! expected a 3D stack, got shape {stack.shape}")
        return
    nz = stack.shape[0]
    if not 0 <= z_test < nz:
        print(f"  !! z={z_test} is outside the stack (0..{nz - 1}); "
              f"using the middle slice instead")
        z_test = nz // 2
    raw = np.asarray(stack[z_test])
    print(f"  raw stack   : {stack.shape}   slice {z_test}: {raw.shape}")

    mask3d = np.load(mask_path) if is_npy else nrrd.read(mask_path)[0]
    print(f"  mask volume : {mask3d.shape}")

    # the z axis is the one whose length equals the number of slices
    cand = [i for i, n in enumerate(mask3d.shape) if n == nz]
    if not cand:
        print(f"  !! no mask axis has length {nz}; shapes are incompatible")
        return
    mask3d = np.moveaxis(mask3d, cand[0], 0)
    msk = np.asarray(mask3d[z_test]) > 0
    print(f"  mask slice  : {msk.shape}   (z axis was {cand[0]})")

    tissue, thr = tissue_mask(raw)
    tc = centroid(tissue)
    cover = tissue.sum() / tissue.size
    print(f"  tissue      : threshold {thr:.1f}, {tissue.sum():,} px "
          f"({cover * 100:.1f}% of frame), centroid "
          f"({tc[0]:.1f}, {tc[1]:.1f})")
    if cover > 0.45:
        print("  !! segmentation covers nearly half the frame; the centroid "
              "test is unreliable here, rely on the intensity ratio")

    gmean = float(raw.mean())
    options = [("as stored", msk)]
    if msk.shape != msk.T.shape:
        pass            # a transpose cannot match a non-square raw slice
    options.append(("transposed", msk.T))

    best, fig_slice, fig_label = None, None, None
    for label, m in options:
        if m.shape != raw.shape:
            print(f"  {label:>10} : shape {m.shape} does not match the raw "
                  f"slice, skipped")
            continue
        mc = centroid(m)
        if mc is None:
            print(f"  {label:>10} : mask is empty on this slice")
            continue
        d = float(np.hypot(mc[0] - tc[0], mc[1] - tc[1]))
        inside = float((m & tissue).sum()) / max(int(m.sum()), 1)
        ratio = float(raw[m].mean()) / gmean
        print(f"  {label:>10} : centroid ({mc[0]:.1f}, {mc[1]:.1f}), "
              f"{m.sum():,} px")
        print(f"               distance {d:6.1f} px | inside tissue "
              f"{inside:.3f} | intensity {ratio:.2f}x slice mean")
        score = (d, -ratio)
        if best is None or score < best[0]:
            best, fig_slice, fig_label = (score, label), m, label

    if fig_slice is None:
        print("  !! no orientation matched the raw slice shape")
        return
    print(f"  -> best orientation: {fig_label}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(raw, cmap="gray")
    ax.contour(fig_slice, colors="red", linewidths=1.5)
    ax.plot(tc[1], tc[0], "+", color="cyan", markersize=14, markeredgewidth=2,
            label="tissue centroid")
    mc = centroid(fig_slice)
    ax.plot(mc[1], mc[0], "x", color="yellow", markersize=14,
            markeredgewidth=2, label="mask centroid")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{sample_name}  z={z_test}  ({fig_label})")
    out = f"{sample_name}_alignment_check_z{z_test}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


if __name__ == "__main__":
    verify(f"{BASE}/Salicornia 1, Inowroclaw 0 mM/Salicornia_1_stack.tif",
           f"{BASE}/Salicornia 1, Inowroclaw 0 mM/S1_cc_labelmap_proper.nrrd",
           z_test=522, sample_name="I-0")

    verify(f"{BASE}/Salicornia 4, Inowroclaw 1000 mM/Salicornia_4_stack.tif",
           f"{BASE}/Salicornia 4, Inowroclaw 1000 mM/cc_lm_v5.nrrd",
           z_test=572, sample_name="I-1000")

    verify(f"{BASE}/Salicornia_5/Salicornia_5_stack.tif",
           f"{BASE}/Salicornia_5/C-0_cc_reference_mask.nrrd",
           z_test=454, sample_name="C-0")

    verify(f"{BASE}/Salicornia_8/Salicornia_8_stack.tif",
           f"{BASE}/Salicornia_8/cc_labelmap.nrrd",
           z_test=750, sample_name="C-1000")
